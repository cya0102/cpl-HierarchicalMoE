"""
CPL with Query-Guided Mixture of Experts (QGMoE).

This module extends the CPL baseline for weakly-supervised video temporal grounding
by replacing the single-FFN proposal predictor with a query-guided MoE that routes
different queries to specialized experts based on query semantics.

Key innovations:
1. QueryGuidedGating: routes queries to different experts based on query semantic content
2. QueryGuidedMoELinear: efficient MoE via weight merging for proposal prediction
3. MoE completion head: specialized semantic completion per expert
4. Load-balancing loss: encourages uniform expert utilization

Adapted from:
- CPL: Contrastive Proposal Learning for Weakly Supervised Temporal Grounding
- HierarchicalMoE: Graph Mixture of Experts with explicit diversity modeling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.transformer import DualTransformer
from models.cpl import SinusoidalPositionalEmbedding, _generate_mask


class QueryGuidedMoELinear(nn.Module):
    """
    Efficient MoE linear layer using weight merging.

    Instead of computing all expert outputs and mixing, this module merges
    expert weight matrices based on gating signals first, then applies a
    single matrix multiplication. This is memory-efficient for layers with
    small output dimensions (e.g., proposal prediction head).

    Args:
        in_features: input dimension
        out_features: output dimension
        num_experts: number of expert linear layers
    """

    def __init__(self, in_features, out_features, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        # All expert weights stored as a single tensor for efficient merging
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(num_experts, out_features))
        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x, gates):
        """
        Args:
            x: [B, in_features] or [B, L, in_features]
            gates: [B, num_experts]
        Returns:
            output: same shape as x but with out_features in last dim
        """
        # Merge expert weights based on gating: W_merged = sum(g_i * W_i)
        mixed_w = torch.einsum('be,eoi->boi', gates, self.weight)  # [B, out, in]
        mixed_b = torch.einsum('be,eo->bo', gates, self.bias)      # [B, out]

        if x.dim() == 2:
            # [B, in] @ [B, in, out] -> [B, out]
            return torch.bmm(x.unsqueeze(1), mixed_w.transpose(1, 2)).squeeze(1) + mixed_b
        elif x.dim() == 3:
            # [B, L, in] @ [B, in, out] -> [B, L, out]
            return torch.bmm(x, mixed_w.transpose(1, 2)) + mixed_b.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported input dimensions: {x.dim()}")


class QueryGuidedGating(nn.Module):
    """
    Query-semantic-conditioned gating network.

    Computes sparse top-k gating weights from query representation,
    routing different query types to different specialized experts.

    Args:
        hidden_size: dimension of the query representation
        num_experts: total number of experts
        top_k: number of experts to activate per sample
        dropout: dropout rate in gating network
    """

    def __init__(self, hidden_size, num_experts, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_experts)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_repr):
        """
        Args:
            query_repr: [bsz, hidden_size] - query semantic representation
        Returns:
            gates: [bsz, num_experts] - sparse gating weights (top-k non-zero)
        """
        logits = self.gate_network(query_repr)  # [bsz, num_experts]
        top_logits, top_indices = logits.topk(self.top_k, dim=-1)
        top_k_gates = self.softmax(top_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_indices, top_k_gates)
        return gates


class CPL_MoE(nn.Module):
    """
    CPL with Query-Guided Mixture of Experts.

    Extends the CPL baseline by:
    1. Replacing the single fc_gauss with a MoE of proposal predictors,
       each expert specializing in different query semantic patterns.
    2. Replacing the single fc_comp with a MoE of completion heads,
       each expert specializing in different semantic completion patterns.
    3. Adding a query-guided gating network that routes queries to experts
       based on the encoded query semantics.
    4. Adding a load-balancing loss to encourage uniform expert utilization.

    Args:
        config: dict containing model hyperparameters including:
            - frames_input_size, words_input_size, hidden_size
            - num_experts (default 4): number of MoE experts
            - moe_top_k (default 2): experts to activate per query
            - moe_loss_coef (default 0.1): load-balancing loss weight
            - Other CPL parameters (dropout, sigma, num_props, etc.)
    """

    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']
        self.sigma = config["sigma"]
        self.use_negative = config['use_negative']
        self.num_props = config['num_props']
        self.max_epoch = config['max_epoch']
        self.gamma = config['gamma']

        # MoE hyperparameters
        self.num_experts = config.get('num_experts', 4)
        self.moe_top_k = config.get('moe_top_k', 2)
        self.moe_loss_coef = config.get('moe_loss_coef', 0.1)

        hidden_size = config['hidden_size']

        # ===== Shared Feature Encoders (same as CPL) =====
        self.frame_fc = nn.Linear(config['frames_input_size'], hidden_size)
        self.word_fc = nn.Linear(config['words_input_size'], hidden_size)
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.pred_vec = nn.Parameter(torch.zeros(config['frames_input_size']).float(), requires_grad=True)

        # ===== Shared DualTransformer Backbone =====
        self.trans = DualTransformer(**config['DualTransformer'])

        # ===== Query-Guided MoE Components =====
        # Gating network: routes queries to experts based on semantics
        self.query_gating = QueryGuidedGating(
            hidden_size, self.num_experts, self.moe_top_k, self.dropout
        )

        # MoE proposal predictor (replaces single fc_gauss)
        # Uses weight merging for efficiency (output dim = num_props*2 is small)
        self.moe_gauss = QueryGuidedMoELinear(
            hidden_size, self.num_props * 2, self.num_experts
        )

        # MoE semantic completion heads (replaces single fc_comp)
        # Uses separate experts with sequential accumulation for memory efficiency
        self.moe_comp = nn.ModuleList([
            nn.Linear(hidden_size, self.vocab_size)
            for _ in range(self.num_experts)
        ])

        # ===== Positional Encoding =====
        self.word_pos_encoder = SinusoidalPositionalEmbedding(hidden_size, 0, 20)

    def _pool_query(self, enc_out, mask):
        """
        Masked mean pooling of encoded query features.

        Args:
            enc_out: [bsz, seq_len, hidden_size] - encoded query
            mask: [bsz, seq_len] - 1 for valid positions, 0 for padding
        Returns:
            pooled: [bsz, hidden_size] - pooled query representation
        """
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()  # [bsz, seq_len, 1]
            pooled = (enc_out * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            pooled = enc_out.mean(dim=1)
        return pooled

    def _moe_comp_forward(self, h, gates):
        """
        Memory-efficient MoE forward for the semantic completion head.
        Sequentially accumulates gated expert outputs to avoid materializing
        the full [B, num_experts, L, vocab_size] tensor.

        Args:
            h: [B, L, hidden] or [B, hidden] - transformer decoder output
            gates: [B, num_experts] - gating weights
        Returns:
            output: [B, L, vocab_size] or [B, vocab_size]
        """
        output = None
        for i, expert in enumerate(self.moe_comp):
            gate_weight = gates[:, i]  # [B]
            expert_out = expert(h)     # [B, L, V] or [B, V]
            # Expand gate weight to match expert output dims
            if h.dim() == 3:
                weighted = gate_weight.unsqueeze(-1).unsqueeze(-1) * expert_out
            else:
                weighted = gate_weight.unsqueeze(-1) * expert_out
            output = weighted if output is None else output + weighted
        return output

    @staticmethod
    def cv_squared(x):
        """Coefficient of variation squared for load balancing."""
        eps = 1e-10
        return x.float().std() / (x.float().mean() + eps)

    def balance_loss(self, gates):
        """
        Compute load-balancing loss encouraging uniform expert usage.
        Adapted from HierarchicalMoE's cv_squared approach.

        Args:
            gates: [bsz, num_experts] - gating weights
        Returns:
            loss: scalar tensor
        """
        importance = gates.sum(0)  # [num_experts] - total importance per expert
        return self.cv_squared(importance) * self.moe_loss_coef

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, **kwargs):
        bsz, n_frames, _ = frames_feat.shape

        # ===== Feature Encoding (same as CPL) =====
        pred_vec = self.pred_vec.view(1, 1, -1).expand(bsz, 1, -1)
        frames_feat = torch.cat([frames_feat, pred_vec], dim=1)
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask = _generate_mask(frames_feat, frames_len)

        words_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)

        # ===== Cross-Modal Interaction (first DualTransformer pass) =====
        # enc_out: self-attended query (word) features
        # h: cross-attended video features conditioned on query
        enc_out, h = self.trans(frames_feat, frames_mask, words_feat + words_pos, words_mask, decoding=1)

        # ===== Query-Guided Gating =====
        # Pool encoded query to get a single query semantic vector
        query_repr = self._pool_query(enc_out, words_mask)  # [bsz, hidden]
        # Compute sparse top-k gating weights
        gates = self.query_gating(query_repr)  # [bsz, num_experts]
        # Load-balancing loss to prevent expert collapse
        moe_loss = self.balance_loss(gates)

        # ===== MoE Proposal Prediction (replaces single fc_gauss) =====
        # Each expert proposes different Gaussian center/width based on query type
        gauss_param = torch.sigmoid(self.moe_gauss(h[:, -1], gates))
        gauss_param = gauss_param.view(bsz * self.num_props, 2)
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1]

        # ===== Downsample for Efficiency (same as CPL) =====
        props_len = n_frames // 4
        keep_idx = torch.linspace(0, n_frames - 1, steps=props_len).long()
        frames_feat = frames_feat[:, keep_idx]
        frames_mask = frames_mask[:, keep_idx]
        props_feat = frames_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz * self.num_props, props_len, -1)
        props_mask = frames_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz * self.num_props, -1)

        gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width)

        # ===== Semantic Completion with Word Masking (same as CPL) =====
        words_feat, masked_words = self._mask_words(words_feat, words_len, weights=weights)
        words_feat = words_feat + words_pos
        words_feat = words_feat[:, :-1]
        words_mask = words_mask[:, :-1]

        words_mask1 = words_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz * self.num_props, -1)
        words_id1 = words_id.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz * self.num_props, -1)
        words_feat1 = words_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz * self.num_props, words_mask1.size(1), -1)

        # Expand gates for the proposal dimension: [bsz, E] -> [bsz*num_props, E]
        gates_props = gates.unsqueeze(1).expand(bsz, self.num_props, -1) \
            .contiguous().view(bsz * self.num_props, -1)

        # ===== Second DualTransformer pass with Gaussian weighting =====
        pos_weight = gauss_weight / gauss_weight.max(dim=-1, keepdim=True)[0]
        _, h, attn_weight = self.trans(props_feat, props_mask, words_feat1, words_mask1,
                                       decoding=2, gauss_weight=pos_weight, need_weight=True)

        # MoE semantic completion (replaces single fc_comp)
        words_logit = self._moe_comp_forward(h, gates_props)

        # ===== Negative Proposal Mining (same as CPL, with MoE completion) =====
        if self.use_negative:
            neg_1_weight, neg_2_weight = self.negative_proposal_mining(
                props_len, gauss_center, gauss_width, kwargs['epoch'])

            _, neg_h_1 = self.trans(props_feat, props_mask, words_feat1, words_mask1,
                                    decoding=2, gauss_weight=neg_1_weight)
            neg_words_logit_1 = self._moe_comp_forward(neg_h_1, gates_props)

            _, neg_h_2 = self.trans(props_feat, props_mask, words_feat1, words_mask1,
                                    decoding=2, gauss_weight=neg_2_weight)
            neg_words_logit_2 = self._moe_comp_forward(neg_h_2, gates_props)

            _, ref_h = self.trans(frames_feat, frames_mask, words_feat, words_mask, decoding=2)
            ref_words_logit = self._moe_comp_forward(ref_h, gates)
        else:
            neg_words_logit_1 = None
            neg_words_logit_2 = None
            ref_words_logit = None

        return {
            'neg_words_logit_1': neg_words_logit_1,
            'neg_words_logit_2': neg_words_logit_2,
            'ref_words_logit': ref_words_logit,
            'words_logit': words_logit,
            'words_id': words_id,
            'words_mask': words_mask,
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,
            'moe_loss': moe_loss,
            'gates': gates,
        }

    def generate_gauss_weight(self, props_len, center, width):
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma

        w = 0.3989422804014327
        weight = w / width * torch.exp(-(weight - center) ** 2 / (2 * width ** 2))

        return weight / weight.max(dim=-1, keepdim=True)[0]

    def negative_proposal_mining(self, props_len, center, width, epoch):
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma / 2)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327
            y1 = w / w1 * torch.exp(-(pos - c) ** 2 / (2 * w1 ** 2))
            return y1 / y1.max(dim=-1, keepdim=True)[0]

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        left_width = torch.clamp(center - width / 2, min=0)
        left_center = left_width * min(epoch / self.max_epoch, 1) ** self.gamma * 0.5
        right_width = torch.clamp(1 - center - width / 2, min=0)
        right_center = 1 - right_width * min(epoch / self.max_epoch, 1) ** self.gamma * 0.5

        left_neg_weight = Gauss(weight, left_center, left_center)
        right_neg_weight = Gauss(weight, 1 - right_center, right_center)

        return left_neg_weight, right_neg_weight

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1)
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            if l < 1:
                continue
            p = weights[i, :l].cpu().numpy() if weights is not None else None
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1

        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words
