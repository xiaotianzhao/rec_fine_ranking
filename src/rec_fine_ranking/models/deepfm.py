"""DeepFM ranking model (classical CTR baseline).

Body spec (per implementation plan):
    seq_pool = seq.mean(dim=1)
    x = cat([nonseq, seq_pool], dim=-1)              # (B, D)
    # FM second-order interactions over F = D // K fields of K dims each.
    V = fm_proj(x).view(B, F, K)                     # (B, F, K)
    sum_sq = V.sum(dim=1) ** 2                       # (B, K)
    sq_sum = (V ** 2).sum(dim=1)                     # (B, K)
    fm_2nd = 0.5 * (sum_sq - sq_sum).sum(-1, keepdim=True)
    linear = Linear(D, 1)(x)
    deep   = MLP([D, 256, 128, 64, 1], ReLU + Dropout(0.1))(x)
    return (linear + fm_2nd + deep).squeeze(-1)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseRanker, FeatureEncoder


class DeepFM(BaseRanker):
    K: int = 8  # field embedding dim used by FM second-order

    def __init__(self, encoder: FeatureEncoder | None = None, dropout: float = 0.1):
        super().__init__(encoder)
        d = self.encoder.non_seq_dim + self.encoder.seq_dim
        K = self.K
        # Pad x to a multiple of K if needed so the FM reshape is exact.
        self._pad = (K - (d % K)) % K
        d_padded = d + self._pad
        self._d = d
        self._d_padded = d_padded
        self._F = d_padded // K

        self.fm_proj = nn.Linear(d_padded, self._F * K)
        self.linear = nn.Linear(d_padded, 1)

        hidden_dims = [d_padded, 256, 128, 64]
        deep_layers: list[nn.Module] = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            deep_layers.append(nn.Linear(in_dim, out_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout))
        deep_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.deep_mlp = nn.Sequential(*deep_layers)

    def body(self, nonseq: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        seq_pool = seq.mean(dim=1)                              # (B, seq_dim)
        x = torch.cat([nonseq, seq_pool], dim=-1)               # (B, D)
        if self._pad:
            x = F.pad(x, (0, self._pad))                        # (B, D_padded)
        B = x.size(0)
        V = self.fm_proj(x).view(B, self._F, self.K)            # (B, F, K)
        sum_sq = V.sum(dim=1) ** 2                              # (B, K)
        sq_sum = (V ** 2).sum(dim=1)                            # (B, K)
        fm_2nd = 0.5 * (sum_sq - sq_sum).sum(dim=-1, keepdim=True)  # (B, 1)
        linear = self.linear(x)                                 # (B, 1)
        deep = self.deep_mlp(x)                                 # (B, 1)
        return (linear + fm_2nd + deep).squeeze(-1)             # (B,)
