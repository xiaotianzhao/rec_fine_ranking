"""DCN-v2 (Wang et al., 2021) on the Wide&Deep+DIN base.

Two upgrades over the `dcn` (v1) baseline in this repo:
  1. **Matrix cross** — each cross layer uses a full weight matrix:
        x_{l+1} = x0 ⊙ (W_l x_l + b_l) + x_l,  W_l ∈ R^{D×D}
     vs v1's vector/scalar interaction `x0 ⊙ (x_l·w_l) + x_l`. This is strictly
     more expressive (DCN-v2's core contribution).
  2. **DIN target-attention pooling** of the behaviour sequence (instead of mean
     pool), reusing `DINAttentionPooling` — so the candidate item drives which
     history steps matter.

Parallel structure (DCN-v2 paper): cross network and deep MLP run in parallel on
the same x0, their outputs concatenated into the final logit.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseRanker, FeatureEncoder
from .din import DINAttentionPooling, cand_field_slices


class _CrossNetV2(nn.Module):
    """L layers of x_{l+1} = x0 ⊙ (W_l x_l + b_l) + x_l (full-matrix cross)."""

    def __init__(self, d: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for lin in self.layers:
            x = x0 * lin(x) + x
        return x


class DCNv2(BaseRanker):
    def __init__(self, encoder: FeatureEncoder | None = None, n_cross: int = 3,
                 dropout: float = 0.1, att_hidden: int = 64):
        super().__init__(encoder)
        self._cand_slices = cand_field_slices()
        self.din = DINAttentionPooling(self.encoder.seq_dim, att_hidden, dropout)
        d = self.encoder.non_seq_dim + self.encoder.seq_dim
        self.cross = _CrossNetV2(d, n_cross)
        dims = [d, 256, 128, 64]
        deep: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            deep += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.deep = nn.Sequential(*deep)
        self.head = nn.Linear(d + dims[-1], 1)   # [cross_out ‖ deep_out] → logit

    def body(self, nonseq: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        cand = torch.cat([nonseq[:, s:e] for (s, e) in self._cand_slices], dim=-1)
        seq_pool = self.din(cand, seq)                          # (B, seq_dim)
        x0 = torch.cat([nonseq, seq_pool], dim=-1)              # (B, D)
        cross_out = self.cross(x0)                              # (B, D)
        deep_out = self.deep(x0)                                # (B, 64)
        out = self.head(torch.cat([cross_out, deep_out], dim=-1))
        return out.squeeze(-1)                                  # (B,)
