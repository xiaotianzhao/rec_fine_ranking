"""DCN (Deep & Cross Network) ranking model.

Body spec (per implementation plan):
    seq_pool = seq.mean(dim=1)
    x0 = cat([nonseq, seq_pool], dim=-1)              # (B, D)
    # Cross network, L = 3 layers:
    #   x_{l+1} = x0 * (x_l @ w_l) + b_l + x_l
    deep = MLP([D, 256, 128, 64], dropout=0.1)(x0)
    logit = Linear(D + 64, 1)(cat([x_L, deep], -1))
    return logit.squeeze(-1)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import BaseRanker, FeatureEncoder


class DCN(BaseRanker):
    def __init__(
        self,
        encoder: FeatureEncoder | None = None,
        n_cross: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(encoder)
        d = self.encoder.non_seq_dim + self.encoder.seq_dim
        self.n_cross = n_cross

        # Cross network parameters: per-layer w (D,) and b (D,).
        self.cross_w = nn.ParameterList(
            [nn.Parameter(torch.zeros(d)) for _ in range(n_cross)]
        )
        self.cross_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(d)) for _ in range(n_cross)]
        )
        # Kaiming-uniform init for cross weights (treat each w as a fan_in=D linear).
        bound = 1.0 / math.sqrt(d)
        for w in self.cross_w:
            nn.init.uniform_(w, -bound, bound)

        # Deep MLP: D -> 256 -> 128 -> 64.
        self.deep_mlp = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
        )

        # Final projection over [cross output (D), deep output (64)].
        self.final = nn.Linear(d + 64, 1)

    def body(self, nonseq: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        seq_pool = seq.mean(dim=1)                            # (B, seq_dim)
        x0 = torch.cat([nonseq, seq_pool], dim=-1)            # (B, D)

        x = x0
        for layer in range(self.n_cross):
            cross_proj = x @ self.cross_w[layer]              # (B,)
            x = x0 * cross_proj.unsqueeze(-1) + self.cross_b[layer] + x

        deep = self.deep_mlp(x0)                              # (B, 64)
        logit = self.final(torch.cat([x, deep], dim=-1))      # (B, 1)
        return logit.squeeze(-1)                              # (B,)
