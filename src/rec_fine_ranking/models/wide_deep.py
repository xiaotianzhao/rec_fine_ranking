"""Wide & Deep ranking model (classical CTR baseline).

Body spec (per implementation plan):
    seq_pool = seq.mean(dim=1)               # (B, seq_dim)
    x = cat([nonseq, seq_pool], dim=-1)      # (B, D)
    wide = Linear(D, 1)(x)
    deep = MLP([D, 256, 128, 64, 1], ReLU + Dropout(0.1))(x)
    return (wide + deep).squeeze(-1)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseRanker, FeatureEncoder


class WideDeep(BaseRanker):
    def __init__(self, encoder: FeatureEncoder | None = None, dropout: float = 0.1):
        super().__init__(encoder)
        d = self.encoder.non_seq_dim + self.encoder.seq_dim

        self.wide = nn.Linear(d, 1)

        hidden_dims = [d, 256, 128, 64]
        deep_layers: list[nn.Module] = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            deep_layers.append(nn.Linear(in_dim, out_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout))
        deep_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.deep = nn.Sequential(*deep_layers)

    def body(self, nonseq: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        seq_pool = seq.mean(dim=1)                       # (B, seq_dim)
        x = torch.cat([nonseq, seq_pool], dim=-1)        # (B, D)
        wide = self.wide(x)                              # (B, 1)
        deep = self.deep(x)                              # (B, 1)
        return (wide + deep).squeeze(-1)                 # (B,)
