"""Field-aware DeepFM — canonical FM over per-feature embedding fields.

Contrast with `deepfm.py` (the plan's simplified variant), which runs the FM
sum-square trick over arbitrary K-dim chunks of a single re-projection of the
concatenated embedding vector. That loses per-feature semantics and, on RecFlow,
made DeepFM underperform Wide&Deep (see execution-log Finding F1).

Here each feature's OWN embedding — every non-sequence feature (user_id, video_id,
…, hour_of_day) and every pooled sequence field — is projected to a common K-dim
latent and treated as one FM field, so the second-order term Σ_{i<j}<v_i,v_j>
captures genuine pairwise *feature* interactions.

The linear + deep towers are identical to WideDeep, so the only difference from
W&D is this field-aware FM term — isolating the FM-implementation variable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseRanker, FeatureEncoder, NON_SEQ_FEATURES, HOUR_DIM
from ..data.feature_config import feature_by_name, SEQUENCE_FIELDS


class DeepFMField(BaseRanker):
    K: int = 16  # common FM latent dim per field

    def __init__(self, encoder: FeatureEncoder | None = None, dropout: float = 0.1):
        super().__init__(encoder)
        K = self.K
        # Slices into the concatenated nonseq / pooled-seq vectors — one per feature field.
        self._nonseq_slices: list[tuple[int, int]] = []
        off = 0
        for n in NON_SEQ_FEATURES:
            dim = HOUR_DIM if n == "hour_of_day" else feature_by_name(n).emb_dim
            self._nonseq_slices.append((off, off + dim)); off += dim
        self._seq_slices: list[tuple[int, int]] = []
        off = 0
        for f in SEQUENCE_FIELDS:
            dim = feature_by_name(f).emb_dim
            self._seq_slices.append((off, off + dim)); off += dim
        self.num_fields = len(self._nonseq_slices) + len(self._seq_slices)

        # One projection per field → K dims (fields have heterogeneous source dims).
        self.field_proj = nn.ModuleList(
            [nn.Linear(e - s, K) for (s, e) in (self._nonseq_slices + self._seq_slices)]
        )

        d = self.encoder.non_seq_dim + self.encoder.seq_dim
        self.linear = nn.Linear(d, 1)
        hidden = [d, 256, 128, 64]
        layers: list[nn.Module] = []
        for a, b in zip(hidden[:-1], hidden[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden[-1], 1))
        self.deep = nn.Sequential(*layers)

    def body(self, nonseq: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        seq_pool = seq.mean(dim=1)                                   # (B, seq_dim)
        fields = [self.field_proj[i](nonseq[:, s:e])
                  for i, (s, e) in enumerate(self._nonseq_slices)]
        base = len(self._nonseq_slices)
        fields += [self.field_proj[base + j](seq_pool[:, s:e])
                   for j, (s, e) in enumerate(self._seq_slices)]
        V = torch.stack(fields, dim=1)                              # (B, F, K)
        fm = 0.5 * ((V.sum(dim=1) ** 2) - (V ** 2).sum(dim=1)).sum(dim=-1, keepdim=True)  # (B,1)
        x = torch.cat([nonseq, seq_pool], dim=-1)                   # (B, D)
        return (self.linear(x) + fm + self.deep(x)).squeeze(-1)     # (B,)
