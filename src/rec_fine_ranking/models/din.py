"""DIN — Deep Interest Network (Zhou et al., KDD 2018), on the Wide&Deep base.

Replaces the mean-pool of the behaviour sequence (used by wide_deep/dcn/deepfm/
rankmixer/unimixer) with DIN's **target-attention** pooling: the candidate item
acts as the query and scores each history step via a local activation unit, then
the history is weighted-summed. Everything else (wide linear + deep MLP towers) is
identical to WideDeep, so this isolates the *sequence-pooling* variable.

The candidate query is built from the same SEQUENCE_FIELDS embeddings as the
sequence (video_id, author_id, category_level_two/one, upload_type), sliced out of
`nonseq`, so query and keys live in the same space.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseRanker, FeatureEncoder, NON_SEQ_FEATURES, HOUR_DIM
from ..data.feature_config import feature_by_name, SEQUENCE_FIELDS


def cand_field_slices() -> list[tuple[int, int]]:
    """[start,end) slices into `nonseq` for the candidate's SEQUENCE_FIELDS, in seq order.

    Concatenating these slices yields a candidate vector aligned channel-for-channel
    with the pooled sequence (dim == encoder.seq_dim).
    """
    off, offsets = 0, {}
    for n in NON_SEQ_FEATURES:
        dim = HOUR_DIM if n == "hour_of_day" else feature_by_name(n).emb_dim
        offsets[n] = (off, off + dim); off += dim
    return [offsets[f] for f in SEQUENCE_FIELDS]


class DINAttentionPooling(nn.Module):
    """Target-attention pooling. query=candidate (B,dim), keys/values=history (B,L,dim).

    Activation unit scores each step from [c, h, c-h, c*h]; padded steps (all-zero
    embedding) are masked to weight 0. Raw weighted sum (no softmax), per the DIN
    paper, to preserve interest intensity.
    """

    def __init__(self, dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.act = nn.Sequential(
            nn.Linear(4 * dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, cand: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        L = hist.size(1)
        c = cand.unsqueeze(1).expand(-1, L, -1)                  # (B, L, dim)
        att_in = torch.cat([c, hist, c - hist, c * hist], dim=-1)  # (B, L, 4*dim)
        score = self.act(att_in).squeeze(-1)                     # (B, L)
        mask = hist.abs().sum(-1) > 0                            # (B, L) True = real step
        score = score.masked_fill(~mask, 0.0)
        return (score.unsqueeze(-1) * hist).sum(dim=1)           # (B, dim)


def _mlp(d: int, dropout: float) -> nn.Sequential:
    dims = [d, 256, 128, 64]
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(dims[-1], 1))
    return nn.Sequential(*layers)


class DIN(BaseRanker):
    def __init__(self, encoder: FeatureEncoder | None = None, dropout: float = 0.1,
                 att_hidden: int = 64):
        super().__init__(encoder)
        self._cand_slices = cand_field_slices()
        self.din = DINAttentionPooling(self.encoder.seq_dim, att_hidden, dropout)
        d = self.encoder.non_seq_dim + self.encoder.seq_dim
        self.wide = nn.Linear(d, 1)
        self.deep = _mlp(d, dropout)

    def body(self, nonseq: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        cand = torch.cat([nonseq[:, s:e] for (s, e) in self._cand_slices], dim=-1)  # (B, seq_dim)
        seq_pool = self.din(cand, seq)                          # (B, seq_dim)
        x = torch.cat([nonseq, seq_pool], dim=-1)               # (B, D)
        return (self.wide(x) + self.deep(x)).squeeze(-1)        # (B,)
