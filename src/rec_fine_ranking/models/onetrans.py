"""OneTrans (ByteDance, WWW 2026; arXiv 2510.26104).

Single-tower causal transformer over [seq_tokens ... | candidate_token].
The candidate is appended last so its representation attends to all history
tokens under a causal mask, and the head reads off the final position.

Architecture (per implementation plan):
    seq_tok  = Linear(seq_dim, d_model)(seq)              # (B, 50, d_model)
    cand_tok = Linear(non_seq_dim, d_model)(nonseq)[:, None]   # (B, 1, d_model)
    x = cat([seq_tok, cand_tok], dim=1)                   # (B, 51, d_model)
    mask = causal_mask(51)                                # upper-tri = -inf
    for layer in layers:  x = layer(x, attn_mask=mask)
    logit = Linear(d_model, 1)(x[:, -1]).squeeze(-1)      # (B,)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseRanker, FeatureEncoder


class _TransformerBlock(nn.Module):
    def __init__(self, d: int, h: int, ffn_mult: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * ffn_mult),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d * ffn_mult, d),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class OneTrans(BaseRanker):
    def __init__(
        self,
        encoder: FeatureEncoder | None = None,
        n_layers: int = 4,
        d_model: int = 320,
        n_heads: int = 8,
        ffn_mult: int = 4,
    ):
        super().__init__(encoder)
        self.d_model = d_model
        self.seq_proj = nn.Linear(self.encoder.seq_dim, d_model)
        self.cand_proj = nn.Linear(self.encoder.non_seq_dim, d_model)
        self.layers = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, ffn_mult) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, 1)
        # Cached causal mask (lazily built on first forward for correct device/dtype).
        self.register_buffer("_mask_cache", torch.empty(0), persistent=False)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((T, T), float("-inf"), device=device), diagonal=1
        )

    def body(self, nonseq: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        seq_tok = self.seq_proj(seq)                       # (B, 50, d_model)
        cand_tok = self.cand_proj(nonseq).unsqueeze(1)     # (B,  1, d_model)
        x = torch.cat([seq_tok, cand_tok], dim=1)          # (B, 51, d_model)

        T = x.size(1)
        if self._mask_cache.numel() != T * T or self._mask_cache.device != x.device:
            self._mask_cache = self._causal_mask(T, x.device)
        mask = self._mask_cache

        for layer in self.layers:
            x = layer(x, attn_mask=mask)

        logit = self.head(x[:, -1])                        # (B, 1) -- candidate token
        return logit.squeeze(-1)                           # (B,)
