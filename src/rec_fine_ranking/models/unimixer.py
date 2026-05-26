"""UniMixer (Ha et al., 2026; arXiv 2604.00590).

Token-mixing ranker that splits the flattened feature vector into ``n_blocks``
blocks of width ``block_dim`` and alternates two complementary mixing operations
per layer:

Architecture (per implementation plan):
    x = cat([nonseq, seq.mean(1)], dim=-1)                       # (B, D)
    x = Linear(D, n_blocks * block_dim)(x).view(B, n_blocks, block_dim)
    for layer in layers:
        # local mixing: independent block_dim x block_dim weight per block
        x_local  = einsum("bnk,nkj->bnj", x, W_B)                # (B, n_blocks, block_dim)
        # global mixing: doubly-stochastic W_G from Sinkhorn-Knopp on learnable logits
        W_G      = sinkhorn_knopp(global_logits, n_iter=3)       # (n_blocks, n_blocks)
        x_global = einsum("nm,bmk->bnk", W_G, x_local)           # (B, n_blocks, block_dim)
        x = x + per_block_ffn(x_global)                          # hidden = block_dim * hidden_mult
    logit = Linear(n_blocks * block_dim, 1)(x.flatten(1)).squeeze(-1)   # (B,)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseRanker, FeatureEncoder


def sinkhorn_knopp(logits: torch.Tensor, n_iter: int = 3) -> torch.Tensor:
    """Project ``logits`` to a (near) doubly-stochastic matrix.

    Starts from ``softmax(logits)`` (rows already sum to 1) and alternates row /
    column normalisation for ``n_iter`` iterations. Fully differentiable; a small
    epsilon keeps the column step numerically stable.

    Args:
        logits: ``(n, n)`` learnable logit matrix.
        n_iter: number of Sinkhorn-Knopp iterations.

    Returns:
        ``(n, n)`` matrix whose rows and columns approximately sum to 1.
    """
    p = torch.softmax(logits, dim=-1)
    eps = 1e-8
    for _ in range(n_iter):
        p = p / (p.sum(dim=0, keepdim=True) + eps)  # column normalize
        p = p / (p.sum(dim=1, keepdim=True) + eps)  # row normalize
    return p


class _PerBlockFFN(nn.Module):
    """Independent two-layer FFN applied to each block via batched matmul."""

    def __init__(self, n_blocks: int, block_dim: int, hidden_mult: int):
        super().__init__()
        hidden = block_dim * hidden_mult
        self.w1 = nn.Parameter(torch.empty(n_blocks, block_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(n_blocks, hidden))
        self.w2 = nn.Parameter(torch.empty(n_blocks, hidden, block_dim))
        self.b2 = nn.Parameter(torch.zeros(n_blocks, block_dim))
        self.act = nn.GELU()
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("bnk,nkj->bnj", x, self.w1) + self.b1
        h = self.act(h)
        return torch.einsum("bnj,njk->bnk", h, self.w2) + self.b2


class _MixerLayer(nn.Module):
    def __init__(self, n_blocks: int, block_dim: int, hidden_mult: int):
        super().__init__()
        # Per-block local-mixing weights: (n_blocks, block_dim, block_dim).
        self.w_local = nn.Parameter(torch.empty(n_blocks, block_dim, block_dim))
        # Learnable logits for the global doubly-stochastic mixing matrix.
        self.global_logits = nn.Parameter(torch.zeros(n_blocks, n_blocks))
        self.norm = nn.LayerNorm(block_dim)
        self.ffn = _PerBlockFFN(n_blocks, block_dim, hidden_mult)
        nn.init.xavier_uniform_(self.w_local)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_local = torch.einsum("bnk,nkj->bnj", x, self.w_local)
        w_g = sinkhorn_knopp(self.global_logits, n_iter=3)
        x_global = torch.einsum("nm,bmk->bnk", w_g, x_local)
        return x + self.ffn(self.norm(x_global))


class UniMixer(BaseRanker):
    def __init__(
        self,
        encoder: FeatureEncoder | None = None,
        n_layers: int = 3,
        n_blocks: int = 8,
        block_dim: int = 24,
        hidden_mult: int = 2,
    ):
        super().__init__(encoder)
        self.n_blocks = n_blocks
        self.block_dim = block_dim
        in_dim = self.encoder.non_seq_dim + self.encoder.seq_dim
        self.in_proj = nn.Linear(in_dim, n_blocks * block_dim)
        self.layers = nn.ModuleList(
            [_MixerLayer(n_blocks, block_dim, hidden_mult) for _ in range(n_layers)]
        )
        self.head = nn.Linear(n_blocks * block_dim, 1)

    def body(self, nonseq: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        x = torch.cat([nonseq, seq.mean(dim=1)], dim=-1)        # (B, D)
        x = self.in_proj(x).view(-1, self.n_blocks, self.block_dim)  # (B, n_blocks, block_dim)
        for layer in self.layers:
            x = layer(x)
        logit = self.head(x.flatten(1))                        # (B, 1)
        return logit.squeeze(-1)                               # (B,)
