"""RankMixer (Zhu et al., 2025; arXiv 2507.15551).

Tokenises the flat feature vector (non-seq fields concatenated with the
mean-pooled behaviour sequence) into ``n_tokens`` tokens of width ``d_token``,
then alternates a parameter-free token-mixing permutation with a per-token FFN.

Architecture (per implementation plan):
    x = cat([nonseq, seq.mean(1)], dim=-1)                 # (B, D)
    x = Linear(D, n_tokens * d_token)(x).view(B, T, d_token)
    for _ in range(n_layers):
        x = token_mix(x)                                   # parameter-free
        x = x + per_token_ffn(x)                           # per-token weights
    logit = Linear(n_tokens * d_token, 1)(x.flatten(1)).squeeze(-1)   # (B,)

``token_mix`` is the MLP-Mixer permutation
``(B, T, T*Dh) -> (B, T, T, Dh) -> permute(0, 2, 1, 3) -> (B, T, T*Dh)`` where
``T*Dh = d_token`` and ``Dh = d_token // n_tokens``. It moves head ``h`` of token
``t`` to position ``(t, h)``, mixing information across tokens without any
learnable parameters. This requires ``d_token`` to be divisible by ``n_tokens``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseRanker, FeatureEncoder


class _PerTokenFFN(nn.Module):
    """Independent two-layer FFN applied to each of the ``n_tokens`` tokens.

    Implemented with batched weights (one weight matrix per token) so that no
    token shares parameters with another, matching the per-token capacity of the
    paper while keeping the forward pass a pair of ``einsum`` ops.
    """

    def __init__(self, n_tokens: int, d_token: int, ffn_mult: int):
        super().__init__()
        hidden = ffn_mult * d_token
        self.norm = nn.LayerNorm(d_token)
        self.w1 = nn.Parameter(torch.empty(n_tokens, d_token, hidden))
        self.b1 = nn.Parameter(torch.zeros(n_tokens, hidden))
        self.w2 = nn.Parameter(torch.empty(n_tokens, hidden, d_token))
        self.b2 = nn.Parameter(torch.zeros(n_tokens, d_token))
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)                                   # (B, T, d_token)
        h = torch.einsum("btd,tdh->bth", h, self.w1) + self.b1
        h = self.drop(self.act(h))
        h = torch.einsum("bth,thd->btd", h, self.w2) + self.b2
        return self.drop(h)


class RankMixer(BaseRanker):
    def __init__(
        self,
        encoder: FeatureEncoder | None = None,
        n_tokens: int = 16,
        d_token: int = 64,
        n_layers: int = 3,
        ffn_mult: int = 4,
    ):
        super().__init__(encoder)
        if d_token % n_tokens != 0:
            raise ValueError(
                f"d_token ({d_token}) must be divisible by n_tokens ({n_tokens}) "
                "for the token_mix permutation"
            )
        self.n_tokens = n_tokens
        self.d_token = d_token
        self.head_dim = d_token // n_tokens  # Dh in the token_mix permutation

        in_dim = self.encoder.non_seq_dim + self.encoder.seq_dim
        self.tokenize = nn.Linear(in_dim, n_tokens * d_token)
        self.ffns = nn.ModuleList(
            [_PerTokenFFN(n_tokens, d_token, ffn_mult) for _ in range(n_layers)]
        )
        self.head = nn.Linear(n_tokens * d_token, 1)

    def _token_mix(self, x: torch.Tensor) -> torch.Tensor:
        """Parameter-free MLP-Mixer permutation across tokens.

        (B, T, T*Dh) -> (B, T, T, Dh) -> permute(0, 2, 1, 3) -> (B, T, T*Dh)
        """
        B = x.size(0)
        T, Dh = self.n_tokens, self.head_dim
        x = x.view(B, T, T, Dh)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, T * Dh)

    def body(self, nonseq: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        B = nonseq.size(0)
        x = torch.cat([nonseq, seq.mean(dim=1)], dim=-1)   # (B, D)
        x = self.tokenize(x).view(B, self.n_tokens, self.d_token)

        for ffn in self.ffns:
            x = self._token_mix(x)                         # parameter-free
            x = x + ffn(x)                                 # per-token FFN

        logit = self.head(x.flatten(1))                    # (B, 1)
        return logit.squeeze(-1)                           # (B,)
