"""HyFormer (Huang et al., 2026; arXiv 2601.12681).

Hybrid two-stream transformer with bidirectional cross-attention between a small
set of learnable-style feature queries and the behaviour sequence. Each layer runs
two cross-attention passes:

    * Query Decoding  -- feat queries attend into the seq stream (q2s projects seq
      keys/values down to the seq space the queries are routed to).
    * Query Boosting  -- the seq stream attends back into the feat queries (s2q
      projects feat keys/values up to the feat space).

Both directions are residual and each is followed by a position-wise FFN.

Architecture (per implementation plan):
    seq_t  = Linear(seq_dim, d_seq)(seq)                            # (B, 50, d_seq)
    feat_t = Linear(non_seq_dim, d_feat)(nonseq)[:, None]
                 .expand(-1, n_query, -1)                           # (B, n_query, d_feat)
    q2s = Linear(d_feat, d_seq); s2q = Linear(d_seq, d_feat)
    for i in range(n_layers):
        # Query Decoding: feat queries cross-attend to seq (in d_seq space)
        feat_t = feat_t + q_dec_out(cross_attn(q=q2s(feat_t), k=seq_t, v=seq_t))
        feat_t = feat_t + feat_ffn(feat_t)
        # Query Boosting: seq cross-attends back to feat (in d_feat space).
        # Only on non-final layers, since the boosted seq_t feeds the *next*
        # decode; the head reads feat_t, so a final-layer boost would be dead.
        if i < n_layers - 1:
            seq_t = seq_t + s_boost_out(cross_attn(q=s2q(seq_t), k=feat_t, v=feat_t))
            seq_t = seq_t + seq_ffn(seq_t)
    pooled = feat_t.mean(1)                                         # (B, d_feat)
    logit  = Linear(d_feat, 1)(pooled).squeeze(-1)                  # (B,)

nn.MultiheadAttention requires q/k/v to share an embed dim, so the query stream is
projected into the key/value stream's space (q2s / s2q), attention runs there, and
the result is projected back to the residual stream's native dim.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseRanker, FeatureEncoder


class _CrossAttnBlock(nn.Module):
    """One residual cross-attention + FFN over a `dst` stream querying a `src` stream.

    Attention is computed in `d_src` space: the dst residual stream (dim `d_dst`) is
    projected to `d_src` to form queries that match the src keys/values, and the
    attention output is projected back to `d_dst` before the residual add.
    """

    def __init__(self, d_dst: int, d_src: int, n_heads: int, ffn_mult: int):
        super().__init__()
        self.q_proj = nn.Linear(d_dst, d_src)
        self.attn = nn.MultiheadAttention(d_src, n_heads, batch_first=True, dropout=0.1)
        self.out_proj = nn.Linear(d_src, d_dst)
        self.norm1 = nn.LayerNorm(d_dst)
        self.norm2 = nn.LayerNorm(d_dst)
        self.ffn = nn.Sequential(
            nn.Linear(d_dst, d_dst * ffn_mult),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_dst * ffn_mult, d_dst),
            nn.Dropout(0.1),
        )

    def forward(self, dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(self.norm1(dst))                  # (B, T_dst, d_src)
        a, _ = self.attn(q, src, src, need_weights=False)  # (B, T_dst, d_src)
        dst = dst + self.out_proj(a)
        dst = dst + self.ffn(self.norm2(dst))
        return dst


class HyFormer(BaseRanker):
    def __init__(
        self,
        encoder: FeatureEncoder | None = None,
        n_layers: int = 2,
        d_seq: int = 64,
        d_feat: int = 128,
        n_heads: int = 4,
        n_query: int = 4,
        ffn_mult: int = 4,
    ):
        super().__init__(encoder)
        self.d_seq = d_seq
        self.d_feat = d_feat
        self.n_query = n_query
        self.seq_proj = nn.Linear(self.encoder.seq_dim, d_seq)
        self.feat_proj = nn.Linear(self.encoder.non_seq_dim, d_feat)
        # Query Decoding: feat queries (d_feat) attend into the seq stream (d_seq).
        self.query_decode = nn.ModuleList(
            [_CrossAttnBlock(d_feat, d_seq, n_heads, ffn_mult) for _ in range(n_layers)]
        )
        # Query Boosting: seq stream (d_seq) attends back into feat queries (d_feat).
        # One fewer block than decode: each boost refines seq_t for the *next* layer's
        # decode, so a final-layer boost would never be read by the feat-pooled head.
        self.query_boost = nn.ModuleList(
            [_CrossAttnBlock(d_seq, d_feat, n_heads, ffn_mult) for _ in range(max(n_layers - 1, 0))]
        )
        self.head = nn.Linear(d_feat, 1)

    def body(self, nonseq: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        seq_t = self.seq_proj(seq)                                   # (B, 50, d_seq)
        feat_t = self.feat_proj(nonseq).unsqueeze(1).expand(
            -1, self.n_query, -1
        )                                                            # (B, n_query, d_feat)

        for i, decode in enumerate(self.query_decode):
            feat_t = decode(feat_t, seq_t)   # feat queries attend to seq
            if i < len(self.query_boost):    # boost refines seq_t for the next decode
                seq_t = self.query_boost[i](seq_t, feat_t)

        pooled = feat_t.mean(1)                                      # (B, d_feat)
        logit = self.head(pooled)                                    # (B, 1)
        return logit.squeeze(-1)                                     # (B,)
