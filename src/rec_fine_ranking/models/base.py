"""Shared FeatureEncoder + BaseRanker.

All seven models share these embedding tables. Each model only implements the body
between (non_seq, seq) tensors and the final logit.
"""
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
from ..data.feature_config import FEATURES, SEQUENCE_FIELDS, SEQUENCE_LEN, feature_by_name

# Order of non-sequence features in the concatenated output (kept stable for reproducibility).
NON_SEQ_FEATURES: Tuple[str, ...] = tuple(f.name for f in FEATURES) + ("hour_of_day",)
HOUR_VOCAB, HOUR_DIM = 24, 4

class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tables = nn.ModuleDict()
        for f in FEATURES:
            self.tables[f.name] = nn.Embedding(f.vocab_size + 1, f.emb_dim, padding_idx=0)
        self.tables["hour_of_day"] = nn.Embedding(HOUR_VOCAB, HOUR_DIM)
        # init: small std for hash buckets (cold-start safety), larger for small vocabs
        for f in FEATURES:
            std = 0.01 if f.kind.value == "hash" else 0.05
            nn.init.normal_(self.tables[f.name].weight, mean=0.0, std=std)
            nn.init.zeros_(self.tables[f.name].weight[0])  # padding row
        nn.init.normal_(self.tables["hour_of_day"].weight, mean=0.0, std=0.05)

    @property
    def non_seq_dim(self) -> int:
        return sum(feature_by_name(n).emb_dim for n in NON_SEQ_FEATURES if n != "hour_of_day") + HOUR_DIM

    @property
    def seq_dim(self) -> int:
        return sum(feature_by_name(n).emb_dim for n in SEQUENCE_FIELDS)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        nonseq_parts = []
        for name in NON_SEQ_FEATURES:
            t = self.tables[name](batch[name])
            nonseq_parts.append(t)
        nonseq = torch.cat(nonseq_parts, dim=-1)  # (B, non_seq_dim)
        seq_parts = [self.tables[f](batch[f"seq_{f}"]) for f in SEQUENCE_FIELDS]
        seq = torch.cat(seq_parts, dim=-1)  # (B, 50, seq_dim)
        return nonseq, seq


class BaseRanker(nn.Module):
    """Subclass and implement `body(nonseq, seq) -> logit (B,)`."""
    def __init__(self, encoder: FeatureEncoder | None = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else FeatureEncoder()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        nonseq, seq = self.encoder(batch)
        return self.body(nonseq, seq)

    def body(self, nonseq: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError

    def count_params(self) -> Dict[str, int]:
        backbone = sum(p.numel() for n, p in self.named_parameters() if not n.startswith("encoder."))
        embedding = sum(p.numel() for n, p in self.named_parameters() if n.startswith("encoder."))
        return {"backbone": backbone, "embedding": embedding, "total": backbone + embedding}


# -------- testing utilities (only imported by tests; safe to live here) --------
def _toy_batch(B: int = 4, device: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
    """Synthetic batch matching the dataset/collate schema."""
    g = torch.Generator(device="cpu").manual_seed(0)
    def ri(hi, shape=(B,)): return torch.randint(0, hi, shape, generator=g, dtype=torch.long, device=device)
    batch = {
        "user_id": ri(50_000), "device_id": ri(50_000),
        "age": ri(8), "gender": ri(3), "province": ri(60),
        "category_level_one": ri(120), "category_level_two": ri(700),
        "upload_type": ri(30),
        "video_id":  torch.randint(1, 1_048_576, (B,),  generator=g, dtype=torch.long, device=device),
        "author_id": torch.randint(1,   262_144, (B,),  generator=g, dtype=torch.long, device=device),
        "duration": ri(64), "hour_of_day": ri(24),
        "seq_video_id":           torch.randint(1, 1_048_576, (B, 50), generator=g, dtype=torch.long, device=device),
        "seq_author_id":          torch.randint(1,   262_144, (B, 50), generator=g, dtype=torch.long, device=device),
        "seq_category_level_two": torch.randint(0,       700, (B, 50), generator=g, dtype=torch.long, device=device),
        "seq_category_level_one": torch.randint(0,       120, (B, 50), generator=g, dtype=torch.long, device=device),
        "seq_upload_type":        torch.randint(0,        30, (B, 50), generator=g, dtype=torch.long, device=device),
        "label": torch.randint(0, 2, (B,), generator=g, dtype=torch.float, device=device),
    }
    return batch
