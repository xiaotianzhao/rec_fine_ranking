"""Single source of truth for feature schema.

Each feature declares: name, kind (vocab/hash/bucketize), vocab_size, emb_dim, source column.
Sequence features reuse the same vocab/embedding tables as their non-sequence counterparts.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

class FeatureKind(str, Enum):
    VOCAB     = "vocab"      # small cardinality, true id-to-index lookup
    HASH      = "hash"       # high cardinality, FNV-1a bucketing
    BUCKETIZE = "bucketize"  # continuous → discrete via log-spaced bins

@dataclass(frozen=True)
class Feature:
    name: str
    kind: FeatureKind
    vocab_size: int
    emb_dim: int
    source_col: str = ""    # column in the raw feather; defaults to `name`

    def __post_init__(self):
        object.__setattr__(self, "source_col", self.source_col or self.name)

FEATURES: tuple[Feature, ...] = (
    Feature("user_id",            FeatureKind.VOCAB,      50_000, 16),
    Feature("device_id",          FeatureKind.VOCAB,      50_000, 16),
    Feature("age",                FeatureKind.VOCAB,         16,  8),
    Feature("gender",             FeatureKind.VOCAB,          4,  4),
    Feature("province",           FeatureKind.VOCAB,         80,  8),
    Feature("category_level_one", FeatureKind.VOCAB,        128,  8),
    Feature("category_level_two", FeatureKind.VOCAB,        768,  8),
    Feature("upload_type",        FeatureKind.VOCAB,         32,  4),
    Feature("video_id",           FeatureKind.HASH,    1_048_576, 16),
    Feature("author_id",          FeatureKind.HASH,      262_144, 16),
    Feature("duration",           FeatureKind.BUCKETIZE,     64,  8),
)

# 50-step long-effective-view sequence; reuses video_id/author_id/category/upload tables.
SEQUENCE_FIELDS: tuple[str, ...] = (
    "video_id", "author_id", "category_level_two",
    "category_level_one", "upload_type",
)
SEQUENCE_LEN: int = 50

def feature_by_name(name: str) -> Feature:
    for f in FEATURES:
        if f.name == name:
            return f
    raise KeyError(name)
