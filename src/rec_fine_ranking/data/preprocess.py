"""Per-frame transform: filter realshow=1, hash big IDs, bucketise continuous, narrow dtypes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .feature_config import FEATURES, FeatureKind, feature_by_name
from .hashing import hash_ids

_KEEP_COLS = [
    # ids / cats
    "user_id","device_id","age","gender","province",
    "category_level_one","category_level_two","upload_type",
    "video_id","author_id","duration",
    # context
    "request_id","request_timestamp","hour_of_day",
    # labels (we keep all three so downstream can pick)
    "effective_view","long_view","like",
]

def _bucketize_duration(seconds: pd.Series, n_buckets: int = 64) -> np.ndarray:
    # log-spaced bins covering 1..7200 sec
    edges = np.logspace(0, np.log10(7200), n_buckets)
    idx = np.searchsorted(edges, seconds.to_numpy().clip(min=1)) - 1
    return idx.clip(0, n_buckets - 1).astype(np.int8)

def transform_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["realshow"] == 1].copy()
    # video_id / author_id → hashed
    df["video_id"]  = hash_ids(df["video_id" ].to_numpy(np.int64), feature_by_name("video_id" ).vocab_size).astype(np.int32)
    df["author_id"] = hash_ids(df["author_id"].to_numpy(np.int64), feature_by_name("author_id").vocab_size).astype(np.int32)
    # small vocabs → int16 / int8
    df["user_id"]            = df["user_id"           ].clip(upper=49_999).astype(np.int32)
    df["device_id"]          = df["device_id"         ].clip(upper=49_999).astype(np.int32)
    df["category_level_two"] = df["category_level_two"].clip(upper=767  ).astype(np.int16)
    df["category_level_one"] = df["category_level_one"].clip(upper=127  ).astype(np.int8)
    df["province"]           = df["province"          ].clip(upper=79   ).astype(np.int8)
    df["upload_type"]        = df["upload_type"       ].clip(upper=31   ).astype(np.int8)
    df["age"]                = df["age"               ].clip(upper=15   ).astype(np.int8)
    df["gender"]             = df["gender"            ].clip(upper=3    ).astype(np.int8)
    # continuous → bucketise
    df["duration"] = _bucketize_duration(df["duration"])
    # context
    df["hour_of_day"] = ((df["request_timestamp"] // 3600) % 24).astype(np.int8)
    # labels → int8
    for c in ("effective_view","long_view","like"):
        df[c] = df[c].astype(np.int8)
    df["request_id"] = df["request_id"].astype(np.int64)
    df["request_timestamp"] = df["request_timestamp"].astype(np.int64)
    return df[_KEEP_COLS]
