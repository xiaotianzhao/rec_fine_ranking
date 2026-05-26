"""Build leakage-safe per-user effective-view history for each sample.

Output schema: dict[(user_id:int, request_id:int)] -> dict[field_name -> np.ndarray of length SEQ_LEN].
History is right-aligned: padding (0) fills the head, the most recent event sits at index -1.
This matches conventional sequence-modelling layouts (DIN, BERT4Rec) and makes causal-attention
masking natural for OneTrans-style models.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

SEQ_LEN = 50
SEQ_FIELDS = ("video_id","author_id","category_level_two","category_level_one","upload_type")

def build_history_for_day(today_df: pd.DataFrame, prior_df: pd.DataFrame) -> Dict[Tuple[int,int], Dict[str, np.ndarray]]:
    """Return history dict keyed by (user_id, request_id) for every row of today_df.

    today_df: rows for the day we want history for (need user_id, request_id, request_timestamp).
    prior_df: pool of effective_view==1 events from earlier-or-same-day prior to each sample.
              Must contain user_id, request_timestamp, effective_view, and all SEQ_FIELDS.
    """
    eff = prior_df[prior_df["effective_view"] == 1].sort_values(["user_id","request_timestamp"])
    # group prior events by user
    groups = {uid: g for uid, g in eff.groupby("user_id", sort=False)}
    out: Dict[Tuple[int,int], Dict[str, np.ndarray]] = {}
    today_sorted = today_df.sort_values(["user_id","request_timestamp"])
    for row in today_sorted.itertuples(index=False):
        g = groups.get(row.user_id)
        if g is None:
            window = None
        else:
            mask = g["request_timestamp"].to_numpy() < row.request_timestamp
            window = g.iloc[mask].tail(SEQ_LEN) if mask.any() else None
        rec: Dict[str, np.ndarray] = {}
        for f in SEQ_FIELDS:
            buf = np.zeros(SEQ_LEN, dtype=np.int32)
            if window is not None and len(window) > 0:
                vals = window[f].to_numpy().astype(np.int32)
                buf[-len(vals):] = vals
            rec[f] = buf
        out[(int(row.user_id), int(row.request_id))] = rec
    return out
