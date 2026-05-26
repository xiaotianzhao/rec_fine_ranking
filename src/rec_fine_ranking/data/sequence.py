"""Build leakage-safe per-user effective-view history for each sample.

Output schema: dict[(user_id:int, request_id:int)] -> dict[field_name -> np.ndarray of length SEQ_LEN].
History is right-aligned: padding (0) fills the head, the most recent event sits at index -1.
This matches conventional sequence-modelling layouts (DIN, BERT4Rec) and makes causal-attention
masking natural for OneTrans-style models.

Implementation: prior effective-view events are extracted into per-user NumPy arrays sorted by
timestamp once; each sample then locates its cutoff via ``np.searchsorted`` (strictly-earlier
events only — no leakage) and right-aligns the last SEQ_LEN into a zero buffer. This avoids
pandas groupby/boolean-mask work per row, which was orders of magnitude slower and memory-heavy.
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

    A sample's history is the last SEQ_LEN prior events with timestamp STRICTLY LESS THAN the
    sample's request_timestamp (leakage-safe), right-aligned with zero padding at the head.
    """
    # Prior effective-view events, sorted by (user, time). Stable sort keeps input order on ties.
    eff = prior_df[prior_df["effective_view"] == 1].sort_values(
        ["user_id", "request_timestamp"], kind="mergesort"
    )
    e_uid = eff["user_id"].to_numpy()
    e_ts = eff["request_timestamp"].to_numpy()
    e_fields = {f: eff[f].to_numpy().astype(np.int32, copy=False) for f in SEQ_FIELDS}

    # Contiguous [start, end) slice into the sorted arrays for each user.
    user_range: Dict[int, Tuple[int, int]] = {}
    if len(e_uid) > 0:
        uniq, starts = np.unique(e_uid, return_index=True)
        for i, u in enumerate(uniq):
            end = int(starts[i + 1]) if i + 1 < len(uniq) else len(e_uid)
            user_range[int(u)] = (int(starts[i]), end)

    t_uid = today_df["user_id"].to_numpy()
    t_rid = today_df["request_id"].to_numpy()
    t_ts = today_df["request_timestamp"].to_numpy()

    out: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    for i in range(len(t_uid)):
        u = int(t_uid[i]); rid = int(t_rid[i]); ts = t_ts[i]
        rng = user_range.get(u)
        rec: Dict[str, np.ndarray] = {}
        if rng is None:
            for f in SEQ_FIELDS:
                rec[f] = np.zeros(SEQ_LEN, dtype=np.int32)
        else:
            s, e = rng
            # count of this user's events with timestamp strictly < ts
            cut = s + int(np.searchsorted(e_ts[s:e], ts, side="left"))
            lo = max(s, cut - SEQ_LEN)
            n = cut - lo
            for f in SEQ_FIELDS:
                buf = np.zeros(SEQ_LEN, dtype=np.int32)
                if n > 0:
                    buf[-n:] = e_fields[f][lo:cut]
                rec[f] = buf
        out[(u, rid)] = rec
    return out
