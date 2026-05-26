"""AUC / GAUC / LogLoss helpers — thin wrappers over sklearn with NaN-safe aggregation."""
from __future__ import annotations
import math
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.min() == y_true.max():
        return math.nan
    return float(roc_auc_score(y_true, y_score))

def compute_gauc(y_true: np.ndarray, y_score: np.ndarray, user_ids: np.ndarray) -> float:
    df_uid = np.asarray(user_ids).astype(np.int64)
    order = np.argsort(df_uid, kind="stable")
    df_uid, y_true, y_score = df_uid[order], y_true[order], y_score[order]
    splits = np.flatnonzero(np.diff(df_uid)) + 1
    groups_idx = np.split(np.arange(len(df_uid)), splits)
    total, total_w = 0.0, 0.0
    for g in groups_idx:
        if len(g) < 2 or y_true[g].min() == y_true[g].max():
            continue
        total   += roc_auc_score(y_true[g], y_score[g]) * len(g)
        total_w += len(g)
    return float(total / total_w) if total_w > 0 else math.nan

def compute_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7), labels=[0, 1]))
