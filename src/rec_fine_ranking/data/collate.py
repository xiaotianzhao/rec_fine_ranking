"""Collate function: stack a list of per-sample dicts into a single batched dict of tensors."""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch

def collate_batch(samples: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    keys = samples[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        arrs = [s[k] for s in samples]
        if arrs[0].ndim == 0:
            t = torch.from_numpy(np.stack(arrs))
        else:
            t = torch.from_numpy(np.stack(arrs, axis=0))
        out[k] = t.float() if k == "label" else t.long()
    return out
