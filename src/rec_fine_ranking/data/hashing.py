"""Deterministic hashing for high-cardinality IDs.

Uses 64-bit FNV-1a, modulo (buckets), +1 offset so bucket 0 is reserved for padding.
NumPy-vectorised for speed; no external deps.
"""
from __future__ import annotations
import numpy as np

_FNV_OFFSET = np.int64(-3750763034362895579)  # 0xcbf29ce484222325 as signed
_FNV_PRIME  = np.int64(1099511628211)

def hash_ids(ids: np.ndarray, buckets: int) -> np.ndarray:
    """Hash an int64 array into [1, buckets] using FNV-1a on the 8 bytes."""
    if ids.dtype != np.int64:
        ids = ids.astype(np.int64, copy=False)
    h = np.full_like(ids, _FNV_OFFSET)
    for shift in range(0, 64, 8):
        byte = (ids >> shift) & 0xFF
        h = (h ^ byte).astype(np.int64) * _FNV_PRIME
    # cast to unsigned conceptually, then mod
    h_u = h.view(np.uint64)
    return (h_u % np.uint64(buckets)).astype(np.int64) + 1
