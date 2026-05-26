"""Streaming IterableDataset over parquet shards + per-day sequence pickles."""
from __future__ import annotations
import pickle, random
from pathlib import Path
from typing import Dict, Iterator
import numpy as np
import pandas as pd
import torch
from .feature_config import FEATURES, SEQUENCE_FIELDS, SEQUENCE_LEN

_NON_SEQ = [f.name for f in FEATURES] + ["hour_of_day"]

class RecFlowDataset(torch.utils.data.IterableDataset):
    """Iterates over preprocessed shards, joining each row with its history sequence.

    Yields a dict of np.ndarrays per sample. Use `collate_batch` to stack into tensors.
    """
    def __init__(self, shard_dir: Path, seq_dir: Path, label: str = "effective_view",
                 shuffle_buffer: int = 8192, shuffle_shards: bool = True, seed: int = 42):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.seq_dir = Path(seq_dir)
        self.label = label
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self._shards = sorted(self.shard_dir.glob("*.parquet"))
        if not self._shards:
            raise FileNotFoundError(f"No parquet shards in {self.shard_dir}")

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        rng = random.Random(self.seed)
        shards = list(self._shards)
        if self.shuffle_shards:
            rng.shuffle(shards)
        buf: list = []
        for shard in shards:
            seq_path = self.seq_dir / (shard.stem + ".pkl")
            with open(seq_path, "rb") as fh:
                hist: Dict = pickle.load(fh)
            df = pd.read_parquet(shard)
            if self.shuffle_buffer:
                df = df.sample(frac=1.0, random_state=self.seed)
            for row in df.itertuples(index=False):
                key = (int(row.user_id), int(row.request_id))
                seq = hist.get(key)
                rec: Dict[str, np.ndarray] = {}
                for f in _NON_SEQ:
                    if hasattr(row, f):
                        rec[f] = np.asarray(getattr(row, f), dtype=np.int64)
                for f in SEQUENCE_FIELDS:
                    rec[f"seq_{f}"] = (seq[f].astype(np.int64) if seq is not None
                                      else np.zeros(SEQUENCE_LEN, dtype=np.int64))
                rec["label"] = np.float32(getattr(row, self.label))
                if self.shuffle_buffer:
                    buf.append(rec)
                    if len(buf) >= self.shuffle_buffer:
                        rng.shuffle(buf)
                        yield from buf
                        buf.clear()
                else:
                    yield rec
        if buf:
            rng.shuffle(buf)
            yield from buf
