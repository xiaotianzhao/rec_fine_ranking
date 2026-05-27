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
_ZERO_SEQ = np.zeros(SEQUENCE_LEN, dtype=np.int32)  # shared read-only pad for missing history


class RecFlowDataset(torch.utils.data.IterableDataset):
    """Iterates over preprocessed shards, joining each row with its history sequence.

    Yields a dict of np.ndarrays per sample. Use `collate_batch` to stack into tensors.

    Performance notes:
    - Columns are pulled out as NumPy arrays per shard and indexed positionally
      (no `itertuples`, which is very slow on ~1M-row shards).
    - Sequence arrays are yielded by reference (int32, no per-row dtype copy);
      `collate_batch` does the single batched `.long()` cast. Missing history shares
      one read-only zero array. Both cut per-sample allocation sharply.
    - `__iter__` is worker-aware: with `num_workers>0` each worker takes a disjoint
      slice of shards (shards[worker_id::num_workers]), so a DataLoader can overlap
      data loading with compute without duplicating or dropping samples.
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
        shards = list(self._shards)
        if self.shuffle_shards:
            random.Random(self.seed).shuffle(shards)
        # Worker-aware sharding: each worker takes a disjoint slice of shards.
        worker = torch.utils.data.get_worker_info()
        wid = 0 if worker is None else worker.id
        if worker is not None:
            shards = shards[worker.id::worker.num_workers]
        rng = random.Random(self.seed + wid)

        buf: list = []
        for shard in shards:
            seq_path = self.seq_dir / (shard.stem + ".pkl")
            with open(seq_path, "rb") as fh:
                hist: Dict = pickle.load(fh)
            df = pd.read_parquet(shard)
            if self.shuffle_buffer:
                df = df.sample(frac=1.0, random_state=self.seed + wid)
            # Pull columns out as NumPy arrays once (positional indexing beats itertuples).
            present = [f for f in _NON_SEQ if f in df.columns]
            cols = {f: df[f].to_numpy() for f in present}
            uid = df["user_id"].to_numpy()
            rid = df["request_id"].to_numpy()
            labels = df[self.label].to_numpy().astype(np.float32, copy=False)
            n = len(df)
            for i in range(n):
                seq = hist.get((int(uid[i]), int(rid[i])))
                rec: Dict[str, np.ndarray] = {f: cols[f][i] for f in present}
                if seq is not None:
                    for f in SEQUENCE_FIELDS:
                        rec[f"seq_{f}"] = seq[f]              # int32 view, batched-cast in collate
                else:
                    for f in SEQUENCE_FIELDS:
                        rec[f"seq_{f}"] = _ZERO_SEQ
                rec["label"] = labels[i]
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
