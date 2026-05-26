"""Walk preprocessed parquet shards and emit per-day sequence pickles.

Usage:
  python scripts/build_sequences.py --data-root experiments/data
"""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import pandas as pd
from rec_fine_ranking.data.sequence import build_history_for_day, SEQ_FIELDS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    args = p.parse_args()
    root = Path(args.data_root)
    for split in ("train","test"):
        shard_dir = root / split
        seq_dir = root / f"{split}_seq"
        seq_dir.mkdir(parents=True, exist_ok=True)
        shards = sorted(shard_dir.glob("*.parquet"))
        prior_frames = []
        for i, f in enumerate(shards):
            today = pd.read_parquet(f)
            prior = pd.concat(prior_frames, copy=False) if prior_frames else today.iloc[:0]
            # also include earlier rows of *today* before each row's timestamp
            prior_for_today = pd.concat([prior, today], copy=False)
            hist = build_history_for_day(today, prior_for_today)
            out = seq_dir / f"{f.stem}.pkl"
            with open(out, "wb") as fh:
                pickle.dump(hist, fh, protocol=4)
            print(f"[{split}] {f.name}: {len(hist):>10,} keys → {out.name}")
            prior_frames.append(today[["user_id","request_timestamp","effective_view", *SEQ_FIELDS]])
            # cap memory: keep last 5 days of prior
            if len(prior_frames) > 5:
                prior_frames.pop(0)

if __name__ == "__main__":
    main()
