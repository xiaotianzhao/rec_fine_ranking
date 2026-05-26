"""CLI driver: walk all_stage/ + learning/realshow/, transform, write parquet shards.

Usage:
  python scripts/preprocess.py \
      --raw-root /Users/zhaoxiaotian.0701/data/rec_flow \
      --out-root experiments/data
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd
from rec_fine_ranking.data.preprocess import transform_frame

def _process_split(raw_dir: Path, out_dir: Path, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(raw_dir.glob("*.feather"))
    print(f"[{label}] {len(files)} files → {out_dir}")
    for f in files:
        out = out_dir / f"{f.stem}.parquet"
        if out.exists():
            print(f"  skip (exists): {out.name}")
            continue
        t0 = time.time()
        df = pd.read_feather(f)
        n_in = len(df)
        df = transform_frame(df)
        df.to_parquet(out, index=False, compression="snappy")
        print(f"  {f.name}: {n_in:>10,} → {len(df):>10,} rows in {time.time()-t0:.1f}s")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", required=True)
    p.add_argument("--out-root", required=True)
    args = p.parse_args()
    raw, out = Path(args.raw_root), Path(args.out_root)
    _process_split(raw / "all_stage",        out / "train", "train")
    _process_split(raw / "learning/realshow", out / "test",  "test")

if __name__ == "__main__":
    main()
