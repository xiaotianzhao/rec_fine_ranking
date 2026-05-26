"""Print params + FLOPs per model; flag modern models that fall outside the budget.

Usage:
  PYTHONPATH=src python scripts/calibrate_capacity.py [--batch-size N]
"""
from __future__ import annotations
import argparse, torch
from rec_fine_ranking.models import MODEL_REGISTRY
from rec_fine_ranking.models.base import _toy_batch
from rec_fine_ranking.utils.flops import count_mflops

TARGET_PARAMS = 5_000_000
TARGET_PCT    = 0.10
TARGET_MFLOPS = 50.0
TARGET_MFLOPS_PCT = 0.15
MODERN = {"onetrans","rankmixer","unimixer","hyformer"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()
    b = _toy_batch(args.batch_size)
    print(f"{'model':<12} {'backbone_params':>16} {'emb_params':>14} {'mflops/sample':>16}  in-budget?")
    for name, cls in MODEL_REGISTRY.items():
        m = cls()
        p = m.count_params()
        mf = count_mflops(m, b)
        budget_ok = "n/a"
        if name in MODERN:
            ok_p = abs(p["backbone"] - TARGET_PARAMS) / TARGET_PARAMS <= TARGET_PCT
            ok_f = abs(mf - TARGET_MFLOPS) / TARGET_MFLOPS <= TARGET_MFLOPS_PCT
            budget_ok = "yes" if (ok_p and ok_f) else "NO"
        print(f"{name:<12} {p['backbone']:>16,} {p['embedding']:>14,} {mf:>16.1f}  {budget_ok}")

if __name__ == "__main__":
    main()
