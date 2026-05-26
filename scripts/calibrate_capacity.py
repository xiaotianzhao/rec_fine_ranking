"""Print params + FLOPs per model; flag whether each modern model meets the budget.

Params and FLOPs are reported as SEPARATE criteria. The four modern architectures
have FLOPs-per-param ratios that differ by ~50x (OneTrans causal attention is
compute-heavy; RankMixer/UniMixer are nearly parameter-free in their mixing), so
hitting BOTH 5M params and 50 MFLOPs at once is architecturally infeasible. The
benchmark therefore controls on backbone params (5M ± 10%) and treats MFLOPs/sample
as a measured dependent variable, reported here for transparency.

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
    print(f"{'model':<12} {'backbone_params':>16} {'emb_params':>14} {'mflops/sample':>16}  "
          f"{'params@5M':>10} {'flops@50':>9}")
    for name, cls in MODEL_REGISTRY.items():
        m = cls()
        p = m.count_params()
        mf = count_mflops(m, b)
        params_ok = flops_ok = "n/a"
        if name in MODERN:
            ok_p = abs(p["backbone"] - TARGET_PARAMS) / TARGET_PARAMS <= TARGET_PCT
            ok_f = abs(mf - TARGET_MFLOPS) / TARGET_MFLOPS <= TARGET_MFLOPS_PCT
            params_ok = "yes" if ok_p else "NO"
            flops_ok = "yes" if ok_f else "NO"
        print(f"{name:<12} {p['backbone']:>16,} {p['embedding']:>14,} {mf:>16.1f}  "
              f"{params_ok:>10} {flops_ok:>9}")
    print("\nControl axis = backbone params (5M ± 10%). MFLOPs/sample is a measured "
          "dependent variable (matching both is architecturally infeasible — see module docstring).")

if __name__ == "__main__":
    main()
