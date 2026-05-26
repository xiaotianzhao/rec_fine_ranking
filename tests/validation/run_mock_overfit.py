"""L1 driver — overfit on 1024 mock samples for 50 steps; assert loss-down + finite.

This script imports core, but core does NOT import this script.
"""
from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
import torch
from rec_fine_ranking.training.trainer import Trainer, TrainerConfig
from rec_fine_ranking.models.base import _toy_batch
from rec_fine_ranking.models import MODEL_REGISTRY

class _MockLoader:
    def __init__(self, B=32, n_batches=1024//32):
        self.B = B; self.n = n_batches
        self._fixed = [_toy_batch(B) for _ in range(self.n)]
    def __iter__(self): return iter(self._fixed)
    def __len__(self): return self.n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    loader = _MockLoader()
    cfg = TrainerConfig(model_name=a.model, out_dir=out, device=a.device,
                        max_steps=50, log_every_steps=1, eval_every_steps=10**9,
                        ckpt_every_steps=10**9, batch_size=32)
    t = Trainer(cfg, loader, loader)
    losses = t.fit_for_test()
    summary = {"model": a.model, "initial": losses[0], "final": losses[-1],
               "all_finite": all(math.isfinite(l) for l in losses),
               "ratio": losses[-1] / max(losses[0], 1e-6)}
    print(json.dumps(summary, indent=2))
    ok = summary["all_finite"] and summary["ratio"] < 0.5
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
