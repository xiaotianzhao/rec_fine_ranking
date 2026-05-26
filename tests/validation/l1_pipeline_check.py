"""L1 pipeline verifier — Stages 1/2/4/5/6 + logging artifacts + embedding stability.

Complements run_mock_overfit.py (which covers Stage 3 loss-down). Runs the full
training pipeline on mock data for a few steps via Trainer.fit(), then verifies:

  Stage 1  data loading       — batch shapes + finite inputs
  Stage 2  model instantiation — forward pass, output shape (B,), finite
  (Stage 3 logging artifacts)  — train.log / metrics.csv / meta.json / TB / ckpt produced
  Stage 4  checkpoint          — save → load into fresh model → params allclose + optim state
  Stage 5  inference           — eval mode, finite, deterministic across repeat passes
  Stage 6  evaluation          — Evaluator metrics finite (logloss), no crash
  Arch     embedding stability — encoder embedding norm finite + bounded after training

This script imports core; core never imports it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from rec_fine_ranking.models import MODEL_REGISTRY
from rec_fine_ranking.models.base import _toy_batch
from rec_fine_ranking.training.evaluator import Evaluator
from rec_fine_ranking.training.trainer import Trainer, TrainerConfig


class _MockLoader:
    """Fixed list of toy batches; deterministic, re-iterable, has __len__."""

    def __init__(self, B: int = 32, n_batches: int = 8, seed_offset: int = 0):
        self._fixed = [_toy_batch(B) for _ in range(n_batches)]

    def __iter__(self):
        return iter(self._fixed)

    def __len__(self):
        return len(self._fixed)


def _embedding_norm(model) -> float:
    total = 0.0
    for p in model.encoder.parameters():
        total += float(p.detach().float().norm().item()) ** 2
    return math.sqrt(total)


def check_model(name: str, out_dir: Path) -> dict:
    cls = MODEL_REGISTRY[name]
    train_loader, val_loader = _MockLoader(), _MockLoader()
    result: dict = {"model": name, "stages": {}}

    # Stage 1 — data loading
    batch = next(iter(train_loader))
    s1 = (
        batch["label"].shape[0] == 32
        and batch["seq_video_id"].shape == (32, 50)
        and all(torch.isfinite(v.float()).all() for v in batch.values())
    )
    result["stages"]["1_data_loading"] = bool(s1)

    # Stage 2 — model instantiation + forward
    probe = cls()
    out = probe(_toy_batch(B=8))
    s2 = out.shape == (8,) and torch.isfinite(out).all()
    result["stages"]["2_model_init"] = bool(s2)

    # Stage 3 (logging artifacts) — short fit() run that triggers eval + checkpoint
    cfg = TrainerConfig(
        model_name=name, out_dir=out_dir, device="cpu",
        max_steps=16, epochs=1, batch_size=32,
        eval_every_steps=8, ckpt_every_steps=8, log_every_steps=2,
        lr_backbone=1e-3, lr_emb=1e-2, grad_clip=1.0,
    )
    emb_before = _embedding_norm(Trainer(cfg, train_loader, val_loader).model)
    trainer = Trainer(cfg, train_loader, val_loader)
    trainer.fit()
    emb_after = _embedding_norm(trainer.model)

    train_log = (out_dir / "train.log").read_text() if (out_dir / "train.log").exists() else ""
    metrics_csv = (out_dir / "metrics.csv").read_text() if (out_dir / "metrics.csv").exists() else ""
    meta = json.loads((out_dir / "meta.json").read_text()) if (out_dir / "meta.json").exists() else {}
    tb_files = list(out_dir.glob("events.out.tfevents.*"))
    ckpts = list(out_dir.glob("ckpt_*.pt"))

    # L.1 loss present + parseable + finite; L.2 step speed present; L.6 TB; L.7 ckpt
    log_has_loss = "loss=" in train_log and "step_time_ms=" in train_log
    parsed_losses = []
    for line in train_log.splitlines():
        for tok in line.split():
            if tok.startswith("loss="):
                parsed_losses.append(float(tok.split("=")[1]))
    log_losses_finite = len(parsed_losses) > 0 and all(math.isfinite(x) for x in parsed_losses)
    result["stages"]["3_logging"] = bool(
        log_has_loss and log_losses_finite
        and metrics_csv.startswith("step,auc,gauc,logloss")
        and "train_time_sec" in meta and "params_backbone" in meta
        and len(tb_files) >= 1 and len(ckpts) >= 1
    )

    # Stage 4 — checkpoint save/load round-trip
    ckpt_path = sorted(ckpts)[-1]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    fresh = cls()
    fresh.load_state_dict(ckpt["model_state"])
    params_match = all(
        torch.allclose(a, b)
        for a, b in zip(trainer.model.state_dict().values(), fresh.state_dict().values())
    )
    optim_ok = "optim_state" in ckpt and len(ckpt["optim_state"]["state"]) > 0
    step_ok = ckpt.get("step", 0) > 0
    result["stages"]["4_checkpoint"] = bool(params_match and optim_ok and step_ok)

    # Stage 5 — inference: eval mode, finite, deterministic
    fresh.eval()
    fixed = _toy_batch(B=16)
    with torch.no_grad():
        o1 = fresh(fixed)
        o2 = fresh(fixed)
    result["stages"]["5_inference"] = bool(
        torch.isfinite(o1).all() and torch.allclose(o1, o2)
    )

    # Stage 6 — evaluation metrics finite (logloss always defined; auc/gauc may be NaN on toy)
    res = Evaluator(device="cpu").run(fresh, _MockLoader())
    result["stages"]["6_evaluation"] = bool(
        set(res.keys()) >= {"auc", "gauc", "logloss", "n_samples", "elapsed_sec"}
        and math.isfinite(res["logloss"]) and res["n_samples"] > 0
    )
    result["eval_metrics"] = {k: res[k] for k in ("auc", "gauc", "logloss", "n_samples")}

    # Arch — embedding stability
    result["emb_norm"] = {"before": emb_before, "after": emb_after}
    result["stages"]["arch_emb_stability"] = bool(
        math.isfinite(emb_after) and emb_after < emb_before * 100 + 1e3
    )

    result["pass"] = all(result["stages"].values())
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="experiments/vp_l1_pipeline")
    ap.add_argument("--models", nargs="*", default=list(MODEL_REGISTRY.keys()))
    a = ap.parse_args()
    root = Path(a.out_root)
    all_pass = True
    results = []
    for name in a.models:
        r = check_model(name, root / name)
        results.append(r)
        all_pass = all_pass and r["pass"]
        flags = " ".join(f"{k.split('_')[0]}={'✓' if v else '✗'}" for k, v in r["stages"].items())
        print(f"[{name:<10}] pass={r['pass']}  {flags}  "
              f"logloss={r['eval_metrics']['logloss']:.4f} "
              f"emb {r['emb_norm']['before']:.1f}->{r['emb_norm']['after']:.1f}")
    print(json.dumps(results, indent=2, default=str))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
