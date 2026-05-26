import io, json, time
from pathlib import Path
import numpy as np, pandas as pd, pickle, torch
import pytest
from rec_fine_ranking.training.trainer import Trainer, TrainerConfig
from rec_fine_ranking.training.evaluator import Evaluator
from rec_fine_ranking.models import MODEL_REGISTRY
from rec_fine_ranking.models.base import _toy_batch

def _toy_loader(n=64, B=8):
    class _L:
        def __iter__(self):
            for _ in range(n // B):
                yield _toy_batch(B)
        def __len__(self): return n // B
    return _L()

@pytest.mark.parametrize("name", list(MODEL_REGISTRY.keys()))
def test_train_step_smoke(name, tmp_path):
    cfg = TrainerConfig(
        model_name=name, out_dir=tmp_path / name, device="cpu",
        max_steps=5, eval_every_steps=100, ckpt_every_steps=100,
        batch_size=8, lr_backbone=1e-3, lr_emb=1e-2,
        log_every_steps=1, grad_clip=1.0)
    trainer = Trainer(cfg, train_loader=_toy_loader(), val_loader=_toy_loader())
    losses = trainer.fit_for_test()
    assert len(losses) == 5
    assert all(np.isfinite(losses))
    # at least 2 of 5 steps showed loss decrease vs previous
    decreases = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])
    assert decreases >= 2, f"only {decreases} decreasing steps: {losses}"

def test_train_writes_artifacts(tmp_path):
    cfg = TrainerConfig(model_name="wide_deep", out_dir=tmp_path / "run",
                        device="cpu", max_steps=3, eval_every_steps=100,
                        ckpt_every_steps=2, batch_size=8, lr_backbone=1e-3,
                        lr_emb=1e-2, log_every_steps=1, grad_clip=1.0)
    Trainer(cfg, train_loader=_toy_loader(), val_loader=_toy_loader()).fit()
    out = tmp_path / "run"
    assert (out / "train.log").exists()
    assert (out / "metrics.csv").exists()
    assert (out / "meta.json").exists()
    ckpts = list(out.glob("ckpt_step*.pt"))
    assert len(ckpts) >= 1
