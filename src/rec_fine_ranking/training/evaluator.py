"""Shared evaluator core. Supports two entry modes:

1. In-memory (during training):  evaluator.run(model, loader)
2. Checkpoint-based (offline):   evaluator.run_from_checkpoint(ckpt_path, model_factory, loader)

Both modes go through the same `_loop` so output schemas match exactly.
"""
from __future__ import annotations
import logging, time
from pathlib import Path
from typing import Callable, Dict, Iterable
import numpy as np
import torch
from tqdm import tqdm
from .metrics import compute_auc, compute_gauc, compute_logloss

log = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, device: str | torch.device = "cpu", silent_threshold_sec: float = 60.0):
        self.device = torch.device(device)
        self.silent_threshold_sec = silent_threshold_sec

    @torch.no_grad()
    def _loop(self, model: torch.nn.Module, loader: Iterable, step: int | None = None) -> Dict[str, float]:
        model.eval()
        # A DataLoader over an IterableDataset has __len__ but raises TypeError when
        # called; probe with try/except. Empty-loader is then caught post-loop below.
        try:
            n_batches = len(loader)
        except TypeError:
            n_batches = None
        if n_batches == 0:
            raise ValueError("Evaluator received an empty validation loader.")
        tag = f"@step={step}" if step is not None else ""
        log.info(f"[eval] starting {tag} | n_batches={n_batches}")
        t0 = time.time()
        last_progress = t0
        y_true, y_score, uids = [], [], []
        bar = tqdm(loader, total=n_batches, desc=f"eval{tag}", leave=False)
        for batch in bar:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            logits = model(batch)
            probs = torch.sigmoid(logits).float().cpu().numpy()
            y_score.append(probs)
            y_true.append(batch["label"].cpu().numpy())
            uids.append(batch["user_id"].cpu().numpy())
            now = time.time()
            if now - last_progress > self.silent_threshold_sec:
                log.warning(f"[eval] silent gap > {self.silent_threshold_sec:.0f}s — possible stall")
            last_progress = now
        if not y_true:
            raise ValueError("Evaluator produced zero samples — loader yielded nothing.")
        y_true_a  = np.concatenate(y_true).astype(np.int64)
        y_score_a = np.concatenate(y_score).astype(np.float64)
        uids_a    = np.concatenate(uids).astype(np.int64)
        try:
            auc     = compute_auc(y_true_a, y_score_a)
            gauc    = compute_gauc(y_true_a, y_score_a, uids_a)
            logloss = compute_logloss(y_true_a, y_score_a)
        except Exception as e:
            log.exception(f"[eval] metric aggregation failed: {e}")
            auc = gauc = logloss = float("nan")
        elapsed = time.time() - t0
        if not (np.isfinite(auc) and np.isfinite(gauc) and np.isfinite(logloss)):
            log.warning(f"[eval] non-finite metric: auc={auc} gauc={gauc} logloss={logloss}")
        log.info(f"[eval] done {tag} | auc={auc:.4f} gauc={gauc:.4f} logloss={logloss:.4f} "
                 f"| n_samples={len(y_true_a):,} | elapsed={elapsed:.1f}s "
                 f"| throughput={len(y_true_a)/max(elapsed,1e-6):,.0f} samples/s")
        return {"auc": auc, "gauc": gauc, "logloss": logloss,
                "n_samples": int(len(y_true_a)), "elapsed_sec": elapsed}

    def run(self, model: torch.nn.Module, loader: Iterable, step: int | None = None) -> Dict[str, float]:
        model.to(self.device)
        return self._loop(model, loader, step=step)

    def run_from_checkpoint(self, ckpt_path: str | Path,
                            model_factory: Callable[[], torch.nn.Module],
                            loader: Iterable) -> Dict[str, float]:
        path = Path(ckpt_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        t0 = time.time()
        try:
            ckpt = torch.load(path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Checkpoint unreadable at {path}: {e}") from e
        model = model_factory()
        try:
            model.load_state_dict(ckpt["model_state"])
        except RuntimeError as e:
            log.error(f"[eval] state_dict mismatch loading {path}: {e}")
            raise
        log.info(f"[eval] checkpoint loaded in {time.time()-t0:.2f}s from {path}")
        return self.run(model, loader, step=ckpt.get("step"))
