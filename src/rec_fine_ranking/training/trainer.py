"""Trainer: training loop, logging, checkpointing, evaluation triggering.

Does NOT decide how evaluation runs — delegates to Evaluator.
"""
from __future__ import annotations
import csv, json, logging, math, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ..models import MODEL_REGISTRY
from ..utils.flops import count_mflops
from ..models.base import _toy_batch
from .evaluator import Evaluator

log = logging.getLogger(__name__)

@dataclass
class TrainerConfig:
    model_name: str
    out_dir: Path
    device: str = "auto"
    max_steps: int = 0          # 0 means run to end of epoch
    epochs: int = 1
    batch_size: int = 4096
    lr_backbone: float = 1e-3
    lr_emb: float = 1e-2
    grad_clip: float = 1.0
    eval_every_steps: int = 5000
    ckpt_every_steps: int = 10_000
    log_every_steps: int = 100
    device_peak_tflops: float = 1.0   # M-series fp32 nominal (this run is fp32); MFU display only
    resume: Optional[str] = None


class Trainer:
    def __init__(self, cfg: TrainerConfig, train_loader: Iterable, val_loader: Iterable):
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device if cfg.device != "auto" else self._auto_device())
        self.train_loader = train_loader
        self.val_loader = val_loader
        cls = MODEL_REGISTRY[cfg.model_name]
        self.model = cls().to(self.device)
        emb_params = list(self.model.encoder.parameters())
        backbone_params = [p for n, p in self.model.named_parameters() if not n.startswith("encoder.")]
        self.optim = torch.optim.Adam([
            {"params": backbone_params, "lr": cfg.lr_backbone, "weight_decay": 1e-5},
            {"params": emb_params,      "lr": cfg.lr_emb,      "weight_decay": 0.0},
        ])
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.evaluator = Evaluator(device=self.device)
        self.writer = SummaryWriter(self.out_dir.as_posix())
        self._step = 0
        # static info
        params = self.model.count_params()
        self.mflops = count_mflops(self.model, _toy_batch(B=1, device="cpu"))
        meta = {"model": cfg.model_name,
                "params_backbone": params["backbone"], "params_emb": params["embedding"],
                "mflops_per_sample": self.mflops, "config": asdict(cfg) | {"out_dir": str(cfg.out_dir)}}
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        # log + metrics files
        self._log_file = open(self.out_dir / "train.log", "a", buffering=1)
        self._metrics_csv = open(self.out_dir / "metrics.csv", "a", newline="")
        self._metrics_w = csv.writer(self._metrics_csv)
        if self._metrics_csv.tell() == 0:
            self._metrics_w.writerow(["step", "auc", "gauc", "logloss"])
        # resume
        if cfg.resume:
            self._load_checkpoint(cfg.resume)

    @staticmethod
    def _auto_device():
        if torch.backends.mps.is_available(): return "mps"
        if torch.cuda.is_available(): return "cuda"
        return "cpu"

    def _train_step(self, batch):
        self.model.train()
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        logits = self.model(batch)
        loss = self.loss_fn(logits, batch["label"])
        if not torch.isfinite(loss):
            # Fail loud: a NaN/Inf loss means the run is corrupt; aborting beats
            # silently stepping the optimizer and wasting a long training run.
            raise RuntimeError(
                f"[trainer] non-finite loss ({loss.item()}) at step {self._step} — aborting."
            )
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optim.step()
        return float(loss.item()), float(gnorm.item())

    def _log_step(self, loss: float, gnorm: float, step_ms: float):
        s = self._step
        samples_per_sec = self.cfg.batch_size / max(step_ms / 1000, 1e-6)
        mfu = (self.mflops * 1e6 * samples_per_sec) / (self.cfg.device_peak_tflops * 1e12) * 100
        line = (f"step={s} loss={loss:.4f} grad_norm={gnorm:.3f} "
                f"lr={self.cfg.lr_backbone:.2e}/{self.cfg.lr_emb:.2e} "
                f"step_time_ms={step_ms:.1f} samples/s={samples_per_sec:,.0f} mfu={mfu:.2f}%")
        self._log_file.write(line + "\n")
        self.writer.add_scalar("loss", loss, s)
        self.writer.add_scalar("grad_norm", gnorm, s)
        self.writer.add_scalar("step_time_ms", step_ms, s)
        self.writer.add_scalar("samples_per_sec", samples_per_sec, s)
        self.writer.add_scalar("mfu", mfu, s)

    def _eval(self):
        res = self.evaluator.run(self.model, self.val_loader, step=self._step)
        self._metrics_w.writerow([self._step, res["auc"], res["gauc"], res["logloss"]])
        self._metrics_csv.flush()
        for k in ("auc", "gauc", "logloss"):
            v = res[k]
            if math.isfinite(v):
                self.writer.add_scalar(f"eval/{k}", v, self._step)

    def _save_checkpoint(self, tag: str):
        path = self.out_dir / f"ckpt_step{self._step}.pt" if tag == "step" else self.out_dir / f"ckpt_{tag}.pt"
        torch.save({"model_state": self.model.state_dict(),
                    "optim_state": self.optim.state_dict(),
                    "step": self._step}, path)
        log.info(f"[trainer] checkpoint saved → {path}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optim.load_state_dict(ckpt["optim_state"])
        self._step = int(ckpt["step"])
        log.info(f"[trainer] resumed from {path} @ step={self._step}")

    def fit(self) -> None:
        start = time.time()
        try:
            for epoch in range(self.cfg.epochs):
                it = iter(self.train_loader)
                # A DataLoader over an IterableDataset has __len__ but raises TypeError
                # when called, so probe with try/except rather than hasattr.
                try:
                    _total = len(self.train_loader)
                except TypeError:
                    _total = None
                bar = tqdm(it, desc=f"train epoch={epoch}", total=_total)
                for batch in bar:
                    t0 = time.time()
                    loss, gnorm = self._train_step(batch)
                    step_ms = (time.time() - t0) * 1000
                    self._step += 1
                    if self._step % self.cfg.log_every_steps == 0:
                        self._log_step(loss, gnorm, step_ms)
                        bar.set_postfix(loss=f"{loss:.4f}", gnorm=f"{gnorm:.2f}")
                    if self._step % self.cfg.eval_every_steps == 0:
                        self._eval()
                    if self._step % self.cfg.ckpt_every_steps == 0:
                        self._save_checkpoint("step")
                    if self.cfg.max_steps and self._step >= self.cfg.max_steps:
                        break
                self._eval()
                self._save_checkpoint("end")
                if self.cfg.max_steps and self._step >= self.cfg.max_steps:
                    break
            elapsed = time.time() - start
            meta_path = self.out_dir / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["train_time_sec"] = elapsed
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
        finally:
            # Close handles + TB writer even if training aborts (e.g. non-finite loss).
            self._log_file.close(); self._metrics_csv.close(); self.writer.close()

    def fit_for_test(self) -> List[float]:
        """Tiny helper for integration test: run max_steps without eval/ckpt logic.

        Cycles through the loader if it is shorter than ``max_steps`` so callers
        (e.g. the L1 mock-overfit driver) can take more steps than the loader has
        batches without exhausting the iterator.
        """
        losses = []
        it = iter(self.train_loader)
        for _ in range(self.cfg.max_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.train_loader)
                batch = next(it)
            loss, _ = self._train_step(batch)
            losses.append(loss)
            self._step += 1
        return losses
