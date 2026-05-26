"""CLI driver: train one model on RecFlow preprocessed shards.

Usage:
  python scripts/train.py --model rankmixer --data-root experiments/data \
      --out-dir experiments/runs/rankmixer --device auto --batch-size 4096
"""
from __future__ import annotations
import argparse, logging, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from rec_fine_ranking.data.dataset import RecFlowDataset
from rec_fine_ranking.data.collate import collate_batch
from rec_fine_ranking.training.trainer import Trainer, TrainerConfig

def _seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--eval-every-steps", type=int, default=5000)
    p.add_argument("--ckpt-every-steps", type=int, default=10_000)
    p.add_argument("--log-every-steps", type=int, default=100)
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    _seed()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    root = Path(args.data_root)
    train_ds = RecFlowDataset(root/"train", root/"train_seq", shuffle_buffer=8192)
    val_ds   = RecFlowDataset(root/"test",  root/"test_seq",  shuffle_buffer=0, shuffle_shards=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_batch, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, collate_fn=collate_batch, num_workers=0)
    cfg = TrainerConfig(
        model_name=args.model, out_dir=Path(args.out_dir), device=args.device,
        batch_size=args.batch_size, max_steps=args.max_steps,
        eval_every_steps=args.eval_every_steps,
        ckpt_every_steps=args.ckpt_every_steps,
        log_every_steps=args.log_every_steps, resume=args.resume,
    )
    Trainer(cfg, train_loader, val_loader).fit()

if __name__ == "__main__":
    main()
