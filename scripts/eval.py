"""Offline checkpoint evaluation entry point (mode 2 of Evaluator)."""
from __future__ import annotations
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from rec_fine_ranking.data.dataset import RecFlowDataset
from rec_fine_ranking.data.collate import collate_batch
from rec_fine_ranking.models import MODEL_REGISTRY
from rec_fine_ranking.training.evaluator import Evaluator
from rec_fine_ranking.utils.device import autodetect_device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    a = ap.parse_args()
    root = Path(a.data_root)
    val = RecFlowDataset(root/"test", root/"test_seq", shuffle_buffer=0, shuffle_shards=False)
    loader = DataLoader(val, batch_size=a.batch_size, collate_fn=collate_batch, num_workers=0)
    factory = MODEL_REGISTRY[a.model]
    res = Evaluator(device=autodetect_device()).run_from_checkpoint(a.checkpoint, factory, loader)
    print(res)

if __name__ == "__main__":
    main()
