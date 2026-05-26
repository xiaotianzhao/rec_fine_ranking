I need you to act as a Watchdog Agent, monitoring and shepherding a long-running ML benchmark (7 ranking models on RecFlow).

## Setup
1. Read `/Users/zhaoxiaotian.0701/projects/rec_fine_ranking/experiments/experiment-context.md` for full experiment context, VP baseline, the benchmark sequence, and per-model commands.
2. Work from the project root `/Users/zhaoxiaotian.0701/projects/rec_fine_ranking` with the venv active (`source .venv/bin/activate`) and `PYTHONPATH=src`.
3. This is a multi-stage benchmark, not a single run. Execute the stages in `experiment-context.md` in order:
   - Stage A: preprocess (skip if `experiments/data/train/*.parquet` exist)
   - Stage B: build_sequences (skip if `experiments/data/train_seq/*.pkl` exist)
   - Stage C: train each of the 7 models in turn (wide_deep, dcn, deepfm, onetrans, rankmixer, unimixer, hyformer) — log at `experiments/runs/<m>/train.log`
   - Stage D: aggregate with `scripts/compare.py`, then commit + push `experiments/results.{csv,md,png}`

## Your Behavior
Use the spml:watchdog skill. For EACH model's training run it will guide you through:
- Launching the training command (`scripts/train.py --model <m> ...`)
- Monitoring `experiments/runs/<m>/train.log` for anomalies (loss not decreasing, NaN/Inf, stalls > 60s, throughput collapse)
- Restarting from the latest `ckpt_step<N>.pt` on environment failures (use `--resume`)
- Running async evaluation on new checkpoints via `scripts/eval.py --model <m> --checkpoint <path> --data-root experiments/data`
- Reporting any non-environment anomaly to the user with a diagnosis (no auto-fix)
- Recording interventions and eval results in experiment-context.md

Move to the next model only after the current one finishes (writes `ckpt_end.pt` and a final eval row in metrics.csv). After all 7 finish, run Stage D and notify the user with the results table.

## Key references (from experiment-context.md)
- Train: `PYTHONPATH=src python scripts/train.py --model <m> --data-root experiments/data --out-dir experiments/runs/<m> --device auto`
- Resume: add `--resume experiments/runs/<m>/ckpt_step<N>.pt`
- Eval: `PYTHONPATH=src python scripts/eval.py --model <m> --checkpoint <path> --data-root experiments/data`
- Sanity baselines: test_auc ≥ 0.65, test_gauc ≥ 0.60, GAUC ≥ AUC×0.9, no NaN/Inf. Initial BCE loss ≈ 0.69.
- The trainer fails loud (RuntimeError) on non-finite loss — if that fires, treat as a real anomaly and report with the diagnosis, do not blindly restart.
