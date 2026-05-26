# Experiment Context: RecFlow Fine-Ranking Benchmark (7 models)

## Experiment Design
- **Hypothesis:** Newer token-based architectures (OneTrans / RankMixer / UniMixer / HyFormer) outperform classical baselines (Wide&Deep / DCN / DeepFM) on test AUC and GAUC when given identical embedding tables and a matched **backbone-parameter budget** (~5M).
- **Independent variable:** model architecture (the `body()` between shared encoder output and logit).
- **Dependent variables:** test AUC, GAUC, LogLoss, backbone params, **MFLOPs/sample**, train time.
- **Control variable:** shared `FeatureEncoder` embedding tables (~22.6M emb params, identical across all 7), data pipeline, optimizer (dual-LR Adam), loss (BCEWithLogits), eval set, seeds (42).
- **Validation scope:** L0 (ml-static-checks) + L1 (ml-runtime-validator) — both PASSED on the integration subtask.

## VP Baseline (reference for the Watchdog)
- **L0:** PASS — all 9 mandatory static checks; dual-LR optimizer split covers every trainable param exactly once across all 7 models; MPS device-consistent; attention models finite through backward.
- **L1:** PASS (6/6 stages, all 7 models). Mock-overfit (50 steps, 1024 toy rows) loss-down for every model (ratios 1e-7…1e-27, all finite). Checkpoint save/load, eval-mode inference (finite+deterministic), Evaluator metrics, logging artifacts, embedding-norm stability all verified.
- **Initial loss (BCE):** ~0.69 (≈ln2) for all models — sane init.
- **Gradient health:** finite throughout; trainer fails loud on non-finite loss.
- **Capacity (post-tuning):** modern backbones 4.65M–5.15M (within 5M±10%); MFLOPs/sample 5–252 (measured, not constrained — see note below).

## Capacity Alignment Note
Control axis is **backbone params (5M±10%)**, not FLOPs. The four modern architectures' FLOPs/param ratios differ ~50× so matching both 5M params and 50 MFLOPs is infeasible. MFLOPs/sample is reported as a measured dependent variable. Verify anytime with `PYTHONPATH=src python scripts/calibrate_capacity.py`.

## Benchmark Sequence (run in order)
This is a multi-stage benchmark, not a single run. Run from project root with venv active and `PYTHONPATH=src`.

**Stage A — Preprocess (one-time, ~30–60 min):** skip if `experiments/data/train/*.parquet` already exist.
```
PYTHONPATH=src python scripts/preprocess.py --raw-root /Users/zhaoxiaotian.0701/data/rec_flow --out-root experiments/data
```

**Stage B — Build sequence history (one-time, ~30–60 min):** skip if `experiments/data/train_seq/*.pkl` exist.
```
PYTHONPATH=src python scripts/build_sequences.py --data-root experiments/data
```

**Stage C — Train all 7 models (the long phase).** For each model in order `wide_deep dcn deepfm onetrans rankmixer unimixer hyformer`:
```
PYTHONPATH=src python scripts/train.py --model <m> --data-root experiments/data --out-dir experiments/runs/<m> --device auto
```
- Resume a crashed run with `--resume experiments/runs/<m>/ckpt_step<N>.pt`.
- Eval fires every 5000 steps + end of epoch on full validation (~2M rows); results appended to `experiments/runs/<m>/metrics.csv`.

**Stage D — Aggregate + publish:**
```
PYTHONPATH=src python scripts/compare.py --runs-dir experiments/runs --out-dir experiments/
git add experiments/results.csv experiments/results.md experiments/results.png && git commit -m "results: 7-model RecFlow benchmark" && git push
```

## Training Configuration
- **Script:** `scripts/train.py` (VP-validated production script). Trainer in `src/rec_fine_ranking/training/trainer.py`.
- **Per-model launch command:** see Stage C.
- **Log file (per model):** `experiments/runs/<m>/train.log` (one line per `log_every_steps`: step, loss, grad_norm, lr, step_time_ms, samples/s, mfu).
- **Metrics CSV (per model):** `experiments/runs/<m>/metrics.csv` (step, auc, gauc, logloss).
- **Checkpoint dir:** `experiments/runs/<m>/` — `ckpt_step<N>.pt` every 10000 steps + `ckpt_end.pt`. Resumable (model+optim+step).
- **TensorBoard:** `experiments/runs/<m>/` event files.
- **Eval command (for async checkpoint eval):** `PYTHONPATH=src python scripts/eval.py --model <m> --checkpoint {checkpoint_path} --data-root experiments/data`
- **Key hyperparameters:** batch_size=4096, lr_backbone=1e-3, lr_emb=1e-2, grad_clip=1.0, epochs=1, optimizer=Adam (wd 1e-5 dense / 0 emb), device=auto (MPS).
- **Expected total steps/model:** ~6,100 (≈25M train rows / 4096), 1 epoch.
- **Estimated duration:** classical models fast (<30 min each); onetrans (252 MF) slowest; full 7-model sweep roughly 6–15 h on M-series MPS. Design budget: <2 h/model.

## Review Criteria (from design doc §7)
- metrics: test_auc ≥ 0.65; test_gauc ≥ 0.60
- performance: preprocess < 30 min; single-model train < 2 h on M5 MPS
- observability: TB scalars (loss/grad_norm/lr/step_time); per-eval AUC/GAUC/LogLoss in metrics.csv; results.csv across all 7
- stability: no NaN/Inf; GAUC ≥ AUC × 0.9 (sanity)
- custom: 7 models forward on toy batch; shared embeddings; modern backbones within 5M±10% (✓ done)

## Code State
- **Git commit:** e8c6ed9 (branch `main`), pushed to https://github.com/xiaotianzhao/rec_fine_ranking
- **Key files:** `scripts/{preprocess,build_sequences,train,eval,compare,calibrate_capacity}.py`, `src/rec_fine_ranking/{data,models,training,utils}/`
- **Gitignored (do NOT commit):** `experiments/runs/`, `experiments/data/`, `*.pt`, `*.log`, TB events. Only `experiments/results.{csv,md,png}` get committed.

## Watchdog Status
- Status: not started

## Diagnosis History
(empty)

## Evaluation History
(populated by Watchdog during training)
