# RecFlow Fine-Ranking Models Benchmark — Design Doc

**Date**: 2026-05-25
**Owner**: zhaoxiaotian.0701
**Status**: Approved for implementation

---

## 1. Hypothesis

Seven fine-ranking (精排) models — Wide&Deep, DCN, DeepFM, OneTrans, RankMixer, UniMixer, HyFormer — can be fairly compared on the RecFlow industrial benchmark when implemented under a shared feature pipeline, shared embedding tables, and aligned parameter / FLOPs budget.

**Expected outcome**: newer token-based architectures (OneTrans / RankMixer / UniMixer / HyFormer) outperform classical baselines (Wide&Deep / DCN / DeepFM) on AUC and GAUC, with HyFormer and OneTrans leading on long-sequence-sensitive metrics.

### Variables
- **Independent variable**: model backbone architecture (one of the seven)
- **Dependent variables**: test AUC, GAUC, LogLoss, backbone parameter count, FLOPs per sample, train time
- **Control variables**: feature pipeline, embedding tables, optimizer, learning rate, batch size, train epochs, train/test split, evaluation protocol

---

## 2. Task & Data

### Task definition
Pointwise CTR prediction. Sample space restricted to `realshow=1` rows (items actually shown to the user). Binary label = `effective_view` (whether the user watched the item effectively).

### Dataset
RecFlow (Kuaishou), already downloaded at `/Users/zhaoxiaotian.0701/data/rec_flow/`.

| Split | Source | Filter | Rows (approx) |
|---|---|---|---|
| Train | `all_stage/*.feather` — 24 days: Jan 13–31 (19 days) + Feb 11–13, 15–16 (5 days; Feb 14 missing) | `realshow == 1` | ~25M |
| Test | `learning/realshow/2024-02-17.feather`, `2024-02-18.feather` | (already filtered to realshow=1) | ~2M |
| User history | `learning/seq_effective_50/2024-02-{17,18}.pkl` for test; training history derived from `seq_effective_50` files keyed by request_id | last 50 effective views | per-sample |

> Note: the user history dictionaries currently materialised on disk cover only the test days. For training, history features are pulled from the same per-day pkl files when present; for older days we will derive a leakage-safe history from each user's prior `effective_view==1` events within the training window during the one-time preprocess step.

### Feature inventory & embedding strategy

| Feature | Vocab strategy | Size | Emb dim |
|---|---|---|---|
| user_id | true vocab | 50K | 16 |
| device_id | true vocab | 50K | 16 |
| age | true vocab | 16 | 8 |
| gender | true vocab | 4 | 4 |
| province | true vocab | 80 | 8 |
| category_level_one | true vocab | 128 | 8 |
| category_level_two | true vocab | 768 | 8 |
| upload_type | true vocab | 32 | 4 |
| **video_id** | hash bucket | **1,048,576 (1M)** | 16 |
| **author_id** | hash bucket | **262,144 (256K)** | 16 |
| duration | log-bucketize → discrete | 64 | 8 |
| playing_time (sequence only) | log-bucketize | 64 | 8 |
| request hour-of-day | derived | 24 | 4 |
| candidate vs. request time gap | log-bucketize | 32 | 4 |
| Sequence | 50 × {video_id, author_id, cat_lv2, cat_lv1, upload_type, playing_time, age_of_item} | — | reuses above tables |

**Cardinality justification** (scanned 3 days, ~143M rows):
- video_id: 10.5M unique with max id 74.8M — full vocab × 16dim × 4B ≈ **3.2 GB → infeasible on 16 GB Mac**. Hashing to 1M buckets compresses to 64 MB.
- author_id: 5.5M unique — hashing to 256K buckets compresses to 16 MB.
- All other categorical features fit in a true vocab table comfortably.

Hashing function: `(murmurhash3(str(id)) % bucket_size) + 1` (reserve 0 for padding).

---

## 3. Models

All models inherit from `BaseRanker` and share the same `FeatureEncoder` (embedding tables + sequence pooling/representation utilities).

### 3.1 Classical baselines (论文原始结构)

| Model | Backbone | Notes |
|---|---|---|
| Wide & Deep (Cheng et al., 2016) | Linear (wide) over cross features + DNN(256, 128, 64) over dense embeddings | sequence pooled by mean |
| DCN (Wang et al., 2017) | Cross network L=3 + DNN(256, 128, 64) | sequence pooled by mean |
| DeepFM (Guo et al., 2017) | FM second-order + DNN(256, 128, 64) | sequence pooled by mean |

### 3.2 Modern token-based architectures (with capacity alignment)

| Model | Paper | Backbone |
|---|---|---|
| OneTrans | Zhang et al., WWW 2026 (arXiv 2510.26104) | unified token sequence (sequence tokens + non-sequence feature tokens) → 4-layer causal Transformer with parameter sharing across sequential tokens |
| RankMixer | Zhu et al., 2025 (arXiv 2507.15551) | feature tokens (T tokens) → multi-head token mixing (parameter-free permutation) → per-token FFN, stacked × N |
| UniMixer | Ha et al., 2026 (arXiv 2604.00590) | local block mixing (per-block W_B) + global mixing (Sinkhorn-Knopp doubly stochastic W_G) |
| HyFormer | Huang et al., 2026 (arXiv 2601.12681) | two-stream: seq-stream + feat-stream; alternating Query Decoding (seq → feat) and Query Boosting (feat → seq) |

### 3.3 Capacity alignment (fair comparison)

- **Backbone params budget** (excludes embedding tables): **5M ± 10%**
- **FLOPs per sample budget** (excludes embedding lookup): **50 MFLOPs ± 15%**
- Embedding tables are identical across all models, contributed by the shared `FeatureEncoder`.
- Each modern model exposes a tunable set of capacity knobs:
  - OneTrans: `(n_layers, d_model, n_heads, ffn_mult)`
  - RankMixer: `(n_layers, n_tokens, d_token, ffn_mult)`
  - UniMixer: `(n_layers, n_blocks, block_dim, ffn_mult)`
  - HyFormer: `(n_layers, d_seq, d_feat, n_heads)`
- `scripts/calibrate_capacity.py` fixes RankMixer at its paper-default ≈5M backbone, then bisects the other three models' layer count / hidden dim until both metrics enter tolerance. Resulting configs are written back to `configs/<model>.yaml`.
- Classical baselines (W&D / DCN / DeepFM) keep their paper-canonical DNN(256, 128, 64) and are not forced into the budget — they are intentionally small baselines.

---

## 4. Training

- Framework: PyTorch ≥ 2.4 (MPS-aware), no `torch.compile` (MPS support uneven).
- Device auto-selection: `mps > cuda > cpu`.
- Loss: `BCEWithLogitsLoss`.
- Optimizer: Adam, lr 1e-3 for backbone, lr 1e-2 for embeddings (sparse-update-friendly). No weight decay on embeddings; 1e-5 on dense layers.
- Batch size: 4096.
- Epochs: 1 (industry convention; RecFlow paper confirms 1 epoch suffices on this scale).
- Gradient clipping: `clip_grad_norm_(1.0)`.
- Mixed precision: off (MPS bf16 still has rough edges in 2.4); revisit if training is too slow.
- Data shuffling: within-day shuffle, then per-day shard concatenation (no global shuffle to bound memory).

---

## 5. Evaluation

### Evaluator design
Evaluator is its own module (`src/rec_fine_ranking/training/evaluator.py`). Trainer decides *when* to evaluate; evaluator decides *how*.

- **Cadence**: every 5000 steps + at epoch end.
- **Scope**: full validation set (Feb 17 + Feb 18, ~2M rows) at every eval — no sub-sampling unless overridden via config.
- **Entry modes**: shared evaluator core supports both (a) in-memory evaluation during training and (b) checkpoint-based evaluation via `scripts/eval.py --checkpoint <path>`.
- **Metrics**:
  - `AUC`: global AUC via `sklearn.metrics.roc_auc_score`.
  - `GAUC`: per-user AUC weighted by per-user sample count; users with all-positive or all-negative labels are excluded from the aggregate.
  - `LogLoss`: `sklearn.metrics.log_loss`.

### Evaluation observability
- Phase-start message: `"[eval] starting @ step=K | val_samples=N"`.
- Dedicated `tqdm` progress bar.
- Phase-end message: `"[eval] done @ step=K | AUC=… GAUC=… LogLoss=… | elapsed=…s"`.
- Efficiency summary: load latency, batches/sec, total wall time.

### Evaluation failure handling
- Checkpoint missing/unreadable → fail loud with explicit path + cause.
- Checkpoint restore failure (state_dict mismatch) → log offending keys, exit non-zero.
- Empty / misconfigured validation dataloader → fail at evaluator init, not mid-loop.
- Metric aggregation failure (e.g., all-positive batch) → log + return `nan` for that metric, do not crash training.
- Non-finite metrics → log warning, persist run as failed, continue training.
- Silent gap > 60 s without progress → emit watchdog warning.

---

## 6. Logging & Comparison

- TensorBoard: `experiments/runs/<model>/` — per-step loss, lr, eval metrics, gradient norm.
- CSV: `experiments/runs/<model>/metrics.csv` — every eval point (step, AUC, GAUC, LogLoss).
- Final aggregation: `scripts/compare.py` reads all `metrics.csv` + `<model>_meta.json`, writes:
  - `experiments/results.csv` with columns `model, params_backbone, params_emb, mflops_per_sample, train_time_sec, AUC, GAUC, LogLoss`.
  - `experiments/results.md` — markdown table sorted by GAUC.
  - `experiments/results.png` — grouped bar chart (matplotlib).

---

## 7. Review Criteria

```yaml
review_criteria:
  metrics:
    - name: test_auc
      direction: ">="
      threshold: 0.65          # Wide&Deep floor for RecFlow CTR
    - name: test_gauc
      direction: ">="
      threshold: 0.60
  performance:
    - "preprocess pipeline completes < 30 min on M5"
    - "single model train (1 epoch on 25M) < 2 hours on M5 MPS"
  observability:
    - "TensorBoard scalar logs: per-step loss / grad_norm / lr / step_time"
    - "per-eval AUC / GAUC / LogLoss appended to metrics.csv"
    - "results.csv aggregated across all 7 models"
  stability:
    - "no NaN / Inf during training"
    - "GAUC >= AUC * 0.9 (sanity)"
  custom:
    - "all 7 models compile and forward on a 32-row toy batch (smoke test)"
    - "embedding tables shared identically across models (verified by inspecting FeatureEncoder identity)"
    - "backbone params within 5M ± 10% and FLOPs within 50 MFLOPs ± 15% for the four modern models"
```

---

## 8. Validation Pyramid

The Validation Pyramid runs once on the single `[INTEGRATION]` subtask — the assembled training pipeline. Individual model classes / dataset / evaluator are validated via standard TDD.

### L0 — `spml:ml-static-checks` (mandatory)
Checks: device consistency, precision, FlashAttention (n/a on MPS), optimizer coverage (all model params covered), scheduler, DataLoader behaviour, loss / speed file output, visualisation tool (TensorBoard ✅), plus 15 advisory checks.

### L1 — `spml:ml-runtime-validator` (mandatory)
- **Data**: mock overfit data — randomly sample 1024 rows from the preprocessed train shard.
- **Training volume**: 50 steps per model.
- **Expectation**: loss decreases monotonically over the window; final loss < initial loss × 0.5; gradient norm finite throughout.
- **Project-specific baselines**: none beyond loss-down + finite gradients; perf baselines are review-criteria-only.

### Per-subtask code validation (not VP)
- `tests/test_dataset.py` — shapes, dtypes, hashing determinism, sequence padding correctness.
- `tests/test_models.py` — forward shape on toy batch, backward gradient flow, param-count assertion.
- `tests/test_metrics.py` — AUC / GAUC / LogLoss against sklearn reference on tiny synthetic data.

---

## 9. Repository

- Local path: `~/projects/rec_fine_ranking/`
- GitHub: public repo `rec_fine_ranking` (created during implementation kickoff).
- Initial commit pushed after preprocess + DataLoader subtasks pass tests.

---

## 10. Open items deferred to implementation planning

- Exact preprocess output schema and on-disk format (likely parquet with int32 columns).
- Per-day sequence assembly for training days (since `learning/seq_effective_50/` only ships test days).
- Whether the calibrate-capacity script bisects or runs a small grid search.

These are routine implementation decisions; `spml:experiment-planning` will produce the detailed step-by-step plan.
