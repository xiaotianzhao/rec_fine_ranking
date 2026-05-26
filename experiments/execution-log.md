# Execution Log — RecFlow Fine-Ranking Benchmark

Subagent-driven execution of `plans/2026-05-25-rec-fine-ranking-implementation.md`.
Subtasks 1–10 completed in prior sessions (see git history). Below: subtasks executed under spml:ml-subagent-dev.

### Subtask 11 Conclusion (code)
**Role:** RankMixer model (arXiv 2507.15551) — token-mix permutation + per-token FFN.
**Result:** implemented
**Evidence:** 3 unit tests pass (forward shape, backward flow, param count). Backbone = 1,762,689.
**Reviews:** Spec ✅ (subclasses BaseRanker, shared encoder, no own embeddings). Quality ✅ (token_mix permutation + einsums verified).
**Files:** `src/rec_fine_ranking/models/rankmixer.py`, `tests/test_rankmixer.py`. Commit `b482ec6`.

### Subtask 12 Conclusion (code)
**Role:** UniMixer model (arXiv 2604.00590) — local mixing + Sinkhorn-Knopp doubly-stochastic global mixing.
**Result:** implemented
**Evidence:** 3 unit tests pass. Backbone = 103,825. Sinkhorn verified near-doubly-stochastic + differentiable.
**Reviews:** Spec ✅. Quality ✅ (einsum dims + grad coverage verified, eps guards divide-by-zero).
**Files:** `src/rec_fine_ranking/models/unimixer.py`, `tests/test_unimixer.py`. Commit `82164a5`.

### Subtask 13 Conclusion (code)
**Role:** HyFormer model (arXiv 2601.12681) — two-stream bidirectional cross-attention.
**Result:** implemented (one minor fixed post-review)
**Evidence:** 3 unit tests pass. Backbone = 465,345 (after dead-weight removal). All params receive gradient.
**Reviews:** Spec ✅ (query→kv-space projection judged functionally-equivalent bidirectional cross-attn). Quality ✅ after fix — final-layer `query_boost` was dead (head pools only feat_t); restructured to n_layers−1 boost blocks each feeding the next decode.
**Files:** `src/rec_fine_ranking/models/hyformer.py`, `tests/test_hyformer.py`. Commits `9568159`, fix `e1d39f2`.

### Subtask 14 Conclusion (code)
**Role:** Evaluator core — AUC/GAUC/LogLoss metrics + Evaluator with two entry modes (in-memory + checkpoint) over one shared `_loop`.
**Result:** implemented
**Evidence:** 7 unit tests pass (4 metrics vs sklearn + GAUC group logic; 3 evaluator incl. empty-loader ValueError and constant-label NaN).
**Reviews:** Spec ✅ (single shared core; mode-aware boundary errors). Quality ✅ (torch.no_grad, NaN-safe aggregation, no leakage).
**Files:** `src/rec_fine_ranking/training/{metrics.py,evaluator.py}`, `tests/test_metrics.py`, `tests/test_evaluator.py`. Commit `e41e865`.

### Registry
`models/__init__.py` populated with `MODEL_REGISTRY` (all 7 models instantiate + forward verified). Commit `fd8da65`.

### Subtask 15 Conclusion (code)
**Role:** Capacity calibrator + run-comparison utilities.
**Result:** implemented
**Evidence:** 2 unit tests pass (FLOPs finite/>0 for all 7 models; compare emits csv/md/png). Full suite 40 passed.
**Reviews:** Spec ✅ + Quality ✅ (combined). fvcore→manual-Linear fallback verified (fvcore tracer IndexErrors on DCN's cross-network → fallback gives finite lower-bound). Nits fixed: dead arithmetic in fallback removed; `compare.run` now raises on empty runs dir. Added `matplotlib>=3.7` to requirements.
**Files:** `src/rec_fine_ranking/utils/flops.py`, `scripts/{calibrate_capacity.py,compare.py}`, `tests/test_flops_and_compare.py`. Commits `4edf334`, `274b13a`.

**Capacity picture (from calibrate_capacity.py, configs unchanged):**
| model | backbone | mflops/sample | in-budget |
|---|---|---|---|
| wide_deep | 84,650 | 0.1 | n/a |
| dcn | 85,657 | 0.2 | n/a |
| deepfm | 113,042 | 0.1 | n/a |
| onetrans | 418,433 | 20.5 | NO |
| rankmixer | 1,762,689 | 1.8 | NO |
| unimixer | 103,825 | 0.1 | NO |
| hyformer | 465,345 | 6.5 | NO |

**Note on capacity budget:** all backbones are currently well under the ~5M / ~50-MFLOP target. Unit tests use `>1000` by design. Budget alignment (tuning each modern model's knobs to ~5M) is the plan's **Subtask 17 Step 3**, performed during the benchmark-run phase — not in the code subtasks. Flagged here so it is not silently dropped.

### Subtask 16 Conclusion (INTEGRATION)
**Hypothesis:** All 7 models, instantiated via `MODEL_REGISTRY[name]()` and trained by the same `Trainer`, converge without NaN/Inf on a controlled run.
**Result:** effective — pipeline runnable and healthy for all 7 models.
**Gates:** TDD (8 integration tests) ✅ → Spec Review ✅ → Quality Review ✅ → L0 ✅ → L1 ✅.

**L0 (ml-static-checks): PASS.** All 9 mandatory checks pass — verified empirically: dual-LR optimizer split covers every trainable param exactly once across all 7 models (e.g. hyformer 66/66, onetrans 42/42, zero overlap/missing); device consistency on MPS; both attention models (OneTrans causal, HyFormer cross-attn) finite through backward+grad-clip. Important findings addressed: MFU baseline corrected to fp32-nominal (`device_peak_tflops` 2.0→1.0, commit `cb04c71`); live tqdm metrics added. Accepted-as-designed: OneTrans/token models attend over zero-padded empty history (phantom token) — applied **uniformly across all 7 models**, so a controlled treatment, not a confound; no key-padding mask, consistent with plan spec.

**L1 (ml-runtime-validator): PASS (6/6 stages, all 7 models).**
- Stage 3 mock-overfit (50 steps, 1024 toy rows) — loss-down + all-finite for every model:

  | model | initial | final | ratio | all_finite |
  |---|---|---|---|---|
  | wide_deep | 0.6953 | 1.1e-07 | 1.6e-07 | ✓ |
  | dcn | 0.6912 | 3.7e-10 | 5.4e-10 | ✓ |
  | deepfm | 0.6952 | 1.2e-27 | 1.7e-27 | ✓ |
  | onetrans | 0.7522 | 2.2e-10 | 2.9e-10 | ✓ |
  | rankmixer | 0.6944 | 9.5e-23 | 1.4e-22 | ✓ |
  | unimixer | 0.6914 | 4.1e-06 | 5.9e-06 | ✓ |
  | hyformer | 0.6908 | 1.4e-11 | 2.0e-11 | ✓ |

  All ratios ≪ 0.5 threshold.
- Stages 1/2/4/5/6 + logging + embedding stability (via `tests/validation/l1_pipeline_check.py`) — **all ✓ for all 7**: data shapes+finite (1); forward (2); train.log/metrics.csv/meta.json/TB/checkpoint produced (3-logging L.1/L.2/L.6/L.7); checkpoint save→load params allclose + optim state + step restored (4); eval-mode inference finite+deterministic (5); Evaluator metrics finite, n_samples>0 (6); embedding norm stable (~78→80, no divergence).

**Anomalies:** none. GAUC is NaN on synthetic toy data (single/uniform user grouping) — expected, guarded before TB logging; logloss is the finite eval signal used here.
**Recommendation:** proceed. Pipeline is the production training script for the benchmark (Subtask 17). Capacity-budget alignment for modern models still pending in Subtask 17 Step 3.

**All 16 dev subtasks complete. 40 unit + 8 integration tests pass. Integration VP passed.**
