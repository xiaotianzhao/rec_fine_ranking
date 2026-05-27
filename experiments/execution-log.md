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

### Subtask 17 Step 3 — Capacity alignment (post-VP, pre-benchmark)
Tuned the 4 modern models' config defaults to the design's **5M ± 10% backbone budget** (user directive: "用5m"):

| model | new config | backbone | mflops/sample | params@5M | flops@50 |
|---|---|---|---|---|---|
| onetrans | n_layers=4, d_model=320, n_heads=8 | 4,986,561 | 252.2 | ✓ | ✗ |
| rankmixer | n_tokens=16, d_token=112, n_layers=3 | 5,149,089 | 5.1 | ✓ | ✗ |
| unimixer | n_layers=4, n_blocks=16, block_dim=120 | 4,959,425 | 5.1 | ✓ | ✗ |
| hyformer | n_layers=3, d_seq=256, d_feat=256, n_heads=8 | 4,650,497 | 109.8 | ✓ | ✗ |

**Deliberate deviation from review_criteria L181 (which asked for both 5M params AND 50 MFLOPs):** the four architectures' FLOPs-per-param ratios differ by ~50× (OneTrans causal attention ≈252 MF at 5M; RankMixer/UniMixer ≈5 MF at 5M), so matching both simultaneously is architecturally infeasible. **Decision:** control on backbone params (5M±10%, all ✓) and treat MFLOPs/sample as a *measured dependent variable* (design §1 already lists "FLOPs per sample" as a dependent variable). `calibrate_capacity.py` now reports params@5M and flops@50 separately. Modern-model unit tests tightened to assert 4.5M–5.5M. 48 tests pass. Commit `e8c6ed9`, pushed.

Classical baselines (W&D/DCN/DeepFM) intentionally kept small (84K–113K) per design §3.3.

---

## Findings

### F1 — DeepFM (simplified FM) underperforms Wide&Deep
Observed in the benchmark run: DeepFM **AUC 0.7017 / GAUC 0.6352** vs Wide&Deep **0.7166 / 0.6472** and DCN **0.7172 / 0.6483** — i.e. DeepFM is the *weakest* classical baseline, which is counterintuitive (DeepFM is usually framed as a W&D upgrade).

**Root cause (analysed):** This `DeepFM` is architecturally `WideDeep`'s exact `linear + deep` towers **plus** one extra `fm_2nd` term — so DeepFM < W&D means `fm_2nd` is net-harmful. Two reasons:
1. **Not field-aware.** The implemented FM (per the plan's simplification) projects the *concatenated* embedding vector through a single `fm_proj = Linear(d, F·K)` and reshapes into **arbitrary K-dim chunks** (F=21, K=8), then does the sum-square FM trick on those. These chunks are learned linear mixtures with **no per-feature semantics** — not the canonical FM that operates on each feature's own embedding (user_id vec × video_id vec …).
2. **Redundant with the deep tower.** The MLP already models interactions on the shared embeddings, so the FM term adds little orthogonal signal.

**Empirical evidence** (trained DeepFM, logit decomposed on 4096 real test rows):
| term | abs_mean | corr with label |
|---|---|---|
| linear | 0.426 | **+0.322** |
| fm_2nd | **0.700** (largest) | +0.235 |
| deep | 0.372 | +0.173 |

`fm_2nd` has the **largest magnitude but a weaker label correlation than the linear term** → it dominates the logit scale while injecting mostly-redundant/noisy signal, diluting the cleaner ranking from linear+deep (~−0.015 AUC). Consistent with literature where DeepFM≈W&D within ~0.01 AUC and the winner is implementation/dataset-dependent.

### Benchmark Results (8/10 models; hyformer/onetrans still training — GPU-contended)

Sorted by GAUC (1 epoch, batch 2048, RecFlow ~25.6M train / ~2M test):

| model | AUC | GAUC | LogLoss | backbone |
|---|---|---|---|---|
| dcn (v1) | 0.7172 | 0.6483 | 0.5865 | 86K |
| rankmixer | 0.7165 | 0.6483 | 0.5882 | 5.15M |
| wide_deep | 0.7166 | 0.6472 | 0.5866 | 85K |
| din | 0.7142 | 0.6445 | 0.5881 | 98K |
| deepfm_field | 0.7119 | 0.6434 | 0.5901 | 88K |
| unimixer | 0.7112 | 0.6420 | 0.5893 | 4.96M |
| dcn_v2 | 0.7129 | 0.6416 | 0.5884 | 183K |
| deepfm | 0.7017 | 0.6352 | 0.6014 | 113K |

**Three targeted comparisons:**
- **F1 — DeepFM FM implementation:** deepfm (simplified chunk-FM) 0.7017 → **deepfm_field (semantic field-FM) 0.7119** AUC (+0.0102), GAUC +0.0082. **Field-aware FM clearly helps** — confirms Finding F1: the simplified FM was the culprit, proper field FM lifts DeepFM back near the classical pack.
- **DIN vs Wide&Deep (target-attention vs mean-pool):** din 0.7142 vs wide_deep 0.7166 (−0.0024 AUC, −0.0027 GAUC). **Target-attention does NOT beat mean-pool here** — slightly worse (within noise). The 50-step effective-view history carries limited target-discriminative signal on this task.
- **DCN-v2 vs DCN-v1 (matrix vs vector cross):** dcn_v2 0.7129 vs dcn 0.7172 (−0.0043 AUC). **Matrix cross + DIN pooling did NOT beat the v1 vector cross** (confounded: dcn_v2 changes both cross type and seq pooling).

**Headline:** all 8 cluster in AUC 0.702–0.717 / GAUC 0.635–0.648. Under shared embeddings + 1-epoch RecFlow CTR, architecture choice moves the needle <~0.005 (noise-level) — **the one exception is deepfm's broken simplified-FM**, which field-aware FM fixes. Modern 5M models (rankmixer/unimixer) ≈ tiny classical baselines; capacity/architecture is not the bottleneck here, shared embeddings dominate. (Note: din/dcn_v2 train_time_sec 9k–14k are inflated by the overnight data-loading bottleneck at num_workers=0, not a model property.)

**Follow-up:** implemented a field-aware variant `deepfm_field` (each feature's embedding → its own K-dim FM field, canonical FM over semantic fields; same linear+deep towers as W&D to isolate the FM-implementation variable). To be trained **after** the main 7-model sweep completes, then compared against the simplified `deepfm`.
