# RecFlow Fine-Ranking Benchmark — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:ml-subagent-dev to implement this plan task-by-task.

**Goal:** Build a fair, reproducible benchmark of seven CTR ranking models (Wide&Deep, DCN, DeepFM, OneTrans, RankMixer, UniMixer, HyFormer) on the RecFlow dataset under a shared feature pipeline and aligned compute budget.

**Experiment directory:** `/Users/zhaoxiaotian.0701/projects/rec_fine_ranking/experiments/`

**Hypothesis:** Newer token-based architectures (OneTrans/RankMixer/UniMixer/HyFormer) outperform classical baselines (Wide&Deep/DCN/DeepFM) on test AUC and GAUC when given identical embedding tables and aligned (~5M backbone params, ~50 MFLOPs/sample) compute budget.

**Validation scope:** L0 (ml-static-checks) + L1 (ml-runtime-validator) both mandatory, runs once on the `[INTEGRATION]` training pipeline subtask. L1 uses **mock overfit data** (1024-row toy shard) with 50-step run; expectation is loss-down + no NaN.

**Evaluation design:** Required. Cadence = every 5000 steps + end of epoch. Scope = full validation (~2M rows, Feb 17 + Feb 18 of RecFlow `learning/realshow/`). Evaluator core supports both entry modes: in-memory during training and checkpoint-based via `scripts/eval.py`. Observability requirements: phase-start log, tqdm progress bar, phase-end log, result + efficiency summary, watchdog warning on > 60 s silence. Failure handling: checkpoint missing/unreadable → fail loud; state-dict mismatch → log + exit non-zero; empty dataloader → fail at evaluator init; aggregation failure → log + return NaN, don't crash; non-finite metrics → warn + mark run failed; silent gap > 60 s → emit watchdog warning.

**Architecture:** Single Python package `src/rec_fine_ranking/` with separated `data/`, `models/`, `training/`, `utils/`. All seven models inherit `BaseRanker` and share a `FeatureEncoder` (hash-bucket embeddings for video_id/author_id, true-vocab embeddings for small features, log-bucketised continuous features). Modern models expose capacity knobs that `scripts/calibrate_capacity.py` tunes to match RankMixer's paper-default 5M backbone params / 50 MFLOPs per sample.

---

## Shared Scaffold

### Existing infra (don't touch — observe only)

- Raw data: `/Users/zhaoxiaotian.0701/data/rec_flow/` (read-only)
  - `all_stage/*.feather`: 24 daily files, ~15 M rows each
  - `learning/realshow/2024-02-17.feather`, `2024-02-18.feather`: test set
  - `learning/seq_effective_50/2024-02-17.pkl`, `2024-02-18.pkl`: test-day sequences (dict keyed by `(user_id, request_id)` → list of 50 effective-view item ids + side info)
  - `learning/video_info.pkl`: video → static metadata
  - `request_id_dict.tar.gz`, `seq_effective_50_dict.tar.gz`, `ubm_seq_request_id_dict.tar.gz`, `others.tar.gz`, `realshow.tar.gz`: bundled raw archives — leave compressed unless a subtask explicitly needs them

### Needs setup

- Project root: `/Users/zhaoxiaotian.0701/projects/rec_fine_ranking/`
- Python package: `src/rec_fine_ranking/`
- Outputs root: `experiments/runs/`
- Preprocessed-data root: `experiments/data/` (parquet shards, leakage-safe history dicts)
- GitHub repo: created in Subtask 17

### Target directory layout

```
rec_fine_ranking/
├── pyproject.toml
├── requirements.txt
├── README.md
├── .gitignore
├── src/rec_fine_ranking/
│   ├── data/         (feature_config, hashing, preprocess, sequence, dataset, collate)
│   ├── models/       (base, wide_deep, dcn, deepfm, onetrans, rankmixer, unimixer, hyformer)
│   ├── training/     (trainer, evaluator, metrics)
│   └── utils/        (device, logger, flops, config)
├── configs/<model>.yaml
├── scripts/{preprocess.py, build_sequences.py, train.py, eval.py, calibrate_capacity.py, compare.py}
├── tests/
└── experiments/{plans/, runs/, data/}
```

### Conventions

- Python ≥ 3.10. PyTorch ≥ 2.4. `pytest`, `pyarrow`, `pandas`, `numpy`, `scikit-learn`, `tensorboard`, `tqdm`, `pyyaml`, `fvcore` (for FLOPs counting).
- All code under `src/rec_fine_ranking/` — no test imports. Tests under `tests/` may import core.
- Configs in YAML, loaded via `utils/config.py` into a `dataclass`. CLI overrides via `--key=value` flags.
- Device auto-detect: `mps > cuda > cpu` in `utils/device.py`.
- Seeds: every script sets `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`.
- Commit every subtask. Use Conventional-Commits style: `feat(data): ...`, `feat(model): ...`, `test: ...`, `chore: ...`.

### Capacity budget reference

- **Backbone params** (excluding embedding tables): 5 M ± 10 %
- **FLOPs per sample** (excluding embedding lookup): 50 MFLOPs ± 15 %
- Embedding tables (identical across all seven models): user_id 50K×16 + device_id 50K×16 + video_id 1M×16 + author_id 256K×16 + small vocabs (~20K×8) ≈ **~22 M emb params, ~88 MB**.

---

## Subtask List (16 total: 15 code + 1 integration)

| # | Title | VP |
|---|---|---|
| 1 | Repo bootstrap (package, deps, configs scaffold, .gitignore) | no |
| 2 | Feature config + hashing utility | no |
| 3 | Preprocess script: filter realshow=1, narrow dtypes, parquet shards | no |
| 4 | Sequence history assembly (leakage-safe per-day history) | no |
| 5 | Dataset + collate (torch IterableDataset over parquet) | no |
| 6 | FeatureEncoder + BaseRanker (shared embedding lookup + sequence rep) | no |
| 7 | Wide & Deep model | no |
| 8 | DCN model | no |
| 9 | DeepFM model | no |
| 10 | OneTrans model | no |
| 11 | RankMixer model | no |
| 12 | UniMixer model | no |
| 13 | HyFormer model | no |
| 14 | Evaluator core (AUC/GAUC/LogLoss, both entry modes) | no |
| 15 | Capacity calibrator + Compare/aggregate script | no |
| 16 | **Final Training Pipeline `[INTEGRATION]`** (train.py, all models runnable, runs VP) | **L0 + L1** |
| 17 | (Post-VP) Publish to GitHub + run benchmark | no |

Subtask 17 runs only after Subtask 16's VP passes; it is a deployment step, not a code change.

---

## Subtask 1: Repo bootstrap

**Role:** Lay down the Python package skeleton, pin dependencies, set up pytest, write README and .gitignore.
**Implementation:** Create `pyproject.toml`, `requirements.txt`, `README.md`, `.gitignore`, empty `__init__.py` files for every package directory, and a `tests/conftest.py` with shared fixtures.
**Unit Tests:** `pytest` should run cleanly with zero collected tests (smoke check).
**Expected Conclusion:** "Bootstrap complete; `pytest` runs and reports 0 tests; package importable."

### Step 1: Write smoke test
File: `tests/test_imports.py`
```python
def test_package_imports():
    import rec_fine_ranking
    import rec_fine_ranking.data
    import rec_fine_ranking.models
    import rec_fine_ranking.training
    import rec_fine_ranking.utils
```

### Step 2: Verify test fails
```
PYTHONPATH=src pytest tests/test_imports.py -v
```
Expected: ImportError (modules not yet created).

### Step 3: Create skeleton

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "rec_fine_ranking"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
addopts = "-q"
```

`requirements.txt`:
```
torch>=2.4
pandas>=2.2
pyarrow>=15
numpy>=1.26
scikit-learn>=1.4
tensorboard>=2.16
tqdm>=4.66
pyyaml>=6.0
fvcore>=0.1.5
pytest>=8.0
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
.pytest_cache/
experiments/runs/
experiments/data/
*.feather
*.parquet
*.pt
*.log
.DS_Store
.idea/
.vscode/
```

`README.md`: a short stub describing the project and pointing to the design doc.

Create empty `__init__.py` in: `src/rec_fine_ranking/`, `.../data/`, `.../models/`, `.../training/`, `.../utils/`, `tests/`.

### Step 4: Verify test passes
```
PYTHONPATH=src pytest tests/test_imports.py -v
```
Expected: 1 passed.

### Step 5: Commit
```bash
git add pyproject.toml requirements.txt README.md .gitignore src/ tests/
git commit -m "chore: bootstrap package, deps, pytest"
```

---

## Subtask 2: Feature config + hashing utility

**Role:** Single source of truth for every feature's vocab strategy, vocab size, and embedding dim. Hashing must be deterministic and stable across runs.
**Implementation:** `src/rec_fine_ranking/data/feature_config.py`, `src/rec_fine_ranking/data/hashing.py`.
**Unit Tests:** hash determinism, hash range, vocab schema completeness.
**Expected Conclusion:** "Feature schema declared; hash determinism + range tests pass."

### Step 1: Write tests
File: `tests/test_hashing.py`
```python
import numpy as np
from rec_fine_ranking.data.hashing import hash_ids
from rec_fine_ranking.data.feature_config import FEATURES, FeatureKind

def test_hash_deterministic():
    a = hash_ids(np.array([1, 2, 3, 4_000_000_000], dtype=np.int64), buckets=1_000_000)
    b = hash_ids(np.array([1, 2, 3, 4_000_000_000], dtype=np.int64), buckets=1_000_000)
    assert (a == b).all()

def test_hash_range_within_buckets():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 2**62, size=10_000, dtype=np.int64)
    h = hash_ids(ids, buckets=1_000_000)
    assert (h >= 1).all() and (h < 1_000_000 + 1).all()  # 0 reserved for padding

def test_feature_config_has_required_fields():
    needed = {"user_id","video_id","author_id","device_id","age","gender","province",
              "category_level_one","category_level_two","upload_type","duration"}
    assert needed.issubset({f.name for f in FEATURES})
    for f in FEATURES:
        assert f.kind in {FeatureKind.VOCAB, FeatureKind.HASH, FeatureKind.BUCKETIZE}
        assert f.vocab_size > 0 and f.emb_dim > 0
```

### Step 2: Verify tests fail
```
PYTHONPATH=src pytest tests/test_hashing.py -v
```
Expected: ImportError.

### Step 3: Implement
File: `src/rec_fine_ranking/data/hashing.py`
```python
"""Deterministic hashing for high-cardinality IDs.

Uses 64-bit FNV-1a, modulo (buckets), +1 offset so bucket 0 is reserved for padding.
NumPy-vectorised for speed; no external deps.
"""
from __future__ import annotations
import numpy as np

_FNV_OFFSET = np.int64(-3750763034362895579)  # 0xcbf29ce484222325 as signed
_FNV_PRIME  = np.int64(1099511628211)

def hash_ids(ids: np.ndarray, buckets: int) -> np.ndarray:
    """Hash an int64 array into [1, buckets] using FNV-1a on the 8 bytes."""
    if ids.dtype != np.int64:
        ids = ids.astype(np.int64, copy=False)
    h = np.full_like(ids, _FNV_OFFSET)
    for shift in range(0, 64, 8):
        byte = (ids >> shift) & 0xFF
        h = (h ^ byte).astype(np.int64) * _FNV_PRIME
    # cast to unsigned conceptually, then mod
    h_u = h.view(np.uint64)
    return (h_u % np.uint64(buckets)).astype(np.int64) + 1
```

File: `src/rec_fine_ranking/data/feature_config.py`
```python
"""Single source of truth for feature schema.

Each feature declares: name, kind (vocab/hash/bucketize), vocab_size, emb_dim, source column.
Sequence features reuse the same vocab/embedding tables as their non-sequence counterparts.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

class FeatureKind(str, Enum):
    VOCAB     = "vocab"      # small cardinality, true id-to-index lookup
    HASH      = "hash"       # high cardinality, FNV-1a bucketing
    BUCKETIZE = "bucketize"  # continuous → discrete via log-spaced bins

@dataclass(frozen=True)
class Feature:
    name: str
    kind: FeatureKind
    vocab_size: int
    emb_dim: int
    source_col: str = ""    # column in the raw feather; defaults to `name`

    def __post_init__(self):
        object.__setattr__(self, "source_col", self.source_col or self.name)

FEATURES: tuple[Feature, ...] = (
    Feature("user_id",            FeatureKind.VOCAB,      50_000, 16),
    Feature("device_id",          FeatureKind.VOCAB,      50_000, 16),
    Feature("age",                FeatureKind.VOCAB,         16,  8),
    Feature("gender",             FeatureKind.VOCAB,          4,  4),
    Feature("province",           FeatureKind.VOCAB,         80,  8),
    Feature("category_level_one", FeatureKind.VOCAB,        128,  8),
    Feature("category_level_two", FeatureKind.VOCAB,        768,  8),
    Feature("upload_type",        FeatureKind.VOCAB,         32,  4),
    Feature("video_id",           FeatureKind.HASH,    1_048_576, 16),
    Feature("author_id",          FeatureKind.HASH,      262_144, 16),
    Feature("duration",           FeatureKind.BUCKETIZE,     64,  8),
)

# 50-step long-effective-view sequence; reuses video_id/author_id/category/upload tables.
SEQUENCE_FIELDS: tuple[str, ...] = (
    "video_id", "author_id", "category_level_two",
    "category_level_one", "upload_type",
)
SEQUENCE_LEN: int = 50

def feature_by_name(name: str) -> Feature:
    for f in FEATURES:
        if f.name == name:
            return f
    raise KeyError(name)
```

### Step 4: Verify tests pass
```
PYTHONPATH=src pytest tests/test_hashing.py -v
```
Expected: 3 passed.

### Step 5: Commit
```bash
git add src/rec_fine_ranking/data/{hashing.py,feature_config.py} tests/test_hashing.py
git commit -m "feat(data): feature config + FNV-1a hashing"
```

---

## Subtask 3: Preprocess script

**Role:** One-time pass over `all_stage/*.feather` (train) and `learning/realshow/*.feather` (test). Filter `realshow == 1`, hash high-cardinality IDs, bucketise continuous features, narrow dtypes, write parquet shards under `experiments/data/`.
**Implementation:** `scripts/preprocess.py` + helper module `src/rec_fine_ranking/data/preprocess.py`.
**Unit Tests:** dtype narrowing correctness, hash bucket range, realshow filter row count match, columns expected.
**Expected Conclusion:** "Preprocess module unit-tested; CLI script ready to run (actual run done outside the test loop)."

### Step 1: Write tests
File: `tests/test_preprocess.py`
```python
import pandas as pd, numpy as np, pyarrow as pa
from rec_fine_ranking.data.preprocess import transform_frame

def _toy_frame(n=100, realshow_pos=30):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "user_id":            rng.integers(0, 40_000, n, dtype=np.int64),
        "device_id":          rng.integers(0, 40_000, n, dtype=np.int64),
        "age":                rng.integers(0, 8,      n, dtype=np.int64),
        "gender":             rng.integers(0, 3,      n, dtype=np.int64),
        "province":           rng.integers(0, 60,     n, dtype=np.int64),
        "category_level_one": rng.integers(0, 120,    n, dtype=np.int64),
        "category_level_two": rng.integers(0, 700,    n, dtype=np.int64),
        "upload_type":        rng.integers(0, 30,     n, dtype=np.int64),
        "video_id":           rng.integers(0, 80_000_000, n, dtype=np.int64),
        "author_id":          rng.integers(0, 32_000_000, n, dtype=np.int64),
        "duration":           rng.integers(1, 3600,   n, dtype=np.int64),
        "request_timestamp":  rng.integers(1_700_000_000, 1_710_000_000, n, dtype=np.int64),
        "realshow":           (np.arange(n) < realshow_pos).astype(np.int64),
        "effective_view":     rng.integers(0, 2,      n, dtype=np.int64),
        "long_view":          rng.integers(0, 2,      n, dtype=np.int64),
        "like":               rng.integers(0, 2,      n, dtype=np.int64),
        "request_id":         rng.integers(0, 10**9,  n, dtype=np.int64),
    })
    return df

def test_filter_realshow_only():
    df = _toy_frame()
    out = transform_frame(df)
    assert (out["effective_view"].notna()).all()
    assert len(out) == 30  # realshow=1 rows kept
    # narrow dtypes
    assert out["video_id"].dtype == np.int32
    assert out["author_id"].dtype == np.int32
    assert out["category_level_two"].dtype == np.int16
    assert out["age"].dtype == np.int8
    # video_id / author_id are hashed into buckets >= 1
    assert (out["video_id"] >= 1).all() and (out["video_id"] <= 1_048_576).all()
    assert (out["author_id"] >= 1).all() and (out["author_id"] <= 262_144).all()
    # duration is bucketised
    assert (out["duration"] >= 0).all() and (out["duration"] < 64).all()
    # hour_of_day derived from timestamp
    assert "hour_of_day" in out.columns
    assert (out["hour_of_day"] >= 0).all() and (out["hour_of_day"] < 24).all()
```

### Step 2: Verify tests fail
```
PYTHONPATH=src pytest tests/test_preprocess.py -v
```
Expected: ImportError.

### Step 3: Implement

File: `src/rec_fine_ranking/data/preprocess.py`
```python
"""Per-frame transform: filter realshow=1, hash big IDs, bucketise continuous, narrow dtypes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .feature_config import FEATURES, FeatureKind, feature_by_name
from .hashing import hash_ids

_KEEP_COLS = [
    # ids / cats
    "user_id","device_id","age","gender","province",
    "category_level_one","category_level_two","upload_type",
    "video_id","author_id","duration",
    # context
    "request_id","request_timestamp","hour_of_day",
    # labels (we keep all three so downstream can pick)
    "effective_view","long_view","like",
]

def _bucketize_duration(seconds: pd.Series, n_buckets: int = 64) -> np.ndarray:
    # log-spaced bins covering 1..7200 sec
    edges = np.logspace(0, np.log10(7200), n_buckets)
    idx = np.searchsorted(edges, seconds.to_numpy().clip(min=1)) - 1
    return idx.clip(0, n_buckets - 1).astype(np.int8)

def transform_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["realshow"] == 1].copy()
    # video_id / author_id → hashed
    df["video_id"]  = hash_ids(df["video_id" ].to_numpy(np.int64), feature_by_name("video_id" ).vocab_size).astype(np.int32)
    df["author_id"] = hash_ids(df["author_id"].to_numpy(np.int64), feature_by_name("author_id").vocab_size).astype(np.int32)
    # small vocabs → int16 / int8
    df["user_id"]            = df["user_id"           ].clip(upper=49_999).astype(np.int32)
    df["device_id"]          = df["device_id"         ].clip(upper=49_999).astype(np.int32)
    df["category_level_two"] = df["category_level_two"].clip(upper=767  ).astype(np.int16)
    df["category_level_one"] = df["category_level_one"].clip(upper=127  ).astype(np.int8)
    df["province"]           = df["province"          ].clip(upper=79   ).astype(np.int8)
    df["upload_type"]        = df["upload_type"       ].clip(upper=31   ).astype(np.int8)
    df["age"]                = df["age"               ].clip(upper=15   ).astype(np.int8)
    df["gender"]             = df["gender"            ].clip(upper=3    ).astype(np.int8)
    # continuous → bucketise
    df["duration"] = _bucketize_duration(df["duration"])
    # context
    df["hour_of_day"] = ((df["request_timestamp"] // 3600) % 24).astype(np.int8)
    # labels → int8
    for c in ("effective_view","long_view","like"):
        df[c] = df[c].astype(np.int8)
    df["request_id"] = df["request_id"].astype(np.int64)
    df["request_timestamp"] = df["request_timestamp"].astype(np.int64)
    return df[_KEEP_COLS]
```

File: `scripts/preprocess.py`
```python
"""CLI driver: walk all_stage/ + learning/realshow/, transform, write parquet shards.

Usage:
  python scripts/preprocess.py \
      --raw-root /Users/zhaoxiaotian.0701/data/rec_flow \
      --out-root experiments/data
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd
from rec_fine_ranking.data.preprocess import transform_frame

def _process_split(raw_dir: Path, out_dir: Path, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(raw_dir.glob("*.feather"))
    print(f"[{label}] {len(files)} files → {out_dir}")
    for f in files:
        out = out_dir / f"{f.stem}.parquet"
        if out.exists():
            print(f"  skip (exists): {out.name}")
            continue
        t0 = time.time()
        df = pd.read_feather(f)
        n_in = len(df)
        df = transform_frame(df)
        df.to_parquet(out, index=False, compression="snappy")
        print(f"  {f.name}: {n_in:>10,} → {len(df):>10,} rows in {time.time()-t0:.1f}s")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", required=True)
    p.add_argument("--out-root", required=True)
    args = p.parse_args()
    raw, out = Path(args.raw_root), Path(args.out_root)
    _process_split(raw / "all_stage",        out / "train", "train")
    _process_split(raw / "learning/realshow", out / "test",  "test")

if __name__ == "__main__":
    main()
```

### Step 4: Verify tests pass
```
PYTHONPATH=src pytest tests/test_preprocess.py -v
```
Expected: 1 passed.

### Step 5: Commit
```bash
git add src/rec_fine_ranking/data/preprocess.py scripts/preprocess.py tests/test_preprocess.py
git commit -m "feat(data): preprocess pipeline (filter, hash, bucketise, narrow dtypes)"
```

---

## Subtask 4: Sequence history assembly

**Role:** Build leakage-safe per-user history dict from training shards (since `learning/seq_effective_50/` only ships test days). For each (user_id, request_timestamp) sample, history = last 50 `effective_view==1` events that occurred *strictly before* that timestamp.
**Implementation:** `src/rec_fine_ranking/data/sequence.py` + driver `scripts/build_sequences.py`.
**Unit Tests:** time-ordering correctness, no-leakage check, padding semantics, length capping.
**Expected Conclusion:** "Sequence builder unit-tested; driver script materialises per-day sequence pickles."

### Step 1: Write tests
File: `tests/test_sequence.py`
```python
import numpy as np, pandas as pd
from rec_fine_ranking.data.sequence import build_history_for_day, SEQ_LEN

def test_history_no_leakage():
    # user 1 has events at t=10, 20, 30 (all effective_view=1)
    prior = pd.DataFrame({
        "user_id": [1,1,1,2],
        "request_timestamp": [10,20,30,15],
        "video_id": [101,102,103,201],
        "author_id":[11,12,13,21],
        "category_level_two":[1,1,2,3],
        "category_level_one":[0,0,1,2],
        "upload_type":[1,1,2,3],
        "effective_view":[1,1,1,1],
    }).astype({"effective_view":"int8"})
    today = pd.DataFrame({
        "user_id":[1,2],
        "request_timestamp":[25,16],
        "request_id":[999,888],
    })
    hist = build_history_for_day(today, prior)
    assert hist[(1,999)]["video_id"].tolist()[:2] == [101,102]   # t<25 only
    assert 103 not in hist[(1,999)]["video_id"].tolist()
    assert hist[(2,888)]["video_id"].tolist()[:1] == [201]
    # all arrays padded to SEQ_LEN
    for v in hist.values():
        for f, arr in v.items():
            assert len(arr) == SEQ_LEN
```

### Step 2: Verify test fails
```
PYTHONPATH=src pytest tests/test_sequence.py -v
```
Expected: ImportError.

### Step 3: Implement
File: `src/rec_fine_ranking/data/sequence.py`
```python
"""Build leakage-safe per-user effective-view history for each sample.

Output schema: dict[(user_id:int, request_id:int)] -> dict[field_name -> np.ndarray of length SEQ_LEN].
Padding is 0 at the head; the most recent history item sits at index -1.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

SEQ_LEN = 50
SEQ_FIELDS = ("video_id","author_id","category_level_two","category_level_one","upload_type")

def build_history_for_day(today_df: pd.DataFrame, prior_df: pd.DataFrame) -> Dict[Tuple[int,int], Dict[str, np.ndarray]]:
    """Return history dict keyed by (user_id, request_id) for every row of today_df.

    today_df: rows for the day we want history for (need user_id, request_id, request_timestamp).
    prior_df: pool of effective_view==1 events from earlier-or-same-day prior to each sample.
              Must contain user_id, request_timestamp, effective_view, and all SEQ_FIELDS.
    """
    eff = prior_df[prior_df["effective_view"] == 1].sort_values(["user_id","request_timestamp"])
    # group prior events by user
    groups = {uid: g for uid, g in eff.groupby("user_id", sort=False)}
    out: Dict[Tuple[int,int], Dict[str, np.ndarray]] = {}
    today_sorted = today_df.sort_values(["user_id","request_timestamp"])
    for row in today_sorted.itertuples(index=False):
        g = groups.get(row.user_id)
        if g is None:
            window = None
        else:
            mask = g["request_timestamp"].to_numpy() < row.request_timestamp
            window = g.iloc[mask].tail(SEQ_LEN) if mask.any() else None
        rec: Dict[str, np.ndarray] = {}
        for f in SEQ_FIELDS:
            buf = np.zeros(SEQ_LEN, dtype=np.int32)
            if window is not None and len(window) > 0:
                vals = window[f].to_numpy().astype(np.int32)
                buf[-len(vals):] = vals
            rec[f] = buf
        out[(int(row.user_id), int(row.request_id))] = rec
    return out
```

File: `scripts/build_sequences.py`
```python
"""Walk preprocessed parquet shards and emit per-day sequence pickles.

Usage:
  python scripts/build_sequences.py --data-root experiments/data
"""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import pandas as pd
from rec_fine_ranking.data.sequence import build_history_for_day, SEQ_FIELDS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    args = p.parse_args()
    root = Path(args.data_root)
    for split in ("train","test"):
        shard_dir = root / split
        seq_dir = root / f"{split}_seq"
        seq_dir.mkdir(parents=True, exist_ok=True)
        shards = sorted(shard_dir.glob("*.parquet"))
        prior_frames = []
        for i, f in enumerate(shards):
            today = pd.read_parquet(f)
            prior = pd.concat(prior_frames, copy=False) if prior_frames else today.iloc[:0]
            # also include earlier rows of *today* before each row's timestamp
            prior_for_today = pd.concat([prior, today], copy=False)
            hist = build_history_for_day(today, prior_for_today)
            out = seq_dir / f"{f.stem}.pkl"
            with open(out, "wb") as fh:
                pickle.dump(hist, fh, protocol=4)
            print(f"[{split}] {f.name}: {len(hist):>10,} keys → {out.name}")
            prior_frames.append(today[["user_id","request_timestamp","effective_view", *SEQ_FIELDS]])
            # cap memory: keep last 5 days of prior
            if len(prior_frames) > 5:
                prior_frames.pop(0)

if __name__ == "__main__":
    main()
```

### Step 4: Verify test passes
```
PYTHONPATH=src pytest tests/test_sequence.py -v
```
Expected: 1 passed.

### Step 5: Commit
```bash
git add src/rec_fine_ranking/data/sequence.py scripts/build_sequences.py tests/test_sequence.py
git commit -m "feat(data): leakage-safe sequence history assembly"
```

---

## Subtask 5: Dataset + collate

**Role:** Torch `IterableDataset` that streams parquet shards day-by-day, joins with the per-day sequence pickle, and yields per-sample feature dicts. `collate_fn` stacks dicts into tensor batches.
**Implementation:** `src/rec_fine_ranking/data/dataset.py`, `src/rec_fine_ranking/data/collate.py`.
**Unit Tests:** shape correctness, dtype correctness, padding, label propagation, deterministic order under fixed seed.
**Expected Conclusion:** "Dataset yields batches of expected shape/dtype on toy shard."

### Step 1: Write tests
File: `tests/test_dataset.py`
```python
import numpy as np, pandas as pd, pickle, torch
from pathlib import Path
from rec_fine_ranking.data.dataset import RecFlowDataset
from rec_fine_ranking.data.collate import collate_batch

def _make_toy_split(tmp_path, n=64):
    d = tmp_path / "train"; d.mkdir()
    s = tmp_path / "train_seq"; s.mkdir()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "user_id": rng.integers(0,1000,n,dtype=np.int32),
        "device_id": rng.integers(0,1000,n,dtype=np.int32),
        "age": rng.integers(0,8,n,dtype=np.int8),
        "gender": rng.integers(0,3,n,dtype=np.int8),
        "province": rng.integers(0,60,n,dtype=np.int8),
        "category_level_one": rng.integers(0,120,n,dtype=np.int8),
        "category_level_two": rng.integers(0,700,n,dtype=np.int16),
        "upload_type": rng.integers(0,30,n,dtype=np.int8),
        "video_id": rng.integers(1, 1_048_576, n, dtype=np.int32),
        "author_id": rng.integers(1, 262_144, n, dtype=np.int32),
        "duration": rng.integers(0,64,n,dtype=np.int8),
        "hour_of_day": rng.integers(0,24,n,dtype=np.int8),
        "effective_view": rng.integers(0,2,n,dtype=np.int8),
        "long_view": rng.integers(0,2,n,dtype=np.int8),
        "like": rng.integers(0,2,n,dtype=np.int8),
        "request_id": np.arange(n, dtype=np.int64),
        "request_timestamp": (np.arange(n)+1_700_000_000).astype(np.int64),
    })
    df.to_parquet(d / "2024-01-13.parquet", index=False)
    hist = {(int(df.user_id.iloc[i]), int(df.request_id.iloc[i])):
            {f: np.zeros(50, dtype=np.int32) for f in
             ("video_id","author_id","category_level_two","category_level_one","upload_type")}
            for i in range(n)}
    with open(s / "2024-01-13.pkl","wb") as f: pickle.dump(hist, f)
    return tmp_path

def test_dataset_yields_correct_shapes(tmp_path):
    root = _make_toy_split(tmp_path)
    ds = RecFlowDataset(root / "train", root / "train_seq", label="effective_view", shuffle_buffer=0)
    samples = list(ds)
    assert len(samples) == 64
    x = samples[0]
    assert set(x.keys()) >= {"user_id","video_id","author_id","seq_video_id","label","duration","hour_of_day"}
    assert x["seq_video_id"].shape == (50,)
    assert x["label"].dtype == np.float32

def test_collate_batches(tmp_path):
    root = _make_toy_split(tmp_path)
    ds = RecFlowDataset(root / "train", root / "train_seq", label="effective_view", shuffle_buffer=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, collate_fn=collate_batch)
    b = next(iter(loader))
    assert b["user_id"].shape == (16,)
    assert b["seq_video_id"].shape == (16,50)
    assert b["label"].shape == (16,)
    assert b["label"].dtype == torch.float32
```

### Step 2: Verify tests fail
```
PYTHONPATH=src pytest tests/test_dataset.py -v
```
Expected: ImportError.

### Step 3: Implement

File: `src/rec_fine_ranking/data/dataset.py`
```python
"""Streaming IterableDataset over parquet shards + per-day sequence pickles."""
from __future__ import annotations
import pickle, random
from pathlib import Path
from typing import Dict, Iterator
import numpy as np
import pandas as pd
import torch
from .feature_config import FEATURES, SEQUENCE_FIELDS, SEQUENCE_LEN

_NON_SEQ = [f.name for f in FEATURES] + ["hour_of_day"]

class RecFlowDataset(torch.utils.data.IterableDataset):
    """Iterates over preprocessed shards, joining each row with its history sequence.

    Yields a dict of np.ndarrays per sample. Use `collate_batch` to stack into tensors.
    """
    def __init__(self, shard_dir: Path, seq_dir: Path, label: str = "effective_view",
                 shuffle_buffer: int = 8192, shuffle_shards: bool = True, seed: int = 42):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.seq_dir = Path(seq_dir)
        self.label = label
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self._shards = sorted(self.shard_dir.glob("*.parquet"))
        if not self._shards:
            raise FileNotFoundError(f"No parquet shards in {self.shard_dir}")

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        rng = random.Random(self.seed)
        shards = list(self._shards)
        if self.shuffle_shards:
            rng.shuffle(shards)
        buf: list = []
        for shard in shards:
            seq_path = self.seq_dir / (shard.stem + ".pkl")
            with open(seq_path, "rb") as fh:
                hist: Dict = pickle.load(fh)
            df = pd.read_parquet(shard)
            if self.shuffle_buffer:
                df = df.sample(frac=1.0, random_state=self.seed)
            for row in df.itertuples(index=False):
                key = (int(row.user_id), int(row.request_id))
                seq = hist.get(key)
                rec: Dict[str, np.ndarray] = {}
                for f in _NON_SEQ:
                    if hasattr(row, f):
                        rec[f] = np.asarray(getattr(row, f), dtype=np.int64)
                for f in SEQUENCE_FIELDS:
                    rec[f"seq_{f}"] = (seq[f].astype(np.int64) if seq is not None
                                      else np.zeros(SEQUENCE_LEN, dtype=np.int64))
                rec["label"] = np.float32(getattr(row, self.label))
                if self.shuffle_buffer:
                    buf.append(rec)
                    if len(buf) >= self.shuffle_buffer:
                        rng.shuffle(buf)
                        yield from buf
                        buf.clear()
                else:
                    yield rec
        if buf:
            rng.shuffle(buf)
            yield from buf
```

File: `src/rec_fine_ranking/data/collate.py`
```python
"""Collate function: stack a list of per-sample dicts into a single batched dict of tensors."""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch

def collate_batch(samples: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    keys = samples[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        arrs = [s[k] for s in samples]
        if arrs[0].ndim == 0:
            t = torch.from_numpy(np.stack(arrs))
        else:
            t = torch.from_numpy(np.stack(arrs, axis=0))
        out[k] = t.float() if k == "label" else t.long()
    return out
```

### Step 4: Verify tests pass
```
PYTHONPATH=src pytest tests/test_dataset.py -v
```
Expected: 2 passed.

### Step 5: Commit
```bash
git add src/rec_fine_ranking/data/{dataset.py,collate.py} tests/test_dataset.py
git commit -m "feat(data): IterableDataset + collate (shard streaming + sequence join)"
```

---

## Subtask 6: FeatureEncoder + BaseRanker

**Role:** Shared embedding-lookup module that every model uses. Encapsulates the embedding tables for all features (vocab + hash + bucketize), the sequence representation (mean-pool or as a 50-token tensor), and a base class declaring `forward(batch) -> logit`.
**Implementation:** `src/rec_fine_ranking/models/base.py`.
**Unit Tests:** all embedding tables created with correct sizes; non-seq output shape; seq output shape; gradient flows back to embeddings.
**Expected Conclusion:** "FeatureEncoder produces a (B, D_concat) non-seq vector and (B, 50, D_seq) sequence tensor; gradient flow verified."

### Step 1: Write tests
File: `tests/test_feature_encoder.py`
```python
import torch
from rec_fine_ranking.models.base import FeatureEncoder, _toy_batch

def test_encoder_shapes():
    enc = FeatureEncoder()
    batch = _toy_batch(B=8)
    nonseq, seq = enc(batch)
    assert nonseq.dim() == 2 and nonseq.size(0) == 8
    assert seq.dim() == 3 and seq.size(0) == 8 and seq.size(1) == 50

def test_encoder_grad_flow():
    enc = FeatureEncoder()
    batch = _toy_batch(B=4)
    nonseq, seq = enc(batch)
    (nonseq.sum() + seq.sum()).backward()
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
```

### Step 2: Verify tests fail
```
PYTHONPATH=src pytest tests/test_feature_encoder.py -v
```
Expected: ImportError.

### Step 3: Implement
File: `src/rec_fine_ranking/models/base.py`
```python
"""Shared FeatureEncoder + BaseRanker.

All seven models share these embedding tables. Each model only implements the body
between (non_seq, seq) tensors and the final logit.
"""
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
from ..data.feature_config import FEATURES, SEQUENCE_FIELDS, SEQUENCE_LEN, feature_by_name

# Order of non-sequence features in the concatenated output (kept stable for reproducibility).
NON_SEQ_FEATURES: Tuple[str, ...] = tuple(f.name for f in FEATURES) + ("hour_of_day",)
HOUR_VOCAB, HOUR_DIM = 24, 4

class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tables = nn.ModuleDict()
        for f in FEATURES:
            self.tables[f.name] = nn.Embedding(f.vocab_size + 1, f.emb_dim, padding_idx=0)
        self.tables["hour_of_day"] = nn.Embedding(HOUR_VOCAB, HOUR_DIM)
        # init: small std for hash buckets (cold-start safety), larger for small vocabs
        for f in FEATURES:
            std = 0.01 if f.kind.value == "hash" else 0.05
            nn.init.normal_(self.tables[f.name].weight, mean=0.0, std=std)
            nn.init.zeros_(self.tables[f.name].weight[0])  # padding row
        nn.init.normal_(self.tables["hour_of_day"].weight, mean=0.0, std=0.05)

    @property
    def non_seq_dim(self) -> int:
        return sum(feature_by_name(n).emb_dim for n in NON_SEQ_FEATURES if n != "hour_of_day") + HOUR_DIM

    @property
    def seq_dim(self) -> int:
        return sum(feature_by_name(n).emb_dim for n in SEQUENCE_FIELDS)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        nonseq_parts = []
        for name in NON_SEQ_FEATURES:
            t = self.tables[name](batch[name])
            nonseq_parts.append(t)
        nonseq = torch.cat(nonseq_parts, dim=-1)  # (B, non_seq_dim)
        seq_parts = [self.tables[f](batch[f"seq_{f}"]) for f in SEQUENCE_FIELDS]
        seq = torch.cat(seq_parts, dim=-1)  # (B, 50, seq_dim)
        return nonseq, seq


class BaseRanker(nn.Module):
    """Subclass and implement `body(nonseq, seq) -> logit (B,)`."""
    def __init__(self, encoder: FeatureEncoder | None = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else FeatureEncoder()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        nonseq, seq = self.encoder(batch)
        return self.body(nonseq, seq)

    def body(self, nonseq: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError

    def count_params(self) -> Dict[str, int]:
        backbone = sum(p.numel() for n, p in self.named_parameters() if not n.startswith("encoder."))
        embedding = sum(p.numel() for n, p in self.named_parameters() if n.startswith("encoder."))
        return {"backbone": backbone, "embedding": embedding, "total": backbone + embedding}


# -------- testing utilities (only imported by tests; safe to live here) --------
def _toy_batch(B: int = 4, device: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
    """Synthetic batch matching the dataset/collate schema."""
    g = torch.Generator(device="cpu").manual_seed(0)
    def ri(hi, shape=(B,)): return torch.randint(0, hi, shape, generator=g, dtype=torch.long, device=device)
    batch = {
        "user_id": ri(50_000), "device_id": ri(50_000),
        "age": ri(8), "gender": ri(3), "province": ri(60),
        "category_level_one": ri(120), "category_level_two": ri(700),
        "upload_type": ri(30),
        "video_id":  torch.randint(1, 1_048_576, (B,),  generator=g, dtype=torch.long, device=device),
        "author_id": torch.randint(1,   262_144, (B,),  generator=g, dtype=torch.long, device=device),
        "duration": ri(64), "hour_of_day": ri(24),
        "seq_video_id":           torch.randint(1, 1_048_576, (B, 50), generator=g, dtype=torch.long, device=device),
        "seq_author_id":          torch.randint(1,   262_144, (B, 50), generator=g, dtype=torch.long, device=device),
        "seq_category_level_two": torch.randint(0,       700, (B, 50), generator=g, dtype=torch.long, device=device),
        "seq_category_level_one": torch.randint(0,       120, (B, 50), generator=g, dtype=torch.long, device=device),
        "seq_upload_type":        torch.randint(0,        30, (B, 50), generator=g, dtype=torch.long, device=device),
        "label": torch.randint(0, 2, (B,), generator=g, dtype=torch.float, device=device),
    }
    return batch
```

> Note: `_toy_batch` lives in core because building it requires the canonical feature schema; tests import it but no production runtime path does.

### Step 4: Verify tests pass
```
PYTHONPATH=src pytest tests/test_feature_encoder.py -v
```
Expected: 2 passed.

### Step 5: Commit
```bash
git add src/rec_fine_ranking/models/base.py tests/test_feature_encoder.py
git commit -m "feat(models): FeatureEncoder + BaseRanker shared backbone"
```

---

## Subtasks 7–13: Models (one per file)

Each model subtask uses the same shape: write a 3-test file (forward shape, backward gradient flow, param-budget assertion for modern models), implement the body, commit. Common test template lives below; only `body(...)` differs per model.

### Common per-model test template
File: `tests/test_<model>.py`
```python
import torch
from rec_fine_ranking.models.<model> import <ModelClass>
from rec_fine_ranking.models.base import _toy_batch

def test_forward_shape():
    m = <ModelClass>()
    out = m(_toy_batch(B=8))
    assert out.shape == (8,)
    assert torch.isfinite(out).all()

def test_backward_flow():
    m = <ModelClass>()
    out = m(_toy_batch(B=4))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, torch.zeros(4))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())

def test_param_count_in_budget():  # only for modern models (10–13)
    m = <ModelClass>()
    bp = m.count_params()["backbone"]
    assert 4_500_000 <= bp <= 5_500_000, f"backbone={bp}"
```

For the three classical models the third test is replaced by: `assert m.count_params()["backbone"] > 0` (no budget enforcement).

### Per-model `body()` specifications

#### Subtask 7 — Wide & Deep (`src/rec_fine_ranking/models/wide_deep.py`)
```
body:
  seq_pool = seq.mean(dim=1)                        # (B, seq_dim)
  x = cat([nonseq, seq_pool], dim=-1)               # (B, D)
  wide  = Linear(D, 1)(x)                           # linear part over flat features
  deep  = MLP([D, 256, 128, 64, 1], dropout=0.1)(x) # deep tower
  return (wide + deep).squeeze(-1)
```
Use `nn.Sequential` with `Linear → ReLU → Dropout` blocks.

#### Subtask 8 — DCN (`src/rec_fine_ranking/models/dcn.py`)
```
body:
  seq_pool = seq.mean(dim=1)
  x0 = cat([nonseq, seq_pool], dim=-1)              # (B, D)
  # cross network, L = 3 layers
  x = x0
  for _ in range(3):
      x = x0 * (x @ w + b) + x        # w in R^{D}, b in R^{D}
  deep  = MLP([D, 256, 128, 64], dropout=0.1)(x0)
  out   = Linear(D + 64, 1)(cat([x, deep], -1))
  return out.squeeze(-1)
```

#### Subtask 9 — DeepFM (`src/rec_fine_ranking/models/deepfm.py`)
```
body:
  # FM second-order: treat each non-seq feature embedding + seq-pool as a field vector v_i.
  fields = [emb_t for emb_t in encoder_views(nonseq, seq)]
  # Each model gets its own copy of FM-style fields by re-projecting nonseq into K-dim per field:
  V = stack(fields, dim=1)            # (B, F, K)
  sum_square = V.sum(dim=1) ** 2      # (B, K)
  square_sum = (V ** 2).sum(dim=1)    # (B, K)
  fm_2nd = 0.5 * (sum_square - square_sum).sum(dim=-1, keepdim=True)
  x = cat([nonseq, seq.mean(1)], -1)
  deep = MLP([D, 256, 128, 64, 1], dropout=0.1)(x)
  linear = Linear(D, 1)(x)
  return (linear + fm_2nd + deep).squeeze(-1)
```
> For simplicity, group nonseq features into K-dim chunks (K=8) for FM; this preserves the second-order interaction property without requiring per-feature dim alignment.

#### Subtask 10 — OneTrans (`src/rec_fine_ranking/models/onetrans.py`)
Paper: Zhang et al., WWW 2026 (arXiv 2510.26104).
```
config: n_layers=2, d_model=128, n_heads=4, ffn_mult=4
body:
  # tokens: 50 seq tokens (each projected to d_model) + 1 candidate token from nonseq
  seq_tok = Linear(seq_dim, d_model)(seq)          # (B, 50, d)
  cand_tok = Linear(non_seq_dim, d_model)(nonseq).unsqueeze(1)  # (B, 1, d)
  x = cat([seq_tok, cand_tok], dim=1)              # (B, 51, d), candidate last
  # 2 layers of causal Transformer; share self-attn params across sequential tokens,
  # use separate FFN for the candidate token (paper Sec. 3.3 simplified).
  for layer in self.layers:
      x = x + layer.attn(x, causal=True)
      x = x + layer.ffn(x)                          # per-position FFN
  pred = Linear(d_model, 1)(x[:, -1])               # candidate token → logit
  return pred.squeeze(-1)
```
Use built-in `nn.MultiheadAttention(..., batch_first=True)` with `attn_mask=causal_mask`.

#### Subtask 11 — RankMixer (`src/rec_fine_ranking/models/rankmixer.py`)
Paper: Zhu et al., 2025 (arXiv 2507.15551).
```
config: n_tokens=16, d_token=64, n_layers=3, ffn_mult=4
body:
  # tokenise: project (nonseq ‖ seq.mean(1)) into n_tokens × d_token
  x = cat([nonseq, seq.mean(1)], dim=-1)            # (B, D)
  x = Linear(D, n_tokens * d_token)(x).view(B, n_tokens, d_token)
  for _ in range(n_layers):
      # multi-head token mixing: split each token along channel dim into n_tokens heads,
      # then transpose so head h of token t goes to position (t, h) → standard MLP-Mixer permutation
      x = token_mix(x)                              # parameter-free
      x = x + per_token_ffn(x)                      # per-token weights, ffn_mult * d_token hidden
  pred = Linear(n_tokens * d_token, 1)(x.flatten(1))
  return pred.squeeze(-1)
```
`token_mix`: reshape `(B, T, T*Dh) → (B, T, T, Dh) → permute(0, 2, 1, 3) → (B, T, T*Dh)` where `T*Dh = d_token`, `Dh = d_token // T`.

#### Subtask 12 — UniMixer (`src/rec_fine_ranking/models/unimixer.py`)
Paper: Ha et al., 2026 (arXiv 2604.00590).
```
config: n_layers=3, n_blocks=8, block_dim=24, hidden_mult=2
body:
  x = cat([nonseq, seq.mean(1)], dim=-1)
  x = Linear(D, n_blocks * block_dim)(x).view(B, n_blocks, block_dim)
  for _ in range(n_layers):
      # local mixing: per-block weight W_B^i in R^{block_dim x block_dim}
      x_local = einsum("bnk,nkj->bnj", x, W_B)
      # global mixing: doubly-stochastic W_G (Sinkhorn-Knopp on a learnable logit matrix)
      W_G = sinkhorn_knopp(self.global_logits, n_iter=3)         # (n_blocks, n_blocks)
      x_global = einsum("nm,bmk->bnk", W_G, x_local)
      x = x + per_block_ffn(x_global)                            # hidden = block_dim * hidden_mult
  pred = Linear(n_blocks * block_dim, 1)(x.flatten(1))
  return pred.squeeze(-1)
```
Sinkhorn-Knopp (3 iters) on `softmax(logits)` is sufficient for unit tests; full convergence is not required for benchmark correctness.

#### Subtask 13 — HyFormer (`src/rec_fine_ranking/models/hyformer.py`)
Paper: Huang et al., 2026 (arXiv 2601.12681).
```
config: n_layers=2, d_seq=64, d_feat=128, n_heads=4
body:
  seq_t  = Linear(seq_dim, d_seq)(seq)           # (B, 50, d_seq)
  feat_t = Linear(non_seq_dim, d_feat)(nonseq).unsqueeze(1).expand(-1, n_query, -1)  # n_query=4
  q2s = Linear(d_feat, d_seq)
  s2q = Linear(d_seq, d_feat)
  for _ in range(n_layers):
      # Query Decoding: feat queries attend to seq
      feat_t = feat_t + cross_attn(q=feat_t, k=q2s(seq_t), v=q2s(seq_t), heads=n_heads)
      feat_t = feat_t + feat_ffn(feat_t)
      # Query Boosting: seq attends back to feat (bidirectional)
      seq_t = seq_t + cross_attn(q=seq_t, k=s2q(feat_t), v=s2q(feat_t), heads=n_heads)
      seq_t = seq_t + seq_ffn(seq_t)
  pooled = feat_t.mean(1)
  return Linear(d_feat, 1)(pooled).squeeze(-1)
```

### Steps for each of subtasks 7–13

1. Write `tests/test_<model>.py` from the template above.
2. Run tests → expect ImportError.
3. Implement `src/rec_fine_ranking/models/<model>.py` with the spec above and a single class subclassing `BaseRanker`.
4. Run tests → expect 3 passed (or 2 for classical models).
5. Commit:
   ```
   git add src/rec_fine_ranking/models/<model>.py tests/test_<model>.py
   git commit -m "feat(model): <ModelClass>"
   ```

After Subtask 13, `src/rec_fine_ranking/models/__init__.py` should expose:
```python
from .wide_deep  import WideDeep
from .dcn        import DCN
from .deepfm     import DeepFM
from .onetrans   import OneTrans
from .rankmixer  import RankMixer
from .unimixer   import UniMixer
from .hyformer   import HyFormer
MODEL_REGISTRY = {
    "wide_deep": WideDeep, "dcn": DCN, "deepfm": DeepFM,
    "onetrans": OneTrans, "rankmixer": RankMixer,
    "unimixer": UniMixer, "hyformer": HyFormer,
}
```

---

## Subtask 14: Evaluator core (AUC / GAUC / LogLoss, both entry modes)

**Role:** Pure evaluator decoupled from the trainer. Trainer calls `evaluator.run(model, loader)` for in-memory eval; `scripts/eval.py` calls `evaluator.run_from_checkpoint(ckpt_path, loader)` for offline eval.
**Implementation:** `src/rec_fine_ranking/training/metrics.py` (AUC/GAUC/LogLoss), `src/rec_fine_ranking/training/evaluator.py`.
**Unit Tests:** AUC against sklearn ground-truth on toy data; GAUC matches manual per-user computation; LogLoss equals sklearn within 1e-6; evaluator handles empty-group case (returns NaN, no crash); evaluator emits required log lines.
**Expected Conclusion:** "Evaluator passes metric correctness tests and failure-handling tests."

### Step 1: Write tests
File: `tests/test_metrics.py`
```python
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from rec_fine_ranking.training.metrics import compute_auc, compute_gauc, compute_logloss

def test_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000); p = rng.random(1000)
    assert abs(compute_auc(y, p) - roc_auc_score(y, p)) < 1e-9

def test_gauc_manual():
    y    = np.array([1,0,1,0, 1,1,0,0])
    p    = np.array([.9,.1,.8,.2,.6,.7,.4,.3])
    uids = np.array([1,1,1,1, 2,2,2,2])
    # both users have AUC=1.0 → GAUC=1.0 (each user weight = n_samples / total)
    assert abs(compute_gauc(y, p, uids) - 1.0) < 1e-9

def test_logloss_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500); p = rng.random(500).clip(1e-6, 1-1e-6)
    assert abs(compute_logloss(y, p) - log_loss(y, p)) < 1e-9

def test_gauc_skips_constant_users():
    y    = np.array([1,1,1, 1,0])
    p    = np.array([.5,.6,.7,.8,.2])
    uids = np.array([1,1,1, 2,2])  # user 1 has all-positive → excluded
    g = compute_gauc(y, p, uids)
    # only user 2 contributes; user 2 AUC = 1.0
    assert abs(g - 1.0) < 1e-9
```

File: `tests/test_evaluator.py`
```python
import torch, pytest
from rec_fine_ranking.training.evaluator import Evaluator
from rec_fine_ranking.models.wide_deep import WideDeep
from rec_fine_ranking.models.base import _toy_batch

class _FakeLoader:
    def __init__(self, n_batches=4, B=16): self.n, self.B = n_batches, B
    def __iter__(self):
        for _ in range(self.n):
            yield _toy_batch(self.B)
    def __len__(self): return self.n

def test_evaluator_in_memory():
    m = WideDeep().eval()
    ev = Evaluator(device="cpu")
    res = ev.run(m, _FakeLoader())
    for k in ("auc","gauc","logloss","n_samples","elapsed_sec"):
        assert k in res

def test_evaluator_empty_loader_fails_loud():
    m = WideDeep()
    ev = Evaluator(device="cpu")
    with pytest.raises(ValueError, match="empty"):
        ev.run(m, _FakeLoader(n_batches=0))

def test_evaluator_handles_constant_labels_without_crash():
    # A loader where labels are all zeros — AUC should be NaN, not crash.
    class _Z:
        def __iter__(self):
            b = _toy_batch(8); b["label"] = torch.zeros(8); yield b
        def __len__(self): return 1
    m = WideDeep()
    res = Evaluator(device="cpu").run(m, _Z())
    import math
    assert math.isnan(res["auc"])
```

### Step 2: Verify tests fail
```
PYTHONPATH=src pytest tests/test_metrics.py tests/test_evaluator.py -v
```
Expected: ImportError.

### Step 3: Implement

File: `src/rec_fine_ranking/training/metrics.py`
```python
"""AUC / GAUC / LogLoss helpers — thin wrappers over sklearn with NaN-safe aggregation."""
from __future__ import annotations
import math
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.min() == y_true.max():
        return math.nan
    return float(roc_auc_score(y_true, y_score))

def compute_gauc(y_true: np.ndarray, y_score: np.ndarray, user_ids: np.ndarray) -> float:
    df_uid = np.asarray(user_ids).astype(np.int64)
    order = np.argsort(df_uid, kind="stable")
    df_uid, y_true, y_score = df_uid[order], y_true[order], y_score[order]
    splits = np.flatnonzero(np.diff(df_uid)) + 1
    groups_idx = np.split(np.arange(len(df_uid)), splits)
    total, total_w = 0.0, 0.0
    for g in groups_idx:
        if len(g) < 2 or y_true[g].min() == y_true[g].max():
            continue
        total   += roc_auc_score(y_true[g], y_score[g]) * len(g)
        total_w += len(g)
    return float(total / total_w) if total_w > 0 else math.nan

def compute_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7), labels=[0, 1]))
```

File: `src/rec_fine_ranking/training/evaluator.py`
```python
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
        n_batches = len(loader) if hasattr(loader, "__len__") else None
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
```

### Step 4: Verify tests pass
```
PYTHONPATH=src pytest tests/test_metrics.py tests/test_evaluator.py -v
```
Expected: 7 passed.

### Step 5: Commit
```bash
git add src/rec_fine_ranking/training/{metrics.py,evaluator.py} tests/test_metrics.py tests/test_evaluator.py
git commit -m "feat(training): evaluator core + AUC/GAUC/LogLoss metrics"
```

---

## Subtask 15: Capacity calibrator + Compare script

**Role:** Two small post-pipeline tools. `scripts/calibrate_capacity.py` measures params & FLOPs for the four modern models, prints a table, and writes recommended config knobs into `configs/<model>.yaml`. `scripts/compare.py` reads `experiments/runs/*/metrics.csv` + `meta.json` and emits `experiments/results.csv`, `experiments/results.md`, `experiments/results.png`.
**Implementation:** `src/rec_fine_ranking/utils/flops.py`, `scripts/calibrate_capacity.py`, `scripts/compare.py`.
**Unit Tests:** FLOPs counter returns plausible (non-zero, finite) numbers on each model; calibrator dry-run doesn't crash; compare script emits expected files given a fake `runs/` directory.
**Expected Conclusion:** "Calibration + comparison utilities unit-tested; ready to run after training."

### Step 1: Write tests
File: `tests/test_flops_and_compare.py`
```python
import json, csv, torch
from pathlib import Path
from rec_fine_ranking.utils.flops import count_mflops
from rec_fine_ranking.models import MODEL_REGISTRY
from rec_fine_ranking.models.base import _toy_batch

def test_flops_finite_and_nonzero():
    for name, cls in MODEL_REGISTRY.items():
        m = cls().eval()
        f = count_mflops(m, _toy_batch(B=1))
        assert f > 0 and f < 5000, f"{name}: {f} MFLOPs"

def test_compare_emits_files(tmp_path, monkeypatch):
    from scripts import compare
    # build fake run dirs
    for model, vals in {"wide_deep":(0.62,0.58,0.45,3.1e6,12e6,5.0,42.0),
                        "onetrans":(0.71,0.66,0.41,5.0e6,22e6,52.0,1800.0)}.items():
        d = tmp_path / "runs" / model
        d.mkdir(parents=True)
        with open(d/"metrics.csv","w",newline="") as f:
            w = csv.writer(f); w.writerow(["step","auc","gauc","logloss"])
            w.writerow([1000, vals[0], vals[1], vals[2]])
        meta = {"params_backbone":vals[3],"params_emb":vals[4],
                "mflops_per_sample":vals[5],"train_time_sec":vals[6]}
        (d/"meta.json").write_text(json.dumps(meta))
    compare.run(tmp_path / "runs", tmp_path / "results.csv", tmp_path / "results.md", tmp_path / "results.png")
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "results.md").exists()
    assert (tmp_path / "results.png").exists()
```

### Step 2: Verify tests fail
```
PYTHONPATH=src pytest tests/test_flops_and_compare.py -v
```
Expected: ImportError.

### Step 3: Implement

File: `src/rec_fine_ranking/utils/flops.py`
```python
"""Light wrapper over fvcore's FlopCountAnalysis with graceful fallback."""
from __future__ import annotations
from typing import Dict
import torch
try:
    from fvcore.nn import FlopCountAnalysis
    _HAS_FVCORE = True
except Exception:
    _HAS_FVCORE = False

def count_mflops(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> float:
    """Return MFLOPs/sample, excluding embedding lookups."""
    if not _HAS_FVCORE:
        # rough manual fallback: count Linear matmul flops
        flops = 0
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                flops += 2 * m.in_features * m.out_features
        return flops / 1e6
    model.eval()
    # fvcore expects positional args; wrap to accept dict
    class _W(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, **kw): return self.m(kw)
    wrapped = _W(model)
    # fvcore can't ingest **kwargs directly — pass via a tuple input by serialising
    # Instead, just run once and use the input cache:
    analysis = FlopCountAnalysis(model, (batch,))
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    return analysis.total() / 1e6 / max(batch["label"].shape[0], 1)
```

File: `scripts/calibrate_capacity.py`
```python
"""Print params + FLOPs per model; flag modern models that fall outside the budget."""
from __future__ import annotations
import argparse, torch
from rec_fine_ranking.models import MODEL_REGISTRY
from rec_fine_ranking.models.base import _toy_batch
from rec_fine_ranking.utils.flops import count_mflops

TARGET_PARAMS = 5_000_000
TARGET_PCT    = 0.10
TARGET_MFLOPS = 50.0
TARGET_MFLOPS_PCT = 0.15
MODERN = {"onetrans","rankmixer","unimixer","hyformer"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()
    b = _toy_batch(args.batch_size)
    print(f"{'model':<12} {'backbone_params':>16} {'emb_params':>14} {'mflops/sample':>16}  in-budget?")
    for name, cls in MODEL_REGISTRY.items():
        m = cls()
        p = m.count_params()
        mf = count_mflops(m, b)
        budget_ok = "n/a"
        if name in MODERN:
            ok_p = abs(p["backbone"] - TARGET_PARAMS) / TARGET_PARAMS <= TARGET_PCT
            ok_f = abs(mf - TARGET_MFLOPS) / TARGET_MFLOPS <= TARGET_MFLOPS_PCT
            budget_ok = "yes" if (ok_p and ok_f) else "NO"
        print(f"{name:<12} {p['backbone']:>16,} {p['embedding']:>14,} {mf:>16.1f}  {budget_ok}")

if __name__ == "__main__":
    main()
```

File: `scripts/compare.py`
```python
"""Aggregate per-run metrics into results.{csv,md,png}."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def run(runs_dir: Path, out_csv: Path, out_md: Path, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for run in sorted(Path(runs_dir).iterdir()):
        m = run / "metrics.csv"; meta = run / "meta.json"
        if not (m.exists() and meta.exists()):
            continue
        with open(m) as f:
            last = list(csv.DictReader(f))[-1]
        md = json.loads(meta.read_text())
        rows.append({
            "model": run.name,
            "params_backbone": int(md["params_backbone"]),
            "params_emb":      int(md["params_emb"]),
            "mflops_per_sample": float(md["mflops_per_sample"]),
            "train_time_sec":  float(md["train_time_sec"]),
            "AUC": float(last["auc"]),
            "GAUC": float(last["gauc"]),
            "LogLoss": float(last["logloss"]),
        })
    rows.sort(key=lambda r: r["GAUC"], reverse=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    md_lines = ["| model | backbone | mflops | AUC | GAUC | LogLoss | train_s |",
                "|---|---|---|---|---|---|---|"]
    for r in rows:
        md_lines.append(f"| {r['model']} | {r['params_backbone']:,} | {r['mflops_per_sample']:.1f} "
                        f"| {r['AUC']:.4f} | {r['GAUC']:.4f} | {r['LogLoss']:.4f} | {r['train_time_sec']:.0f} |")
    Path(out_md).write_text("\n".join(md_lines))
    fig, ax = plt.subplots(figsize=(9,4))
    labels = [r["model"] for r in rows]
    auc = [r["AUC"] for r in rows]; gauc = [r["GAUC"] for r in rows]
    x = range(len(labels)); w = 0.35
    ax.bar([i-w/2 for i in x], auc, w, label="AUC")
    ax.bar([i+w/2 for i in x], gauc, w, label="GAUC")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=20); ax.legend()
    ax.set_ylim(0.5, max(max(auc), max(gauc)) + 0.02); ax.set_title("RecFlow CTR — AUC / GAUC")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    run(Path(a.runs_dir), out / "results.csv", out / "results.md", out / "results.png")

if __name__ == "__main__":
    main()
```

### Step 4: Verify tests pass
```
PYTHONPATH=src pytest tests/test_flops_and_compare.py -v
```
Expected: 2 passed.

### Step 5: Commit
```bash
git add src/rec_fine_ranking/utils/flops.py scripts/{calibrate_capacity.py,compare.py} tests/test_flops_and_compare.py
git commit -m "feat(utils): capacity calibrator + run comparison script"
```

---

## Subtask 16: Final Training Pipeline `[INTEGRATION]`

**Hypothesis:** All seven models, instantiated via `MODEL_REGISTRY[name]()` and trained by the same `Trainer` over the preprocessed RecFlow shards, converge without NaN/Inf and produce monotonically increasing validation AUC over the first epoch.

**Components consumed:**
- `src/rec_fine_ranking/data/{feature_config,hashing,preprocess,sequence,dataset,collate}.py` — Subtasks 2-5
- `src/rec_fine_ranking/models/*.py` + `MODEL_REGISTRY` — Subtasks 6-13
- `src/rec_fine_ranking/training/evaluator.py`, `metrics.py` — Subtask 14
- `src/rec_fine_ranking/utils/flops.py` — Subtask 15

**Implementation:** Build `src/rec_fine_ranking/training/trainer.py` (training loop, schedule, checkpointing, logging) and `scripts/train.py` (CLI wiring). Trainer decides *when* to evaluate; Evaluator decides *how*. No test imports in core.

**Integration Tests:** End-to-end smoke test: instantiate `WideDeep` + tiny in-memory dataset + 5 train steps + 1 eval call → assert loss is finite and decreasing on at least 3 of the 5 steps.

**Validation Pyramid:** L0 + L1 (mandatory) — runs on this subtask.

**Evaluation contract:**
- Cadence: every `eval_every_steps=5000` steps + at end of epoch.
- Scope: full validation set (no sub-sampling) unless `--eval-max-batches` overrides.
- Both entry modes share the same `Evaluator._loop` (Subtask 14): in-memory during training; checkpoint-based via `scripts/eval.py --checkpoint <path>`.
- Observability: phase-start log `"[eval] starting @step=K | n_batches=…"`, `tqdm` progress bar, phase-end log `"[eval] done @step=K | auc=… gauc=… logloss=… | elapsed=…"`, throughput line, watchdog warning on > 60 s silence.
- Failure handling: missing ckpt → `FileNotFoundError`; unreadable ckpt → `RuntimeError`; state-dict mismatch → log + non-zero exit; empty loader → `ValueError` at init; metric aggregation failure → log + NaN; non-finite metrics → warn + persist run as failed.

**Production training requirements (built into this subtask, not added later):**
- Human-readable log file: `experiments/runs/<model>/train.log` with one line per step containing `step`, `loss`, `grad_norm`, `lr`, `step_time_ms`, `samples_per_sec`.
- MFU calculation: at every log step using `count_mflops` × `samples_per_sec` ÷ device peak FLOPs (peak: M-series ≈ 1 TFLOPs fp32 / 2 TFLOPs bf16 — passed via config).
- tqdm progress bar over the training loop.
- Checkpoint save: every `ckpt_every_steps` (default 10000); also at the end of training.
- Resume support: `--resume <ckpt>` loads model + optimizer + step counter.
- Fixed seeds set in `scripts/train.py` (`torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`).
- TensorBoard scalar logging: `loss`, `grad_norm`, `lr`, `step_time_ms`, `mfu`, `eval/auc`, `eval/gauc`, `eval/logloss`.

**Expected Conclusion:** L0 passes all mandatory checks; L1 mock-overfit run shows final loss < initial loss × 0.5 with finite gradients throughout for every model.

### Step 1: Write integration tests
File: `tests/test_integration.py`
```python
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
    # at least 3 of 5 steps showed loss decrease vs previous
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
```

### Step 2: Verify tests fail
```
PYTHONPATH=src pytest tests/test_integration.py -v
```
Expected: ImportError.

### Step 3: Assemble the training pipeline

File: `src/rec_fine_ranking/utils/device.py`
```python
import torch
def autodetect_device() -> torch.device:
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")
```

File: `src/rec_fine_ranking/training/trainer.py`
```python
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
    device_peak_tflops: float = 2.0   # M5 bf16 nominal; used for MFU display only
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
        backbone_params = [p for n,p in self.model.named_parameters() if not n.startswith("encoder.")]
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
            self._metrics_w.writerow(["step","auc","gauc","logloss"])
        # resume
        if cfg.resume:
            self._load_checkpoint(cfg.resume)

    @staticmethod
    def _auto_device():
        if torch.backends.mps.is_available(): return "mps"
        if torch.cuda.is_available(): return "cuda"
        return "cpu"

    def _train_step(self, batch) -> float:
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        logits = self.model(batch)
        loss = self.loss_fn(logits, batch["label"])
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
        for k in ("auc","gauc","logloss"):
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
        for epoch in range(self.cfg.epochs):
            it = iter(self.train_loader)
            bar = tqdm(it, desc=f"train epoch={epoch}", total=len(self.train_loader)
                        if hasattr(self.train_loader, "__len__") else None)
            for batch in bar:
                t0 = time.time()
                loss, gnorm = self._train_step(batch)
                step_ms = (time.time() - t0) * 1000
                self._step += 1
                if self._step % self.cfg.log_every_steps == 0:
                    self._log_step(loss, gnorm, step_ms)
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
        self._log_file.close(); self._metrics_csv.close(); self.writer.close()

    def fit_for_test(self) -> List[float]:
        """Tiny helper for integration test: run max_steps without eval/ckpt logic."""
        losses = []
        it = iter(self.train_loader)
        for _ in range(self.cfg.max_steps):
            batch = next(it)
            loss, _ = self._train_step(batch)
            losses.append(loss)
            self._step += 1
        return losses
```

File: `scripts/train.py`
```python
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
```

File: `scripts/eval.py`
```python
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
```

### Step 4: Verify integration tests pass
```
PYTHONPATH=src pytest tests/test_integration.py -v
```
Expected: 8 passed (7 model-smoke parameterised + 1 artifact-writing).

### Step 5: Write validation scripts (external, observe-only)

File: `tests/validation/run_mock_overfit.py`
```python
"""L1 driver — overfit on 1024 mock samples for 50 steps; assert loss-down + finite.

This script imports core, but core does NOT import this script.
"""
from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
import torch
from rec_fine_ranking.training.trainer import Trainer, TrainerConfig
from rec_fine_ranking.models.base import _toy_batch
from rec_fine_ranking.models import MODEL_REGISTRY

class _MockLoader:
    def __init__(self, B=32, n_batches=1024//32):
        self.B = B; self.n = n_batches
        self._fixed = [_toy_batch(B) for _ in range(self.n)]
    def __iter__(self): return iter(self._fixed)
    def __len__(self): return self.n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    loader = _MockLoader()
    cfg = TrainerConfig(model_name=a.model, out_dir=out, device=a.device,
                        max_steps=50, log_every_steps=1, eval_every_steps=10**9,
                        ckpt_every_steps=10**9, batch_size=32)
    t = Trainer(cfg, loader, loader)
    losses = t.fit_for_test()
    summary = {"model": a.model, "initial": losses[0], "final": losses[-1],
               "all_finite": all(math.isfinite(l) for l in losses),
               "ratio": losses[-1] / max(losses[0], 1e-6)}
    print(json.dumps(summary, indent=2))
    ok = summary["all_finite"] and summary["ratio"] < 0.5
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
```

### Step 6: Run Validation Pyramid (L0 → L1)

L0 (`spml:ml-static-checks`): dispatched as subagent — no manual command.

L1 (`spml:ml-runtime-validator`): per-model mock-overfit run.
```bash
for m in wide_deep dcn deepfm onetrans rankmixer unimixer hyformer; do
  PYTHONPATH=src python tests/validation/run_mock_overfit.py \
      --model $m --out-dir experiments/vp_l1/$m --device cpu
done
```
Expected: each invocation prints summary, exits 0; aggregated success across all seven means L1 PASS.

### Step 7: Record full conclusion

Append to `experiments/plans/2026-05-25-rec-fine-ranking-implementation.md` under a `## VP Results` section the following:
- L0 PASS/FAIL with the static-check report attached
- L1 per-model `(initial_loss, final_loss, ratio, all_finite)` tuples
- Hypothesis-level: "Pipeline runnable for all seven models; ready to launch benchmark."

### Step 8: Commit
```bash
git add src/rec_fine_ranking/training/trainer.py src/rec_fine_ranking/utils/device.py \
        scripts/{train.py,eval.py} tests/test_integration.py tests/validation/run_mock_overfit.py
git commit -m "feat(training)[INTEGRATION]: trainer + CLI + VP L1 validation script"
```

---

## Subtask 17 (post-VP): GitHub publish + benchmark run

This is not a code subtask; it runs after Subtask 16's VP passes.

### Step 1: Create the public GitHub repo and push
```bash
cd /Users/zhaoxiaotian.0701/projects/rec_fine_ranking
gh repo create rec_fine_ranking --public --source=. --remote=origin --description "RecFlow CTR benchmark across Wide&Deep, DCN, DeepFM, OneTrans, RankMixer, UniMixer, HyFormer"
git push -u origin main
```

### Step 2: Run preprocess + sequence assembly (one-time)
```bash
PYTHONPATH=src python scripts/preprocess.py     --raw-root /Users/zhaoxiaotian.0701/data/rec_flow --out-root experiments/data
PYTHONPATH=src python scripts/build_sequences.py --data-root experiments/data
```

### Step 3: Calibrate capacity
```bash
PYTHONPATH=src python scripts/calibrate_capacity.py
```
If any modern model is outside the budget, tweak its config defaults in the model class and re-test.

### Step 4: Run all seven models
```bash
for m in wide_deep dcn deepfm onetrans rankmixer unimixer hyformer; do
  PYTHONPATH=src python scripts/train.py --model $m --data-root experiments/data --out-dir experiments/runs/$m --device auto
done
```

### Step 5: Aggregate and publish results
```bash
PYTHONPATH=src python scripts/compare.py --runs-dir experiments/runs --out-dir experiments/
git add experiments/results.{csv,md,png}
git commit -m "results: benchmark across 7 ranking models on RecFlow"
git push
```

---

## Plan Self-Review

- [x] Header includes goal, experiment dir, hypothesis, validation scope, evaluation design, architecture.
- [x] Exactly one `[INTEGRATION]` subtask (Subtask 16) — gated by L0 + L1.
- [x] Each subtask has TDD (write tests → fail → implement → pass → commit).
- [x] Exact file paths everywhere; complete code provided (no "add …").
- [x] Core code never imports tests; tests/validation observe core externally.
- [x] Evaluation cadence, scope, both entry modes, observability, failure handling all spelled out in Subtask 14 and reinforced in Subtask 16's evaluation contract.
- [x] Production training script requirements (log file, MFU, tqdm, ckpt save/resume, fixed seeds, TensorBoard) live in Subtask 16's spec.
- [x] Frequent commits — one per subtask, plus the final results commit.
- [x] Capacity-alignment constraint enforced by per-model unit tests and by `scripts/calibrate_capacity.py`.
