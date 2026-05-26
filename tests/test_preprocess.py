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
