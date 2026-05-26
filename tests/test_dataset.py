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
