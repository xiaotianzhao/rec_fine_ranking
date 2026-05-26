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
