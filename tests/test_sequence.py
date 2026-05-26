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
