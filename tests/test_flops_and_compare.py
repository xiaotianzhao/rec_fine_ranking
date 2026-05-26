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
