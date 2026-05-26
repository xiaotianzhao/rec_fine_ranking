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
