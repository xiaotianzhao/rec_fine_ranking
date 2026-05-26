import torch
from rec_fine_ranking.models.unimixer import UniMixer
from rec_fine_ranking.models.base import _toy_batch


def test_forward_shape():
    m = UniMixer()
    out = m(_toy_batch(B=8))
    assert out.shape == (8,)
    assert torch.isfinite(out).all()


def test_backward_flow():
    m = UniMixer()
    out = m(_toy_batch(B=4))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, torch.zeros(4))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_backbone_param_count():
    m = UniMixer()
    bp = m.count_params()["backbone"]
    assert 4_500_000 <= bp <= 5_500_000, f"backbone={bp:,} outside 5M±10%"
