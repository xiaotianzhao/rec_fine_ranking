import torch
from rec_fine_ranking.models.din import DIN, DINAttentionPooling
from rec_fine_ranking.models.dcn_v2 import DCNv2
from rec_fine_ranking.models.base import _toy_batch


def test_din_forward_shape():
    out = DIN()(_toy_batch(B=8))
    assert out.shape == (8,) and torch.isfinite(out).all()


def test_dcnv2_forward_shape():
    out = DCNv2()(_toy_batch(B=8))
    assert out.shape == (8,) and torch.isfinite(out).all()


def test_backward_flow():
    for cls in (DIN, DCNv2):
        m = cls()
        out = m(_toy_batch(B=4))
        torch.nn.functional.binary_cross_entropy_with_logits(out, torch.zeros(4)).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters()), cls.__name__


def test_din_pooling_masks_padding():
    # An all-zero (fully padded) history must pool to exactly zero.
    pool = DINAttentionPooling(dim=6)
    cand = torch.randn(3, 6)
    hist = torch.zeros(3, 50, 6)
    assert torch.allclose(pool(cand, hist), torch.zeros(3, 6))


def test_candidate_aligns_with_seq_dim():
    m = DIN()
    assert sum(e - s for (s, e) in m._cand_slices) == m.encoder.seq_dim
