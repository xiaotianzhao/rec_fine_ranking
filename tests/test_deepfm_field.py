import torch
from rec_fine_ranking.models.deepfm_field import DeepFMField
from rec_fine_ranking.models.base import _toy_batch, NON_SEQ_FEATURES
from rec_fine_ranking.data.feature_config import SEQUENCE_FIELDS


def test_forward_shape():
    m = DeepFMField()
    out = m(_toy_batch(B=8))
    assert out.shape == (8,)
    assert torch.isfinite(out).all()


def test_backward_flow():
    m = DeepFMField()
    out = m(_toy_batch(B=4))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, torch.zeros(4))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_fields_are_per_feature():
    # One FM field per non-seq feature (incl. hour_of_day) + per pooled sequence field.
    m = DeepFMField()
    assert m.num_fields == len(NON_SEQ_FEATURES) + len(SEQUENCE_FIELDS)
    assert m.count_params()["backbone"] > 0
