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
