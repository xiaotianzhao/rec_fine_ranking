import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from rec_fine_ranking.training.metrics import compute_auc, compute_gauc, compute_logloss

def test_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000); p = rng.random(1000)
    assert abs(compute_auc(y, p) - roc_auc_score(y, p)) < 1e-9

def test_gauc_manual():
    y    = np.array([1,0,1,0, 1,1,0,0])
    p    = np.array([.9,.1,.8,.2,.6,.7,.4,.3])
    uids = np.array([1,1,1,1, 2,2,2,2])
    # both users have AUC=1.0 → GAUC=1.0 (each user weight = n_samples / total)
    assert abs(compute_gauc(y, p, uids) - 1.0) < 1e-9

def test_logloss_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500); p = rng.random(500).clip(1e-6, 1-1e-6)
    assert abs(compute_logloss(y, p) - log_loss(y, p)) < 1e-9

def test_gauc_skips_constant_users():
    y    = np.array([1,1,1, 1,0])
    p    = np.array([.5,.6,.7,.8,.2])
    uids = np.array([1,1,1, 2,2])  # user 1 has all-positive → excluded
    g = compute_gauc(y, p, uids)
    # only user 2 contributes; user 2 AUC = 1.0
    assert abs(g - 1.0) < 1e-9
