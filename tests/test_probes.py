import numpy as np
from wm2cx.probes import ridge_fit_predict


def _primal(Xtr, Ytr, Xte, lam):
    """The ridge of scripts/eval_hd.py before the dual branch was added."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]); Wt = np.linalg.solve(A, Xtr.T @ (Ytr - Ytr.mean(0)))
    return Xte @ Wt + Ytr.mean(0)


def test_primal_branch_is_the_old_ridge():
    r = np.random.default_rng(0)
    Xtr, Xte, Ytr = r.normal(size=(200, 30)).astype(np.float32), r.normal(size=(50, 30)).astype(np.float32), r.normal(size=(200, 2))
    assert np.allclose(ridge_fit_predict(Xtr, Ytr, Xte, 100.0), _primal(Xtr, Ytr, Xte, 100.0))


def test_dual_branch_matches_primal_when_features_exceed_samples():
    r = np.random.default_rng(1)
    W = r.normal(size=(400, 2)) / 20
    Xtr, Xte = r.normal(size=(120, 400)).astype(np.float32), r.normal(size=(60, 400)).astype(np.float32)
    Ytr = Xtr @ W + 0.1 * r.normal(size=(120, 2))
    a, b = ridge_fit_predict(Xtr, Ytr, Xte, 100.0), _primal(Xtr, Ytr, Xte, 100.0)
    assert np.abs(a - b).max() < 1e-3 * np.abs(b).max()
