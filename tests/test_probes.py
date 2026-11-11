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


def test_ridge_weights_primal_and_dual_match_the_closed_form():
    import torch
    from wm2cx.probes import ridge_weights
    r = np.random.default_rng(2)
    for n, f in ((300, 40), (80, 500)):                      # primal branch, dual branch
        X = (r.normal(size=(n, f)) + 0.5).astype(np.float32); Y = (X[:, :3] @ r.normal(size=(3, 4)) + 0.1 * r.normal(size=(n, 4))).astype(np.float32)
        mu, W, ym = ridge_weights(torch.from_numpy(X), torch.from_numpy(Y), 100.0, "cpu", chunk=64)
        Xc = X.astype(np.float64) - X.astype(np.float64).mean(0)
        ref = np.linalg.solve(Xc.T @ Xc + 100.0 * np.eye(f), Xc.T @ (Y - Y.mean(0)))
        assert np.abs(W.numpy() - ref).max() < 1e-4 * np.abs(ref).max() + 1e-6
        assert np.allclose(mu.numpy(), X.mean(0), atol=1e-5) and np.allclose(ym.numpy(), Y.mean(0), atol=1e-5)
