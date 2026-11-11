"""Ridge probe used by scripts/eval_hd.py.

`ridge_fit_predict` solves the same ridge regression in whichever form is smaller: the primal normal equations
(features x features, the original code path) when there are fewer features than training samples, and the dual form
(samples x samples) otherwise. The two give the same weights, W = (X'X + lam I)^-1 X'Y = X'(XX' + lam I)^-1 Y; the dual is
what makes the probe feasible on the whole-brain state (139,248 columns: the primal matrix alone would be 155 GB).
"""
import numpy as np


def ridge_fit_predict(Xtr, Ytr, Xte, lam):
    """Standardise on the training set, fit ridge with penalty lam on centred targets, return test predictions."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ym = Ytr.mean(0); Yc = Ytr - ym
    n, f = Xtr.shape
    if f <= n:
        A = Xtr.T @ Xtr + lam * np.eye(f)
        return Xte @ np.linalg.solve(A, Xtr.T @ Yc) + ym
    K = (Xtr @ Xtr.T).astype(np.float64); K[np.diag_indices(n)] += lam
    alpha = np.linalg.solve(K, Yc.astype(np.float64)).astype(Xtr.dtype)
    return Xte @ (Xtr.T @ alpha) + ym
