"""Ridge probe used by scripts/eval_hd.py.

`ridge_fit_predict` solves the same ridge regression in whichever form is smaller: the primal normal equations
(features x features, the original code path) when there are fewer features than training samples, and the dual form
(samples x samples) otherwise. The two give the same weights, W = (X'X + lam I)^-1 X'Y = X'(XX' + lam I)^-1 Y; the dual is
what makes the probe feasible on the whole-brain state (139,248 columns: the primal matrix alone would be 155 GB).
The dual form runs on the GPU when one is available (same arithmetic: float32 Gram matrix, float64 solve); on the CPU
one whole-brain probe takes about 25 minutes.
"""
import numpy as np


def _dual_gpu(Xtr, Ytr, Xte, lam, device):
    import torch
    with torch.no_grad():
        X = torch.from_numpy(np.ascontiguousarray(Xtr)).to(device)
        mu, sd = X.mean(0), X.std(0, correction=0) + 1e-6
        X.sub_(mu).div_(sd)
        Y = torch.from_numpy(np.asarray(Ytr, dtype=np.float64)).to(device); ym = Y.mean(0)
        K = (X @ X.T).double(); K.diagonal().add_(lam)
        alpha = torch.linalg.solve(K, Y - ym).to(X.dtype); del K
        W = X.T @ alpha; del X
        Xe = torch.from_numpy(np.ascontiguousarray(Xte)).to(device); Xe.sub_(mu).div_(sd)
        pred = (Xe @ W).double() + ym
        out = pred.cpu().numpy()
    torch.cuda.empty_cache()
    return out


def ridge_fit_predict(Xtr, Ytr, Xte, lam, device=None):
    """Standardise on the training set, fit ridge with penalty lam on centred targets, return test predictions."""
    n, f = Xtr.shape
    if f > n and device is not None and str(device).startswith("cuda"):
        return _dual_gpu(Xtr, Ytr, Xte, lam, device)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ym = Ytr.mean(0); Yc = Ytr - ym
    if f <= n:
        A = Xtr.T @ Xtr + lam * np.eye(f)
        return Xte @ np.linalg.solve(A, Xtr.T @ Yc) + ym
    K = (Xtr @ Xtr.T).astype(np.float64); K[np.diag_indices(n)] += lam
    alpha = np.linalg.solve(K, Yc.astype(np.float64)).astype(Xtr.dtype)
    return Xte @ (Xtr.T @ alpha) + ym
