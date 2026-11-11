"""Ridge probe used by scripts/eval_hd.py.

`ridge_fit_predict` solves the same ridge regression in whichever form is smaller: the primal normal equations
(features x features, the original code path) when there are fewer features than training samples, and the dual form
(samples x samples) otherwise. The two give the same weights, W = (X'X + lam I)^-1 X'Y = X'(XX' + lam I)^-1 Y; the dual is
what makes the probe feasible on the whole-brain state (139,248 columns: the primal matrix alone would be 155 GB).

The dual form runs on the GPU when one is available. It is computed entirely in float64, in column chunks, with TF32
switched off: the training scripts export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, and a Gram matrix accumulated in TF32
(10-bit mantissa) gives a meaningless solve. The first GPU version of this file (commit e2e47f1) had exactly that
defect; its outputs are kept under results/hd_invalid_tf32_2026-11-11/ and must not be used.
"""
import numpy as np


def _dual_gpu(Xtr, Ytr, Xte, lam, device, chunk=4096):
    import torch
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            n, f = Xtr.shape
            K = torch.zeros(n, n, dtype=torch.float64, device=device)
            mus, sds = [], []
            for j in range(0, f, chunk):                       # Gram matrix of the standardised features, float64
                c = torch.from_numpy(np.ascontiguousarray(Xtr[:, j:j + chunk])).to(device).double()
                mu, sd = c.mean(0), c.std(0, correction=0) + 1e-6
                c.sub_(mu).div_(sd); K.addmm_(c, c.T); mus.append(mu); sds.append(sd)
            K.diagonal().add_(lam)
            Y = torch.from_numpy(np.asarray(Ytr, dtype=np.float64)).to(device); ym = Y.mean(0)
            alpha = torch.linalg.solve(K, Y - ym); del K
            pred = torch.zeros(Xte.shape[0], Y.shape[1], dtype=torch.float64, device=device)
            for i, j in enumerate(range(0, f, chunk)):        # pred = Xte W with W = Xtr' alpha, chunk by chunk
                c = torch.from_numpy(np.ascontiguousarray(Xtr[:, j:j + chunk])).to(device).double().sub_(mus[i]).div_(sds[i])
                e = torch.from_numpy(np.ascontiguousarray(Xte[:, j:j + chunk])).to(device).double().sub_(mus[i]).div_(sds[i])
                pred.addmm_(e, c.T @ alpha)
            out = (pred + ym).cpu().numpy()
        torch.cuda.empty_cache()
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


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


def ridge_weights(X, Y, lam, device, chunk=4096):
    """Centred (not standardised) ridge used by scripts/eval_homing.py: returns (mu_x [f], W [f, k], mu_y [k]) as float32 on
    `device`, so that the readout is (h - mu_x) @ W + mu_y. X is a float32 torch tensor [n, f] on any device (kept on the
    CPU for whole-brain states), Y is [n, k]. Everything is accumulated in float64 with TF32 off (see the module note);
    primal normal equations when f <= n, dual form otherwise (the 139,248-column primal matrix would be 72 GB in fp32)."""
    import torch
    cd = "cpu" if str(device).startswith("mps") else device      # no float64 on MPS
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            n, f = X.shape
            Yd = Y.to(cd).double(); ym = Yd.mean(0); Yc = Yd - ym
            mu = torch.cat([X[:, j:j + chunk].to(cd).double().mean(0) for j in range(0, f, chunk)])
            if f <= n:
                A = torch.zeros(f, f, dtype=torch.float64, device=cd); B = torch.zeros(f, Yc.shape[1], dtype=torch.float64, device=cd)
                for i in range(0, n, chunk):
                    r = X[i:i + chunk].to(cd).double() - mu
                    A.addmm_(r.T, r); B.addmm_(r.T, Yc[i:i + chunk])
                A.diagonal().add_(lam); W = torch.linalg.solve(A, B)
            else:
                on_dev = True                                                 # does the n x n Gram matrix fit on the (possibly shared) GPU?
                if str(cd).startswith("cuda"): on_dev = torch.cuda.mem_get_info(cd)[0] > 8 * n * n + 3 * 8 * n * chunk + (2 << 30)
                K = torch.zeros(n, n, dtype=torch.float64, device=cd if on_dev else "cpu")
                for j in range(0, f, chunk):
                    c = X[:, j:j + chunk].to(cd).double() - mu[j:j + chunk]
                    if on_dev: K.addmm_(c, c.T)
                    else:                                                     # products on the GPU in row blocks, accumulated on the CPU
                        for i in range(0, n, chunk): K[i:i + chunk] += (c[i:i + chunk] @ c.T).cpu()
                K.diagonal().add_(lam); Kc = K.cpu().numpy(); del K            # solve on the CPU, in place: the GPU may be shared
                from scipy.linalg.lapack import dposv                         # Cholesky solve in place (K is symmetric, so K.T is the
                _, al, info = dposv(Kc.T, Yc.cpu().numpy(), overwrite_a=1)   # same matrix in Fortran order: no 8 n^2 copy)
                assert info == 0, f"dposv info {info}"
                alpha = torch.from_numpy(al).to(cd); del Kc
                W = torch.cat([(X[:, j:j + chunk].to(cd).double() - mu[j:j + chunk]).T @ alpha for j in range(0, f, chunk)])
            return mu.float().to(device), W.float().to(device), ym.float().to(device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
