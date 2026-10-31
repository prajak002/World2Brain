"""Trivial predictors for every prediction table (#6): constant train-mean and copy-last-visible latent.

Both are evaluated in exactly the student's setting (same PCA-256 whitening fit on train, same blind windows,
same test episodes) and additionally in the original teacher space (3,840-d for ViT-B mean + 2x2 pooled tokens).
Per-episode errors are stored so summarize.py can run paired bootstrap tests against the students (#8).

  PYTHONPATH=. python scripts/eval_trivial.py                                   # arena  -> results/baselines/*_K0_s0.json
  PYTHONPATH=. python scripts/eval_trivial.py --train data/nyc_train.npz --test data/nyc_test.npz --tag _nyc
"""
import argparse, json, os, numpy as np, torch
from wm2cx.data import load_split, parse_windows, window_errors

p = argparse.ArgumentParser()
p.add_argument("--train", nargs="+", default=["data/nav_train.npz", "data/arrange_train.npz"])
p.add_argument("--test", nargs="+", default=["data/nav_test.npz", "data/arrange_test.npz"])
p.add_argument("--k", type=int, default=256); p.add_argument("--tag", default="")
p.add_argument("--eval_windows", nargs="+", default=["8:16", "14:22"])
a = p.parse_args()
tr, lat = load_split(a.train, k=a.k); te, _ = load_split(a.test, lat, keep_raw=True)
T = te["z"].shape[1]; windows = parse_windows(a.eval_windows, T)
z, z_raw = te["z"], te["z_raw"]
mean_pca = tr["z"].reshape(-1, a.k).mean(0)                         # train mean in whitened PCA space (~0 by construction)


def predictors(t0, t1):
    """Each returns z_hat [E,T,k] in PCA space (teacher-space error is taken via the PCA inverse, like the students)."""
    const = mean_pca.expand_as(z).clone()
    copy = z.clone(); copy[:, t0:t1] = z[:, t0 - 1:t0]                # hold the last visible latent through the blind window
    return {"const_mean": const, "copy_last": copy}


os.makedirs("results/baselines", exist_ok=True)
for name in ("const_mean", "copy_last"):
    ev = {}
    for t0, t1 in windows:
        zh = predictors(t0, t1)[name]
        e = window_errors(zh, z, t0, t1, lat, z_raw)
        ev[f"{t0}:{t1}"] = dict(mse=e["mse"].mean(0).tolist(), cos=e["cos"].mean(0).tolist(), mse_teacher=e["mse_teacher"].mean(0).tolist(),
                                per_ep_mse=e["mse"].numpy().round(5).tolist())
    W0 = f"{windows[0][0]}:{windows[0][1]}"
    # variance of the raw teacher latents = teacher-space MSE of the constant predictor at the *test* mean (reference scale)
    var_teacher = float(z_raw.float().reshape(-1, z_raw.shape[-1]).var(0).mean())
    out = dict(args=dict(wiring=name, n_brains=0, seed=0, tag=a.tag, k=a.k, test=a.test, train=a.train, eval_windows=a.eval_windows),
               neurons=0, params=0, test_mse=ev[W0]["mse"], test_cos=ev[W0]["cos"], test_mse_teacher=ev[W0]["mse_teacher"],
               test_mse_per_ep=ev[W0]["per_ep_mse"], test_windows=ev, n_test_episodes=int(len(z)), pca_explained=lat.explained,
               teacher_space_variance=var_teacher)
    fn = f"results/baselines/{name}_K0_s0{a.tag}.json"; json.dump(out, open(fn, "w"), indent=1)
    m = ev[W0]
    print(f"{name:11s} {W0}: PCA-space MSE h1 {m['mse'][0]:.4f} h4 {m['mse'][3]:.4f} h8 {m['mse'][-1]:.4f} mean {np.mean(m['mse']):.4f} | "
          f"teacher-space mean {np.mean(m['mse_teacher']):.4f} (teacher variance {var_teacher:.4f}) -> {fn}")
