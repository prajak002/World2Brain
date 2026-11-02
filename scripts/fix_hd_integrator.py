"""Recompute the `actions_integrated` position R² in results/hd/*.json with the correct step geometry (arena step_len is 0.6,
not 1.0; NYC twin 3 m / -2 m). Heading is unaffected. Real-video probes are skipped (their actions are quantised estimates)."""
import glob, json, os, numpy as np, torch
from wm2cx.ckpt import load_run
from wm2cx.data import load_split

TURN = np.pi / 8


def integrated(act, step):
    E, T = act.shape; th = np.zeros((E, T)); pos = np.zeros((E, T, 2))
    for t in range(1, T):
        a_ = act[:, t - 1]; th[:, t] = th[:, t - 1] + TURN * ((a_ == 2).astype(float) - (a_ == 3))
        fwd = np.where(a_ == 1, step[0], np.where(a_ == 4, step[1], 0.0))
        pos[:, t] = pos[:, t - 1] + fwd[:, None] * np.stack([np.cos(th[:, t]), np.sin(th[:, t])], -1)
    return pos / 6.0


def rel_pos(d):
    th0 = d["heading"].numpy()[:, :1]; dp = d["pos"].numpy() - d["pos"].numpy()[:, :1]; c, s = np.cos(th0), np.sin(th0)
    return np.stack([c * dp[..., 0] + s * dp[..., 1], -s * dp[..., 0] + c * dp[..., 1]], -1) / 6.0


cache = {}
for f in sorted(glob.glob("results/hd/*.json")):
    n = os.path.basename(f)[:-5]
    if "_rv_" in n or "_test" in n: continue
    d = json.load(open(f)); run = n
    for t in ("_mlp", "_b8_24", "_b4_24"):
        if run.endswith(t): run = run[:-len(t)]
    if not os.path.exists(f"results/runs/{run}.pt"): continue
    if run not in cache:
        _, args, _, lat = load_run(run, device="cpu"); te, _ = load_split(args["test"], lat, with_z=False)
        step = (3.0, -2.0) if "nyc" in run else (0.6, -0.36); cache[run] = (integrated(te["act"].numpy(), step), rel_pos(te))
    ip, pt = cache[run]; b0, b1 = (int(x) for x in d.get("blind_window", "8:16").split(":"))
    for key, sl in (("vis_actions_integrated", slice(0, pt.shape[1])), ("blind_actions_integrated", slice(b0, b1))):
        if key not in d: continue
        P, Y = ip[:, sl].reshape(-1, 2), pt[:, sl].reshape(-1, 2)
        old = d[key]["pos_r2"]; d[key]["pos_r2"] = float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
        print(f"{n:40s} {key:26s} pos R2 {old:.3f} -> {d[key]['pos_r2']:.3f}")
    json.dump(d, open(f, "w"), indent=1)
