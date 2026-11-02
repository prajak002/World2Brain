"""Decode head direction / position from the student's internal state (linear probe, ridge), visible and blind.

Controls (#7):
  * `actions_only`: ridge on the cumulative one-hot action counts alone (no neural state). Turns are discrete and
    fixed-angle, so relative heading is a linear function of (#left - #right); any leaky integrator of the one-hot
    action input exposes it. A population only "encodes heading" if it beats this floor.
  * `all@1400`: random subset of 1,400 neurons so the FlyBrain probe has the same feature count as the GRU (1,400 hidden).
  * `actions+all@1400`: both, to see whether the state adds anything beyond the action record.
"""
import argparse, json, numpy as np, torch, os
from wm2cx.ckpt import load_run
from wm2cx.data import load_split

p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--device", default="mps")
p.add_argument("--match_n", type=int, default=1400, help="feature-matched subset size (GRU hidden size)"); p.add_argument("--lam", type=float, default=100.0)
p.add_argument("--blind", default="8:16"); a = p.parse_args()
dev = a.device; model, args, conn, lat = load_run(a.run, device=dev)
te, _ = load_split(args["test"], lat); tr, _ = load_split(args["train"], lat)
b0, b1 = (int(x) for x in a.blind.split(":")); T = te["obs"].shape[1]


def states(d, blind):
    Hs = []
    with torch.no_grad():
        for i in range(0, len(d["obs"]), 64):
            obs, act = d["obs"][i:i + 64].to(dev), d["act"][i:i + 64].to(dev)
            vis = torch.ones(obs.shape[:2], device=dev)
            if blind: vis[:, b0:b1] = 0
            _, H = model(obs, act, vis); Hs.append(H.cpu())
    return torch.cat(Hs)


def action_features(d):
    """[E, T, 6]: cumulative count of each action taken before step t (act[:, t-1] is the action that led to state t) + t."""
    act = d["act"].numpy(); E, T_ = act.shape
    oh = np.eye(5, dtype=np.float32)[act]                         # [E,T,5]
    prev = np.concatenate([np.zeros((E, 1, 5), np.float32), oh[:, :-1]], 1)
    cum = np.cumsum(prev, 1)
    return np.concatenate([cum, np.arange(T_, dtype=np.float32)[None, :, None].repeat(E, 0) / T_], -1)


def ridge_r2(Xtr, Ytr, Xte, Yte, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]); Wt = np.linalg.solve(A, Xtr.T @ (Ytr - Ytr.mean(0)))
    pred = Xte @ Wt + Ytr.mean(0); return float(1 - ((pred - Yte) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()), pred


region = np.tile(conn["region"], args["n_brains"]); ctype = np.tile(conn["cell_type"], args["n_brains"])
groups = {"all": np.ones(model.N, bool)}
if args["wiring"] != "gru":
    groups.update({"CX": region == "CX", "EPG": np.char.startswith(ctype, "EPG"), "VIS": region == "VIS", "MB": region == "MB"})
if model.N > a.match_n:
    sub = np.zeros(model.N, bool); sub[np.random.default_rng(0).choice(model.N, a.match_n, replace=False)] = True; groups[f"all@{a.match_n}"] = sub
Atr, Ate = action_features(tr), action_features(te)
out = {}
for blind in (False, True):
    Htr, Hte = states(tr, blind).numpy(), states(te, blind).numpy()
    sl = slice(b0, b1) if blind else slice(0, T)
    # heading relative to the episode's first heading (a ring attractor is anchored arbitrarily per environment),
    # position relative to the start, expressed in the start-heading frame (path integration)
    def rel(d):
        th = d["heading"].numpy(); th0 = th[:, :1]; dth = th - th0
        dp = d["pos"].numpy() - d["pos"].numpy()[:, :1]; c, s_ = np.cos(th0), np.sin(th0)
        dpr = np.stack([c * dp[..., 0] + s_ * dp[..., 1], -s_ * dp[..., 0] + c * dp[..., 1]], -1) / 6.0
        return np.stack([np.cos(dth), np.sin(dth)], -1)[:, sl].reshape(-1, 2), dpr[:, sl].reshape(-1, 2)
    ytr, ptr = rel(tr); yte, pte = rel(te)
    feats = {g: (Htr[:, sl][:, :, m].reshape(-1, m.sum()), Hte[:, sl][:, :, m].reshape(-1, m.sum())) for g, m in groups.items() if m.sum() > 0}
    feats["actions_only"] = (Atr[:, sl].reshape(-1, 6), Ate[:, sl].reshape(-1, 6))
    if f"all@{a.match_n}" in feats:
        feats[f"actions+all@{a.match_n}"] = tuple(np.concatenate([x, y], 1) for x, y in zip(feats["actions_only"], feats[f"all@{a.match_n}"]))
    for g, (Xtr, Xte) in feats.items():
        lam = 1e-3 if g == "actions_only" else a.lam                 # 6 features: effectively OLS
        r2h, pred = ridge_r2(Xtr, ytr, Xte, yte, lam); r2p, _ = ridge_r2(Xtr, ptr, Xte, pte, lam)
        err = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(pred[:, 1], pred[:, 0]) - np.arctan2(yte[:, 1], yte[:, 0]))))))
        out[f"{'blind' if blind else 'vis'}_{g}"] = dict(n=int(Xtr.shape[1]), heading_r2=r2h, heading_err_deg=float(np.median(err)), pos_r2=r2p)
        print(f"{'blind' if blind else 'vis '} {g:18s} n={Xtr.shape[1]:5d} heading R2 {r2h:.3f} median err {np.median(err):5.1f} deg  pos R2 {r2p:.3f}")
out["blind_window"] = a.blind
os.makedirs("results/hd", exist_ok=True); json.dump(out, open(f"results/hd/{a.run}.json", "w"), indent=1)
