"""Decode head direction / position from the student's internal state (linear probe, ridge), visible and blind.
Issue #7 additions: an action-only baseline (ridge on cumulative one-hot actions: what a leaky integrator of the
action input would give for free), and a feature-count-matched probe (random subset of n_match neurons, several draws)."""
import argparse, json, numpy as np, torch, os
from wm2cx.ckpt import load_run
from wm2cx.data import load_split

p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--device", default="mps")
p.add_argument("--n_match", type=int, default=0, help="feature-matched probe size (0 = the GRU hidden size if a *_gru run exists, else 762)")
p.add_argument("--draws", type=int, default=3); a = p.parse_args()
dev = a.device; model, args, conn, lat = load_run(a.run, device=dev)
te, _ = load_split(args["test"], lat); tr, _ = load_split(args["train"], lat)
rng = np.random.default_rng(0)

def states(d, blind):
    Hs = []
    with torch.no_grad():
        for i in range(0, len(d["obs"]), 64):
            obs, act = d["obs"][i:i + 64].to(dev), d["act"][i:i + 64].to(dev)
            vis = torch.ones(obs.shape[:2], device=dev)
            if blind: vis[:, 8:16] = 0
            _, H = model(obs, act, vis); Hs.append(H.cpu())
    return torch.cat(Hs)

def ridge_r2(Xtr, Ytr, Xte, Yte, lam=100.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]); Wt = np.linalg.solve(A, Xtr.T @ Ytr)
    pred = Xte @ Wt; return float(1 - ((pred - Yte) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()), pred

def action_features(d, sl):
    """cumulative one-hot action counts up to each step (the action the student received before step t)."""
    act = d["act"].numpy(); E, T = act.shape; oh = np.zeros((E, T, 5), np.float32)
    for t in range(1, T): oh[np.arange(E), t, act[:, t - 1]] = 1
    cum = np.cumsum(oh, 1); return cum[:, sl].reshape(-1, 5)

def integrated_features(d, sl):
    """analytic path integration of the discrete turns: cos/sin of (n_left - n_right) * pi/8 (what a perfect
    leaky integrator of the action input would expose linearly). Upper bound: heading is a deterministic function of actions."""
    cum = action_features(d, slice(0, None)).reshape(d["act"].shape[0], -1, 5); dth = (cum[..., 2] - cum[..., 3]) * np.pi / 8
    return np.stack([np.cos(dth), np.sin(dth)], -1)[:, sl].reshape(-1, 2)

region = np.tile(conn["region"], args["n_brains"]); ctype = np.tile(conn["cell_type"], args["n_brains"])
n_match = a.n_match or 762
groups = {"all": np.ones(model.N, bool)}
if args["wiring"] != "gru":
    groups.update({"CX": region == "CX", "EPG": np.char.startswith(ctype, "EPG"), "VIS": region == "VIS", "MB": region == "MB"})
if model.N > n_match:
    for k in range(a.draws): groups[f"match{n_match}_{k}"] = np.isin(np.arange(model.N), rng.choice(model.N, n_match, replace=False))
out = {}
for blind in (False, True):
    Htr, Hte = states(tr, blind).numpy(), states(te, blind).numpy()
    sl = slice(8, 16) if blind else slice(0, 24); tag = "blind" if blind else "vis"
    # heading relative to the episode's first heading (a ring attractor is anchored arbitrarily per environment),
    # position relative to the start, expressed in the start-heading frame (path integration)
    def rel(d):
        th = d["heading"].numpy(); th0 = th[:, :1]; dth = th - th0
        dp = d["pos"].numpy() - d["pos"].numpy()[:, :1]; c, s_ = np.cos(th0), np.sin(th0)
        dpr = np.stack([c * dp[..., 0] + s_ * dp[..., 1], -s_ * dp[..., 0] + c * dp[..., 1]], -1) / 6.0
        return np.stack([np.cos(dth), np.sin(dth)], -1)[:, sl].reshape(-1, 2), dpr[:, sl].reshape(-1, 2)
    ytr, ptr = rel(tr); yte, pte = rel(te)
    def report(name, Xtr, Xte):
        r2h, pred = ridge_r2(Xtr, ytr, Xte, yte); r2p, _ = ridge_r2(Xtr, ptr, Xte, pte)
        err = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(pred[:, 1], pred[:, 0]) - np.arctan2(yte[:, 1], yte[:, 0]))))))
        out[f"{tag}_{name}"] = dict(n=int(Xtr.shape[1]), heading_r2=r2h, heading_err_deg=float(np.median(err)), pos_r2=r2p)
        print(f"{tag:5s} {name:12s} n={Xtr.shape[1]:5d} heading R2 {r2h:.3f} median err {np.median(err):5.1f} deg  pos R2 {r2p:.3f}")
    report("actions_only", action_features(tr, sl), action_features(te, sl))  # issue #7 baseline: linear in cumulative action counts
    report("actions_integ", integrated_features(tr, sl), integrated_features(te, sl))  # issue #7: analytic integrator (ceiling)
    for g, m in groups.items():
        if m.sum() == 0: continue
        report(g, Htr[:, sl][:, :, m].reshape(-1, m.sum()), Hte[:, sl][:, :, m].reshape(-1, m.sum()))
    mk = [v for k, v in out.items() if k.startswith(f"{tag}_match")]
    if mk: out[f"{tag}_matched"] = dict(n=n_match, draws=len(mk), heading_r2=float(np.mean([v["heading_r2"] for v in mk])),
                                        heading_err_deg=float(np.mean([v["heading_err_deg"] for v in mk])), pos_r2=float(np.mean([v["pos_r2"] for v in mk])))
os.makedirs("results/hd", exist_ok=True); json.dump(out, open(f"results/hd/{a.run}.json", "w"), indent=1)
