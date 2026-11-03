"""Decode head direction / position from the student's internal state (linear probe, ridge), visible and blind.

Controls (#7):
  * `actions_only`: ridge on the cumulative one-hot action counts alone (no neural state). Turns are discrete and
    fixed-angle, so relative heading is a linear function of (#left - #right); any leaky integrator of the one-hot
    action input exposes it. A population only "encodes heading" if it beats this floor.
  * `all@1400`: random subset of 1,400 neurons so the FlyBrain probe has the same feature count as the GRU (1,400 hidden).
  * `actions+all@1400`: both, to see whether the state adds anything beyond the action record.
  * `actions_integrated`: closed-form integrator of the nominal actions (heading = 22.5 deg x (#left - #right), position by dead
    reckoning), no fit. With fixed-angle turns this is exact, so it is the information ceiling for the blind window: the probes
    measure how much of that integral the state keeps in a *linearly readable* (population) code, not whether the information exists.
  * `rand@CX`, `rand@EPG`: random neuron subsets of the same size as the CX / EPG populations (is the region special, or its size?).
  * `--probe mlp`: a 2-layer MLP probe instead of ridge, to tell a linear (ring-like) code from a merely non-linear one.
  * `--blind t0:t1` and `--tag`: other dark windows (8:24, 4:24) written to results/hd/<run><tag>.json.
  * `--test data/realvid_all.npz`: probe on real handheld video, whose heading / position come from continuous egomotion
    (phase correlation), not from the quantised action record. `--train` defaults to the run's own training split (the
    heading readout learned in the simulator is read out on real video: transfer); `--train data/realvid_all.npz
    --split_by_clip` fits it on the first clips of the same file and tests on the last two (in-domain, no episode overlap).
"""
import argparse, json, numpy as np, torch, os
from wm2cx.ckpt import load_run
from wm2cx.data import load_split

p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--device", default="mps")
p.add_argument("--match_n", type=int, default=1400, help="feature-matched subset size (GRU hidden size)"); p.add_argument("--lam", type=float, default=100.0)
p.add_argument("--blind", default="8:16"); p.add_argument("--probe", default="ridge", choices=["ridge", "mlp"]); p.add_argument("--tag", default="")
p.add_argument("--test", nargs="*", default=None, help="episode files to probe on (default: the run's test split)")
p.add_argument("--train", nargs="*", default=None, help="episode files to fit the probe on (default: the run's training split)")
p.add_argument("--split_by_clip", type=int, default=0, help="--train and --test are the same real-video file: fit on all but the last N clips, test on them")
a = p.parse_args()
dev = a.device; model, args, conn, lat = load_run(a.run, device=dev)
te, _ = load_split(a.test or args["test"], lat, with_z=False); tr, _ = load_split(a.train or args["train"], lat, with_z=False)   # teacher latents not needed
if a.split_by_clip:                                               # real video: episodes overlap within a clip (stride 12), so split by clip
    clip = np.concatenate([np.load(f)["clip"] for f in a.test]); test_clips = np.unique(clip)[-a.split_by_clip:]; m = np.isin(clip, test_clips)
    te = {k: v[torch.from_numpy(m)] for k, v in te.items() if torch.is_tensor(v)}; tr = {k: v[torch.from_numpy(~m)] for k, v in tr.items() if torch.is_tensor(v)}
    print(f"split by clip: {int((~m).sum())} train episodes (clips {np.unique(clip[~m]).tolist()}), {int(m.sum())} test (clips {test_clips.tolist()})")
TURN = np.pi / 8; STEP = {1: 0.6, 4: -0.36} if "nyc" not in a.run else {1: 3.0, 4: -2.0}   # action geometry: arena step_len 0.6 (wm2cx/env.py), NYC twin 3 m / -2 m
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


def integrated(d):
    """[E, T, 4]: nominal heading (cos, sin) and dead-reckoned position (start frame, /6 like `rel`) from the action record."""
    act = d["act"].numpy(); E, T_ = act.shape; th = np.zeros((E, T_)); pos = np.zeros((E, T_, 2))
    for t in range(1, T_):
        a_ = act[:, t - 1]; th[:, t] = th[:, t - 1] + TURN * ((a_ == 2).astype(float) - (a_ == 3))
        fwd = np.vectorize(lambda k: STEP.get(int(k), 0.0))(a_)
        pos[:, t] = pos[:, t - 1] + fwd[:, None] * np.stack([np.cos(th[:, t]), np.sin(th[:, t])], -1)
    return np.concatenate([np.cos(th)[..., None], np.sin(th)[..., None], pos / 6.0], -1)


def r2(pred, Y): return float(1 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def ridge_r2(Xtr, Ytr, Xte, Yte, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]); Wt = np.linalg.solve(A, Xtr.T @ (Ytr - Ytr.mean(0)))
    pred = Xte @ Wt + Ytr.mean(0); return r2(pred, Yte), pred


def mlp_r2(Xtr, Ytr, Xte, Yte, lam=None, hidden=256, epochs=30):
    """2-layer tanh MLP probe (Adam, weight decay 1e-5), same standardisation as the ridge."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr_, Xte_ = (Xtr - mu) / sd, (Xte - mu) / sd
    X = torch.from_numpy(Xtr_.astype(np.float32)).to(dev); Y = torch.from_numpy(Ytr.astype(np.float32)).to(dev); n = len(X)
    torch.manual_seed(0); net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, Y.shape[1])).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 1024):
            j = perm[i:i + 1024]; loss = ((net(X[j]) - Y[j]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): pred = net(torch.from_numpy(Xte_.astype(np.float32)).to(dev)).cpu().numpy()
    return r2(pred, Yte), pred


probe = ridge_r2 if a.probe == "ridge" else mlp_r2


region = np.tile(conn["region"], args["n_brains"]); ctype = np.tile(conn["cell_type"], args["n_brains"])
groups = {"all": np.ones(model.N, bool)}
if args["wiring"] != "gru":
    groups.update({"CX": region == "CX", "EPG": np.char.startswith(ctype, "EPG"), "VIS": region == "VIS", "MB": region == "MB"})
if model.N > a.match_n:
    sub = np.zeros(model.N, bool); sub[np.random.default_rng(0).choice(model.N, a.match_n, replace=False)] = True; groups[f"all@{a.match_n}"] = sub
for reg in ("CX", "EPG"):                                        # size-matched random subsets for the anatomical populations
    if reg in groups and 0 < groups[reg].sum() < model.N:
        sub = np.zeros(model.N, bool); sub[np.random.default_rng(1).choice(model.N, int(groups[reg].sum()), replace=False)] = True; groups[f"rand@{reg}"] = sub
Atr, Ate = action_features(tr), action_features(te); Ite = integrated(te)
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
    ih, ip = Ite[:, sl, :2].reshape(-1, 2), Ite[:, sl, 2:].reshape(-1, 2)   # closed-form integrator of the nominal actions (no fit)
    err_i = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(ih[:, 1], ih[:, 0]) - np.arctan2(yte[:, 1], yte[:, 0]))))))
    out[f"{'blind' if blind else 'vis'}_actions_integrated"] = dict(n=0, heading_r2=r2(ih, yte), heading_err_deg=float(np.median(err_i)), pos_r2=r2(ip, pte))
    print(f"{'blind' if blind else 'vis '} {'actions_integrated':18s} n=    0 heading R2 {r2(ih, yte):.3f} median err {np.median(err_i):5.1f} deg  pos R2 {r2(ip, pte):.3f}")
    for g, (Xtr, Xte) in feats.items():
        lam = 1e-3 if g == "actions_only" else a.lam                 # 6 features: effectively OLS
        r2h, pred = probe(Xtr, ytr, Xte, yte, lam); r2p, _ = probe(Xtr, ptr, Xte, pte, lam)
        err = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(pred[:, 1], pred[:, 0]) - np.arctan2(yte[:, 1], yte[:, 0]))))))
        out[f"{'blind' if blind else 'vis'}_{g}"] = dict(n=int(Xtr.shape[1]), heading_r2=r2h, heading_err_deg=float(np.median(err)), pos_r2=r2p)
        print(f"{'blind' if blind else 'vis '} {g:18s} n={Xtr.shape[1]:5d} heading R2 {r2h:.3f} median err {np.median(err):5.1f} deg  pos R2 {r2p:.3f}")
out["blind_window"] = a.blind; out["probe"] = a.probe
os.makedirs("results/hd", exist_ok=True); json.dump(out, open(f"results/hd/{a.run}{a.tag}.json", "w"), indent=1)
