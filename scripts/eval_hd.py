"""Decode head direction / position from the student's internal state (linear probe, ridge), visible and blind.
Also reports whether the real EPG population forms a ring (circular-variance of first 2 PCs vs heading)."""
import argparse, json, numpy as np, torch, os
from wm2cx.student import FlyBrain, GRUBaseline
from wm2cx.data import load_split, Latents
from wm2cx.connectome import load as load_conn

p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--device", default="mps"); a = p.parse_args()
dev = a.device; ck = torch.load(f"results/runs/{a.run}.pt", weights_only=False); args = ck["args"]
conn = load_conn(args["conn"]); lat = Latents.__new__(Latents); lat.mu, lat.P, lat.scale = ck["lat"]["mu"], ck["lat"]["P"], ck["lat"]["scale"]
te, _ = load_split(args["test"], lat); tr, _ = load_split(args["train"], lat)
if args["wiring"] == "gru": model = GRUBaseline(te["obs"].shape[-1], args["k"], args["gru_hidden"])
else: model = FlyBrain(conn, te["obs"].shape[-1], args["k"], wiring=args["wiring"], n_brains=args["n_brains"], seed=args["seed"])
model.load_state_dict(ck["state"]); model.to(dev).eval()

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

region = np.tile(conn["region"], args["n_brains"]); ctype = np.tile(conn["cell_type"], args["n_brains"])
groups = {"all": np.ones(model.N, bool)} if args["wiring"] == "gru" else {"all": np.ones(model.N, bool), "CX": region == "CX", "EPG": np.char.startswith(ctype, "EPG"), "VIS": region == "VIS", "MB": region == "MB"}
out = {}
for blind in (False, True):
    Htr, Hte = states(tr, blind).numpy(), states(te, blind).numpy()
    sl = slice(8, 16) if blind else slice(0, 24)
    # heading relative to the episode's first heading (a ring attractor is anchored arbitrarily per environment),
    # position relative to the start, expressed in the start-heading frame (path integration)
    def rel(d):
        th = d["heading"].numpy(); th0 = th[:, :1]; dth = th - th0
        dp = d["pos"].numpy() - d["pos"].numpy()[:, :1]; c, s_ = np.cos(th0), np.sin(th0)
        dpr = np.stack([c * dp[..., 0] + s_ * dp[..., 1], -s_ * dp[..., 0] + c * dp[..., 1]], -1) / 6.0
        return np.stack([np.cos(dth), np.sin(dth)], -1)[:, sl].reshape(-1, 2), dpr[:, sl].reshape(-1, 2)
    ytr, ptr = rel(tr); yte, pte = rel(te)
    for g, m in groups.items():
        if m.sum() == 0: continue
        Xtr = Htr[:, sl][:, :, m].reshape(-1, m.sum()); Xte = Hte[:, sl][:, :, m].reshape(-1, m.sum())
        r2h, pred = ridge_r2(Xtr, ytr, Xte, yte); r2p, _ = ridge_r2(Xtr, ptr, Xte, pte)
        err = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(pred[:, 1], pred[:, 0]) - np.arctan2(yte[:, 1], yte[:, 0]))))))
        out[f"{'blind' if blind else 'vis'}_{g}"] = dict(n=int(m.sum()), heading_r2=r2h, heading_err_deg=float(np.median(err)), pos_r2=r2p)
        print(f"{'blind' if blind else 'vis '} {g:4s} n={m.sum():5d} heading R2 {r2h:.3f} median err {np.median(err):5.1f} deg  pos R2 {r2p:.3f}")
os.makedirs("results/hd", exist_ok=True); json.dump(out, open(f"results/hd/{a.run}.json", "w"), indent=1)
