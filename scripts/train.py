import argparse, json, time, numpy as np, torch, os
from wm2cx.student import FlyBrain, GRUBaseline, gru_hidden_for, WIRINGS
from wm2cx.data import load_split, blackout
from wm2cx.connectome import load as load_conn

p = argparse.ArgumentParser()
p.add_argument("--wiring", default="connectome", choices=WIRINGS + ("gru",)); p.add_argument("--n_brains", type=int, default=1)
p.add_argument("--epochs", type=int, default=30); p.add_argument("--bs", type=int, default=32); p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--H", type=int, default=8); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="mps")
p.add_argument("--k", type=int, default=256)
p.add_argument("--gru_hidden", type=int, default=0, help="0 = largest hidden size with <= the FlyBrain parameter count of this config (issue #4)")
p.add_argument("--vis_flag", type=int, default=1, help="feed the blackout flag to every unit of both FlyBrain and GRU (issue #4)")
p.add_argument("--in_rank", type=int, default=0, help="low-rank W_in bottleneck (0 = full); shrinks the shared input layer (issue #3)")
p.add_argument("--val", type=float, default=0.0, help="fraction of train episodes held out as a validation split for hyper-parameter selection")
p.add_argument("--train", nargs="+", default=["data/nav_train.npz", "data/arrange_train.npz"])
p.add_argument("--test", nargs="+", default=["data/nav_test.npz", "data/arrange_test.npz"])
p.add_argument("--conn", default="data/connectome.npz"); p.add_argument("--tag", default="")
a = p.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = a.device
tr, lat = load_split(a.train, k=a.k); te, _ = load_split(a.test, lat)
va = None
if a.val > 0:  # validation split carved from the training episodes (fixed by seed), never from test
    E0 = len(tr["obs"]); perm = np.random.default_rng(1000 + a.seed).permutation(E0); nv = int(a.val * E0)
    vi, ti = torch.from_numpy(perm[:nv]), torch.from_numpy(perm[nv:])
    va = {k: v[vi] for k, v in tr.items()}; tr = {k: v[ti] for k, v in tr.items()}
print(f"train eps {len(tr['obs'])} val eps {0 if va is None else len(va['obs'])} test eps {len(te['obs'])} PCA{a.k} explained {lat.explained:.3f}")
conn = load_conn(a.conn); obs_dim = tr["obs"].shape[-1]
fly = FlyBrain(conn, obs_dim, a.k, wiring=a.wiring if a.wiring != "gru" else "connectome", n_brains=a.n_brains, seed=a.seed, vis_flag=bool(a.vis_flag), in_rank=a.in_rank)
if a.wiring == "gru":
    if a.gru_hidden <= 0: a.gru_hidden = gru_hidden_for(fly.n_params(), obs_dim, a.k, vis_flag=bool(a.vis_flag))
    model = GRUBaseline(obs_dim, a.k, a.gru_hidden, vis_flag=bool(a.vis_flag)); del fly
else:
    model = fly
model.to(dev); pb = model.param_breakdown(); print("neurons", model.N, "params", pb)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
E, T = tr["obs"].shape[:2]
P = torch.from_numpy(lat.P).to(dev); scale = float(lat.scale)  # for MSE in the original 3,840-d teacher space (issue #6)


def predict(d, t0=8):
    """8 visible steps then H blind steps starting at t0. Returns per-episode, per-horizon MSE / cosine [E,H], and raw-space MSE."""
    model.eval(); errs, cos, raw = [], [], []
    with torch.no_grad():
        for i in range(0, len(d["obs"]), 64):
            obs, act, z = d["obs"][i:i + 64].to(dev), d["act"][i:i + 64].to(dev), d["z"][i:i + 64].to(dev)
            vis = torch.ones(obs.shape[:2], device=dev); vis[:, t0:t0 + a.H] = 0
            zh, _ = model(obs, act, vis); sl = slice(t0, t0 + a.H)
            errs.append(((zh - z) ** 2).mean(-1)[:, sl].cpu()); cos.append(torch.nn.functional.cosine_similarity(zh, z, -1)[:, sl].cpu())
            raw.append((((zh - z)[:, sl] * scale) @ P.T).pow(2).mean(-1).cpu())  # exact: P is orthonormal, so this equals the 3,840-d MSE of the PCA-reconstructed target
    model.train()
    return torch.cat(errs), torch.cat(cos), torch.cat(raw)


def trivial_baselines(d, t0=8):
    """const-mean (train-set mean latent) and copy-last-visible predictors, same windows, same space (issue #6)."""
    z = d["z"]; sl = slice(t0, t0 + a.H); mu = tr["z"].mean((0, 1))
    cm = ((mu - z[:, sl]) ** 2).mean(-1); cl = ((z[:, t0 - 1:t0] - z[:, sl]) ** 2).mean(-1)
    return dict(const_mean=cm.mean(0).tolist(), copy_last=cl.mean(0).tolist(),
                const_mean_raw=float(((mu - z[:, sl]) * scale @ P.T.cpu()).pow(2).mean()), copy_last_raw=float(((z[:, t0 - 1:t0] - z[:, sl]) * scale @ P.T.cpu()).pow(2).mean()))


def run_eval(d, t0=8):
    e, c, r = predict(d, t0); return e.mean(0).tolist(), c.mean(0).tolist(), float(r.mean())


t0 = time.time(); log = []
for ep in range(a.epochs):
    perm = torch.randperm(E); tot = 0; n = 0
    for i in range(0, E, a.bs):
        idx = perm[i:i + a.bs]
        obs, act, z = tr["obs"][idx].to(dev), tr["act"][idx].to(dev), tr["z"][idx].to(dev)
        vis = blackout(len(idx), T, a.H, rng, dev)
        zh, Hs = model(obs, act, vis)
        w = 1.0 + 2.0 * (1 - vis)  # emphasise blind (imagination) steps
        loss = (((zh - z) ** 2).mean(-1) * w).mean() + 1e-4 * (Hs ** 2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += loss.item(); n += 1
    sched.step()
    if ep % 5 == 4 or ep == a.epochs - 1:
        mse, cs, raw = run_eval(te)
        rec = dict(ep=ep, loss=tot / n, mse=mse, cos=cs, mse_raw=raw)
        if va is not None: rec["val_mse"] = float(np.mean(run_eval(va)[0]))
        print(f"ep {ep} loss {tot/n:.4f} test blind-MSE h1 {mse[0]:.4f} h4 {mse[3]:.4f} h8 {mse[-1]:.4f} cos h1 {cs[0]:.3f} h8 {cs[-1]:.3f}" + (f" val {rec['val_mse']:.4f}" if va is not None else "") + f" {time.time()-t0:.0f}s", flush=True)
        log.append(rec)
errs, cos, raw = predict(te); mse, cs = errs.mean(0).tolist(), cos.mean(0).tolist()
alt_t0 = 12 if T >= 12 + a.H else 4  # a second blind window to check for position-specific overfitting (issue #8)
mse_alt, cs_alt, raw_alt = run_eval(te, alt_t0)
name = f"{a.wiring}_K{a.n_brains}_s{a.seed}{a.tag}"
os.makedirs("results/runs", exist_ok=True)
out = dict(args=vars(a), neurons=model.N, params=model.n_params(), param_breakdown=pb, test_mse=mse, test_cos=cs, test_mse_raw=float(raw.mean()),
           alt_window=dict(t0=alt_t0, mse=mse_alt, cos=cs_alt, mse_raw=raw_alt), baselines=trivial_baselines(te), baselines_alt=trivial_baselines(te, alt_t0),
           log=log, pca_explained=lat.explained, val_mse=(log[-1].get("val_mse") if va is not None else None))
json.dump(out, open(f"results/runs/{name}.json", "w"), indent=1)
np.savez_compressed(f"results/runs/{name}_episodes.npz", mse=errs.numpy(), cos=cos.numpy(), mse_raw=raw.numpy(), task=te["task"].numpy())  # per-episode errors for paired bootstrap CIs
torch.save(dict(state=model.state_dict(), lat=dict(mu=lat.mu, P=lat.P, scale=lat.scale), args=vars(a)), f"results/runs/{name}.pt")
print("saved", name, "baselines", {k: (np.mean(v) if isinstance(v, list) else v) for k, v in out["baselines"].items()})
