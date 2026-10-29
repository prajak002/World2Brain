import argparse, json, time, numpy as np, torch, os
from wm2cx.student import FlyBrain, GRUBaseline
from wm2cx.data import load_split, blackout
from wm2cx.connectome import load as load_conn

p = argparse.ArgumentParser()
p.add_argument("--wiring", default="connectome"); p.add_argument("--n_brains", type=int, default=1)
p.add_argument("--epochs", type=int, default=30); p.add_argument("--bs", type=int, default=32); p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--H", type=int, default=8); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="mps")
p.add_argument("--k", type=int, default=256); p.add_argument("--gru_hidden", type=int, default=0)
p.add_argument("--train", nargs="+", default=["data/nav_train.npz", "data/arrange_train.npz"])
p.add_argument("--test", nargs="+", default=["data/nav_test.npz", "data/arrange_test.npz"])
p.add_argument("--conn", default="data/connectome.npz"); p.add_argument("--tag", default="")
a = p.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = a.device
tr, lat = load_split(a.train, k=a.k); te, _ = load_split(a.test, lat)
print(f"train eps {len(tr['obs'])} test eps {len(te['obs'])} PCA{a.k} explained {lat.explained:.3f}")
conn = load_conn(a.conn)
if a.wiring == "gru":
    model = GRUBaseline(tr["obs"].shape[-1], a.k, a.gru_hidden)
else:
    model = FlyBrain(conn, tr["obs"].shape[-1], a.k, wiring=a.wiring, n_brains=a.n_brains, seed=a.seed)
model.to(dev); print("neurons", model.N, "params", model.n_params())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
E, T = tr["obs"].shape[:2]

def run_eval(d):
    """Blind-horizon prediction error: 8 visible steps then H blind steps; cosine sim + MSE per horizon."""
    model.eval(); errs = []; cos = []
    with torch.no_grad():
        for i in range(0, len(d["obs"]), 64):
            obs, act, z = d["obs"][i:i + 64].to(dev), d["act"][i:i + 64].to(dev), d["z"][i:i + 64].to(dev)
            vis = torch.ones(obs.shape[:2], device=dev); vis[:, 8:8 + a.H] = 0
            zh, _ = model(obs, act, vis)
            errs.append(((zh - z) ** 2).mean(-1)[:, 8:8 + a.H].cpu()); cos.append(torch.nn.functional.cosine_similarity(zh, z, -1)[:, 8:8 + a.H].cpu())
    model.train(); errs = torch.cat(errs).mean(0); cos = torch.cat(cos).mean(0)
    return errs.tolist(), cos.tolist()

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
        mse, cs = run_eval(te)
        print(f"ep {ep} loss {tot/n:.4f} test blind-MSE h1 {mse[0]:.4f} h4 {mse[3]:.4f} h8 {mse[-1]:.4f} cos h1 {cs[0]:.3f} h8 {cs[-1]:.3f} {time.time()-t0:.0f}s", flush=True)
        log.append(dict(ep=ep, loss=tot / n, mse=mse, cos=cs))
mse, cs = run_eval(te)
name = f"{a.wiring}_K{a.n_brains}_s{a.seed}{a.tag}"
os.makedirs("results/runs", exist_ok=True)
json.dump(dict(args=vars(a), neurons=model.N, params=model.n_params(), test_mse=mse, test_cos=cs, log=log, pca_explained=lat.explained),
          open(f"results/runs/{name}.json", "w"), indent=1)
torch.save(dict(state=model.state_dict(), lat=dict(mu=lat.mu, P=lat.P, scale=lat.scale), args=vars(a)), f"results/runs/{name}.pt")
print("saved", name)
