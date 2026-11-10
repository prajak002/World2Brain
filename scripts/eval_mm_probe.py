"""Memory Maze B5: offline probing of a frozen student (PREREG section 2, protocol of Pasukonis et al. 2023 unchanged).

Probe input = frozen model state at step t concatenated with agent_pos (2) and agent_dir (2); no gradient into the model.
Two probes, each an MLP of 4 hidden layers x 1024 with LayerNorm (eps 1e-3) and ELU, Adam lr 3e-4 eps 1e-5:
  Walls   -> 81 logits of maze_layout, BCE;     score = tile accuracy (%)
  Objects -> 3 x 2 agent-centric targets_vec, MSE; score = squared error per object, mean over objects
Trained on states of the training trajectories (all steps, all visible), --passes over --n_probe_train trajectories in
batches of --bs trajectories x --window-step windows (one probe update per window); the final probe is scored once on the
1,000 eval trajectories, each score averaged over steps 500-1000. Conditions:
  C S B R G T  --ckpt <checkpoint from train_mm.py>       N  teacher PCA-256 latent of step t (no recurrence)
  K  constant predictor: train-mean layout probabilities (thresholded) and train-mean object vectors
Writes results/memmaze/probe/<label>.json with per-trajectory scores (for the paired bootstrap in mm_stats.py).
"""
import argparse, json, os, time, numpy as np, torch, torch.nn as nn
from wm2cx.memmaze import load_shards, walls_acc, objects_mse, SCORE_FROM, SCORE_TO
from wm2cx.mm_models import build, rollout, act_prev_of, transformer_states, probe_mlp
from wm2cx.connectome import load as load_conn

p = argparse.ArgumentParser()
p.add_argument("--cond", required=True, choices=["C", "S", "B", "R", "G", "T", "N", "K"]); p.add_argument("--ckpt", default="")
p.add_argument("--label", default="", help="output name; default <ckpt stem> or <cond>_s<seed>")
p.add_argument("--seed", type=int, default=0, help="probe seed (PREREG: one probe seed per model seed)")
p.add_argument("--data", default="data/mm9"); p.add_argument("--conn", default="data/connectome_full.npz")
p.add_argument("--n_probe_train", type=int, default=0, help="training trajectories used for the probe (0 = all); PREREG Amendment 1")
p.add_argument("--passes", type=int, default=1); p.add_argument("--bs", type=int, default=32); p.add_argument("--window", type=int, default=48)
p.add_argument("--n_eval", type=int, default=0, help="smoke tests only: first N eval trajectories"); p.add_argument("--device", default="cuda")
p.add_argument("--out", default="results/memmaze/probe")
a = p.parse_args(); dev = a.device; torch.backends.cuda.matmul.allow_tf32 = True; torch.manual_seed(a.seed)
label = a.label or (os.path.basename(a.ckpt).rsplit(".", 1)[0] if a.ckpt else f"{a.cond}_s{a.seed}")
os.makedirs(a.out, exist_ok=True); J = f"{a.out}/{label}.json"
if os.path.exists(J): print("done already:", J); raise SystemExit
sp = json.load(open(f"{a.data}/splits.json")); META = ("obs", "act", "agent_pos", "agent_dir", "maze_layout", "targets_vec")
need_z = a.cond == "N"
tr = load_shards(a.data, sp["train"], lat=need_z, meta=META); ev = load_shards(a.data, sp["eval"], lat=need_z, meta=META)
if a.n_probe_train: tr = {k: v[:a.n_probe_train] for k, v in tr.items()}
if a.n_eval: ev = {k: v[:a.n_eval] for k, v in ev.items()}
Etr, T = tr["act"].shape; print(f"{label}: probe-train traj {Etr} eval traj {len(ev['act'])}", flush=True)
t0 = time.time()

if a.cond == "K":
    lay_p = tr["maze_layout"].reshape(Etr, -1).mean(0); vec_m = tr["targets_vec"].mean((0, 1))
    sw = walls_acc(lay_p[None], ev["maze_layout"].reshape(len(ev["act"]), -1), is_logit=False)
    so = np.array([objects_mse(np.broadcast_to(vec_m, v[SCORE_FROM:SCORE_TO + 1].shape), v[SCORE_FROM:SCORE_TO + 1]).mean() for v in ev["targets_vec"]])
    json.dump(dict(cond="K", label=label, walls_acc=sw.tolist(), objects_mse=so.tolist(), mean_walls=float(sw.mean()), mean_objects=float(so.mean()),
                   n_eval=len(sw), n_probe_train=int(Etr)), open(J, "w"), indent=1)
    print(f"K walls {sw.mean():.2f} objects {so.mean():.2f}"); raise SystemExit

model = None; feat_dim = 256
if a.cond != "N":
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); assert ck["cond"] == a.cond, (ck["cond"], a.cond)
    model, _ = build(a.cond, load_conn(a.conn), ck["args"]["seed"], ck["args"]["d_model"], ck["args"]["window"])
    model.load_state_dict(ck["state"]); model.to(dev).eval().requires_grad_(False); feat_dim = model.N
to = lambda x, idx: torch.from_numpy(np.ascontiguousarray(x[idx])).to(dev)


@torch.no_grad()
def windows(d, idx):
    """Yield (features [B,W,F+4], layout [B,81], targets [B,W,3,2], step slice) over whole trajectories, state carried."""
    B = len(idx); obs = to(d["obs"], idx).float() / 255.0; act = to(d["act"], idx).long(); aux = torch.cat([to(d["agent_pos"], idx), to(d["agent_dir"], idx)], -1)
    lay = to(d["maze_layout"], idx).reshape(B, -1).float(); tv = to(d["targets_vec"], idx)
    if a.cond == "N": S_all = to(d["z"], idx).float()
    elif a.cond == "T": S_all = transformer_states(model, obs, act_prev_of(act), torch.ones(B, T, device=dev), ctx=a.window)
    h = None; last = None
    for t0 in range(0, T, a.window):
        s = slice(t0, min(t0 + a.window, T))
        if a.cond in ("N", "T"): S = S_all[:, s]
        else:
            _, S, h = rollout(model, obs[:, s], act_prev_of(act[:, s], last), torch.ones(B, s.stop - t0, device=dev), h); last = act[:, s.stop - 1]
        yield torch.cat([S, aux[:, s]], -1), lay, tv[:, s], s


mlp = lambda i, o: probe_mlp(i, o).to(dev)
pw, po = mlp(feat_dim + 4, 81), mlp(feat_dim + 4, 6)
opt = torch.optim.Adam(list(pw.parameters()) + list(po.parameters()), lr=3e-4, eps=1e-5)
rng = np.random.default_rng(a.seed); n_upd = 0; lw, lo = [], []
for ps in range(a.passes):
    perm = rng.permutation(Etr)
    for i in range(0, Etr - a.bs + 1, a.bs):
        for x, lay, tv, _ in windows(tr, np.sort(perm[i:i + a.bs])):
            yw = pw(x); yo = po(x).view(*x.shape[:2], 3, 2)
            l_w = nn.functional.binary_cross_entropy_with_logits(yw, lay[:, None].expand_as(yw)); l_o = ((yo - tv) ** 2).mean()
            opt.zero_grad(); (l_w + l_o).backward(); opt.step(); n_upd += 1; lw.append(l_w.item()); lo.append(l_o.item())
        if (i // a.bs) % 20 == 0:
            print(f"pass {ps} traj {i + a.bs}/{Etr} upd {n_upd} bce {np.mean(lw[-200:]):.4f} mse {np.mean(lo[-200:]):.4f} {time.time() - t0:.0f}s", flush=True)

pw.eval(); po.eval(); SW, SO = [], []
with torch.no_grad():
    for i in range(0, len(ev["act"]), a.bs):
        idx = np.arange(i, min(i + a.bs, len(ev["act"]))); aw, ao = [], []
        for x, lay, tv, s in windows(ev, idx):
            aw.append(walls_acc(pw(x).cpu().numpy(), lay[:, None].expand(-1, x.shape[1], -1).cpu().numpy()))
            ao.append(objects_mse(po(x).view(*x.shape[:2], 3, 2).cpu().numpy(), tv.cpu().numpy()))
        aw, ao = np.concatenate(aw, 1), np.concatenate(ao, 1)                  # [B, 1001]
        SW += aw[:, SCORE_FROM:SCORE_TO + 1].mean(1).tolist(); SO += ao[:, SCORE_FROM:SCORE_TO + 1].mean(1).tolist()
res = dict(cond=a.cond, label=label, ckpt=a.ckpt, probe_seed=a.seed, walls_acc=SW, objects_mse=SO, mean_walls=float(np.mean(SW)), mean_objects=float(np.mean(SO)),
           n_eval=len(SW), n_probe_train=int(Etr), passes=a.passes, probe_updates=n_upd, feat_dim=feat_dim, final_train_bce=float(np.mean(lw[-200:])),
           final_train_mse=float(np.mean(lo[-200:])), wall_h=(time.time() - t0) / 3600)
torch.save(dict(pw=pw.state_dict(), po=po.state_dict(), feat_dim=feat_dim, cond=a.cond, ckpt=a.ckpt), f"{a.out}/{label}_probe.pt")   # for the lesion test (Amendment 2)
json.dump(res, open(J, "w"), indent=1); print(f"{label}: walls {res['mean_walls']:.2f} objects {res['mean_objects']:.2f} ({res['wall_h']:.2f} h)")
