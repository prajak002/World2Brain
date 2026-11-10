"""Memory Maze in-silico lesions of a trained whole-brain student (PREREG Amendment 2).

For one checkpoint (C, S or B) and its already-trained probe pair (eval_mm_probe.py saves <label>_probe.pt), roll the frozen
student over the 1,000 eval trajectories once per lesion and score the frozen probes on the lesioned state. Nothing is
trained here. A lesion silences a set of neurons: their state is set to 0 after every step from t = 0, so they send
nothing through the recurrent edges and the probe reads 0 from them. Lesions (all run in one batch, one row per lesion):
  intact                   no lesion; must reproduce the probe JSON
  CX, MB, OUT              anatomical groups (wm2cx.mm_models.lesion_groups)
  CXdeaf                   CX neurons kept, but their direct action input removed (separates "CX lost its input" from
                           "the CX circuit is gone")
  randCX_i, randMB_i       --n_rand random lesions from the neurons with no direct input, matched to CX / MB in neuron
                           count and out-degree deciles (seeded by model seed and i)
Writes results/memmaze/lesion/<label>.json: per lesion, per eval trajectory, walls_acc and objects_mse (steps 500-1000).
"""
import argparse, json, os, time, numpy as np, torch
from wm2cx.memmaze import load_shards, walls_acc, objects_mse, SCORE_FROM, SCORE_TO
from wm2cx.mm_models import build, rollout, act_prev_of, probe_mlp, lesion_groups, matched_random, COND_WIRING
from wm2cx.connectome import load as load_conn

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True); p.add_argument("--probe", default="", help="default results/memmaze/probe/<label>_probe.pt")
p.add_argument("--label", default=""); p.add_argument("--data", default="data/mm9"); p.add_argument("--conn", default="data/connectome_full.npz")
p.add_argument("--n_rand", type=int, default=10); p.add_argument("--bs", type=int, default=8, help="trajectories per batch (x lesions rows)")
p.add_argument("--window", type=int, default=48); p.add_argument("--n_eval", type=int, default=0, help="smoke tests only")
p.add_argument("--device", default="cuda"); p.add_argument("--out", default="results/memmaze/lesion")
a = p.parse_args(); dev = a.device; torch.backends.cuda.matmul.allow_tf32 = True
label = a.label or os.path.basename(a.ckpt).rsplit(".", 1)[0]
os.makedirs(a.out, exist_ok=True); J = f"{a.out}/{label}.json"
if os.path.exists(J): print("done already:", J); raise SystemExit
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); cond = ck["cond"]; assert cond in COND_WIRING, cond
conn = load_conn(a.conn); seed = ck["args"]["seed"]
model, _ = build(cond, conn, seed, ck["args"]["d_model"], ck["args"]["window"])
model.load_state_dict(ck["state"]); model.to(dev).eval().requires_grad_(False); N = model.N
pk = torch.load(a.probe or f"results/memmaze/probe/{label}_probe.pt", map_location="cpu", weights_only=False)
pw, po = probe_mlp(N + 4, 81).to(dev), probe_mlp(N + 4, 6).to(dev); pw.load_state_dict(pk["pw"]); po.load_state_dict(pk["po"]); pw.eval(); po.eval()

G = lesion_groups(conn); names = ["intact", "CX", "MB", "OUT", "CXdeaf"]; sets = {"CX": G["CX"], "MB": G["MB"], "OUT": G["OUT"]}
for g in ("CX", "MB"):
    for i in range(a.n_rand):
        sets[f"rand{g}_{i}"] = matched_random(conn, G[g], G["POOL"], np.random.default_rng([seed, i, int(g == "MB")])); names.append(f"rand{g}_{i}")
L = len(names); keep = torch.ones(L, N, device=dev); act_keep = torch.ones(L, N, device=dev)
for j, n in enumerate(names):
    if n in sets: keep[j, torch.from_numpy(sets[n]).to(dev)] = 0
act_keep[names.index("CXdeaf"), torch.from_numpy(G["CX"]).to(dev)] = 0
base_act = model.act_mask.clone()
deg = np.bincount(conn["pre"].astype(np.int64), minlength=N)
sizes = {n: dict(neurons=int(len(s)), out_edges=int(deg[s].sum())) for n, s in sets.items()}
print(f"{label}: cond {cond} lesions {L} sizes", {k: sizes[k] for k in ("CX", "MB", "OUT")}, flush=True)

META = ("obs", "act", "agent_pos", "agent_dir", "maze_layout", "targets_vec")
ev = load_shards(a.data, json.load(open(f"{a.data}/splits.json"))["eval"], lat=False, meta=META)
if a.n_eval: ev = {k: v[:a.n_eval] for k, v in ev.items()}
E, T = ev["act"].shape; to = lambda x, idx: torch.from_numpy(np.ascontiguousarray(x[idx])).to(dev)
rep = lambda x: x.repeat_interleave(L, 0)                                      # row b*L + j = trajectory b under lesion j
SW, SO = np.zeros((L, E)), np.zeros((L, E)); t0 = time.time()
with torch.no_grad():
    for i in range(0, E, a.bs):
        idx = np.arange(i, min(i + a.bs, E)); B = len(idx)
        obs = rep(to(ev["obs"], idx).float() / 255.0); act = rep(to(ev["act"], idx).long())
        aux = rep(torch.cat([to(ev["agent_pos"], idx), to(ev["agent_dir"], idx)], -1)); lay = rep(to(ev["maze_layout"], idx).reshape(B, -1).float()).cpu().numpy()
        tv = rep(to(ev["targets_vec"], idx)); K = keep.repeat(B, 1); model.act_mask = base_act[None] * act_keep.repeat(B, 1)
        h = None; last = None; aw, ao = [], []
        for s0 in range(0, T, a.window):
            s = slice(s0, min(s0 + a.window, T))
            _, S, h = rollout(model, obs[:, s], act_prev_of(act[:, s], last), torch.ones(B * L, s.stop - s0, device=dev), h, keep=K); last = act[:, s.stop - 1]
            x = torch.cat([S, aux[:, s]], -1)
            aw.append(walls_acc(pw(x).cpu().numpy(), np.broadcast_to(lay[:, None], (B * L, x.shape[1], lay.shape[-1]))))
            ao.append(objects_mse(po(x).view(*x.shape[:2], 3, 2).cpu().numpy(), tv[:, s].cpu().numpy()))
        aw, ao = np.concatenate(aw, 1)[:, SCORE_FROM:SCORE_TO + 1].mean(1), np.concatenate(ao, 1)[:, SCORE_FROM:SCORE_TO + 1].mean(1)
        SW[:, idx] = aw.reshape(B, L).T; SO[:, idx] = ao.reshape(B, L).T
        if (i // a.bs) % 10 == 0: print(f"traj {i + B}/{E} intact walls {SW[0, :i + B].mean():.2f} objects {SO[0, :i + B].mean():.2f} {time.time() - t0:.0f}s", flush=True)
model.act_mask = base_act
res = dict(cond=cond, label=label, ckpt=a.ckpt, seed=seed, lesions=names, sizes=sizes, n_eval=E, n_rand=a.n_rand,
           walls_acc={n: SW[j].tolist() for j, n in enumerate(names)}, objects_mse={n: SO[j].tolist() for j, n in enumerate(names)},
           mean={n: [float(SW[j].mean()), float(SO[j].mean())] for j, n in enumerate(names)}, wall_h=(time.time() - t0) / 3600)
json.dump(res, open(J, "w")); print(json.dumps(res["mean"], indent=1))
