"""Memory Maze B4: train one student (PREREG section 3) on teacher-latent prediction with truncated BPTT.

Objective = scripts/train.py unchanged (blind-step-weighted latent MSE under random blackouts + 1e-4 activity penalty,
AdamW lr 2e-3 wd 1e-4, grad clip 1, cosine schedule); windows of --window steps carry the state across the whole 1001-step
trajectory (reset only at the trajectory start). The budget is a fixed number of parameter updates (one per window), equal
for every condition (PREREG Amendment 1). Validation = splits.json 'val' shards; eval trajectories are never loaded here.
Writes results/memmaze/runs/<cond>_s<seed><tag>.json and checkpoints <name>_u<update>.pt; resumes from <name>_resume.pt.
"""
import argparse, json, os, time, numpy as np, torch
from wm2cx.memmaze import load_shards
from wm2cx.mm_models import build, rollout, act_prev_of
from wm2cx.data import blackout
from wm2cx.connectome import load as load_conn

p = argparse.ArgumentParser()
p.add_argument("--cond", required=True, choices=["C", "S", "B", "R", "G", "T"]); p.add_argument("--seed", type=int, default=0)
p.add_argument("--data", default="data/mm9"); p.add_argument("--conn", default="data/connectome_full.npz")
p.add_argument("--updates", type=int, required=True, help="fixed parameter-update budget (PREREG Amendment 1)")
p.add_argument("--bs", type=int, default=32); p.add_argument("--window", type=int, default=48); p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--H", type=int, default=8, help="blackout length inside a window (train.py default)")
p.add_argument("--d_model", type=int, default=768); p.add_argument("--eval_every", type=int, default=2500); p.add_argument("--n_val", type=int, default=250)
p.add_argument("--out", default="results/memmaze/runs"); p.add_argument("--tag", default=""); p.add_argument("--device", default="cuda")
p.add_argument("--n_train", type=int, default=0, help="smoke tests only: first N training trajectories")
a = p.parse_args()
name = f"{a.cond}_s{a.seed}{a.tag}"; os.makedirs(a.out, exist_ok=True); J = f"{a.out}/{name}.json"
if os.path.exists(J): print("done already:", J); raise SystemExit
torch.manual_seed(a.seed); dev = a.device; torch.backends.cuda.matmul.allow_tf32 = True
sp = json.load(open(f"{a.data}/splits.json"))
tr = load_shards(a.data, sp["train"]); va = load_shards(a.data, sp["val"])
if a.n_train: tr = {k: v[:a.n_train] for k, v in tr.items()}
E, T = tr["act"].shape; print(f"{name}: train traj {E} val traj {len(va['act'])} steps {T}", flush=True)
model, info = build(a.cond, load_conn(a.conn), a.seed, a.d_model, a.window); model.to(dev)
pb = model.param_breakdown(); print("params", pb, info, flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.updates)
is_T = a.cond == "T"
to = lambda x, i, s: torch.from_numpy(np.ascontiguousarray(x[i][:, s])).to(dev, non_blocking=True)


def batch(d, idx, s):
    return (to(d["obs"], idx, s).float() / 255.0, to(d["act"], idx, s).long(), to(d["z"], idx, s).float())


@torch.no_grad()
def val_loss():
    """Latent MSE on --n_val validation trajectories: all steps visible (mse_vis) and with a fixed 8-step blackout per window."""
    model.eval(); rng = np.random.default_rng(12345); tot = dict(mse_vis=0.0, mse_blind=0.0); n = dict(mse_vis=0, mse_blind=0)
    for i in range(0, a.n_val, a.bs):
        idx = np.arange(i, min(i + a.bs, a.n_val)); B = len(idx)
        for key in ("mse_vis", "mse_blind"):
            h = None; last = None
            for t0 in range(0, T, a.window):
                s = slice(t0, min(t0 + a.window, T)); obs, act, z = batch(va, idx, s); W = obs.shape[1]
                vis = torch.ones(B, W, device=dev) if key == "mse_vis" else blackout(B, W, min(a.H, W - 5), rng, dev, p_blind=1.0)
                zh, _, h = rollout(model, obs, act_prev_of(act, last), vis, h); last = act[:, -1]
                m = ((zh - z) ** 2).mean(-1); m = m if key == "mse_vis" else m[vis < 0.5]
                tot[key] += float(m.sum()); n[key] += m.numel()
    model.train(); return {k: tot[k] / max(n[k], 1) for k in tot}


R = f"{a.out}/{name}_resume.pt"; u = 0; log = []; rng = np.random.default_rng(a.seed); perm = []
if os.path.exists(R):
    st = torch.load(R, map_location=dev, weights_only=False); model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    u, log, rng, perm = st["u"], st["log"], st["rng"], st["perm"]; print(f"resumed at update {u}", flush=True)
t_start = time.time(); u0 = u; run_loss = []
while u < a.updates:
    if len(perm) < a.bs: perm = list(rng.permutation(E)) + list(perm)
    idx = np.sort(np.array([perm.pop() for _ in range(a.bs)])); h = None; last = None
    for t0 in range(0, T, a.window):                                       # one trajectory batch = ceil(1001/48) = 21 updates
        if u >= a.updates: break
        s = slice(t0, min(t0 + a.window, T)); obs, act, z = batch(tr, idx, s); W = obs.shape[1]
        vis = blackout(a.bs, W, min(a.H, W - 5), rng, dev)
        zh, Hs, h = rollout(model, obs, act_prev_of(act, last), vis, h); last = act[:, -1]
        w = 1.0 + 2.0 * (1 - vis)
        loss = (((zh - z) ** 2).mean(-1) * w).mean() + 1e-4 * (Hs ** 2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if h is not None: h = h.detach()                                    # truncated BPTT: state carried, gradient cut
        u += 1; run_loss.append(loss.item())
        if u % 100 == 0:
            el = time.time() - t_start; print(f"u {u} loss {np.mean(run_loss[-100:]):.4f} lr {sched.get_last_lr()[0]:.2e} {el:.0f}s "
                                              f"eta {(a.updates - u) * el / max(u - u0, 1) / 3600:.1f}h", flush=True)
        if u % a.eval_every == 0 or u == a.updates:
            v = val_loss(); log.append(dict(u=u, loss=float(np.mean(run_loss[-a.eval_every:])), **v, t=time.time() - t_start)); print("val", log[-1], flush=True)
            torch.save(dict(state=model.state_dict(), cond=a.cond, info=info, args=vars(a), u=u), f"{a.out}/{name}_u{u}.pt")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), u=u, log=log, rng=rng, perm=perm), R + ".tmp"); os.replace(R + ".tmp", R)

# checkpoint rule (PREREG Amendment 1): the last checkpoint, unless the final validation blind MSE exceeds 1.5 x its minimum
best = min(log, key=lambda r: r["mse_blind"]); chosen = log[-1] if log[-1]["mse_blind"] <= 1.5 * best["mse_blind"] else best
json.dump(dict(cond=a.cond, seed=a.seed, args=vars(a), info=info, params=model.n_params(), param_breakdown=pb, neurons=model.N,
               n_train_traj=int(E), updates=u, log=log, chosen_ckpt=f"{name}_u{chosen['u']}.pt", chosen_u=chosen["u"],
               wall_h=(time.time() - t_start) / 3600), open(J, "w"), indent=1)
print("saved", J, "chosen", chosen["u"])
