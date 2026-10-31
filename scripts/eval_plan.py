"""Teacher-free latent MPC with the trained student in the live arena, plus the oracle upper bounds asked for in issue #9.
--dynamics student       : CEM over the student's imagined latents (the paper's E2 setting; no teacher, no simulator at test time)
--dynamics oracle_state  : CEM over *true* environment rollouts scored by ground-truth goal distance (validates the search)
--dynamics oracle_render : CEM over true rollouts, scored by the student's latent of the *rendered* candidate view vs the goal latent
                           (validates goal-latent matching as a cost, with perfect dynamics)
Metrics: success, SPL (success * shortest / actual path length), goal distance, steps to goal."""
import argparse, copy, json, numpy as np, torch, os
from wm2cx.env import FlyArena, scripted_policy
from wm2cx.ckpt import load_run

p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--task", default="nav"); p.add_argument("--episodes", type=int, default=50)
p.add_argument("--steps", type=int, default=40); p.add_argument("--horizon", type=int, default=6); p.add_argument("--n_cand", type=int, default=128)
p.add_argument("--iters", type=int, default=2); p.add_argument("--dynamics", default="student", choices=["student", "oracle_state", "oracle_render"])
p.add_argument("--device", default="mps"); p.add_argument("--policy", default="mpc"); p.add_argument("--seed", type=int, default=7)
a = p.parse_args()
dev = a.device; rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
model, args, conn, lat = load_run(a.run, device=dev)
eh, ew = 8, 32

def eye(img):
    R = img.shape[0]; band = img[R // 4: 3 * R // 4]
    return torch.from_numpy(band.reshape(eh, band.shape[0] // eh, ew, R // ew, 3).mean((1, 3)).astype(np.float32) / 255.0).flatten().to(dev)

@torch.no_grad()
def encode_view(img, W, steps=6):
    """student latent of a static view (observe it for a few steps from a blank state)."""
    o = eye(img)[None]; h = model.init_state(1, dev)
    for _ in range(steps): h = model.step(h, o, torch.zeros(1, dtype=torch.long, device=dev), torch.ones(1, device=dev), W)
    return model.W_out(h)[0]

@torch.no_grad()
def encode_goal(env, W):
    """student latent of goal views at 8 headings (box placed in zone for arrangement)."""
    if a.task == "nav": pos = env.lm[env.goal]; saved = None
    else:
        saved = env.box.copy(); env.box[0] = env.zone[0]; u = (env.zone[0] - saved[0]); u = u / (np.linalg.norm(u) + 1e-6)
        pos = np.clip(env.zone[0] - 1.3 * u, 0.6, env.size - 0.6)
    zs = [encode_view(env.render(pos, hd), W) for hd in np.linspace(-np.pi, np.pi, 8, endpoint=False)]
    if saved is not None: env.box = saved
    return torch.stack(zs)  # [8,k]

def sample_seqs(mean, B, Hh, best_seq):
    seqs = np.stack([[rng.choice(5, p=mean[t] / mean[t].sum()) for t in range(Hh)] for _ in range(B)])
    if best_seq is not None: seqs[0] = best_seq
    return seqs

@torch.no_grad()
def cost_student(h, zg, W, seqs):
    B, Hh = seqs.shape; acts = torch.from_numpy(seqs).to(dev); hh = h.repeat(B, 1); cost = torch.zeros(B, device=dev)
    o0 = torch.zeros(B, eh * ew * 3, device=dev); vis0 = torch.zeros(B, device=dev)
    for t in range(Hh):
        hh = model.step(hh, o0, acts[:, t], vis0, W)
        z = model.W_out(hh); d = ((z[:, None] - zg[None]) ** 2).mean(-1).min(1).values
        cost += d * (1.0 if t < Hh - 1 else 3.0)
    return cost.cpu().numpy()

def cost_oracle(env, zg, W, seqs):
    cost = np.zeros(len(seqs))
    for i, seq in enumerate(seqs):
        e = copy.deepcopy(env)
        for t, ac in enumerate(seq):
            if a.dynamics == "oracle_state": e.move(int(ac)); c = e.goal_distance()  # no render needed
            else: img, s = e.step(int(ac)); c = float(((encode_view(img, W)[None] - zg) ** 2).mean(-1).min())
            cost[i] += c * (1.0 if t < len(seq) - 1 else 3.0)
    return cost

def mpc(env, h, zg, W):
    B = a.n_cand; Hh = a.horizon; probs = np.array([0.0, 0.5, 0.25, 0.25, 0.0]); best_seq = None; mean = np.tile(probs, (Hh, 1))
    for it in range(a.iters):
        seqs = sample_seqs(mean, B, Hh, best_seq)
        cost = cost_student(h, zg, W, seqs) if a.dynamics == "student" else cost_oracle(env, zg, W, seqs)
        top = np.argsort(cost)[: max(B // 8, 1)]; best_seq = seqs[top[0]]
        mean = np.stack([np.bincount(seqs[top, t], minlength=5) + 0.5 for t in range(Hh)])
    return int(best_seq[0])

succ, dist0, dist1, spl, nsteps = [], [], [], [], []
for ep in range(a.episodes):
    env = FlyArena(task=a.task, res=256, n_boxes=1, seed=10_000 + ep); img = env.reset(); d0 = env.goal_distance(); dist0.append(d0)
    path = 0.0; p_prev = env.pos.copy()
    with torch.no_grad():
        W = model.weights(); zg = encode_goal(env, W)
        h = model.init_state(1, dev); s = 0
        prev = torch.zeros(1, dtype=torch.long, device=dev); t_done = a.steps
        for t in range(a.steps):
            h = model.step(h, eye(img)[None], prev, torch.ones(1, device=dev), W)
            if a.policy == "mpc": ac = mpc(env, h, zg, W)
            elif a.policy == "random": ac = int(rng.integers(1, 4))
            else: ac = scripted_policy(env, rng, 0.0)
            img, s = env.step(ac); prev = torch.tensor([ac], device=dev)
            path += float(np.linalg.norm(env.pos - p_prev)); p_prev = env.pos.copy()
            if s: t_done = t + 1; break
    succ.append(s); dist1.append(env.goal_distance()); nsteps.append(t_done)
    spl.append(s * max(d0 - 1.2, 0.0) / max(path, max(d0 - 1.2, 0.0), 1e-6) if a.task == "nav" else float("nan"))  # SPL defined for nav only (straight line to the success radius)
    if ep % 10 == 0: print(f"ep {ep} success so far {np.mean(succ):.2f} spl {np.mean(spl):.2f}", flush=True)
res = dict(run=a.run, task=a.task, policy=a.policy, dynamics=a.dynamics, success=float(np.mean(succ)), spl=float(np.mean(spl)), dist_start=float(np.mean(dist0)),
           dist_end=float(np.mean(dist1)), steps_to_goal=float(np.mean(nsteps)), episodes=a.episodes, steps=a.steps, horizon=a.horizon, n_cand=a.n_cand, iters=a.iters)
print(json.dumps(res)); os.makedirs("results/plan", exist_ok=True)
tag = "" if a.dynamics == "student" else f"_{a.dynamics}"
json.dump(res, open(f"results/plan/{a.run}_{a.task}_{a.policy}{tag}.json", "w"))
