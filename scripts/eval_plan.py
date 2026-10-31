"""Closed-loop planning in the live arena with CEM-MPC, and the oracle ladder that validates the planner (#9).

--dynamics decides what the planner rolls out:
  oracle          ground-truth state: env copies stepped without rendering, cost = goal distance. Upper bound for the
                  CEM search itself (if this fails, the planner, not the student, is the problem).
  render_student  ground-truth state + rendered frames encoded by the *student with eyes open*; cost in the student's
                  latent space against the goal latent. Tests the latent-space goal cost with perfect dynamics.
  render_teacher  ground-truth state + rendered frames encoded by the V-JEPA teacher (PCA-projected like training
                  targets); cost in teacher latent space. Tests whether the teacher's latent is a usable goal metric.
  student         (default, teacher-free) the student imagines the blind rollout from its own state. The real E2 number.
--goal_mode: `headings` = nearest of 8 rendered headings at the goal position (original); `pose` = one frame at the goal
pose, facing the way the agent arrives (direction start -> goal).

  PYTHONPATH=. python scripts/eval_plan.py --run connectome_K1_s0 --dynamics oracle --episodes 30 --horizon 10 --iters 4
"""
import argparse, copy, json, numpy as np, torch, os
from wm2cx.env import FlyArena, scripted_policy

p = argparse.ArgumentParser()
p.add_argument("--run", default="connectome_K1_s0"); p.add_argument("--task", default="nav"); p.add_argument("--episodes", type=int, default=50)
p.add_argument("--steps", type=int, default=40); p.add_argument("--horizon", type=int, default=6); p.add_argument("--n_cand", type=int, default=128)
p.add_argument("--iters", type=int, default=2); p.add_argument("--elite_frac", type=float, default=0.125)
p.add_argument("--device", default="mps"); p.add_argument("--policy", default="mpc", choices=["mpc", "random", "scripted"]); p.add_argument("--seed", type=int, default=7)
p.add_argument("--dynamics", default="student", choices=["student", "oracle", "render_student", "render_teacher"])
p.add_argument("--goal_mode", default="headings", choices=["headings", "pose"]); p.add_argument("--turn_noise", type=float, default=0.0)
p.add_argument("--tag", default="")
a = p.parse_args()
dev = a.device; rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
eh, ew = 8, 32
need_model = a.policy == "mpc" and a.dynamics in ("student", "render_student")
model = lat = teacher = None
if need_model or a.dynamics == "render_teacher":
    from wm2cx.student import FlyBrain, GRUBaseline
    from wm2cx.connectome import load as load_conn
    from wm2cx.data import Latents
    ck = torch.load(f"results/runs/{a.run}.pt", weights_only=False); args = ck["args"]
    lat = Latents.__new__(Latents); lat.mu, lat.P, lat.scale = ck["lat"]["mu"], ck["lat"]["P"], ck["lat"]["scale"]
    if need_model:
        conn = load_conn(args["conn"])
        if args["wiring"] == "gru": model = GRUBaseline(eh * ew * 3, args["k"], args["gru_hidden"])
        else: model = FlyBrain(conn, eh * ew * 3, args["k"], wiring=args["wiring"], n_brains=args["n_brains"], seed=args["seed"])
        model.load_state_dict(ck["state"]); model.to(dev).eval()
if a.dynamics == "render_teacher":
    from wm2cx.teacher import VJEPA21Encoder
    teacher = VJEPA21Encoder("vitb", crop=256, frames=16).to(dev)


def eye(img):
    R = img.shape[0]; band = img[R // 4: 3 * R // 4]
    return torch.from_numpy(band.reshape(eh, band.shape[0] // eh, ew, R // ew, 3).mean((1, 3)).astype(np.float32) / 255.0).flatten().to(dev)


@torch.no_grad()
def encode_frames(imgs, W=None):
    """Latent of a batch of panoramas with the active encoder (student with eyes open, 6 settle steps; or teacher)."""
    if a.dynamics == "render_teacher":
        clips = torch.from_numpy(np.stack(imgs))[:, None].repeat(1, 16, 1, 1, 1).to(dev)           # static clip
        z = teacher.per_step_latents(clips)[:, 0].cpu().numpy(); return torch.from_numpy(lat(z)).to(dev)
    o = torch.stack([eye(i) for i in imgs]); h = model.init_state(len(imgs), dev); W = model.edge_weights() if W is None else W
    for _ in range(6): h = model.step(h, o, torch.zeros(len(imgs), dtype=torch.long, device=dev), torch.ones(len(imgs), device=dev), W)
    return model.W_out(h)


def goal_pose(env):
    """Position (and arrival heading) the agent should reach; box placed in its zone for arrangement."""
    if a.task == "nav": pos = env.lm[env.goal]; saved = None
    else:
        saved = env.box.copy(); env.box[0] = env.zone[0]; u = (env.zone[0] - saved[0]); u = u / (np.linalg.norm(u) + 1e-6)
        pos = np.clip(env.zone[0] - 1.3 * u, 0.6, env.size - 0.6)
    v = pos - env.pos; hd = float(np.arctan2(v[1], v[0]))
    return pos, hd, saved


@torch.no_grad()
def encode_goal(env, W=None):
    pos, hd, saved = goal_pose(env)
    hds = [hd] if a.goal_mode == "pose" else list(np.linspace(-np.pi, np.pi, 8, endpoint=False))
    zg = encode_frames([env.render(pos, h_) for h_ in hds], W)
    if saved is not None: env.box = saved
    return zg  # [G,k]


def rollout_oracle(env, seqs):
    """Ground-truth rollouts of candidate action sequences from the current state: returns cost [B] and, if asked, frames."""
    s0 = env.state(); B, Hh = seqs.shape; cost = np.zeros(B); frames = [[None] * Hh for _ in range(B)]
    want_frames = a.dynamics != "oracle"
    for b in range(B):
        env.set_state(s0)
        for t in range(Hh):
            img, _ = env.step(int(seqs[b, t]), render=want_frames); frames[b][t] = img
            if a.dynamics == "oracle": cost[b] += env.goal_distance() * (1.0 if t < Hh - 1 else 3.0)
    env.set_state(s0)
    return cost, frames


@torch.no_grad()
def plan(env, h, zg, W):
    B, Hh = a.n_cand, a.horizon; n_elite = max(2, int(B * a.elite_frac))
    probs = np.array([0.0, 0.5, 0.25, 0.25, 0.0]); best_seq = None; mean = np.tile(probs, (Hh, 1))
    for it in range(a.iters):
        seqs = np.stack([[rng.choice(5, p=mean[t] / mean[t].sum()) for t in range(Hh)] for _ in range(B)])
        if best_seq is not None: seqs[0] = best_seq
        if a.dynamics == "student":
            acts = torch.from_numpy(seqs).to(dev); hh = h.repeat(B, 1); cost = torch.zeros(B, device=dev)
            o0 = torch.zeros(B, eh * ew * 3, device=dev); vis0 = torch.zeros(B, device=dev)
            for t in range(Hh):
                hh = model.step(hh, o0, acts[:, t], vis0, W)
                z = model.W_out(hh); d = ((z[:, None] - zg[None]) ** 2).mean(-1).min(1).values
                cost += d * (1.0 if t < Hh - 1 else 3.0)
            cost = cost.cpu().numpy()
        else:
            cost, frames = rollout_oracle(env, seqs)
            if a.dynamics != "oracle":
                cost = np.zeros(B)
                for t in range(Hh):
                    z = encode_frames([frames[b][t] for b in range(B)], W); d = ((z[:, None] - zg[None]) ** 2).mean(-1).min(1).values.cpu().numpy()
                    cost += d * (1.0 if t < Hh - 1 else 3.0)
        top = np.argsort(cost)[:n_elite]; best_seq = seqs[top[0]]
        mean = np.stack([np.bincount(seqs[top, t], minlength=5) + 0.5 for t in range(Hh)]); mean[:, [0, 4]] = 0.0
    return int(best_seq[0])


succ, dist0, dist1 = [], [], []
for ep in range(a.episodes):
    env = FlyArena(task=a.task, res=256, n_boxes=1, seed=10_000 + ep, turn_noise=a.turn_noise); img = env.reset(); dist0.append(env.goal_distance())
    with torch.no_grad():
        W = model.edge_weights() if model is not None else None
        zg = encode_goal(env, W) if (a.policy == "mpc" and a.dynamics != "oracle") else None
        h = model.init_state(1, dev) if model is not None else None; s = 0
        prev = torch.zeros(1, dtype=torch.long, device=dev)
        for t in range(a.steps):
            if model is not None: h = model.step(h, eye(img)[None], prev, torch.ones(1, device=dev), W)
            if a.policy == "mpc": ac = plan(env, h, zg, W)
            elif a.policy == "random": ac = int(rng.integers(1, 4))
            else: ac = scripted_policy(env, rng, 0.0)
            img, s = env.step(ac); prev = torch.tensor([ac], device=dev)
            if s: break
    succ.append(s); dist1.append(env.goal_distance())
    if ep % 10 == 0: print(f"ep {ep} success so far {np.mean(succ):.2f}", flush=True)
res = dict(run=a.run, task=a.task, policy=a.policy, dynamics=a.dynamics if a.policy == "mpc" else "-", goal_mode=a.goal_mode, horizon=a.horizon,
           iters=a.iters, n_cand=a.n_cand, success=float(np.mean(succ)), dist_start=float(np.mean(dist0)), dist_end=float(np.mean(dist1)),
           episodes=a.episodes, steps=a.steps, turn_noise=a.turn_noise)
print(json.dumps(res)); os.makedirs("results/plan", exist_ok=True)
suffix = "" if a.dynamics == "student" or a.policy != "mpc" else f"_{a.dynamics}"
json.dump(res, open(f"results/plan/{a.run}_{a.task}_{a.policy}{suffix}{a.tag}.json", "w"))
