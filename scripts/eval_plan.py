"""Teacher-free latent MPC with the trained student in the live arena.
Goal latent is produced by the student itself from the goal panorama (no teacher at test time)."""
import argparse, json, numpy as np, torch, os
from wm2cx.env import FlyArena, scripted_policy
from wm2cx.student import FlyBrain, GRUBaseline
from wm2cx.connectome import load as load_conn

p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--task", default="nav"); p.add_argument("--episodes", type=int, default=50)
p.add_argument("--steps", type=int, default=40); p.add_argument("--horizon", type=int, default=6); p.add_argument("--n_cand", type=int, default=128)
p.add_argument("--device", default="mps"); p.add_argument("--policy", default="mpc"); p.add_argument("--seed", type=int, default=7)
a = p.parse_args()
dev = a.device; rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
ck = torch.load(f"results/runs/{a.run}.pt", weights_only=False); args = ck["args"]
conn = load_conn(args["conn"])
eh, ew = 8, 32
if args["wiring"] == "gru": model = GRUBaseline(eh * ew * 3, args["k"], args["gru_hidden"])
else: model = FlyBrain(conn, eh * ew * 3, args["k"], wiring=args["wiring"], n_brains=args["n_brains"], seed=args["seed"])
model.load_state_dict(ck["state"]); model.to(dev).eval()

def eye(img):
    R = img.shape[0]; band = img[R // 4: 3 * R // 4]
    return torch.from_numpy(band.reshape(eh, band.shape[0] // eh, ew, R // ew, 3).mean((1, 3)).astype(np.float32) / 255.0).flatten().to(dev)

@torch.no_grad()
def encode_goal(env):
    """student latent of goal views at 8 headings (box placed in zone for arrangement)."""
    if a.task == "nav": pos = env.lm[env.goal]; saved = None
    else:
        saved = env.box.copy(); env.box[0] = env.zone[0]; u = (env.zone[0] - saved[0]); u = u / (np.linalg.norm(u) + 1e-6)
        pos = np.clip(env.zone[0] - 1.3 * u, 0.6, env.size - 0.6)
    zs = []
    for hd in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        o = eye(env.render(pos, hd))[None].repeat(1, 1); h = model.init_state(1, dev)
        W = model.W() if hasattr(model, "W") else None
        for _ in range(6): h = model.step(h, o, torch.zeros(1, dtype=torch.long, device=dev), torch.ones(1, device=dev), W)
        zs.append(model.W_out(h)[0])
    if saved is not None: env.box = saved
    return torch.stack(zs)  # [8,k]

@torch.no_grad()
def mpc(h, zg, W):
    B = a.n_cand; Hh = a.horizon
    probs = np.array([0.0, 0.5, 0.25, 0.25, 0.0]); best_seq = None; mean = np.tile(probs, (Hh, 1))
    for it in range(2):
        seqs = np.stack([[rng.choice(5, p=mean[t] / mean[t].sum()) for t in range(Hh)] for _ in range(B)])
        if best_seq is not None: seqs[0] = best_seq
        acts = torch.from_numpy(seqs).to(dev); hh = h.repeat(B, 1); cost = torch.zeros(B, device=dev)
        o0 = torch.zeros(B, eh * ew * 3, device=dev); vis0 = torch.zeros(B, device=dev)
        for t in range(Hh):
            hh = model.step(hh, o0, acts[:, t], vis0, W)
            z = model.W_out(hh); d = ((z[:, None] - zg[None]) ** 2).mean(-1).min(1).values
            cost += d * (1.0 if t < Hh - 1 else 3.0)
        top = cost.argsort()[: B // 8].cpu().numpy(); best_seq = seqs[top[0]]
        mean = np.stack([np.bincount(seqs[top, t], minlength=5) + 0.5 for t in range(Hh)])
    return int(best_seq[0])

succ, dist0, dist1 = [], [], []
for ep in range(a.episodes):
    env = FlyArena(task=a.task, res=256, n_boxes=1, seed=10_000 + ep); img = env.reset(); dist0.append(env.goal_distance())
    with torch.no_grad():
        zg = encode_goal(env); W = model.W() if hasattr(model, "W") else None
        h = model.init_state(1, dev); s = 0
        prev = torch.zeros(1, dtype=torch.long, device=dev)
        for t in range(a.steps):
            h = model.step(h, eye(img)[None], prev, torch.ones(1, device=dev), W)
            if a.policy == "mpc": ac = mpc(h, zg, W)
            elif a.policy == "random": ac = int(rng.integers(1, 4))
            else: ac = scripted_policy(env, rng, 0.0)
            img, s = env.step(ac); prev = torch.tensor([ac], device=dev)
            if s: break
    succ.append(s); dist1.append(env.goal_distance())
    if ep % 10 == 0: print(f"ep {ep} success so far {np.mean(succ):.2f}", flush=True)
res = dict(run=a.run, task=a.task, policy=a.policy, success=float(np.mean(succ)), dist_start=float(np.mean(dist0)), dist_end=float(np.mean(dist1)), episodes=a.episodes, steps=a.steps)
print(json.dumps(res)); os.makedirs("results/plan", exist_ok=True)
json.dump(res, open(f"results/plan/{a.run}_{a.task}_{a.policy}.json", "w"))
