"""E10 Goal-vector navigation from a linear readout of the student's state, with blackouts in the loop.

Teacher-free latent MPC fails for every student (E2) because 6-step latent rollouts are too inaccurate to plan with. This
evaluation asks a different question of the same state: does it carry an egocentric *goal vector* that a controller can
follow, and does it keep that vector when vision is cut?
   readout   h_t -> (gx', gy')   goal position in the agent's current egocentric frame (x forward, y left), ridge on the
             training episodes (goal in the episode files) plus `--fit_rollouts` closed-loop episodes in fresh arenas
             (oracle controller, 20 % random actions, same blackout protocol); nothing about the test arenas is seen.
   control   turn toward the decoded vector while |angle| > --tol_deg (at most --max_turns in a row), else step forward;
             the episode ends when the environment reports success (1.2 units from the landmark; 12 m on the NYC twin).
   blackout  vision cut for --blackout steps from step --blackout_at (fixed trigger, as eval_plan.py); the readout keeps
             being decoded from the state, so the agent navigates in the dark from path integration relative to the goal.
Baselines with the same controller: --readout oracle (true goal vector, with and without vision: the controller ceiling),
--readout random (random actions). Metrics as eval_plan.py: success, SPL, final goal distance.

  python scripts/eval_goalnav.py --run connectome_K1_s0 --device cuda                    # arena, no blackout
  python scripts/eval_goalnav.py --run connectome_K1_s0 --blackout 5 --device cuda
  python scripts/eval_goalnav.py --run connectome_K1_s0_nyc --env nyc --blackout 5 --device cuda
"""
import argparse, json, os, time, numpy as np, torch
from wm2cx.ckpt import load_run
from wm2cx.data import load_split
from wm2cx.env import FlyArena, scripted_policy

p = argparse.ArgumentParser()
p.add_argument("--run", default="connectome_K1_s0"); p.add_argument("--device", default="cuda")
p.add_argument("--env", default="arena", choices=["arena", "nyc"]); p.add_argument("--episodes", type=int, default=50); p.add_argument("--seed", type=int, default=13)
p.add_argument("--steps", type=int, default=0, help="episode length (0 = 24 arena / 40 NYC)")
p.add_argument("--readout", default="state", choices=["state", "mlp", "policy", "oracle", "random"],
               help="state: ridge goal vector; mlp: MLP goal vector; policy: behaviour-cloned action head (state -> action) trained on the agent's own rollouts (#18 lever 1)")
p.add_argument("--blackout", type=int, default=0); p.add_argument("--blackout_at", type=int, default=6)
p.add_argument("--lam", type=float, default=100.0); p.add_argument("--fit_rollouts", type=int, default=200)
p.add_argument("--tol_deg", type=float, default=45.0); p.add_argument("--max_turns", type=int, default=8)
p.add_argument("--controller", default="greedy", choices=["greedy", "bumper"],
               help="greedy: turn toward the decoded goal vector, else forward. bumper: same, but a blocked forward step (the env's collision "
                    "counter, a proprioceptive signal that stays available in the blackout) triggers a fixed turn away and a short forward commitment, "
                    "so the agent slides along building fronts instead of pushing into them (#9/#10)")
p.add_argument("--avoid_turns", type=int, default=2, help="bumper: turns (pi/8 each) after a collision"); p.add_argument("--commit", type=int, default=3, help="bumper: forward steps before re-aiming")
p.add_argument("--appearance", default="clear", help="nyc: clear|overcast|fog|dusk|night (#10)")
p.add_argument("--tag", default=""); a = p.parse_args()
dev = a.device; rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
eh, ew = 8, 32; TURN = np.pi / 8; TOL = np.deg2rad(a.tol_deg); T = a.steps or (24 if a.env == "arena" else 40)
R_SUCC = 1.2 if a.env == "arena" else 12.0


def eye(img):
    R = img.shape[0]; band = img[R // 4: 3 * R // 4]
    return torch.from_numpy(band.reshape(eh, band.shape[0] // eh, ew, R // ew, 3).mean((1, 3)).astype(np.float32) / 255.0).flatten().to(dev)


def ego_goal(pos, hd, goal):
    """goal in the agent's egocentric frame (x forward, y left), scaled to arena units / 10 m."""
    dp = np.asarray(goal) - np.asarray(pos); c, s = np.cos(hd), np.sin(hd)
    return np.array([c * dp[0] + s * dp[1], -s * dp[0] + c * dp[1]], np.float32) / (1.0 if a.env == "arena" else 10.0)


def make_env(ep):
    if a.env == "nyc":
        from wm2cx.nyc import NYCArena
        env = NYCArena(res=256, seed=a.seed * 1000 + ep, appearance=a.appearance); img = env.reset(); pol = lambda: env.policy(rng, 0.0)
    else:
        env = FlyArena(task="nav", res=256, seed=a.seed * 1000 + ep); img = env.reset(); pol = lambda: scripted_policy(env, rng, 0.0)
    return env, img, pol


goal_of = lambda env: env.goal if a.env == "nyc" else env.lm[env.goal]
visible_at = lambda t: not (a.blackout and a.blackout_at <= t < a.blackout_at + a.blackout)


def controller(g, n_turn):
    gx, gy = float(g[0]), float(g[1]); err = np.arctan2(gy, gx)
    if abs(err) > TOL and n_turn < a.max_turns: return 2 if err > 0 else 3
    return 1


class Bumper:
    """Greedy controller plus collision recovery: after a blocked forward step, turn `avoid_turns` times away from the goal side
    that was blocked (alternating sides on repeated hits) and then commit `commit` forward steps before aiming again."""
    def __init__(self): self.turns_left = 0; self.commit_left = 0; self.side = 1; self.last_coll = 0; self.last_act = 0
    def __call__(self, env, g, n_turn, act=None):
        coll = getattr(env, "collisions", 0)
        if self.last_act == 1 and coll > self.last_coll:                       # forward step was blocked
            self.side = -self.side if self.turns_left or self.commit_left else (1 if g is None or float(g[1]) >= 0 else -1)
            self.turns_left, self.commit_left = a.avoid_turns, a.commit
        self.last_coll = coll
        if self.turns_left: self.turns_left -= 1; ac = 2 if self.side > 0 else 3
        elif self.commit_left: self.commit_left -= 1; ac = 1
        else: ac = act if act is not None else controller(g, n_turn)
        self.last_act = ac; return ac


# ---------------- readout fit: training episodes (+ closed-loop rollouts) ----------------
model = readout = W = None
if a.readout in ("state", "mlp", "policy"):
    model, args, conn, lat = load_run(a.run, device=dev); W = model.edge_weights()
    files = [f for f in args["train"] if ("nyc" in f if a.env == "nyc" else "nav" in f)]
    tr, _ = load_split(files, lat, with_z=False); goal = np.concatenate([np.load(f)["goal"] for f in files]); E = len(tr["obs"])
    pos, hd = tr["pos"].numpy(), tr["heading"].numpy()
    Y = np.stack([[ego_goal(pos[e, t], hd[e, t], goal[e]) for t in range(tr["obs"].shape[1])] for e in range(E)])
    Hs = []
    with torch.no_grad():
        for i in range(0, E, 64):
            obs, act = tr["obs"][i:i + 64].to(dev), tr["act"][i:i + 64].to(dev)
            vis = torch.tensor([[float(visible_at(t)) for t in range(obs.shape[1])]] * len(obs), device=dev)
            _, H = model(obs, act, vis); Hs.append(H.reshape(-1, model.N).float())
    X = torch.cat(Hs); Yt = torch.from_numpy(Y.reshape(-1, 2)).to(dev); n_tr = len(X)
    if a.fit_rollouts > 0:
        Xs, Ys, As = [], [], []; r = np.random.default_rng(a.seed + 1)
        for i in range(a.fit_rollouts):
            env, img, pol = make_env(100_000 + i); g = goal_of(env)
            h = model.init_state(1, dev); prev = torch.zeros(1, dtype=torch.long, device=dev); n_turn = 0
            for t in range(T):
                visible = visible_at(t); o = eye(img) if visible else torch.zeros(eh * ew * 3, device=dev)
                with torch.no_grad(): h = model.step(h, o[None], prev, torch.full((1,), float(visible), device=dev), W)
                true = ego_goal(env.pos, env.heading, g); Xs.append(h[0]); Ys.append(true); expert = controller(true, n_turn); As.append(expert)
                ac = int(r.integers(0, 4)) if r.random() < 0.2 else expert; n_turn = n_turn + 1 if ac in (2, 3) else 0
                step_out = env.step(ac); img = step_out[0] if step_out[0] is not None else img; prev = torch.tensor([ac], device=dev)
                if step_out[1]: break
        X, Yt = torch.cat([X, torch.stack(Xs).float()]), torch.cat([Yt, torch.from_numpy(np.stack(Ys)).to(dev)])
        Xp, Ap = torch.stack(Xs).float(), torch.tensor(As, device=dev)
    mu_x, mu_y = X.mean(0), Yt.mean(0)
    r2 = lambda pred, sl: [round(float(v), 3) for v in 1 - ((pred[sl] - Yt[sl]) ** 2).sum(0) / ((Yt[sl] - Yt[sl].mean(0)) ** 2).sum(0)]
    if a.readout == "policy":                                              # action head on the closed-loop rollouts only
        assert a.fit_rollouts > 0, "--readout policy needs --fit_rollouts"
        mu_x, sd_x = Xp.mean(0), Xp.std(0) + 1e-6; n = len(Xp); cut = int(0.9 * n)
        net = torch.nn.Sequential(torch.nn.Linear(model.N, 256), torch.nn.Tanh(), torch.nn.Linear(256, 4)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
        for ep in range(40):
            perm = torch.randperm(cut, device=dev)
            for i in range(0, cut, 1024):
                j = perm[i:i + 1024]; loss = torch.nn.functional.cross_entropy(net((Xp[j] - mu_x) / sd_x), Ap[j]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): acc = float((net((Xp[cut:] - mu_x) / sd_x).argmax(-1) == Ap[cut:]).float().mean())
        print(f"policy head: held-out action accuracy {acc:.3f} on {n - cut} states", flush=True)
        policy = lambda h: int(net((h - mu_x) / sd_x).argmax(-1)); pred = None
    elif a.readout == "state":
        Xc = X - mu_x; Wr = torch.linalg.solve(Xc.T @ Xc + a.lam * torch.eye(model.N, device=dev), Xc.T @ (Yt - mu_y))
        readout = lambda h: (h - mu_x) @ Wr + mu_y; pred = readout(X)
    else:
        sd_x = X.std(0) + 1e-6; net = torch.nn.Sequential(torch.nn.Linear(model.N, 256), torch.nn.Tanh(), torch.nn.Linear(256, 2)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5); n = len(X)
        for ep in range(30):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, 1024):
                j = perm[i:i + 1024]; loss = ((net((X[j] - mu_x) / sd_x) - Yt[j]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); readout = lambda h: net((h - mu_x) / sd_x)
        with torch.no_grad(): pred = readout(X)
    if pred is not None: print(f"{a.readout} goal-vector readout: R2 gx/gy = {r2(pred, slice(0, n_tr))} on {E} train episodes"
          + (f", {r2(pred, slice(n_tr, None))} on {a.fit_rollouts} closed-loop rollouts" if a.fit_rollouts else ""), flush=True)
    del X, Hs

# ---------------- closed-loop test ----------------
t0 = time.time(); succ, spl, dfin, d0s = [], [], [], []
for ep in range(a.episodes):
    env, img, pol = make_env(ep); g = goal_of(env); d0 = env.goal_distance()
    shortest = max((env.shortest_path_length() - R_SUCC) if a.env == "nyc" else (d0 - R_SUCC), 1e-6); travelled = 0.0; prev_pos = env.pos.copy()
    h = model.init_state(1, dev) if model is not None else None; prev = torch.zeros(1, dtype=torch.long, device=dev); n_turn = 0; s = 0.0
    bump = Bumper(); ctrl = (lambda g: bump(env, g, n_turn)) if a.controller == "bumper" else (lambda g: controller(g, n_turn))
    for t in range(T):
        visible = visible_at(t)
        if model is not None:
            o = eye(img) if visible else torch.zeros(eh * ew * 3, device=dev)
            with torch.no_grad(): h = model.step(h, o[None], prev, torch.full((1,), float(visible), device=dev), W)
        if a.readout == "policy":
            with torch.no_grad(): ac = bump(env, None, n_turn, policy(h)) if a.controller == "bumper" else policy(h)
        elif a.readout in ("state", "mlp"):
            with torch.no_grad(): ac = ctrl(readout(h)[0].cpu().numpy())
        elif a.readout == "oracle": ac = ctrl(ego_goal(env.pos, env.heading, g))
        else: ac = int(rng.integers(1, 4))
        n_turn = n_turn + 1 if ac in (2, 3) else 0
        img_, s = env.step(ac); img = img_ if img_ is not None else img; prev = torch.tensor([ac], device=dev)
        travelled += float(np.linalg.norm(env.pos - prev_pos)); prev_pos = env.pos.copy()
        if s: break
    succ.append(float(s)); spl.append(float(s) * shortest / max(shortest, travelled, 1e-6)); dfin.append(env.goal_distance()); d0s.append(d0)
    if ep % 10 == 0: print(f"ep {ep}: goal dist {d0:.2f} -> {dfin[-1]:.2f}  success so far {np.mean(succ):.2f}  SPL {np.mean(spl):.2f}  {time.time()-t0:.0f}s", flush=True)
res = dict(run=a.run, env=a.env, readout=a.readout, controller=a.controller, appearance=a.appearance, blackout=a.blackout, blackout_at=a.blackout_at, steps=T, episodes=a.episodes, tol_deg=a.tol_deg,
           max_turns=a.max_turns, fit_rollouts=a.fit_rollouts, success=float(np.mean(succ)), spl=float(np.mean(spl)),
           dist_start=float(np.mean(d0s)), dist_final=float(np.mean(dfin)))
print(json.dumps(res))
os.makedirs("results/goalnav", exist_ok=True)
sfx = ("" if a.controller == "greedy" else f"_{a.controller}") + ("" if a.appearance == "clear" else f"_{a.appearance}")
json.dump(res, open(f"results/goalnav/{a.run}_{a.env}_{a.readout}_b{a.blackout}{sfx}{a.tag}.json", "w"), indent=1)
