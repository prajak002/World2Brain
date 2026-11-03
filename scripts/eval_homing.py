"""E9 Blind homing from the student's internal state ("the fly in the dark" as a behavioural test).

The agent walks away from its start under a goal-directed outbound policy with vision for `--t_vis` steps, then vision is
cut and it must return to the start within `--t_blind` steps using *only* a linear readout of the student's state:
   readout  h_t -> (cos d_theta, sin d_theta, dx', dy')      (pose relative to the start, start-heading frame)
   control  turn toward the decoded home vector while |heading error| > --tol_deg (45 deg) and for at most --max_turns steps in a
            row (one half turn), else step forward; step backward when home is within tol_deg of straight behind; stay
            when the decoded distance < stop_r.  (A tolerance of half a turn step, as
            a perfect integrator would allow, makes a noisily decoded heading spin the agent in place: the error never settles inside
            11 deg, so it turns on every blind step and never moves. Oracle and dead reckoning are unaffected by the wider tolerance.)
The readout is a ridge regression fit on the student's states over the *training* episodes (visible-then-blind protocol,
same as the probes in eval_hd.py) plus `--fit_rollouts` closed-loop episodes in fresh arenas (scripted outbound, then the
oracle controller with random actions mixed in, in the dark): the scripted training episodes almost never turn several
times in a row, which is exactly what homing does, and a readout fit without such states under-reads turn bursts. Nothing
about the test episodes (their own seeds) is seen. Baselines with the same controller:
   --readout oracle   true pose (controller ceiling)            --readout actions  dead reckoning from the discrete action record
   --readout random   random actions (floor)                   --readout mlp      2-layer MLP readout of the same state (is the information
                                                                                  present but non-linear?)
Success = true distance to start < `--radius` at the end of the blind phase (or when the agent stops). Metrics: success, final
distance, path efficiency (straight-line / path length during the blind phase). Arena and NYC raycast twin (--env nyc).

  python scripts/eval_homing.py --run connectome_K1_s0 --device cuda                       # student state, linear readout
  python scripts/eval_homing.py --run gru_K1_s0 --readout mlp --device cuda
  python scripts/eval_homing.py --run connectome_K1_s0 --readout actions                   # dead-reckoning ceiling (no network)
"""
import argparse, json, os, time, numpy as np, torch
from wm2cx.ckpt import load_run
from wm2cx.data import load_split
from wm2cx.env import FlyArena, scripted_policy

p = argparse.ArgumentParser()
p.add_argument("--run", default="connectome_K1_s0"); p.add_argument("--device", default="cuda")
p.add_argument("--env", default="arena", choices=["arena", "nyc"]); p.add_argument("--episodes", type=int, default=50); p.add_argument("--seed", type=int, default=11)
p.add_argument("--t_vis", type=int, default=12); p.add_argument("--t_blind", type=int, default=24)
p.add_argument("--readout", default="state", choices=["state", "mlp", "oracle", "actions", "random"])
p.add_argument("--lam", type=float, default=100.0); p.add_argument("--radius", type=float, default=0.0, help="success radius (0 = 1.0 arena units / 12 m twin)")
p.add_argument("--stop_frac", type=float, default=0.6, help="stop when the decoded distance is below stop_frac * radius")
p.add_argument("--turn_noise", type=float, default=0.0, help="arena: relative sd of the turn angle (makes dead reckoning imperfect)")
p.add_argument("--tol_deg", type=float, default=45.0, help="step forward when the decoded heading error is below this, else turn toward home")
p.add_argument("--max_turns", type=int, default=8, help="never turn more than this many blind steps in a row (8 x 22.5 deg = a half turn), then step forward")
p.add_argument("--fit_rollouts", type=int, default=300, help="closed-loop episodes (fresh arenas, oracle controller in the dark) added to the readout fit")
p.add_argument("--backward", type=int, default=1, help="step backward when home is within tol_deg of straight behind (fewer turns to integrate in the dark)")
p.add_argument("--verbose", type=int, default=0, help="print the decoded pose and action at every blind step of the first episodes")
p.add_argument("--tag", default=""); a = p.parse_args()
dev = a.device; rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
eh, ew = 8, 32; R_SUCC = a.radius or (1.0 if a.env == "arena" else 12.0); R_STOP = a.stop_frac * R_SUCC
TURN = np.pi / 8; TOL = np.deg2rad(a.tol_deg)


def eye(img):
    R = img.shape[0]; band = img[R // 4: 3 * R // 4]
    return torch.from_numpy(band.reshape(eh, band.shape[0] // eh, ew, R // ew, 3).mean((1, 3)).astype(np.float32) / 255.0).flatten().to(dev)


def rel_pose(pos, hd, pos0, hd0):
    """(cos d_theta, sin d_theta, dx', dy') of a pose relative to the start pose, in the start-heading frame."""
    dth = hd - hd0; dp = pos - pos0; c, s = np.cos(hd0), np.sin(hd0)
    return np.array([np.cos(dth), np.sin(dth), c * dp[0] + s * dp[1], -s * dp[0] + c * dp[1]], np.float32)


def make_env(ep):
    if a.env == "nyc":
        from wm2cx.nyc import NYCArena
        env = NYCArena(res=256, seed=a.seed * 1000 + ep); img = env.reset(); pol = lambda: env.policy(rng, 0.2)
    else:
        env = FlyArena(task="nav", res=256, seed=a.seed * 1000 + ep, turn_noise=a.turn_noise); img = env.reset(); pol = lambda: scripted_policy(env, rng, 0.1)
    return env, img, pol


def controller(rp, n_turn=0):
    """rp = (cos, sin, dx', dy') decoded pose, n_turn = turns taken in a row -> action. Home vector in the start frame is (-dx', -dy')."""
    c, s, dx, dy = (float(v) for v in rp); dist = float(np.hypot(dx, dy))
    if dist < R_STOP: return 0, dist
    phi_home = np.arctan2(-dy, -dx); th = np.arctan2(s, c); err = (phi_home - th + np.pi) % (2 * np.pi) - np.pi
    if a.backward and abs(err) > np.pi - TOL: return 4, dist                      # home is behind: step back rather than turn around
    if abs(err) > TOL and n_turn < a.max_turns: return (2 if err > 0 else 3), dist
    return 1, dist


def fit_rollouts(n):
    """States and true relative poses along n closed-loop episodes in fresh arenas (seeds disjoint from the test episodes):
    scripted outbound for t_vis steps, then the oracle controller in the dark with 20% random actions. Gives the readout the
    action statistics of homing (turn bursts), which the scripted training episodes lack."""
    Xs, Ys = [], []; r = np.random.default_rng(a.seed + 1)
    for i in range(n):
        env, img, pol = make_env(100_000 + i); pos0, hd0 = env.pos.copy(), float(env.heading)
        h = model.init_state(1, dev); prev = torch.zeros(1, dtype=torch.long, device=dev); n_turn = 0
        for t in range(a.t_vis + a.t_blind):
            visible = t < a.t_vis; o = eye(img) if visible else torch.zeros(eh * ew * 3, device=dev)
            with torch.no_grad(): h = model.step(h, o[None], prev, torch.full((1,), float(visible), device=dev), W)
            true = rel_pose(env.pos, env.heading, pos0, hd0); Xs.append(h[0]); Ys.append(true)
            if visible: ac = pol()
            else: ac = int(r.integers(0, 4)) if r.random() < 0.2 else controller(true, n_turn)[0]; n_turn = n_turn + 1 if ac in (2, 3) else 0
            step_out = env.step(ac) if a.env == "arena" else env.step(ac, render=t + 1 < a.t_vis)
            img = step_out[0] if step_out[0] is not None else img; prev = torch.tensor([ac], device=dev)
    return torch.stack(Xs).float(), torch.from_numpy(np.stack(Ys)).to(dev)


# ---------------- readout fit on the training episodes ----------------
model = readout = None
if a.readout in ("state", "mlp"):
    model, args, conn, lat = load_run(a.run, device=dev); W = model.edge_weights()
    tr, _ = load_split(args["train"], lat, with_z=False); E, T = tr["obs"].shape[:2]      # teacher latents are not needed here (and slow to load)
    Y = np.zeros((E, T, 4), np.float32)
    pos, hd = tr["pos"].numpy(), tr["heading"].numpy()
    for e in range(E):
        for t in range(T): Y[e, t] = rel_pose(pos[e, t], hd[e, t], pos[e, 0], hd[e, 0])
    Hs = []
    with torch.no_grad():
        W = model.edge_weights()
        for i in range(0, E, 64):
            obs, act = tr["obs"][i:i + 64].to(dev), tr["act"][i:i + 64].to(dev)
            vis = torch.ones(obs.shape[:2], device=dev); vis[:, a.t_vis:] = 0            # deployment protocol: visible, then blind
            _, H = model(obs, act, vis); Hs.append(H.reshape(-1, model.N).float())
    X = torch.cat(Hs); Yt = torch.from_numpy(Y.reshape(-1, 4)).to(dev); n_tr = len(X)
    if a.fit_rollouts > 0: Xr, Yr = fit_rollouts(a.fit_rollouts); X, Yt = torch.cat([X, Xr]), torch.cat([Yt, Yr])
    mu_x, mu_y = X.mean(0), Yt.mean(0)
    def r2_report(pred):
        r2 = lambda sl: [round(float(v), 3) for v in 1 - ((pred[sl] - Yt[sl]) ** 2).sum(0) / ((Yt[sl] - Yt[sl].mean(0)) ** 2).sum(0)]
        return f"R2 cos/sin/dx/dy = {r2(slice(0, n_tr))} on {E} train episodes" + (f", {r2(slice(n_tr, None))} on {a.fit_rollouts} closed-loop rollouts" if a.fit_rollouts > 0 else "")
    if a.readout == "state":
        Xc = X - mu_x; A = Xc.T @ Xc + a.lam * torch.eye(model.N, device=dev)
        Wr = torch.linalg.solve(A, Xc.T @ (Yt - mu_y))
        readout = lambda h: (h - mu_x) @ Wr + mu_y
        print(f"ridge readout fit: {r2_report(readout(X))}", flush=True)
    else:
        sd_x = X.std(0) + 1e-6; net = torch.nn.Sequential(torch.nn.Linear(model.N, 256), torch.nn.Tanh(), torch.nn.Linear(256, 4)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5); n = len(X)
        for ep in range(30):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, 1024):
                j = perm[i:i + 1024]; loss = ((net((X[j] - mu_x) / sd_x) - Yt[j]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); readout = lambda h: net((h - mu_x) / sd_x)
        with torch.no_grad(): print(f"mlp readout fit: {r2_report(readout(X))}", flush=True)
    del X, Hs


t0 = time.time(); succ, dfinal, eff, dstart_blind = [], [], [], []; ep = -1; done = 0
while done < a.episodes:
    ep += 1; env, img, pol = make_env(ep); pos0, hd0 = env.pos.copy(), float(env.heading)
    # outbound phase must take the agent at least 2 success radii from home, otherwise the episode is trivial: peek with a copy of the env RNG
    import copy as _copy; probe = _copy.deepcopy(env); prng = np.random.default_rng(a.seed * 7919 + ep)
    for _ in range(a.t_vis): probe.step((scripted_policy(probe, prng, 0.1) if a.env == "arena" else probe.policy(prng, 0.2)), render=False) if a.env == "nyc" else probe.step(scripted_policy(probe, prng, 0.1), render=False)
    if np.linalg.norm(probe.pos - pos0) < 2 * R_SUCC: continue
    rng = np.random.default_rng(a.seed * 7919 + ep)                                   # replay the same outbound policy draws
    h = model.init_state(1, dev) if model is not None else None; W = model.edge_weights() if model is not None else None
    prev = torch.zeros(1, dtype=torch.long, device=dev); dr = np.array([1.0, 0.0, 0.0, 0.0], np.float32)   # dead-reckoned pose (start frame)
    dr_pos, dr_th = np.zeros(2), 0.0
    for t in range(a.t_vis + a.t_blind):
        visible = t < a.t_vis
        if model is not None:
            o = eye(img) if visible else torch.zeros(eh * ew * 3, device=dev)
            with torch.no_grad(): h = model.step(h, o[None], prev, torch.full((1,), float(visible), device=dev), W)
        if visible:
            ac = pol()
        else:
            if t == a.t_vis: d0 = float(np.linalg.norm(env.pos - pos0)); path = 0.0; n_turn = 0
            if a.readout in ("state", "mlp"):
                with torch.no_grad(): rp = readout(h)[0].cpu().numpy()
            elif a.readout == "oracle": rp = rel_pose(env.pos, env.heading, pos0, hd0)
            elif a.readout == "actions": rp = np.array([np.cos(dr_th), np.sin(dr_th), dr_pos[0], dr_pos[1]], np.float32)
            else: rp = None
            ac = int(rng.integers(1, 4)) if rp is None else controller(rp, n_turn)[0]
            n_turn = n_turn + 1 if ac in (2, 3) else 0
            if a.verbose and done < a.verbose:
                print(f"  ep {ep} t {t} decoded {np.round(rp, 2) if rp is not None else '-'} true {np.round(rel_pose(env.pos, env.heading, pos0, hd0), 2)} -> action {ac}", flush=True)
        before = env.pos.copy()
        step_out = env.step(ac) if a.env == "arena" else env.step(ac, render=visible or t + 1 < a.t_vis)
        img = step_out[0] if step_out[0] is not None else img
        # dead reckoning of the discrete action in the start frame (ideal integrator of the action record)
        d_fwd = {0: 0.0, 1: (0.6 if a.env == "arena" else 3.0), 4: (-0.36 if a.env == "arena" else -2.0)}.get(ac, 0.0)   # arena step_len 0.6
        d_th = {2: TURN, 3: -TURN}.get(ac, 0.0)
        dr_th += d_th; dr_pos += d_fwd * np.array([np.cos(dr_th), np.sin(dr_th)])
        if not visible: path += float(np.linalg.norm(env.pos - before))
        prev = torch.tensor([ac], device=dev)
    dfin = float(np.linalg.norm(env.pos - pos0)); succ.append(float(dfin < R_SUCC)); dfinal.append(dfin); dstart_blind.append(d0)
    eff.append(min(1.0, max(0.0, (d0 - dfin) / max(path, 1e-6))))
    done += 1
    if done % 10 == 1: print(f"ep {ep}: blind-start dist {d0:.2f} -> final {dfin:.2f}  success so far {np.mean(succ):.2f}  {time.time()-t0:.0f}s", flush=True)
res = dict(run=a.run, env=a.env, readout=a.readout, t_vis=a.t_vis, t_blind=a.t_blind, radius=R_SUCC, turn_noise=a.turn_noise, episodes=a.episodes, tol_deg=a.tol_deg, max_turns=a.max_turns, backward=a.backward, fit_rollouts=a.fit_rollouts,
           success=float(np.mean(succ)), dist_blind_start=float(np.mean(dstart_blind)), dist_final=float(np.mean(dfinal)), path_efficiency=float(np.mean(eff)),
           per_episode_final=[round(x, 3) for x in dfinal])
print(json.dumps({k: v for k, v in res.items() if k != "per_episode_final"}))
os.makedirs("results/homing", exist_ok=True); json.dump(res, open(f"results/homing/{a.run}_{a.env}_{a.readout}{a.tag}.json", "w"), indent=1)
