"""Collect episodes: low-res fly-eye observations, actions, ground-truth state, and V-JEPA 2.1 per-step latents."""
import argparse, numpy as np, torch, time, os
from wm2cx.env import FlyArena, scripted_policy
from wm2cx.teacher import VJEPA21Encoder

p = argparse.ArgumentParser()
p.add_argument("--task", default="nav"); p.add_argument("--episodes", type=int, default=400)
p.add_argument("--steps", type=int, default=24); p.add_argument("--clip", type=int, default=8)
p.add_argument("--res", type=int, default=256); p.add_argument("--eye", default="32x8")
p.add_argument("--n_boxes", type=int, default=1); p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None); p.add_argument("--device", default="mps")
p.add_argument("--env", default="arena", choices=["arena", "valdrada"]); p.add_argument("--save_frames", type=int, default=0, help="also store frames downsampled to this size")
p.add_argument("--lat", type=float, default=40.7580); p.add_argument("--lon", type=float, default=-73.9855)
p.add_argument("--causal", type=int, default=1, help="1: target at step t = latent of the clip ENDING at t (frames t-7..t, first frame repeated before the start); "
                                                     "0: legacy non-overlapping clips whose targets encode future frames (issue #5)")
a = p.parse_args()
ew, eh = map(int, a.eye.split("x"))
out = a.out or f"data/{a.task}_s{a.seed}{'' if a.causal else '_noncausal'}.npz"
teacher = VJEPA21Encoder("vitb", crop=a.res, frames=2 * a.clip).to(a.device)
rng = np.random.default_rng(a.seed)
OBS, ACT, POS, HD, BOX, GOAL, Z, SUCC, FR = [], [], [], [], [], [], [], [], []
venv = None
if a.env == "valdrada":
    from wm2cx.valdrada_env import ValdradaArena
    venv = ValdradaArena(lat=a.lat, lon=a.lon, seed=a.seed)

def eye(img):  # compound-eye downsample: [H,W,3] -> [eh,ew,3] (rows cover the horizon band)
    R = img.shape[0]; band = img[R // 4: 3 * R // 4]
    return band.reshape(eh, band.shape[0] // eh, ew, R // ew, 3).mean((1, 3)).astype(np.uint8)

t0 = time.time()
for ep in range(a.episodes):
    if venv is not None:
        env = venv; pol = lambda: env.policy(rng, eps)
        for _try in range(5):
            try: img = env.reset(); break
            except RuntimeError as e: print("reset retry:", e, flush=True)
    else:
        env = FlyArena(task=a.task, res=a.res, n_boxes=a.n_boxes, seed=a.seed * 100000 + ep); img = env.reset(); pol = lambda: scripted_policy(env, rng, eps)
    eps = float(rng.choice([0.1, 0.3, 0.6]))
    for _ in range(int(rng.integers(0, 40)) if venv is None else 0):  # random warm-up so arrangement episodes start mid-task
        img, _ = env.step(pol())
    bx = lambda: env.box.copy() if hasattr(env, "box") else np.zeros((0, 2))
    frames, obs, act, pos, hd, box, succ = [img], [eye(img)], [], [env.pos.copy()], [env.heading], [bx()], [env.success()]
    for t in range(a.steps - 1):
        ac = pol()
        img, s = env.step(ac)
        frames.append(img); obs.append(eye(img)); act.append(ac); pos.append(env.pos.copy()); hd.append(env.heading); box.append(bx()); succ.append(s)
    act.append(0)
    fr = np.stack(frames)  # [T,R,R,3]
    if a.causal:  # sliding window: clip for step t ends at t, so the target never sees frames after t
        padded = np.concatenate([np.repeat(fr[:1], a.clip - 1, 0), fr]); idx = np.arange(a.steps)[:, None] + np.arange(a.clip)[None]
        clips = padded[idx].repeat(2, axis=1)  # [T, 2*clip, R, R, 3]
        z = np.concatenate([teacher.per_step_latents(torch.from_numpy(clips[i:i + 8]).to(a.device))[:, -1].cpu().numpy() for i in range(0, a.steps, 8)]).astype(np.float16)
    else:
        clips = fr.reshape(a.steps // a.clip, a.clip, a.res, a.res, 3).repeat(2, axis=1)  # duplicate frames -> 1 tubelet = 1 env step
        z = teacher.per_step_latents(torch.from_numpy(clips).to(a.device)).cpu().numpy().reshape(a.steps, -1).astype(np.float16)
    OBS.append(np.stack(obs)); ACT.append(np.array(act, np.int8)); POS.append(np.stack(pos)); HD.append(np.array(hd)); BOX.append(np.stack(box))
    GOAL.append(env.goal if venv is not None else (env.lm[env.goal] if a.task == "nav" else env.zone[0])); Z.append(z); SUCC.append(np.array(succ, np.int8))
    if a.save_frames:
        f = a.save_frames; FR.append(np.stack([fr[i].reshape(f, a.res // f, f, a.res // f, 3).mean((1, 3)).astype(np.uint8) for i in range(a.steps)]))
    if ep % 25 == 0: print(f"ep {ep} {time.time()-t0:.0f}s", flush=True)
extra = dict(frames=np.stack(FR)) if FR else {}
np.savez_compressed(out, obs=np.stack(OBS), act=np.stack(ACT), pos=np.stack(POS), heading=np.stack(HD), box=np.stack(BOX), goal=np.stack(GOAL), z=np.stack(Z), succ=np.stack(SUCC), **extra)
print("saved", out, np.stack(Z).shape, f"{time.time()-t0:.0f}s")
