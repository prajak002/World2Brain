"""Render the raycast NYC digital twin (wm2cx/nyc.py) for figures and the explainer video (#10).

Writes to results/nyc_twin/: one 360-degree panorama per appearance shift from the same pose, a scripted episode with a
3-step blackout at the first obstacle encounter (frames, top-down map, pose, goal, collisions, path length), and a
summary JSON. Needs data/nyc/*.json from scripts/fetch_nyc_data.sh; no checkpoints.
"""
import argparse, json, os, numpy as np
from wm2cx.nyc import NYCArena

p = argparse.ArgumentParser()
p.add_argument("--out", default="results/nyc_twin"); p.add_argument("--res", type=int, default=256)
p.add_argument("--seed", type=int, default=3); p.add_argument("--steps", type=int, default=30)
p.add_argument("--blackout", type=int, default=3)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)

env = NYCArena(res=a.res, seed=a.seed); env.reset(); pose = env.state()
try: from PIL import Image
except ImportError: Image = None
def save(name, img):
    if Image is not None: Image.fromarray(img).save(os.path.join(a.out, name))

# 1. appearance shifts from one pose
for k in NYCArena.SHIFTS:
    e = NYCArena(res=a.res, seed=a.seed, appearance=k); e.set_state(pose); save(f"appearance_{k}.png", e.render())

# 2. scripted episode with blackout-in-the-loop (controller acts randomly while blind, as eval_plan.py does for non-student planners)
rng = np.random.default_rng(a.seed); frames, tops, poses, acts, blind = [], [], [], [], []
blind_left, triggered = 0, False; shortest = env.shortest_path_length(); start = env.pos.copy()
for t in range(a.steps):
    img = env.render(); frames.append(img); tops.append(env.topdown()); poses.append(dict(pos=env.pos.tolist(), heading=float(env.heading), dist=env.goal_distance()))
    if a.blackout and not triggered and (env.obstacle_ahead() or t == 8): blind_left, triggered = a.blackout, True   # obstacle, else step 8 (eval_plan --blackout_trigger)
    vis = blind_left <= 0; blind.append(not vis)
    ac = env.policy(rng, 0.0) if vis else int(rng.integers(1, 4)); acts.append(ac)
    if not vis: blind_left -= 1
    env.step(ac, render=False)
    if env.success(): frames.append(env.render()); tops.append(env.topdown()); poses.append(dict(pos=env.pos.tolist(), heading=float(env.heading), dist=env.goal_distance())); blind.append(False); break
# building footprints near the episode (flattened rings + offsets) so the video/figures can draw a real map without data/
near = [i for i, r in enumerate(env.polys) if np.linalg.norm(r.mean(0) - start) < 300.0]
verts = np.concatenate([env.polys[i] for i in near]).astype(np.float32); offs = np.cumsum([0] + [len(env.polys[i]) for i in near])
np.savez_compressed(os.path.join(a.out, "episode.npz"), frames=np.array(frames), topdown=np.array(tops), blind=np.array(blind), act=np.array(acts),
                    poly_verts=verts, poly_offsets=offs, poly_height=env.heights[near].astype(np.float32), poly_year=env.years[near],
                    street_segs=np.array(env.street_segs, np.float32), pos=np.array([q["pos"] for q in poses], np.float32), heading=np.array([q["heading"] for q in poses], np.float32),
                    goal=env.goal.astype(np.float32))
spl = env.success() * max(shortest - 12.0, 1e-6) / max(max(shortest - 12.0, 1e-6), env.path_len)
summary = dict(seed=a.seed, lat0=env.lat0, lon0=env.lon0, buildings=len(env.polys), street_points=int(len(env.street_pts)), walk_grid=list(env.walk.shape),
               start=start.tolist(), goal=env.goal.tolist(), shortest_path_m=shortest, steps=len(acts), success=env.success(), spl=float(spl),
               collisions=int(env.collisions), path_len_m=float(env.path_len), blackout=a.blackout, blind_steps=int(sum(blind)), poses=poses)
json.dump(summary, open(os.path.join(a.out, "episode.json"), "w"), indent=1)
for t in (0, len(frames) // 2, len(frames) - 1): save(f"episode_t{t:02d}.png", frames[t]); save(f"topdown_t{t:02d}.png", tops[t])
print(json.dumps({k: v for k, v in summary.items() if k != "poses"}))
