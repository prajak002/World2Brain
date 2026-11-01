"""Real handheld video -> wm2cx episodes (E7: real-video transfer benchmark).

Turns phone / GoPro walking footage into the same episode format `scripts/collect.py` writes for the arena, so every
student checkpoint can be evaluated on *real* video with `scripts/eval_ckpt.py` and the result lands in the same
tables (`summarize.py` tag `_realvid`). There are no ground-truth actions, so egomotion is estimated from the video
itself with patch-wise phase correlation (pure numpy, no OpenCV):

  * yaw   = mean horizontal flow over a 4x4 patch grid, in degrees via the horizontal field of view (`--fov_deg`);
  * forward motion = mean radial (expanding) flow component, a divergence proxy (positive = moving forward);
  * both are accumulated over `--sub` sub-frames per step so large handheld displacements stay trackable.

The continuous estimates are stored as `ego` [E,T,2] (yaw deg/step, expansion px/step); `heading` is the integrated yaw
(radians, relative to the episode start) so the E3 heading probe has a target on real video, and `pos` is dead-reckoned
from heading x expansion (arbitrary units). Discrete actions follow the arena's semantics (0 stay, 1 forward,
2 turn left = +yaw, 3 turn right, 4 back) with thresholds `--turn_deg` / `--fwd_px`; the continuous `ego` columns let
later work replace the quantiser.

Frames are centre-cropped to a square and resized to `--res` (teacher input) and compound-eye downsampled for the
student exactly like collect.py. Teacher latents are computed when `--device` is a torch device; `--device none` skips
the teacher and keeps the 256 px frames in the file so `--from_npz` can add `z` later on a GPU box.

  python scripts/ingest_video.py --videos a.mp4 b.mp4 --out data/realvid_all.npz --device none       # frames + ego
  python scripts/ingest_video.py --from_npz data/realvid_all.npz --device cuda                       # add teacher z
"""
import argparse, json, os, subprocess, time, numpy as np
from wm2cx.ego import egomotion, quantise

p = argparse.ArgumentParser()
p.add_argument("--videos", nargs="*", default=[]); p.add_argument("--from_npz", default=None, help="add teacher latents to an existing ingest file")
p.add_argument("--out", default="data/realvid_all.npz")
p.add_argument("--fps", type=float, default=2.0, help="env steps per second of video (arena episodes are 24 steps)")
p.add_argument("--sub", type=int, default=4, help="sub-frames per step used for flow accumulation")
p.add_argument("--steps", type=int, default=24); p.add_argument("--stride", type=int, default=24, help="steps between episode starts (24 = no overlap)")
p.add_argument("--clip", type=int, default=8); p.add_argument("--res", type=int, default=256); p.add_argument("--eye", default="32x8")
p.add_argument("--fov_deg", type=float, default=60.0, help="horizontal FoV of the square crop (phone main camera, portrait: ~60 deg)")
p.add_argument("--turn_deg", type=float, default=6.0, help="|yaw| per step above which the step is a turn")
p.add_argument("--fwd_px", type=float, default=1.0, help="radial expansion per step above which the step is forward (below -x: back)")
p.add_argument("--grid", type=int, default=4); p.add_argument("--device", default="none")
p.add_argument("--save_frames", type=int, default=64, help="also store frames at this size for videos/figures (0 = off)")
p.add_argument("--causal", type=int, default=1, help="teacher target at step t from the clip ending at t (#5); must match the students' training data")
a = p.parse_args()
ew, eh = map(int, a.eye.split("x"))


def decode(path, fps, res):
    """ffmpeg -> [N,res,res,3] uint8 at `fps`, centre square crop."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={fps},crop='min(iw,ih)':'min(iw,ih)',scale={res}:{res}:flags=area",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, res, res, 3)


def eye(img):  # compound-eye downsample: [R,R,3] -> [eh,ew,3] (rows cover the horizon band), identical to collect.py
    R = img.shape[0]; band = img[R // 4: 3 * R // 4]
    return band.reshape(eh, band.shape[0] // eh, ew, R // ew, 3).mean((1, 3)).astype(np.uint8)


def ingest(paths):
    OBS, ACT, HD, POS, EGO, CLIP, FR, FULL, SRC = [], [], [], [], [], [], [], [], []
    for ci, path in enumerate(paths):
        t0 = time.time(); fr = decode(path, a.fps * a.sub, a.res)
        n_steps = (len(fr) - 1) // a.sub
        yaw = np.zeros(n_steps, np.float32); exp_ = np.zeros(n_steps, np.float32)
        for s in range(n_steps):                                            # accumulate flow over the sub-frames of each step
            for k in range(a.sub):
                y_, e_ = egomotion(fr[s * a.sub + k], fr[s * a.sub + k + 1], a.grid, a.fov_deg); yaw[s] += y_; exp_[s] += e_
        step_frames = fr[::a.sub][:n_steps + 1]
        starts = list(range(0, n_steps + 1 - a.steps, a.stride))
        print(f"{os.path.basename(path)}: {len(fr)} frames, {n_steps} steps, {len(starts)} episodes, "
              f"median |yaw| {np.median(np.abs(yaw)):.1f} deg/step, median expansion {np.median(exp_):.2f} px/step, {time.time()-t0:.0f}s", flush=True)
        for s0 in starts:
            f = step_frames[s0:s0 + a.steps]; y = yaw[s0:s0 + a.steps]; e = exp_[s0:s0 + a.steps]
            act = np.array([quantise(y[t], e[t], a.turn_deg, a.fwd_px) for t in range(a.steps - 1)] + [0], np.int8)   # act[t] = motion between t and t+1
            hd = np.concatenate([[0.0], np.cumsum(np.deg2rad(y[:-1]))]).astype(np.float32)
            speed = np.clip(e[:-1], 0, None) / max(a.fwd_px, 1e-6)
            pos = np.concatenate([np.zeros((1, 2)), np.cumsum(np.stack([speed * np.cos(hd[:-1]), speed * np.sin(hd[:-1])], 1), 0)]).astype(np.float32)
            OBS.append(np.stack([eye(x) for x in f])); ACT.append(act); HD.append(hd); POS.append(pos)
            EGO.append(np.stack([y, e], 1)); CLIP.append(ci); FULL.append(f); SRC.append(os.path.basename(path))
            if a.save_frames:
                g = a.save_frames; FR.append(np.stack([x.reshape(g, a.res // g, g, a.res // g, 3).mean((1, 3)).astype(np.uint8) for x in f]))
    E = len(OBS); assert E > 0, "no episodes: clips shorter than --steps/--fps seconds?"
    acts = np.concatenate(ACT); hist = np.bincount(acts, minlength=5) / len(acts)
    print(f"{E} episodes from {len(paths)} clips; action histogram stay/fwd/left/right/back = {np.round(hist, 2).tolist()}")
    d = dict(obs=np.stack(OBS), act=np.stack(ACT), pos=np.stack(POS), heading=np.stack(HD), ego=np.stack(EGO), clip=np.array(CLIP, np.int16),
             box=np.zeros((E, a.steps, 0, 2), np.float32), goal=np.zeros((E, 2), np.float32), succ=np.zeros((E, a.steps), np.int8),
             frames_full=np.stack(FULL), src=np.array(SRC), meta=np.array(json.dumps(vars(a))))
    if FR: d["frames"] = np.stack(FR)
    return d


def add_teacher(d, device):
    import torch
    from wm2cx.teacher import VJEPA21Encoder
    teacher = VJEPA21Encoder("vitb", crop=a.res, frames=2 * a.clip).to(device)
    full = d["frames_full"]; E, T = full.shape[:2]; Z = []
    for i in range(E):
        fr = full[i]
        if a.causal:   # clip ending at t, like collect.py --causal 1
            padded = np.concatenate([np.repeat(fr[:1], a.clip - 1, 0), fr]); idx = np.arange(T)[:, None] + np.arange(a.clip)[None]; clips = padded[idx].repeat(2, axis=1)
            Z.append(np.concatenate([teacher.per_step_latents(torch.from_numpy(np.ascontiguousarray(clips[j:j + 8])).to(device))[:, -1].cpu().numpy() for j in range(0, T, 8)]).astype(np.float16))
        else:
            clips = fr.reshape(T // a.clip, a.clip, a.res, a.res, 3).repeat(2, axis=1)       # 1 env step = 1 tubelet
            Z.append(teacher.per_step_latents(torch.from_numpy(np.ascontiguousarray(clips)).to(device)).cpu().numpy().reshape(T, -1).astype(np.float16))
    d["z"] = np.stack(Z); print("teacher latents", d["z"].shape)
    return d


if a.from_npz:
    d = dict(np.load(a.from_npz, allow_pickle=False)); out = a.out if a.out != p.get_default("out") else a.from_npz
else:
    d = ingest(a.videos); out = a.out
if a.device != "none": d = add_teacher(d, a.device)
elif "z" not in d: print("no teacher latents yet (--device none); rerun with --from_npz", out, "--device cuda on the GPU box")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True); np.savez_compressed(out, **d); print("saved", out, {k: v.shape for k, v in d.items() if hasattr(v, "shape") and v.ndim})
