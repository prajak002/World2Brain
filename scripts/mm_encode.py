"""Memory Maze B2: teacher latents for every step of every trajectory (docs/RUNBOOK_GPU.md B2, PREREG Amendment 1).

Stage `encode` (resumable per shard, several workers can share one GPU via --worker/--n_workers):
  for each shard of --shard trajectories write, atomically,
    <out>/raw/<split>_<i>.npy   fp16 [n, 1001, 3840]   V-JEPA 2.1 ViT-B per-step latent (mean token + 2x2 pooled), causal
                                                        sliding window of --clip frames ending at t (frame 0 repeated before
                                                        the start), each frame duplicated so one tubelet = one step, exactly
                                                        as scripts/collect.py --causal 1; frames upsampled 64 -> --crop
    <out>/meta/<split>_<i>.npz  obs (16x16x3 uint8), act, agent_pos, agent_dir, maze_layout, targets_vec, names
    <out>/stats/<split>_<i>.npz float64 sum and sum of outer products of the raw latents (PCA sufficient statistics)
Stage `pca`: PCA-k fit on the TRAIN shards only (never val, never eval) from the sufficient statistics -> <out>/pca.npz
  (mu, P, scale with the same whitening as wm2cx.data.Latents).
Stage `project`: <out>/lat/<split>_<i>.npy fp16 [n, 1001, k] whitened PCA coordinates for every shard.
Shards are fixed 100-file chunks inside each download part, so a shard always holds the same trajectories.
"""
import argparse, glob, json, os, time, numpy as np, torch
from concurrent.futures import ThreadPoolExecutor
from wm2cx.memmaze import list_split, load_traj, causal_clip_index, T_STEPS

p = argparse.ArgumentParser()
p.add_argument("--stage", default="encode", choices=["encode", "pca", "project"])
p.add_argument("--root", default="/root/mm9"); p.add_argument("--out", default="data/mm9")
p.add_argument("--splits", nargs="+", default=["eval", "train_all"], help="encode: eval / train_all; pca uses train; project: any")
p.add_argument("--pool", type=int, nargs="*", default=[], help="download parts forming the training pool (PREREG Amendment 1); val = its last 1000 files")
p.add_argument("--crop", type=int, default=128); p.add_argument("--clip", type=int, default=8); p.add_argument("--bf16", type=int, default=1)
p.add_argument("--bs", type=int, default=128); p.add_argument("--shard", type=int, default=100); p.add_argument("--k", type=int, default=256)
p.add_argument("--worker", type=int, default=0); p.add_argument("--n_workers", type=int, default=1); p.add_argument("--device", default="cuda")
a = p.parse_args()
for d in ("raw", "meta", "stats", "lat"): os.makedirs(f"{a.out}/{d}", exist_ok=True)


def part_shards(part_dir):
    f = sorted(glob.glob(os.path.join(part_dir, "*.npz"))); part = os.path.basename(part_dir.rstrip("/"))
    return [(f"train_{part}_{i // a.shard:02d}", f[i:i + a.shard]) for i in range(0, len(f), a.shard)]


def shards(split):
    """Shards are cut inside each download part (train/part<i>/), so they do not move when parts arrive out of order.
    eval -> eval_<i>; train_all -> every complete part; train / val -> the --pool parts in order, val = the last 1000 files."""
    if split == "eval":
        f = list_split(a.root, "eval"); return [(f"eval_{i // a.shard:04d}", f[i:i + a.shard]) for i in range(0, len(f), a.shard)]
    parts = sorted(glob.glob(os.path.join(a.root, "train", "part*")), key=lambda d: int(d.rstrip("/").split("part")[-1]))
    parts = [d for d in parts if len(glob.glob(os.path.join(d, "*.npz"))) == 2900]          # complete parts only
    if split == "train_all": return [s for d in parts for s in part_shards(d)]
    assert a.pool, "--pool <part ids> is required for train/val (PREREG Amendment 1)"
    sel = [s for i in a.pool for s in part_shards(os.path.join(a.root, "train", f"part{i}"))]
    assert sum(len(f) for _, f in sel) == 2900 * len(a.pool), "a pool part is incomplete"
    nv = 1000 // a.shard
    return sel[-nv:] if split == "val" else sel[:-nv]


if a.stage == "encode":
    from wm2cx.teacher import VJEPA21Encoder
    torch.backends.cuda.matmul.allow_tf32 = True
    teacher = VJEPA21Encoder("vitb", crop=a.crop, frames=2 * a.clip).to(a.device)
    idx = torch.from_numpy(causal_clip_index(T_STEPS, a.clip)).to(a.device)
    json.dump(dict(crop=a.crop, clip=a.clip, bf16=a.bf16, shard=a.shard, teacher="apiantonio/vjepa2.1-vit-base-384",
                   causal=True, upsample="bilinear"), open(f"{a.out}/encode_args.json", "w"), indent=1)

    @torch.no_grad()
    def encode(img):  # [1001,64,64,3] uint8 numpy -> [1001, 3840] fp16
        x = torch.from_numpy(img).to(a.device).permute(0, 3, 1, 2).float()
        x = torch.nn.functional.interpolate(x, size=(a.crop, a.crop), mode="bilinear", align_corners=False).round().clamp(0, 255).byte().permute(0, 2, 3, 1)
        x = torch.cat([x[:1].expand(a.clip - 1, -1, -1, -1), x])                  # pad with frame 0
        out = []
        with torch.autocast("cuda", torch.bfloat16, enabled=bool(a.bf16)):
            for i in range(0, T_STEPS, a.bs):
                clips = x[idx[i:i + a.bs]].repeat_interleave(2, 1)                 # [b, 2*clip, crop, crop, 3]
                out.append(teacher.per_step_latents(clips)[:, -1].float())
        return torch.cat(out)

    todo = [s for sp in a.splits for s in shards(sp)]
    todo = [s for j, s in enumerate(todo) if j % a.n_workers == a.worker and not os.path.exists(f"{a.out}/stats/{s[0]}.npz")]
    print(f"worker {a.worker}/{a.n_workers}: {len(todo)} shards to encode", flush=True)
    pool = ThreadPoolExecutor(8); t0 = time.time(); done = 0
    for name, files in todo:
        Z, M = [], {k: [] for k in ("obs", "act", "agent_pos", "agent_dir", "maze_layout", "targets_vec", "name")}
        s1 = torch.zeros(3840, dtype=torch.float64, device=a.device); s2 = torch.zeros(3840, 3840, dtype=torch.float64, device=a.device)
        for d in pool.map(load_traj, files):
            z = encode(d["image"]); s1 += z.sum(0).double(); s2 += (z.T.double() @ z.double())
            Z.append(z.half().cpu().numpy())
            for k in M: M[k].append(d[k])
        np.save(f"{a.out}/raw/{name}.tmp.npy", np.stack(Z)); os.replace(f"{a.out}/raw/{name}.tmp.npy", f"{a.out}/raw/{name}.npy")
        np.savez(f"{a.out}/meta/{name}.tmp.npz", **{k: np.stack(v) for k, v in M.items()}); os.replace(f"{a.out}/meta/{name}.tmp.npz", f"{a.out}/meta/{name}.npz")
        np.savez(f"{a.out}/stats/{name}.tmp.npz", s1=s1.cpu().numpy(), s2=s2.cpu().numpy(), n=len(files) * T_STEPS)
        os.replace(f"{a.out}/stats/{name}.tmp.npz", f"{a.out}/stats/{name}.npz")   # written last: marks the shard complete
        done += len(files); el = time.time() - t0
        print(f"{name} {done} traj {el:.0f}s {done * T_STEPS / el:.0f} steps/s", flush=True)

elif a.stage == "pca":
    names = [n for n, _ in shards("train")]
    miss = [n for n in names if not os.path.exists(f"{a.out}/stats/{n}.npz")]; assert not miss, f"train shards not encoded: {miss[:5]}..."
    s1 = 0; s2 = 0; n = 0
    for nm in names:
        d = np.load(f"{a.out}/stats/{nm}.npz"); s1 = s1 + d["s1"]; s2 = s2 + d["s2"]; n += int(d["n"])
    mu = s1 / n; C = s2 / n - np.outer(mu, mu)
    ev, V = np.linalg.eigh(C); o = np.argsort(ev)[::-1]; ev, V = np.clip(ev[o], 0, None), V[:, o]
    P = np.ascontiguousarray(V[:, :a.k]).astype(np.float32); scale = float(np.sqrt(ev[:a.k].sum() / a.k)) + 1e-6
    np.savez(f"{a.out}/pca.npz", mu=mu.astype(np.float32), P=P, scale=scale, explained=ev[:a.k].sum() / ev.sum(), n_steps=n, shards=np.array(names))
    json.dump(dict(pool=a.pool, train=names, val=[n_ for n_, _ in shards("val")], eval=[n_ for n_, _ in shards("eval")]), open(f"{a.out}/splits.json", "w"), indent=1)
    print(f"PCA-{a.k} on {len(names)} train shards ({n} steps): explained {ev[:a.k].sum() / ev.sum():.3f}")

else:  # project
    pc = np.load(f"{a.out}/pca.npz"); mu, P, scale = [torch.from_numpy(np.asarray(pc[k])).float().to(a.device) for k in ("mu", "P", "scale")]
    todo = [n for sp in a.splits for n, _ in shards(sp)]
    todo = [n for j, n in enumerate(todo) if j % a.n_workers == a.worker and not os.path.exists(f"{a.out}/lat/{n}.npy")]
    for nm in todo:
        z = torch.from_numpy(np.load(f"{a.out}/raw/{nm}.npy")).to(a.device).float()
        y = ((z - mu) @ P / scale).half().cpu().numpy()
        np.save(f"{a.out}/lat/{nm}.tmp.npy", y); os.replace(f"{a.out}/lat/{nm}.tmp.npy", f"{a.out}/lat/{nm}.npy"); print(nm, y.shape, flush=True)
