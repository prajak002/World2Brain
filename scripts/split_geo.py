"""Geographically disjoint train / val / test split for NYC episodes (#10).

Episodes are assigned to square blocks of `--block` metres by their start position; whole blocks go to one split
(70 / 15 / 15 by episode count, greedy over shuffled blocks), so no test episode starts near a training episode. The
split is defined and saved BEFORE training; `--margin` reports (and optionally enforces) the minimum distance between
any test-episode trajectory point and any train-episode trajectory point.

  PYTHONPATH=. python scripts/split_geo.py data/nyc_raw_*.npz --out data/nyc --block 120 --margin 40
  -> data/nyc_train.npz data/nyc_val.npz data/nyc_test.npz data/nyc_split.json
"""
import argparse, json, os, numpy as np

p = argparse.ArgumentParser()
p.add_argument("raw", nargs="+", help="collected episode files (any number of locations)")
p.add_argument("--out", default="data/nyc"); p.add_argument("--block", type=float, default=120.0, help="block edge, metres")
p.add_argument("--frac", nargs=3, type=float, default=[0.7, 0.15, 0.15]); p.add_argument("--seed", type=int, default=0)
p.add_argument("--margin", type=float, default=0.0, help="min train-test trajectory distance (m) to enforce; 0 = report only")
p.add_argument("--loc_key", default="loc", help="optional per-episode location id array in the npz")
a = p.parse_args()


def load_all(paths):
    ds = [np.load(f, allow_pickle=False) for f in paths]; keys = set(ds[0].files)
    for d in ds[1:]: keys &= set(d.files)
    out = {k: np.concatenate([d[k] for d in ds]) for k in keys if ds[0][k].ndim >= 1 and all(d[k].shape[1:] == ds[0][k].shape[1:] for d in ds)}
    out["src"] = np.concatenate([np.full(len(d["obs"]), i) for i, d in enumerate(ds)])
    return out


D = load_all(a.raw); E = len(D["obs"]); pos = D["pos"]                     # [E, T, 2] metres in the twin's frame
loc = D[a.loc_key] if a.loc_key in D else D["src"]
start = pos[:, 0]
# block id = (location, floor(x / block), floor(y / block)); different locations never share a block
bx, by = np.floor(start[:, 0] / a.block).astype(int), np.floor(start[:, 1] / a.block).astype(int)
blocks = {}
for e in range(E): blocks.setdefault((int(loc[e]), int(bx[e]), int(by[e])), []).append(e)
rng = np.random.default_rng(a.seed); keys = list(blocks); rng.shuffle(keys)
target = np.array(a.frac) * E; counts = np.zeros(3); assign = {}
for k in keys:                                    # greedy: give each block to the split furthest below its target
    s = int(np.argmax(target - counts)); assign[k] = s; counts[s] += len(blocks[k])
split = np.array([assign[(int(loc[e]), int(bx[e]), int(by[e]))] for e in range(E)])
names = ["train", "val", "test"]


def min_dist(A, B):
    """min over all pairs of trajectory points (chunked)."""
    A = A.reshape(-1, 2); B = B.reshape(-1, 2); m = np.inf
    for i in range(0, len(A), 2000):
        d = np.sqrt(((A[i:i + 2000, None] - B[None]) ** 2).sum(-1)); m = min(m, float(d.min()))
    return m


md = min_dist(pos[split == 0], pos[split == 2]) if (split == 0).any() and (split == 2).any() else float("nan")
if a.margin > 0 and md < a.margin:
    # drop test episodes whose trajectory comes within `margin` of any training trajectory (they become unused)
    tr = pos[split == 0].reshape(-1, 2); drop = []
    for e in np.where(split == 2)[0]:
        if np.sqrt(((pos[e][:, None] - tr[None]) ** 2).sum(-1)).min() < a.margin: drop.append(int(e))
    split[drop] = -1; md = min_dist(pos[split == 0], pos[split == 2])
    print(f"dropped {len(drop)} test episodes within {a.margin} m of a training trajectory")
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
for s, n in enumerate(names):
    idx = np.where(split == s)[0]
    np.savez_compressed(f"{a.out}_{n}.npz", **{k: v[idx] for k, v in D.items()})
    print(f"{n:5s} {len(idx):4d} episodes, {len({(int(loc[e]), int(bx[e]), int(by[e])) for e in idx})} blocks, {len(set(loc[idx].tolist()))} locations")
meta = dict(files=a.raw, block_m=a.block, frac=a.frac, seed=a.seed, n_episodes=int(E), n_blocks=len(blocks),
            per_split={n: int((split == s).sum()) for s, n in enumerate(names)}, unused=int((split == -1).sum()),
            min_train_test_distance_m=md, blocks={f"{k[0]}:{k[1]}:{k[2]}": names[v] for k, v in assign.items()})
json.dump(meta, open(f"{a.out}_split.json", "w"), indent=1)
print(f"min train-test trajectory distance {md:.1f} m -> {a.out}_split.json")
