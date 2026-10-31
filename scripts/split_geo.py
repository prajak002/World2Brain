"""Issue #10: geographically disjoint train / val / test split for the NYC episodes.
Episodes are binned by the grid cell of their start position; whole cells are assigned to splits (70/15/15 by episode count),
so no test episode starts within `--cell` metres of a training start. Writes <out>_{train,val,test}.npz and a split json."""
import argparse, json, numpy as np

p = argparse.ArgumentParser(); p.add_argument("--inputs", nargs="+", required=True); p.add_argument("--out", default="data/nyc_geo")
p.add_argument("--cell", type=float, default=60.0); p.add_argument("--frac", nargs=3, type=float, default=[0.7, 0.15, 0.15]); p.add_argument("--seed", type=int, default=0)
a = p.parse_args(); rng = np.random.default_rng(a.seed)
ds = [np.load(f) for f in a.inputs]; keys = ds[0].files
D = {k: np.concatenate([d[k] for d in ds]) for k in keys if all(k in d.files for d in ds)}
start = D["pos"][:, 0]; cell = np.floor(start / a.cell).astype(int); cid = np.unique(cell, axis=0, return_inverse=True)[1]
cells = rng.permutation(cid.max() + 1); counts = np.bincount(cid); n = len(cid); split = {}; acc = 0; bounds = np.cumsum(a.frac) * n; names = ["train", "val", "test"]; si = 0
for c in cells:
    split[c] = names[si]; acc += counts[c]
    if acc >= bounds[si] and si < 2: si += 1
assign = np.array([split[c] for c in cid]); meta = {}
for s in names:
    m = assign == s; np.savez_compressed(f"{a.out}_{s}.npz", **{k: v[m] for k, v in D.items()})
    meta[s] = dict(episodes=int(m.sum()), cells=int(len(set(cid[m]))))
    if m.any() and s != "train":
        d = np.linalg.norm(start[m][:, None] - start[assign == "train"][None], axis=-1).min(1); meta[s]["min_dist_to_train_start_m"] = float(d.min())
json.dump(dict(cell_m=a.cell, frac=a.frac, seed=a.seed, **meta), open(f"{a.out}_split.json", "w"), indent=1); print(json.dumps(meta, indent=1))
