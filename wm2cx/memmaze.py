"""Memory Maze offline-probing data (Pasukonis et al., ICLR 2023; docs/PREREG.md section 2).

Released trajectories are NPZ files of 1001 steps: image (1001,64,64,3) uint8, action (1001,6) one-hot, agent_pos (1001,2),
agent_dir (1001,2), maze_layout (1001,9,9), targets_vec (1001,3,2) agent-centric object vectors, reset/terminal flags.
`action[t]` is the action that LED TO frame t (zeros at reset). The repo's students take act[t] = action taken AFTER
observing step t, so `load_traj` shifts it by one (act[t] = argmax action[t+1], last step 0).

Split (unchanged from the release): eval = the 1,000 files of eval.zip; train = the 29,000 files of train-part*.zip.
Validation (hyper-parameters and checkpoint rule only) = the LAST 1,000 train files in filename (= time) order.
"""
from __future__ import annotations
import glob, os, numpy as np

T_STEPS = 1001
N_ACT = 6
OBS_RES = 16                     # student input: 64x64 frame mean-pooled to 16x16x3 = 768 values (= the paper's 32x8 fly eye)
SCORE_FROM, SCORE_TO = 500, 1000  # PREREG: scores averaged over steps 500-1000


def list_split(root, split, n_val=1000):
    """Sorted trajectory paths. split in {train, val, train_all, eval}. root holds train/ and eval/ directories of NPZs."""
    if split == "eval":
        return sorted(glob.glob(os.path.join(root, "eval", "**", "*.npz"), recursive=True))
    tr = sorted(glob.glob(os.path.join(root, "train", "**", "*.npz"), recursive=True), key=os.path.basename)
    if split == "train_all": return tr
    if split == "val": return tr[-n_val:]
    if split == "train": return tr[:-n_val]
    raise ValueError(split)


def downsample(img, res=OBS_RES):
    """[T,64,64,3] uint8 -> [T,res*res*3] uint8 by block-mean pooling."""
    T, H, W, C = img.shape; f = H // res
    return img.reshape(T, res, f, res, f, C).mean((2, 4)).round().astype(np.uint8).reshape(T, -1)


def load_traj(path, with_image=True):
    d = np.load(path)
    a = d["action"].argmax(-1).astype(np.int8); act = np.zeros(len(a), np.int8); act[:-1] = a[1:]
    out = dict(act=act, agent_pos=d["agent_pos"].astype(np.float32), agent_dir=d["agent_dir"].astype(np.float32),
               maze_layout=d["maze_layout"][0].astype(np.uint8), targets_vec=d["targets_vec"].astype(np.float32),
               name=os.path.basename(path))
    assert (d["maze_layout"] == d["maze_layout"][:1]).all(), f"maze layout changes within {path}"
    assert d["reset"][0] and not d["reset"][1:].any(), f"unexpected reset inside {path}"
    if with_image:
        out["image"] = d["image"]; out["obs"] = downsample(d["image"])
    return out


def causal_clip_index(T, clip=8):
    """Index [T, clip] into a frame array padded with clip-1 copies of frame 0 at the front: row t = frames t-clip+1..t,
    so the teacher latent for step t never sees frames after t (same rule as scripts/collect.py --causal 1)."""
    return np.arange(T)[:, None] + np.arange(clip)[None]


def walls_acc(logits_or_prob, layout, threshold=0.5, is_logit=True):
    """Per-step tile accuracy (%) of a 9x9 layout prediction. pred [..., 81], layout [..., 81] in {0,1}."""
    p = 1 / (1 + np.exp(-logits_or_prob)) if is_logit else logits_or_prob
    return 100.0 * ((p > threshold) == (layout > 0.5)).mean(-1)


def objects_mse(pred, target):
    """Per-step object error: squared Euclidean error of each agent-centric object vector, mean over objects.
    pred, target [..., n_obj, 2] -> [...]."""
    return ((pred - target) ** 2).sum(-1).mean(-1)
