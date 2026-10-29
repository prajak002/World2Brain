import numpy as np, torch

class Latents:
    """PCA projection of teacher latents (fit on train) + obs normalisation."""
    def __init__(self, z_train, k=256):
        z = z_train.reshape(-1, z_train.shape[-1]).astype(np.float32)
        self.mu = z.mean(0); zc = z - self.mu
        U, Sv, Vt = np.linalg.svd(zc[np.random.default_rng(0).choice(len(zc), min(len(zc), 20000), replace=False)], full_matrices=False)
        self.P = Vt[:k].T.astype(np.float32)
        proj = zc @ self.P; self.scale = proj.std() + 1e-6
        self.explained = float((Sv[:k] ** 2).sum() / (Sv ** 2).sum())
    def __call__(self, z): return ((z.astype(np.float32) - self.mu) @ self.P) / self.scale
    def inverse(self, y): return y * self.scale @ self.P.T + self.mu


def load_split(paths, lat=None, k=256):
    ds = [np.load(p) for p in paths]
    obs = np.concatenate([d["obs"] for d in ds]).astype(np.float32) / 255.0  # [E,T,eh,ew,3]
    obs = obs.reshape(obs.shape[0], obs.shape[1], -1)
    act = np.concatenate([d["act"] for d in ds]).astype(np.int64)
    z = np.concatenate([d["z"] for d in ds])
    lat = Latents(z, k) if lat is None else lat
    out = dict(obs=torch.from_numpy(obs), act=torch.from_numpy(act), z=torch.from_numpy(lat(z)).float(),
               heading=torch.from_numpy(np.concatenate([d["heading"] for d in ds])).float(),
               pos=torch.from_numpy(np.concatenate([d["pos"] for d in ds])).float(),
               task=torch.from_numpy(np.concatenate([np.full(len(d["obs"]), i) for i, d in enumerate(ds)])))
    return out, lat


def blackout(B, T, H, rng, device, p_blind=0.7):
    """visible mask [B,T]: a random blind window of length H in a fraction of episodes (always a visible warm-up)."""
    vis = torch.ones(B, T, device=device)
    for b in range(B):
        if rng.random() < p_blind:
            t0 = int(rng.integers(4, max(5, T - H))); vis[b, t0:t0 + H] = 0
    return vis
