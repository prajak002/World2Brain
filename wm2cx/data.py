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
    def inverse_torch(self, y):
        """Whitened PCA coordinates -> original teacher space (torch, any device)."""
        P = torch.as_tensor(self.P, device=y.device, dtype=y.dtype); mu = torch.as_tensor(self.mu, device=y.device, dtype=y.dtype)
        return y * float(self.scale) @ P.T + mu


def load_split(paths, lat=None, k=256, keep_raw=False):
    """Concatenate episode files. `keep_raw=True` also returns the un-projected teacher latents (`z_raw`, float16)
    so errors can be reported in the original teacher space, not only in whitened PCA space (#6)."""
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
    if keep_raw: out["z_raw"] = torch.from_numpy(np.ascontiguousarray(z.astype(np.float16)))
    return out, lat


def blackout(B, T, H, rng, device, p_blind=0.7):
    """visible mask [B,T]: a random blind window of length H in a fraction of episodes (always a visible warm-up)."""
    vis = torch.ones(B, T, device=device)
    for b in range(B):
        if rng.random() < p_blind:
            t0 = int(rng.integers(4, max(5, T - H))); vis[b, t0:t0 + H] = 0
    return vis


def parse_windows(ws, T=None):
    """'8:16 14:22' -> [(8, 16), (14, 22)]; the first window is the primary one reported as test_mse."""
    out = []
    for w in ws:
        t0, t1 = (int(x) for x in w.split(":")); assert 0 < t0 < t1 and (T is None or t1 <= T), w
        out.append((t0, t1))
    return out


def window_errors(zh, z, t0, t1, lat=None, z_raw=None):
    """Per-episode, per-horizon errors inside a blind window. zh, z [B,T,k] (whitened PCA); z_raw [B,T,D] original.
    Returns dict of [B, t1-t0] tensors: mse (PCA space), cos, and mse_teacher (original teacher space) if lat is given."""
    out = dict(mse=((zh - z) ** 2).mean(-1)[:, t0:t1], cos=torch.nn.functional.cosine_similarity(zh, z, -1)[:, t0:t1])
    if lat is not None and z_raw is not None:
        out["mse_teacher"] = ((lat.inverse_torch(zh[:, t0:t1]) - z_raw[:, t0:t1].to(zh.dtype)) ** 2).mean(-1)
    return out
