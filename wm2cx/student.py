"""Connectome-wired recurrent student ("FlyBrain") distilled from V-JEPA 2.1 latents.

h_{t+1} = (1-a) h_t + a * tanh( W h_t + W_in o_t * visible_t + W_act a_t + b )
z_hat_t = W_out h_t
W = M * S * softplus(W_raw)   (M: FlyWire adjacency mask, S: presynaptic neurotransmitter sign -> Dale's law)
Wiring variants keep neuron count, region assignment, I/O pathways and edge count fixed, only the recurrent wiring changes.
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

WIRINGS = ("connectome", "connectome_nosign", "shuffled", "random", "dense")


def build_mask(conn, wiring, n_brains=1, seed=0):
    rng = np.random.default_rng(seed)
    N = len(conn["region"]); pre, post, sign = conn["pre"].copy(), conn["post"].copy(), conn["sign"].copy()
    E = len(pre)
    if wiring == "shuffled":           # degree-preserving: in/out-degree multisets kept, pairing destroyed
        post = rng.permutation(post)
    elif wiring == "random":           # Erdos-Renyi with the same edge count and sign fractions
        pre = rng.integers(0, N, E); post = rng.integers(0, N, E); sign = rng.permutation(sign)
    M = np.zeros((N, N), np.float32); S = np.ones((N, N), np.float32)
    if wiring == "dense":
        M[:] = 1.0
    else:
        M[post, pre] = 1.0; S[post, pre] = sign
    if wiring == "connectome_nosign": S[:] = 1.0
    # multi-brain: K copies, coupled through the real contralateral (cross-hemisphere) edges brain i -> brain i+1
    if n_brains > 1:
        side = conn["side"]; cross = (side[pre] != side[post]) & (side[pre] != "center") & (side[post] != "center")
        Mk = np.zeros((n_brains * N, n_brains * N), np.float32); Sk = np.ones_like(Mk)
        for k in range(n_brains):
            Mk[k * N:(k + 1) * N, k * N:(k + 1) * N] = M; Sk[k * N:(k + 1) * N, k * N:(k + 1) * N] = S
            j = (k + 1) % n_brains
            Mk[j * N + post[cross], k * N + pre[cross]] = 1.0; Sk[j * N + post[cross], k * N + pre[cross]] = sign[cross]
            Mk[k * N + post[cross], k * N + pre[cross]] = 0.0  # move cross-hemisphere edges to the partner brain
        M, S = Mk, Sk
    return M, S


class FlyBrain(nn.Module):
    def __init__(self, conn, obs_dim, z_dim, act_dim=5, wiring="connectome", n_brains=1, alpha=0.5, seed=0, signed=None):
        super().__init__()
        M, S = build_mask(conn, wiring, n_brains, seed)
        self.N = M.shape[0]; self.alpha = alpha; self.wiring = wiring
        self.signed = (wiring in ("connectome", "shuffled", "random")) if signed is None else signed
        region = np.tile(conn["region"], n_brains)
        self.register_buffer("M", torch.from_numpy(M)); self.register_buffer("S", torch.from_numpy(S))
        self.register_buffer("in_mask", torch.from_numpy((region == "VIS").astype(np.float32)))
        self.register_buffer("act_mask", torch.from_numpy(np.isin(region, ["CX", "OUT"]).astype(np.float32)))
        indeg = torch.from_numpy(M.sum(1)).clamp(min=1)
        g = torch.Generator().manual_seed(seed)
        self.dense = wiring == "dense"
        if self.dense:
            self.W_raw = nn.Parameter(torch.randn(self.N, self.N, generator=g) / indeg.sqrt()[:, None] * 0.8)
        else:  # edge-list parametrisation: one weight per real synapse pair
            post_i, pre_i = np.nonzero(M)
            self.register_buffer("e_post", torch.from_numpy(post_i)); self.register_buffer("e_pre", torch.from_numpy(pre_i))
            self.register_buffer("e_sign", self.S[self.e_post, self.e_pre])
            w0 = torch.randn(len(post_i), generator=g) / indeg[self.e_post].sqrt()
            self.W_raw = nn.Parameter(w0.abs() if self.signed else w0 * 0.8)
        self.W_in = nn.Linear(obs_dim, self.N); self.W_act = nn.Linear(act_dim, self.N, bias=False)
        self.b = nn.Parameter(torch.zeros(self.N)); self.gain = nn.Parameter(torch.ones(self.N))
        self.W_out = nn.Linear(self.N, z_dim)
        self.act_dim = act_dim

    def W(self):
        if self.dense: return self.W_raw
        w = F.softplus(self.W_raw) * self.e_sign if self.signed else self.W_raw
        return torch.zeros(self.N, self.N, device=w.device, dtype=w.dtype).index_put((self.e_post, self.e_pre), w)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def init_state(self, B, device):
        return torch.zeros(B, self.N, device=device)

    def step(self, h, obs, act, visible, W=None):
        W = self.W() if W is None else W
        inp = self.W_in(obs) * self.in_mask * visible[:, None] + self.W_act(F.one_hot(act, self.act_dim).float()) * self.act_mask
        pre = h @ W.T + inp + self.b
        return (1 - self.alpha) * h + self.alpha * torch.tanh(self.gain * pre)

    def forward(self, obs, act, visible, h=None):
        """obs [B,T,D], act [B,T] (action taken after observing step t), visible [B,T] -> z_hat [B,T,z], H [B,T,N]"""
        B, T, _ = obs.shape; W = self.W()
        h = self.init_state(B, obs.device) if h is None else h
        hs = []
        for t in range(T):
            h = self.step(h, obs[:, t], act[:, t - 1] if t > 0 else torch.zeros_like(act[:, 0]), visible[:, t], W)
            hs.append(h)
        H = torch.stack(hs, 1)
        return self.W_out(H), H


class GRUBaseline(nn.Module):
    """Parameter-matched unconstrained recurrent baseline."""
    def __init__(self, obs_dim, z_dim, hidden, act_dim=5):
        super().__init__()
        self.cell = nn.GRUCell(obs_dim + act_dim + 1, hidden); self.W_out = nn.Linear(hidden, z_dim)
        self.N = hidden; self.act_dim = act_dim; self.wiring = "gru"; self.M = torch.zeros(1)

    def n_params(self): return sum(p.numel() for p in self.parameters())
    def init_state(self, B, device): return torch.zeros(B, self.N, device=device)
    def step(self, h, obs, act, visible, W=None):
        x = torch.cat([obs * visible[:, None], F.one_hot(act, self.act_dim).float(), visible[:, None]], -1)
        return self.cell(x, h)
    def forward(self, obs, act, visible, h=None):
        B, T, _ = obs.shape; h = self.init_state(B, obs.device) if h is None else h; hs = []
        for t in range(T):
            h = self.step(h, obs[:, t], act[:, t - 1] if t > 0 else torch.zeros_like(act[:, 0]), visible[:, t]); hs.append(h)
        H = torch.stack(hs, 1); return self.W_out(H), H
