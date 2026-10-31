"""Connectome-wired recurrent student ("FlyBrain") distilled from V-JEPA 2.1 latents.

h_{t+1} = (1-a) h_t + a * tanh( W h_t + W_in o_t * visible_t + W_act a_t + b )
z_hat_t = W_out h_t
W = M * S * softplus(W_raw)   (M: FlyWire adjacency mask, S: presynaptic neurotransmitter sign -> Dale's law)
Wiring variants keep neuron count, region assignment, I/O pathways and edge count fixed, only the recurrent wiring changes.

Compute: the recurrence is evaluated on the edge list (gather / scatter-add over E synapse pairs), never on a
materialised N x N matrix, so per-step cost is O(E + N * (obs + act + z)) and memory is O(E) (#11). `W()` still
returns the dense matrix for visualisation only. Checkpoints hold only the edge list, not the N x N mask.
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

WIRINGS = ("connectome", "connectome_nosign", "shuffled", "random", "dense")


def build_edges(conn, wiring, n_brains=1, seed=0):
    """Edge-list wiring: returns (N, post, pre, sign) with unique (post, pre) pairs sorted by (post, pre).

    The ordering equals np.nonzero(M) on the dense mask of the previous implementation, so W_raw in old
    checkpoints lines up with the new edge list.
    """
    rng = np.random.default_rng(seed)
    N = len(conn["region"]); pre, post, sign = conn["pre"].copy(), conn["post"].copy(), conn["sign"].copy()
    E = len(pre)
    if wiring == "shuffled":           # degree-preserving: in/out-degree multisets kept, pairing destroyed
        post = rng.permutation(post)
    elif wiring == "random":           # Erdos-Renyi with the same edge count and sign fractions
        pre = rng.integers(0, N, E); post = rng.integers(0, N, E); sign = rng.permutation(sign)
    elif wiring == "dense":
        post, pre = np.divmod(np.arange(N * N), N); sign = np.ones(N * N, np.int8)
    if wiring == "connectome_nosign": sign = np.ones_like(sign)
    pre = pre.astype(np.int64); post = post.astype(np.int64); sign = sign.astype(np.float32)
    if n_brains > 1:
        # multi-brain: K copies, coupled through the real contralateral (cross-hemisphere) edges brain i -> brain i+1
        side = conn["side"]; cross = (side[pre] != side[post]) & (side[pre] != "center") & (side[post] != "center")
        P, Q, Sg = [], [], []
        for k in range(n_brains):
            j = (k + 1) % n_brains
            P.append(k * N + post[~cross]); Q.append(k * N + pre[~cross]); Sg.append(sign[~cross])      # intra-brain
            P.append(j * N + post[cross]); Q.append(k * N + pre[cross]); Sg.append(sign[cross])         # moved to partner
        post, pre, sign = np.concatenate(P), np.concatenate(Q), np.concatenate(Sg); N = n_brains * N
    # unique (post, pre) pairs, last duplicate's sign wins (matches dense fancy-index assignment semantics)
    key = post * N + pre
    _, last = np.unique(key[::-1], return_index=True); keep = len(key) - 1 - last   # ascending key = (post, pre) order
    return N, post[keep], pre[keep], sign[keep]


def build_mask(conn, wiring, n_brains=1, seed=0):
    """Dense (N, N) mask and sign matrices. Visualisation / tests only: O(N^2) memory."""
    N, post, pre, sign = build_edges(conn, wiring, n_brains, seed)
    M = np.zeros((N, N), np.float32); S = np.ones((N, N), np.float32)
    M[post, pre] = 1.0; S[post, pre] = sign
    return M, S


class FlyBrain(nn.Module):
    def __init__(self, conn, obs_dim, z_dim, act_dim=5, wiring="connectome", n_brains=1, alpha=0.5, seed=0, signed=None):
        super().__init__()
        N, post, pre, sign = build_edges(conn, wiring, n_brains, seed)
        self.N = N; self.alpha = alpha; self.wiring = wiring; self.n_edges = len(post)
        self.signed = (wiring in ("connectome", "shuffled", "random")) if signed is None else signed
        region = np.tile(conn["region"], n_brains)
        self.register_buffer("in_mask", torch.from_numpy((region == "VIS").astype(np.float32)))
        self.register_buffer("act_mask", torch.from_numpy(np.isin(region, ["CX", "OUT"]).astype(np.float32)))
        indeg = torch.from_numpy(np.bincount(post, minlength=N).astype(np.float32)).clamp(min=1)
        g = torch.Generator().manual_seed(seed)
        self.dense = wiring == "dense"
        if self.dense:
            self.W_raw = nn.Parameter(torch.randn(N, N, generator=g) / indeg.sqrt()[:, None] * 0.8)
        else:  # edge-list parametrisation: one weight per real synapse pair
            self.register_buffer("e_post", torch.from_numpy(post)); self.register_buffer("e_pre", torch.from_numpy(pre))
            self.register_buffer("e_sign", torch.from_numpy(sign))
            w0 = torch.randn(len(post), generator=g) / indeg[self.e_post].sqrt()
            self.W_raw = nn.Parameter(w0.abs() if self.signed else w0 * 0.8)
        self.W_in = nn.Linear(obs_dim, N); self.W_act = nn.Linear(act_dim, N, bias=False)
        self.b = nn.Parameter(torch.zeros(N)); self.gain = nn.Parameter(torch.ones(N))
        self.W_out = nn.Linear(N, z_dim)
        self.act_dim = act_dim

    # ---------------- weights ----------------
    def edge_weights(self):
        """Effective synaptic weights, one per edge [E] (dense wiring: the full [N, N] matrix)."""
        if self.dense: return self.W_raw
        return F.softplus(self.W_raw) * self.e_sign if self.signed else self.W_raw

    def W(self):
        """Dense [N, N] recurrent matrix. Visualisation only; never used inside the recurrence."""
        if self.dense: return self.W_raw
        w = self.edge_weights()
        return torch.zeros(self.N, self.N, device=w.device, dtype=w.dtype).index_put((self.e_post, self.e_pre), w)

    def recur(self, h, w):
        """W h for a batch of states: h [B, N], w = edge_weights() -> [B, N]. Gather/scatter over the edge list."""
        if w.dim() == 2: return h @ w.T          # dense wiring or a legacy dense matrix
        msg = h[:, self.e_pre] * w               # [B, E]
        return torch.zeros_like(h).index_add_(1, self.e_post, msg)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_step(self, obs_dim=None):
        """Multiply-adds per step for one sample (recurrence + input/action/readout projections)."""
        obs_dim = self.W_in.in_features if obs_dim is None else obs_dim
        rec = self.N * self.N if self.dense else self.n_edges
        return 2 * (rec + obs_dim * self.N + self.act_dim * self.N + self.N * self.W_out.out_features)

    # ---------------- dynamics ----------------
    def init_state(self, B, device):
        return torch.zeros(B, self.N, device=device)

    def step(self, h, obs, act, visible, W=None):
        """One recurrent step. `W` may be edge_weights() (recommended, compute once per rollout) or None."""
        w = self.edge_weights() if W is None else W
        inp = self.W_in(obs) * self.in_mask * visible[:, None] + self.W_act(F.one_hot(act, self.act_dim).float()) * self.act_mask
        pre = self.recur(h, w) + inp + self.b
        return (1 - self.alpha) * h + self.alpha * torch.tanh(self.gain * pre)

    def forward(self, obs, act, visible, h=None):
        """obs [B,T,D], act [B,T] (action taken after observing step t), visible [B,T] -> z_hat [B,T,z], H [B,T,N]"""
        B, T, _ = obs.shape; w = self.edge_weights()
        h = self.init_state(B, obs.device) if h is None else h
        hs = []
        for t in range(T):
            h = self.step(h, obs[:, t], act[:, t - 1] if t > 0 else torch.zeros_like(act[:, 0]), visible[:, t], w)
            hs.append(h)
        H = torch.stack(hs, 1)
        return self.W_out(H), H

    # ---------------- checkpoints ----------------
    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Accept checkpoints from the dense-mask implementation (which stored N x N `M` and `S` buffers)."""
        state_dict = {k: v for k, v in state_dict.items() if k not in ("M", "S")}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class GRUBaseline(nn.Module):
    """Parameter-matched unconstrained recurrent baseline."""
    def __init__(self, obs_dim, z_dim, hidden, act_dim=5):
        super().__init__()
        self.cell = nn.GRUCell(obs_dim + act_dim + 1, hidden); self.W_out = nn.Linear(hidden, z_dim)
        self.N = hidden; self.act_dim = act_dim; self.wiring = "gru"; self.M = torch.zeros(1)

    def n_params(self): return sum(p.numel() for p in self.parameters())
    def flops_per_step(self, obs_dim=None):
        return 2 * (3 * self.N * (self.cell.input_size + self.N) + self.N * self.W_out.out_features)
    def edge_weights(self): return None
    def init_state(self, B, device): return torch.zeros(B, self.N, device=device)
    def step(self, h, obs, act, visible, W=None):
        x = torch.cat([obs * visible[:, None], F.one_hot(act, self.act_dim).float(), visible[:, None]], -1)
        return self.cell(x, h)
    def forward(self, obs, act, visible, h=None):
        B, T, _ = obs.shape; h = self.init_state(B, obs.device) if h is None else h; hs = []
        for t in range(T):
            h = self.step(h, obs[:, t], act[:, t - 1] if t > 0 else torch.zeros_like(act[:, 0]), visible[:, t]); hs.append(h)
        H = torch.stack(hs, 1); return self.W_out(H), H
