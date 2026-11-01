"""Connectome-wired recurrent student ("FlyBrain") distilled from V-JEPA 2.1 latents.

h_{t+1} = (1-a) h_t + a * tanh( W h_t + W_in o_t * visible_t + W_act a_t + w_vis (1 - visible_t) + b )
z_hat_t = W_out h_t
W is an edge list, one weight per synapse pair: W[post, pre] = sign[pre] * softplus(w_raw)   (Dale's law).
The recurrence is a gather / index_add over the edge list, never a materialised N x N matrix (#11); `W()` returns
the dense matrix for visualisation only. W_in feeds the VIS neurons only (#3); the blackout flag w_vis is fed to
every unit of both the FlyBrain and the GRU (#4).

Wiring variants keep neuron count, region labels, I/O pathways and the exact edge count fixed; every control is a
simple digraph (no self-loops, no duplicate pairs) with one sign per presynaptic neuron (Dale) and the inhibitory
*edge* fraction matched to the real graph (#1, #2):
  connectome           real FlyWire edges, signs from the predicted neurotransmitter
  connectome_nosign    real edges, all +1, raw-weight parametrisation (legacy unsigned control)
  connectome_randsign  real edges, softplus parametrisation, Dale-consistent signs with neurotransmitter identity
                       permuted across neurons (isolates the sign *pattern* under identical optimisation)
  shuffled             Maslov-Sneppen switch rewire: every in/out-degree exact, region block structure destroyed
  block_shuffled       switch rewire inside each (pre-region, post-region) block: degrees and block counts exact
  random               Erdos-Renyi simple digraph with the same edge count
  dense                full N x N matrix (unconstrained control)
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from wm2cx.rewire import maslov_sneppen, block_rewire, erdos_renyi

WIRINGS = ("connectome", "connectome_nosign", "connectome_randsign", "shuffled", "block_shuffled", "random", "dense")


def dale_signs(pre, N, neg_frac, rng):
    """Random per-neuron signs (Dale's law) such that the inhibitory *edge* fraction matches neg_frac: neurons are
    visited in random order and made inhibitory until their summed out-degree reaches the target."""
    outdeg = np.bincount(pre, minlength=N); order = rng.permutation(N); nt = np.ones(N, np.int8)
    cum = np.cumsum(outdeg[order]); k = int(np.searchsorted(cum, neg_frac * len(pre))) + 1
    nt[order[:k]] = -1
    return nt


def build_edges(conn, wiring, n_brains=1, seed=0):
    """Edge-list wiring -> (N, post, pre, sign), unique (post, pre) pairs sorted by (post, pre). `dense` returns the
    full N x N pattern lazily (post, pre = None) because FlyBrain never needs its edge list."""
    rng = np.random.default_rng(seed)
    N = len(conn["region"]); pre, post = conn["pre"].astype(np.int64), conn["post"].astype(np.int64); E = len(pre)
    if "nt" in conn: nt = conn["nt"].astype(np.int8)
    else: nt = np.ones(N, np.int8); nt[pre] = conn["sign"]                 # per-neuron sign recovered from the edge signs
    neg_frac = float((nt[pre] < 0).mean())                                 # inhibitory edge fraction (0.50 real; neuron fraction 0.20)
    if wiring == "dense":
        return n_brains * N, None, None, None
    if wiring == "shuffled": pre, post = maslov_sneppen(pre, post, seed)
    elif wiring == "block_shuffled": pre, post = block_rewire(pre, post, conn["region"], seed)
    elif wiring == "random": pre, post = erdos_renyi(N, E, seed); nt = dale_signs(pre, N, neg_frac, rng)
    elif wiring == "connectome_randsign": nt = dale_signs(pre, N, neg_frac, rng)
    elif wiring not in ("connectome", "connectome_nosign"): raise ValueError(wiring)
    sign = nt[pre].astype(np.float32)
    if wiring == "connectome_nosign": sign[:] = 1.0
    pre = pre.astype(np.int64); post = post.astype(np.int64)
    if n_brains > 1:   # K copies, coupled through the real contralateral (cross-hemisphere) edges brain k -> brain k+1
        side = conn["side"]; cross = (side[pre] != side[post]) & (side[pre] != "center") & (side[post] != "center")
        P, Q, Sg = [], [], []
        for k in range(n_brains):
            j = (k + 1) % n_brains
            P.append(k * N + post[~cross]); Q.append(k * N + pre[~cross]); Sg.append(sign[~cross])      # intra-brain
            P.append(j * N + post[cross]); Q.append(k * N + pre[cross]); Sg.append(sign[cross])         # moved to partner
        post, pre, sign = np.concatenate(P), np.concatenate(Q), np.concatenate(Sg); N = n_brains * N
    key = post * N + pre                                                   # unique pairs, last duplicate's sign wins
    _, last = np.unique(key[::-1], return_index=True); keep = len(key) - 1 - last
    return N, post[keep], pre[keep], sign[keep]


def build_mask(conn, wiring, n_brains=1, seed=0):
    """Dense (N, N) mask and sign matrices. Visualisation / tests only: O(N^2) memory."""
    N, post, pre, sign = build_edges(conn, wiring, n_brains, seed)
    M = np.zeros((N, N), np.float32); S = np.ones((N, N), np.float32)
    if post is None: M[:] = 1.0
    else: M[post, pre] = 1.0; S[post, pre] = sign
    return M, S


def block_counts(post, pre, region):
    """{(pre_region>post_region): n_edges} for reporting the macro-architecture of a wiring."""
    u, c = np.unique(np.char.add(np.char.add(region[pre].astype(str), ">"), region[post].astype(str)), return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))


class FlyBrain(nn.Module):
    def __init__(self, conn, obs_dim, z_dim, act_dim=5, wiring="connectome", n_brains=1, alpha=0.5, seed=0, signed=None,
                 vis_flag=True, in_rank=0, edges=None):
        super().__init__()
        N, post, pre, sign = build_edges(conn, wiring, n_brains, seed) if edges is None else edges
        self.N = N; self.alpha = alpha; self.wiring = wiring; self.act_dim = act_dim
        self.signed = (wiring not in ("connectome_nosign", "dense")) if signed is None else signed
        region = np.tile(conn["region"], n_brains)
        vis_idx = np.nonzero(region == "VIS")[0]
        self.register_buffer("vis_idx", torch.from_numpy(vis_idx.astype(np.int64)))
        self.register_buffer("act_mask", torch.from_numpy(np.isin(region, ["CX", "OUT"]).astype(np.float32)))
        g = torch.Generator().manual_seed(seed)
        self.dense = wiring == "dense"
        if self.dense:
            self.W_raw = nn.Parameter(torch.randn(N, N, generator=g) / float(N) ** 0.5 * 0.8); self.n_edges = N * N
        else:  # edge-list parametrisation: one weight per synapse pair
            self.register_buffer("e_post", torch.from_numpy(post)); self.register_buffer("e_pre", torch.from_numpy(pre))
            self.register_buffer("e_sign", torch.from_numpy(sign))
            indeg = torch.bincount(self.e_post, minlength=N).clamp(min=1).float()
            w0 = torch.randn(len(post), generator=g) / indeg[self.e_post].sqrt()
            self.W_raw = nn.Parameter(w0.abs() if self.signed else w0 * 0.8); self.n_edges = len(post)
        nv = len(vis_idx); self.in_rank = in_rank
        self.W_in = nn.Linear(obs_dim, nv) if in_rank <= 0 else nn.Sequential(nn.Linear(obs_dim, in_rank, bias=False), nn.Linear(in_rank, nv))
        self.W_act = nn.Linear(act_dim, N, bias=False)
        self.w_vis = nn.Parameter(torch.zeros(N)) if vis_flag else None
        self.b = nn.Parameter(torch.zeros(N)); self.gain = nn.Parameter(torch.ones(N))
        self.W_out = nn.Linear(N, z_dim)

    # ---------------- weights ----------------
    def edge_weights(self):
        """Effective synaptic weights, one per edge [E] (dense wiring: the full [N, N] matrix)."""
        if self.dense: return self.W_raw
        return F.softplus(self.W_raw) * self.e_sign if self.signed else self.W_raw
    weights = edge_weights

    def W(self):
        """Dense [N, N] recurrent matrix. Visualisation only; never used inside the recurrence."""
        if self.dense: return self.W_raw
        w = self.edge_weights()
        return torch.zeros(self.N, self.N, device=w.device, dtype=w.dtype).index_put((self.e_post, self.e_pre), w)

    def recur(self, h, w):
        """W h for a batch of states: h [B, N], w = edge_weights() -> [B, N]. Gather/scatter over the edge list."""
        if w.dim() == 2: return h @ w.T
        msg = h[:, self.e_pre] * w
        return torch.zeros_like(h).index_add_(1, self.e_post, msg)

    # ---------------- accounting (#3) ----------------
    def n_params(self): return sum(p.numel() for p in self.parameters())

    def param_breakdown(self):
        rec = self.W_raw.numel(); inp = sum(p.numel() for p in self.W_in.parameters()) + self.W_act.weight.numel()
        out = sum(p.numel() for p in self.W_out.parameters())
        return dict(recurrent=rec, input=inp, output=out, other=self.n_params() - rec - inp - out, total=self.n_params(), edges=self.n_edges)

    def flops_per_step(self, obs_dim=None):
        """Multiply-adds per step for one sample (recurrence + input/action/readout projections)."""
        lin = self.W_in if self.in_rank <= 0 else self.W_in[0]; obs_dim = lin.in_features if obs_dim is None else obs_dim
        nv = len(self.vis_idx); inp = obs_dim * nv if self.in_rank <= 0 else self.in_rank * (obs_dim + nv)
        return 2 * (self.n_edges + inp + self.act_dim * self.N + self.N * self.W_out.out_features)

    # ---------------- dynamics ----------------
    def init_state(self, B, device): return torch.zeros(B, self.N, device=device)

    def step(self, h, obs, act, visible, W=None):
        """One recurrent step. `W` may be edge_weights() (recommended, compute once per rollout) or None."""
        w = self.edge_weights() if W is None else W
        inp = torch.zeros_like(h).index_add_(1, self.vis_idx, self.W_in(obs) * visible[:, None])
        inp = inp + self.W_act(F.one_hot(act, self.act_dim).float()) * self.act_mask
        if self.w_vis is not None: inp = inp + self.w_vis * (1 - visible)[:, None]
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
        """Accept checkpoints from earlier implementations: dense N x N `M`/`S` buffers, an `in_mask` buffer, a full-N
        W_in (only the VIS rows were ever non-zero), and models trained without the blackout flag."""
        s = {k: v for k, v in state_dict.items() if k not in ("M", "S", "in_mask")}
        if "W_in.weight" in s and s["W_in.weight"].shape[0] == self.N and self.N != len(self.vis_idx):
            idx = self.vis_idx.cpu(); s["W_in.weight"] = s["W_in.weight"][idx]; s["W_in.bias"] = s["W_in.bias"][idx]
        if "vis_idx" not in s: s["vis_idx"] = self.vis_idx.cpu()
        if self.w_vis is not None and "w_vis" not in s: s["w_vis"] = torch.zeros(self.N)
        if self.w_vis is None: s.pop("w_vis", None)
        for k in ("e_pre", "e_post"):
            if k in s: s[k] = s[k].long()
        return super().load_state_dict(s, strict=strict, assign=assign)


def gru_params(hidden, obs_dim, z_dim, act_dim=5, vis_flag=True):
    i = obs_dim + act_dim + int(vis_flag)
    return 3 * (i * hidden + hidden * hidden) + 6 * hidden + hidden * z_dim + z_dim


def gru_hidden_for(target_params, obs_dim, z_dim, act_dim=5, vis_flag=True):
    """Largest hidden size whose GRUBaseline has <= target_params parameters (#4)."""
    i = obs_dim + act_dim + int(vis_flag); a, b, c = 3, 3 * i + 6 + z_dim, z_dim - target_params
    return int((-b + (b * b - 4 * a * c) ** 0.5) / (2 * a))


class GRUBaseline(nn.Module):
    """Parameter-matched unconstrained recurrent baseline (same inputs as the FlyBrain: masked obs, action, blackout flag)."""
    def __init__(self, obs_dim, z_dim, hidden, act_dim=5, vis_flag=True):
        super().__init__()
        self.vis_flag = vis_flag
        self.cell = nn.GRUCell(obs_dim + act_dim + int(vis_flag), hidden); self.W_out = nn.Linear(hidden, z_dim)
        self.N = hidden; self.act_dim = act_dim; self.wiring = "gru"; self.n_edges = hidden * hidden; self.dense = False

    def n_params(self): return sum(p.numel() for p in self.parameters())
    def param_breakdown(self):
        h = self.N; rec = 3 * h * h; inp = self.cell.weight_ih.numel(); out = sum(p.numel() for p in self.W_out.parameters())
        return dict(recurrent=rec, input=inp, output=out, other=self.n_params() - rec - inp - out, total=self.n_params(), edges=self.n_edges)
    def flops_per_step(self, obs_dim=None):
        return 2 * (3 * self.N * (self.cell.input_size + self.N) + self.N * self.W_out.out_features)
    def edge_weights(self): return None
    weights = edge_weights
    def init_state(self, B, device): return torch.zeros(B, self.N, device=device)
    def step(self, h, obs, act, visible, W=None):
        parts = [obs * visible[:, None], F.one_hot(act, self.act_dim).float()] + ([visible[:, None]] if self.vis_flag else [])
        return self.cell(torch.cat(parts, -1), h)
    def forward(self, obs, act, visible, h=None):
        B, T, _ = obs.shape; h = self.init_state(B, obs.device) if h is None else h; hs = []
        for t in range(T):
            h = self.step(h, obs[:, t], act[:, t - 1] if t > 0 else torch.zeros_like(act[:, 0]), visible[:, t]); hs.append(h)
        H = torch.stack(hs, 1); return self.W_out(H), H
