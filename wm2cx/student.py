"""Connectome-wired recurrent student ("FlyBrain") distilled from V-JEPA 2.1 latents.

h_{t+1} = (1-a) h_t + a * tanh( W h_t + W_in o_t * visible_t + W_act a_t + w_vis (1-visible_t) + b )
z_hat_t = W_out h_t
W is stored as an edge list (one weight per synapse pair), W[post,pre] = sign[pre] * softplus(w_raw)  (Dale's law).
The recurrence is evaluated as a gather / index_add over the edge list, never as a dense N x N matrix (issue #11).
W_in only feeds VIS neurons (issue #3); GRU/FlyBrain both receive the blackout flag (issue #4).

Wiring variants keep neuron count, region assignment, I/O pathways and the exact edge count fixed;
only the recurrent wiring changes (issues #1, #2):
  connectome          real FlyWire edges, Dale-consistent signs
  connectome_nosign   real edges, all signs +1 (raw-weight parametrisation, legacy)
  connectome_randsign real edges, softplus parametrisation, Dale-consistent signs but neurotransmitter identity
                      permuted across neurons (controls for sign *pattern* with identical optimisation)
  shuffled            Maslov-Sneppen switch rewire: in/out-degrees exact, block structure destroyed
  block_shuffled      switch rewire inside each (pre-region, post-region) block: degrees and block structure exact
  random              Erdos-Renyi simple digraph with the same edge count, Dale-consistent signs
  dense               full N x N matrix (unconstrained control)
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from wm2cx.rewire import maslov_sneppen, block_rewire, erdos_renyi

WIRINGS = ("connectome", "connectome_nosign", "connectome_randsign", "shuffled", "block_shuffled", "random", "dense")


def build_edges(conn, wiring, n_brains=1, seed=0):
    """-> (pre, post, sign) int arrays over n_brains*N neurons. Signs are per presynaptic neuron (Dale's law)."""
    rng = np.random.default_rng(seed)
    N = len(conn["region"]); pre, post = conn["pre"].copy(), conn["post"].copy(); E = len(pre)
    if "nt" in conn: nt = conn["nt"].astype(np.int8)
    else: nt = np.ones(N, np.int8); nt[conn["pre"]] = conn["sign"]  # per-neuron sign recovered from the edge signs
    neg_frac = float((nt[pre] < 0).mean())  # real inhibitory *edge* fraction (0.50; the neuron fraction is 0.20)
    if wiring == "shuffled": pre, post = maslov_sneppen(pre, post, seed)
    elif wiring == "block_shuffled": pre, post = block_rewire(pre, post, conn["region"], seed)
    elif wiring == "random": pre, post = erdos_renyi(N, E, seed); nt = dale_signs(pre, N, neg_frac, rng)
    elif wiring == "connectome_randsign": nt = dale_signs(pre, N, neg_frac, rng)
    elif wiring not in ("connectome", "connectome_nosign", "dense"): raise ValueError(wiring)
    sign = nt[pre].astype(np.float32)
    if wiring == "connectome_nosign": sign[:] = 1.0
    if n_brains > 1:  # K copies, coupled through the real contralateral edges brain k -> brain k+1
        side = conn["side"]; cross = (side[pre] != side[post]) & (side[pre] != "center") & (side[post] != "center")
        P, Q, S = [], [], []
        for k in range(n_brains):
            j = (k + 1) % n_brains
            P.append(pre + k * N); Q.append(np.where(cross, post + j * N, post + k * N)); S.append(sign)
        pre, post, sign = np.concatenate(P), np.concatenate(Q), np.concatenate(S)
    return pre.astype(np.int64), post.astype(np.int64), sign


def dale_signs(pre, N, neg_frac, rng):
    """Random per-neuron signs (Dale's law) chosen so the inhibitory *edge* fraction matches neg_frac:
    neurons are visited in random order and made inhibitory until their summed out-degree reaches the target."""
    outdeg = np.bincount(pre, minlength=N); order = rng.permutation(N); nt = np.ones(N, np.int8)
    cum = np.cumsum(outdeg[order]); k = int(np.searchsorted(cum, neg_frac * len(pre))) + 1
    nt[order[:k]] = -1
    return nt


def build_mask(conn, wiring, n_brains=1, seed=0):
    """Dense (M, S) view of build_edges, for plotting / analysis only."""
    pre, post, sign = build_edges(conn, wiring, n_brains, seed); N = n_brains * len(conn["region"])
    M = np.zeros((N, N), np.float32); S = np.ones((N, N), np.float32)
    if wiring == "dense": M[:] = 1.0
    else: M[post, pre] = 1.0; S[post, pre] = sign
    return M, S


class FlyBrain(nn.Module):
    def __init__(self, conn, obs_dim, z_dim, act_dim=5, wiring="connectome", n_brains=1, alpha=0.5, seed=0,
                 signed=None, edges=None, vis_flag=True, in_rank=0):
        super().__init__()
        N0 = len(conn["region"]); self.N = n_brains * N0; self.alpha = alpha; self.wiring = wiring; self.act_dim = act_dim
        self.signed = (wiring not in ("connectome_nosign", "dense")) if signed is None else signed
        region = np.tile(conn["region"], n_brains)
        vis_idx = np.nonzero(region == "VIS")[0]
        self.register_buffer("vis_idx", torch.from_numpy(vis_idx))
        self.register_buffer("act_mask", torch.from_numpy(np.isin(region, ["CX", "OUT"]).astype(np.float32)))
        g = torch.Generator().manual_seed(seed)
        self.dense = wiring == "dense"
        if self.dense:
            indeg = torch.full((self.N,), float(self.N))
            self.W_raw = nn.Parameter(torch.randn(self.N, self.N, generator=g) / indeg.sqrt()[:, None] * 0.8)
            self.n_edges = self.N * self.N
        else:
            pre, post, sign = build_edges(conn, wiring, n_brains, seed) if edges is None else edges
            self.register_buffer("e_pre", torch.as_tensor(pre, dtype=torch.long)); self.register_buffer("e_post", torch.as_tensor(post, dtype=torch.long))
            self.register_buffer("e_sign", torch.as_tensor(sign, dtype=torch.float32))
            indeg = torch.bincount(self.e_post, minlength=self.N).clamp(min=1).float()
            w0 = torch.randn(len(pre), generator=g) / indeg[self.e_post].sqrt()
            self.W_raw = nn.Parameter(w0.abs() if self.signed else w0 * 0.8)
            self.n_edges = len(pre)
        nv = len(vis_idx)
        self.W_in = nn.Linear(obs_dim, nv) if in_rank <= 0 else nn.Sequential(nn.Linear(obs_dim, in_rank, bias=False), nn.Linear(in_rank, nv))
        self.W_act = nn.Linear(act_dim, self.N, bias=False)
        self.w_vis = nn.Parameter(torch.zeros(self.N)) if vis_flag else None
        self.b = nn.Parameter(torch.zeros(self.N)); self.gain = nn.Parameter(torch.ones(self.N))
        self.W_out = nn.Linear(self.N, z_dim)

    # ---- weights ----
    def weights(self):
        """Edge weights [E] (or dense [N,N]) with signs applied; computed once per forward."""
        if self.dense: return self.W_raw
        return F.softplus(self.W_raw) * self.e_sign if self.signed else self.W_raw

    def W(self):
        """Dense N x N matrix (analysis / plotting only; never used in the recurrence)."""
        w = self.weights()
        if self.dense: return w
        return torch.zeros(self.N, self.N, device=w.device, dtype=w.dtype).index_put((self.e_post, self.e_pre), w)

    def recur(self, h, w):
        if self.dense: return h @ w.T
        m = h[:, self.e_pre] * w  # [B,E]
        return torch.zeros_like(h).index_add_(1, self.e_post, m)

    # ---- accounting ----
    def n_params(self): return sum(p.numel() for p in self.parameters())

    def param_breakdown(self):
        rec = self.W_raw.numel(); inp = sum(p.numel() for p in self.W_in.parameters()) + self.W_act.weight.numel()
        out = sum(p.numel() for p in self.W_out.parameters())
        return dict(recurrent=rec, input=inp, output=out, other=self.n_params() - rec - inp - out, total=self.n_params(), edges=self.n_edges)

    def init_state(self, B, device): return torch.zeros(B, self.N, device=device)

    # ---- dynamics ----
    def step(self, h, obs, act, visible, W=None):
        w = self.weights() if W is None else W
        inp = torch.zeros_like(h).index_add_(1, self.vis_idx, self.W_in(obs) * visible[:, None])
        inp = inp + self.W_act(F.one_hot(act, self.act_dim).float()) * self.act_mask
        if self.w_vis is not None: inp = inp + self.w_vis * (1 - visible)[:, None]
        pre = self.recur(h, w) + inp + self.b
        return (1 - self.alpha) * h + self.alpha * torch.tanh(self.gain * pre)

    def forward(self, obs, act, visible, h=None):
        """obs [B,T,D], act [B,T] (action taken after observing step t), visible [B,T] -> z_hat [B,T,z], H [B,T,N]"""
        B, T, _ = obs.shape; w = self.weights()
        h = self.init_state(B, obs.device) if h is None else h
        hs = []
        for t in range(T):
            h = self.step(h, obs[:, t], act[:, t - 1] if t > 0 else torch.zeros_like(act[:, 0]), visible[:, t], w)
            hs.append(h)
        H = torch.stack(hs, 1)
        return self.W_out(H), H


def gru_params(hidden, obs_dim, z_dim, act_dim=5, vis_flag=True):
    i = obs_dim + act_dim + int(vis_flag)
    return 3 * (i * hidden + hidden * hidden) + 6 * hidden + hidden * z_dim + z_dim


def gru_hidden_for(target_params, obs_dim, z_dim, act_dim=5, vis_flag=True):
    """Largest hidden size whose GRUBaseline has <= target_params parameters (issue #4)."""
    i = obs_dim + act_dim + int(vis_flag); a, b, c = 3, 3 * i + 6 + z_dim, z_dim - target_params
    return int((-b + (b * b - 4 * a * c) ** 0.5) / (2 * a))


class GRUBaseline(nn.Module):
    """Parameter-matched unconstrained recurrent baseline."""
    def __init__(self, obs_dim, z_dim, hidden, act_dim=5, vis_flag=True):
        super().__init__()
        self.vis_flag = vis_flag
        self.cell = nn.GRUCell(obs_dim + act_dim + int(vis_flag), hidden); self.W_out = nn.Linear(hidden, z_dim)
        self.N = hidden; self.act_dim = act_dim; self.wiring = "gru"; self.n_edges = hidden * hidden

    def n_params(self): return sum(p.numel() for p in self.parameters())
    def param_breakdown(self):
        h = self.N; rec = 3 * h * h; inp = self.cell.weight_ih.numel(); out = sum(p.numel() for p in self.W_out.parameters())
        return dict(recurrent=rec, input=inp, output=out, other=self.n_params() - rec - inp - out, total=self.n_params(), edges=self.n_edges)
    def weights(self): return None
    def init_state(self, B, device): return torch.zeros(B, self.N, device=device)
    def step(self, h, obs, act, visible, W=None):
        parts = [obs * visible[:, None], F.one_hot(act, self.act_dim).float()] + ([visible[:, None]] if self.vis_flag else [])
        return self.cell(torch.cat(parts, -1), h)
    def forward(self, obs, act, visible, h=None):
        B, T, _ = obs.shape; h = self.init_state(B, obs.device) if h is None else h; hs = []
        for t in range(T):
            h = self.step(h, obs[:, t], act[:, t - 1] if t > 0 else torch.zeros_like(act[:, 0]), visible[:, t]); hs.append(h)
        H = torch.stack(hs, 1); return self.W_out(H), H
