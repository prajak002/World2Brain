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


class AttnInput(nn.Module):
    """Cross-attention input encoder replacing the linear W_in: every VIS neuron owns a learned query that attends over
    patch tokens of the fly-eye image and reads one scalar drive. About 80 k parameters against 1.8 M for the linear map,
    so the shared input layer stops dominating the parameter budget (#3) and VIS neurons can read spatial structure."""
    def __init__(self, n_out, eh=8, ew=32, ch=3, patch=2, d=32):
        super().__init__()
        self.eh, self.ew, self.ch, self.patch, self.d = eh, ew, ch, patch, d
        self.n_tok = (eh // patch) * (ew // patch); self.pd = patch * patch * ch; self.in_features = eh * ew * ch
        self.embed = nn.Linear(self.pd, d); self.pos = nn.Parameter(torch.randn(self.n_tok, d) * 0.02)
        self.q = nn.Parameter(torch.randn(n_out, d) * 0.02); self.k = nn.Linear(d, d, bias=False); self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, 1)

    def forward(self, obs):                                                # obs [B, eh*ew*ch] laid out as [eh, ew, ch]
        B = obs.shape[0]; p = self.patch
        x = obs.view(B, self.eh // p, p, self.ew // p, p, self.ch).permute(0, 1, 3, 2, 4, 5).reshape(B, self.n_tok, self.pd)
        t = self.embed(x) + self.pos; k, v = self.k(t), self.v(t)
        att = torch.softmax(torch.einsum("nd,btd->bnt", self.q, k) / self.d ** 0.5, -1)
        return self.out(torch.einsum("bnt,btd->bnd", att, v)).squeeze(-1)

    def flops(self):
        return self.n_tok * self.pd * self.d + 2 * self.n_tok * self.d * self.d + 2 * self.q.shape[0] * self.n_tok * self.d + self.q.shape[0] * self.d


class FlyBrain(nn.Module):
    def __init__(self, conn, obs_dim, z_dim, act_dim=5, wiring="connectome", n_brains=1, alpha=0.5, seed=0, signed=None,
                 vis_flag=True, in_rank=0, edges=None, learn_alpha=False, gate=False, in_mode="linear", hybrid=False):
        """Neuron-model ablation (wiring fixed): learn_alpha = per-neuron learnable time constant alpha_i = sigmoid(a_i);
        gate = GRU-style update gate g_i = sigmoid(gain_g (W_g h + W_in_g o + b_g)) carried on a second (unsigned) weight per
        connectome edge and a second VIS input map, so h <- (1-g) h + g tanh(...). Both keep the edge list fixed.
        in_mode = "attn": cross-attention input encoder (AttnInput) instead of the linear W_in.
        hybrid = True: one shared attention layer over cell-type-pooled activity every step (global routing the sparse
        graph lacks), added to the recurrent drive; the edge list is unchanged."""
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
        nv = len(vis_idx); self.in_rank = in_rank; self.in_mode = in_mode
        if in_mode == "attn": self.W_in = AttnInput(nv)
        else: self.W_in = nn.Linear(obs_dim, nv) if in_rank <= 0 else nn.Sequential(nn.Linear(obs_dim, in_rank, bias=False), nn.Linear(in_rank, nv))
        self.hybrid = hybrid
        if hybrid:
            ct = np.tile(conn["cell_type"], n_brains); _, gid = np.unique(ct, return_inverse=True); self.G = int(gid.max()) + 1; d = 32
            self.register_buffer("grp", torch.from_numpy(gid.astype(np.int64)))
            self.register_buffer("grp_n", torch.bincount(self.grp, minlength=self.G).float().clamp(min=1))
            self.h_up = nn.Parameter(torch.randn(d) * 0.1); self.h_emb = nn.Parameter(torch.randn(self.G, d) * 0.02)
            self.h_ln = nn.LayerNorm(d); self.h_attn = nn.MultiheadAttention(d, 4, batch_first=True)
            self.h_down = nn.Linear(d, 1); self.h_gain = nn.Parameter(torch.zeros(1))
        self.W_act = nn.Linear(act_dim, N, bias=False)
        self.w_vis = nn.Parameter(torch.zeros(N)) if vis_flag else None
        self.b = nn.Parameter(torch.zeros(N)); self.gain = nn.Parameter(torch.ones(N))
        self.W_out = nn.Linear(N, z_dim)
        self.learn_alpha = learn_alpha; self.gate = gate
        if learn_alpha: self.alpha_logit = nn.Parameter(torch.full((N,), float(np.log(alpha / (1 - alpha)))))
        if gate:
            if self.dense: self.Wg_raw = nn.Parameter(torch.randn(N, N, generator=g) / float(N) ** 0.5 * 0.8)
            else: self.Wg_raw = nn.Parameter(torch.randn(self.n_edges, generator=g) / indeg[self.e_post].sqrt() * 0.8)
            self.W_in_g = nn.Linear(obs_dim, nv); self.b_g = nn.Parameter(torch.zeros(N)); self.gain_g = nn.Parameter(torch.ones(N))

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
        rec = self.W_raw.numel() + (self.Wg_raw.numel() if self.gate else 0)
        inp = sum(p.numel() for p in self.W_in.parameters()) + self.W_act.weight.numel() + (sum(p.numel() for p in self.W_in_g.parameters()) if self.gate else 0)
        out = sum(p.numel() for p in self.W_out.parameters())
        return dict(recurrent=rec, input=inp, output=out, other=self.n_params() - rec - inp - out, total=self.n_params(), edges=self.n_edges)

    def flops_per_step(self, obs_dim=None):
        """Multiply-adds per step for one sample (recurrence + input/action/readout projections)."""
        nv = len(self.vis_idx)
        if self.in_mode == "attn": inp = self.W_in.flops()
        else:
            lin = self.W_in if self.in_rank <= 0 else self.W_in[0]; obs_dim = lin.in_features if obs_dim is None else obs_dim
            inp = obs_dim * nv if self.in_rank <= 0 else self.in_rank * (obs_dim + nv)
        hyb = (self.N + self.G * (32 + 4 * 32 * 32) + self.G * self.G * 32) if self.hybrid else 0
        return 2 * (self.n_edges + inp + self.act_dim * self.N + self.N * self.W_out.out_features + hyb)

    def global_msg(self, h):
        """Hybrid ablation: cell-type-pooled activity -> one self-attention layer over the type tokens -> one scalar per type,
        broadcast back to its neurons. Starts at zero gain so the model begins as the plain connectome student."""
        pooled = torch.zeros(h.shape[0], self.G, device=h.device, dtype=h.dtype).index_add_(1, self.grp, h) / self.grp_n
        x = self.h_ln(pooled[..., None] * self.h_up + self.h_emb)
        y, _ = self.h_attn(x, x, x, need_weights=False)
        return self.h_gain * self.h_down(y).squeeze(-1)[:, self.grp]

    # ---------------- dynamics ----------------
    def init_state(self, B, device): return torch.zeros(B, self.N, device=device)

    def step(self, h, obs, act, visible, W=None):
        """One recurrent step. `W` may be edge_weights() (recommended, compute once per rollout) or None."""
        w = self.edge_weights() if W is None else W
        inp = torch.zeros_like(h).index_add_(1, self.vis_idx, self.W_in(obs) * visible[:, None])
        inp = inp + self.W_act(F.one_hot(act, self.act_dim).float()) * self.act_mask
        if self.w_vis is not None: inp = inp + self.w_vis * (1 - visible)[:, None]
        pre = self.recur(h, w) + inp + self.b
        if self.hybrid: pre = pre + self.global_msg(h)
        if self.gate:
            gi = torch.zeros_like(h).index_add_(1, self.vis_idx, self.W_in_g(obs) * visible[:, None])
            a = torch.sigmoid(self.gain_g * (self.recur(h, self.Wg_raw) + gi + self.b_g))
        elif self.learn_alpha: a = torch.sigmoid(self.alpha_logit)
        else: a = self.alpha
        return (1 - a) * h + a * torch.tanh(self.gain * pre)

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


def lstm_hidden_for(target_params, obs_dim, z_dim, act_dim=5, vis_flag=True):
    """Largest hidden size whose LSTMBaseline has <= target_params parameters (#24); same rule as gru_hidden_for."""
    i = obs_dim + act_dim + int(vis_flag); a, b, c = 4, 4 * i + 8 + z_dim, z_dim - target_params
    return int((-b + (b * b - 4 * a * c) ** 0.5) / (2 * a))


class LSTMBaseline(GRUBaseline):
    """Parameter-matched unidirectional LSTM (#24); causal like the GRU (a bidirectional LSTM would read frames inside the
    blind window). The state handed to probes, homing and rollouts is [h | c] (2 x hidden): the cell state is part of its
    memory. The latent readout uses h only, as in a standard LSTM."""
    def __init__(self, obs_dim, z_dim, hidden, act_dim=5, vis_flag=True):
        nn.Module.__init__(self)
        self.vis_flag = vis_flag; self.hidden = hidden
        self.cell = nn.LSTMCell(obs_dim + act_dim + int(vis_flag), hidden); self.W_h = nn.Linear(hidden, z_dim)
        self.N = 2 * hidden; self.act_dim = act_dim; self.wiring = "lstm"; self.n_edges = hidden * hidden; self.dense = False

    def W_out(self, H): return self.W_h(H[..., :self.hidden])
    def param_breakdown(self):
        h = self.hidden; rec = 4 * h * h; inp = self.cell.weight_ih.numel(); out = sum(p.numel() for p in self.W_h.parameters())
        return dict(recurrent=rec, input=inp, output=out, other=self.n_params() - rec - inp - out, total=self.n_params(), edges=self.n_edges)
    def flops_per_step(self, obs_dim=None):
        return 2 * (4 * self.hidden * (self.cell.input_size + self.hidden) + self.hidden * self.W_h.out_features)
    def step(self, h, obs, act, visible, W=None):
        parts = [obs * visible[:, None], F.one_hot(act, self.act_dim).float()] + ([visible[:, None]] if self.vis_flag else [])
        hn, cn = self.cell(torch.cat(parts, -1), (h[:, :self.hidden], h[:, self.hidden:]))
        return torch.cat([hn, cn], -1)




class FlyGNN(nn.Module):
    """Graph-network student on the connectome's edge list (#23): the same graph as FlyBrain, but every neuron carries a
    d-dimensional state, so the real wiring gets vector capacity without changing who talks to whom.

      message   m_ij = s_ij * softplus(w_ij) * phi_g(j)(h_j)     one trainable scalar per real edge, fixed sign (Dale)
      aggregate a_i  = sum_j m_ij                                 index_add over the edge list, O(E d)
      update    h_i <- GRU_g(i)(a_i + input_i, h_i)               (update="gru"), or the FlyBrain leaky tanh (update="leaky")
      readout   z_hat = W_o (r_g(i) . h_i)                        one scalar per neuron, then the usual N -> z linear map

    g(i) is the neuron's group: its cell type (group="cell_type") or its region (group="region"); phi, the GRU cell, the
    readout vector r and the input directions are shared inside a group, which is what keeps the parameter count at the
    FlyBrain budget. Input enters as in FlyBrain (visual drive on VIS neurons, action on CX and OUT, blackout flag on all),
    each a scalar per neuron spread over the d channels by a group direction. phi is one linear map + tanh per *pre* group
    (the issue asks for one per (pre type, post type) pair: with 796 cell types that is 634 k pairs, far over budget).
    The state handed to callers is flat, [B, n_neurons * d], neuron-major (neuron i owns columns i*d .. i*d+d-1).
    With d = 1, update="leaky", phi="identity" and unit directions it is FlyBrain exactly (tested)."""
    def __init__(self, conn, obs_dim, z_dim, act_dim=5, wiring="connectome", d=8, group="cell_type", update="gru", phi="mlp",
                 alpha=0.5, seed=0, vis_flag=True):
        super().__init__()
        n, post, pre, sign = build_edges(conn, wiring, 1, seed)
        self.n_neurons = n; self.d = d; self.N = n * d; self.act_dim = act_dim; self.wiring = "gnn"; self.graph = wiring
        self.update = update; self.phi = phi; self.alpha = alpha; self.dense = False; self.n_edges = len(post)
        region = conn["region"]; _, gid = np.unique(conn["cell_type"] if group == "cell_type" else region, return_inverse=True)
        self.G = G = int(gid.max()) + 1
        vis_idx = np.nonzero(region == "VIS")[0]; nv = len(vis_idx)
        self.register_buffer("grp", torch.from_numpy(gid.astype(np.int64)))
        self.register_buffer("vis_idx", torch.from_numpy(vis_idx.astype(np.int64)))
        self.register_buffer("act_mask", torch.from_numpy(np.isin(region, ["CX", "OUT"]).astype(np.float32)))
        self.register_buffer("e_post", torch.from_numpy(post)); self.register_buffer("e_pre", torch.from_numpy(pre))
        self.register_buffer("e_sign", torch.from_numpy(sign))
        g = torch.Generator().manual_seed(seed); rn = lambda *s: torch.randn(*s, generator=g)
        indeg = torch.bincount(self.e_post, minlength=n).clamp(min=1).float()
        self.W_raw = nn.Parameter((rn(len(post)) / indeg[self.e_post].sqrt()).abs())
        self.W_in = nn.Linear(obs_dim, nv); self.W_act = nn.Linear(act_dim, n, bias=False)
        self.w_vis = nn.Parameter(torch.zeros(n)) if vis_flag else None
        self.dir_in = nn.Parameter(rn(G, d)); self.dir_act = nn.Parameter(rn(G, d)); self.dir_vis = nn.Parameter(rn(G, d))
        self.r = nn.Parameter(rn(G, d) / d ** 0.5); self.W_o = nn.Linear(n, z_dim)
        if phi == "mlp": self.phi_W = nn.Parameter(torch.eye(d).repeat(G, 1, 1) + 0.1 * rn(G, d, d)); self.phi_b = nn.Parameter(torch.zeros(G, d))
        if update == "gru":
            self.Wi = nn.Parameter(rn(G, d, 3 * d) / d ** 0.5); self.Wh = nn.Parameter(rn(G, d, 3 * d) / d ** 0.5)
            self.bi = nn.Parameter(torch.zeros(G, 3 * d)); self.bh = nn.Parameter(torch.zeros(G, 3 * d))
        else: self.b = nn.Parameter(torch.zeros(n)); self.gain = nn.Parameter(torch.ones(n))

    def n_params(self): return sum(p.numel() for p in self.parameters())
    def param_breakdown(self):
        rec = self.W_raw.numel(); inp = sum(p.numel() for p in self.W_in.parameters()) + self.W_act.weight.numel()
        out = sum(p.numel() for p in self.W_o.parameters()) + self.r.numel()
        return dict(recurrent=rec, input=inp, output=out, other=self.n_params() - rec - inp - out, total=self.n_params(), edges=self.n_edges)
    def flops_per_step(self, obs_dim=None):
        n, d = self.n_neurons, self.d; upd = 6 * n * d * d if self.update == "gru" else n * d
        return 2 * (self.n_edges * d + (n * d * d if self.phi == "mlp" else 0) + upd + self.W_in.in_features * len(self.vis_idx)
                    + self.act_dim * n + n * d + n * self.W_o.out_features)
    def edge_weights(self): return F.softplus(self.W_raw) * self.e_sign
    weights = edge_weights
    def init_state(self, B, device): return torch.zeros(B, self.N, device=device)
    def W_out(self, H): return self.W_o((H.view(*H.shape[:-1], self.n_neurons, self.d) * self.r[self.grp]).sum(-1))

    def step(self, h, obs, act, visible, W=None):
        w = self.edge_weights() if W is None else W; B = h.shape[0]; h = h.view(B, self.n_neurons, self.d)
        m = torch.tanh(torch.einsum("bnd,nde->bne", h, self.phi_W[self.grp]) + self.phi_b[self.grp]) if self.phi == "mlp" else h
        x = torch.zeros_like(h).index_add_(1, self.e_post, m[:, self.e_pre] * w[:, None])
        s_in = torch.zeros(B, self.n_neurons, device=h.device, dtype=h.dtype).index_add_(1, self.vis_idx, self.W_in(obs) * visible[:, None])
        x = x + s_in[..., None] * self.dir_in[self.grp] + (self.W_act(F.one_hot(act, self.act_dim).float()) * self.act_mask)[..., None] * self.dir_act[self.grp]
        if self.w_vis is not None: x = x + (self.w_vis * (1 - visible)[:, None])[..., None] * self.dir_vis[self.grp]
        if self.update == "gru":
            gi = torch.einsum("bnd,nde->bne", x, self.Wi[self.grp]) + self.bi[self.grp]
            gh = torch.einsum("bnd,nde->bne", h, self.Wh[self.grp]) + self.bh[self.grp]
            ir, iz, inn = gi.chunk(3, -1); hr, hz, hn = gh.chunk(3, -1)
            r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz); out = (1 - z) * torch.tanh(inn + r * hn) + z * h
        else: out = (1 - self.alpha) * h + self.alpha * torch.tanh(self.gain[:, None] * (x + self.b[:, None]))
        return out.reshape(B, self.N)

    def forward(self, obs, act, visible, h=None):
        B, T, _ = obs.shape; w = self.edge_weights(); h = self.init_state(B, obs.device) if h is None else h; hs = []
        for t in range(T):
            h = self.step(h, obs[:, t], act[:, t - 1] if t > 0 else torch.zeros_like(act[:, 0]), visible[:, t], w); hs.append(h)
        H = torch.stack(hs, 1); return self.W_out(H), H
