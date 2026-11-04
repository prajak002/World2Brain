"""Causal transformer world model: the field-standard unconstrained baseline (IRIS / STORM / TWM / DINO-WM-style predictor).

One token per environment step, [obs_t * visible_t, one-hot(a_{t-1}), visible_t], embedded and processed by a pre-LN causal
transformer; the readout predicts the teacher latent at every step. The "state" exposed to the probes and behavioural
readouts is the last layer's token (d_model features), so every evaluation script that works on a FlyBrain or GRU state
works on this model unchanged. In closed loop (`step`) the model keeps its own context and re-runs attention over it,
truncated to `max_len` tokens, which is the recurrent memory a transformer world model actually has.

Parameter matching follows the GRU logic (#4): `transformer_layers_for` picks the deepest model at a given width whose
parameter count does not exceed the FlyBrain it is compared against.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


def transformer_params(d_model, n_layers, obs_dim, z_dim, act_dim=5, vis_flag=True, max_len=64):
    i = obs_dim + act_dim + int(vis_flag)
    per_layer = 4 * d_model * d_model + 4 * d_model + 2 * (d_model * 4 * d_model) + 4 * d_model + d_model + 4 * d_model   # attn + mlp + 2 LN
    return (i * d_model + d_model) + max_len * d_model + n_layers * per_layer + (2 * d_model) + (d_model * z_dim + z_dim)


def transformer_layers_for(target_params, obs_dim, z_dim, d_model=256, act_dim=5, vis_flag=True, max_len=64):
    """Largest number of layers whose TransformerWM at width d_model has <= target_params parameters (at least 1)."""
    n = 1
    while transformer_params(d_model, n + 1, obs_dim, z_dim, act_dim, vis_flag, max_len) <= target_params: n += 1
    return n


class TransformerWM(nn.Module):
    def __init__(self, obs_dim, z_dim, d_model=256, n_layers=4, n_heads=4, act_dim=5, vis_flag=True, max_len=64):
        super().__init__()
        self.vis_flag, self.act_dim, self.max_len = vis_flag, act_dim, max_len
        self.N = d_model; self.wiring = "transformer"; self.dense = False; self.n_layers = n_layers
        self.n_edges = n_layers * 4 * d_model * d_model                     # nominal "recurrent" budget: the attention projections
        in_dim = obs_dim + act_dim + int(vis_flag)
        self.embed = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model); self.W_out = nn.Linear(d_model, z_dim)
        self._ctx = None

    # ---------------- accounting ----------------
    def n_params(self): return sum(p.numel() for p in self.parameters())

    def param_breakdown(self):
        rec = sum(p.numel() for p in self.blocks.parameters()); inp = sum(p.numel() for p in self.embed.parameters()) + self.pos.numel()
        out = sum(p.numel() for p in self.W_out.parameters())
        return dict(recurrent=rec, input=inp, output=out, other=self.n_params() - rec - inp - out, total=self.n_params(), edges=self.n_edges)

    def flops_per_step(self, obs_dim=None, context=24):
        """Multiply-adds for one new token with `context` tokens already in the window (attention is linear in the context)."""
        d = self.N
        return 2 * (self.embed.in_features * d + self.n_layers * (12 * d * d + 2 * context * d) + d * self.W_out.out_features)

    def edge_weights(self): return None
    weights = edge_weights

    # ---------------- dynamics ----------------
    def tokens(self, obs, act_prev, visible):
        parts = [obs * visible[..., None], F.one_hot(act_prev, self.act_dim).float()] + ([visible[..., None]] if self.vis_flag else [])
        return torch.cat(parts, -1)

    def encode(self, tok):
        """tok [B,T,in] -> H [B,T,d] with a causal mask."""
        T = tok.shape[1]
        x = self.embed(tok) + self.pos[:T]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=tok.device), 1)
        return self.norm(self.blocks(x, mask=mask, is_causal=True))

    def forward(self, obs, act, visible, h=None):
        """obs [B,T,D], act [B,T] (action taken after observing step t), visible [B,T] -> z_hat [B,T,z], H [B,T,d]"""
        act_prev = torch.cat([torch.zeros_like(act[:, :1]), act[:, :-1]], 1)
        H = self.encode(self.tokens(obs, act_prev, visible))
        return self.W_out(H), H

    def init_state(self, B, device):
        self._ctx = []
        return torch.zeros(B, self.N, device=device)

    def step(self, h, obs, act, visible, W=None):
        """Closed-loop step with the same signature as FlyBrain.step: appends the token to the kept context (truncated to
        max_len) and returns the last hidden state. `init_state` must be called first for every new episode."""
        if self._ctx is None: self._ctx = []
        self._ctx.append(self.tokens(obs, act, visible)[:, None])
        if len(self._ctx) > self.max_len: self._ctx = self._ctx[-self.max_len:]
        return self.encode(torch.cat(self._ctx, 1))[:, -1]
