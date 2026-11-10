"""Students for Memory Maze (PREREG section 3) and their state rollouts over long trajectories.

C / S / B / R = FlyBrain on the full-brain graph with wiring connectome / shuffled / block_shuffled / random; G = GRU with the
largest hidden size not exceeding C's parameter count; T = causal transformer, width --d_model, depth = the deepest not
exceeding C's parameter count, context = the TBTT window. Inputs: 16x16x3 frame (768), previous action (6), visible flag.

`rollout` advances a recurrent student over a window while carrying state, using the action that led to each frame
(act_prev[t] = act[t-1] within the trajectory; the first step of a window gets the previous window's last action, which
FlyBrain.forward would reset to 0). For the transformer the "state" at step t is its last-layer token computed over the
last `ctx` tokens (no state crosses the context window, which is the memory such a model has).
"""
from __future__ import annotations
import torch
from wm2cx.student import FlyBrain, GRUBaseline, gru_hidden_for
from wm2cx.transformer import TransformerWM, transformer_layers_for

COND_WIRING = {"C": "connectome", "S": "shuffled", "B": "block_shuffled", "R": "random"}
OBS_DIM, Z_DIM, ACT_DIM = 768, 256, 6


def build(cond, conn, seed, d_model=512, ctx=48):
    """-> (model, info). Every model is built from the same C reference so parameter matching is exact and logged."""
    ref = None
    if cond in ("G", "T"):
        ref = FlyBrain(conn, OBS_DIM, Z_DIM, act_dim=ACT_DIM, wiring="connectome", seed=seed); target = ref.n_params(); del ref
    if cond in COND_WIRING:
        m = FlyBrain(conn, OBS_DIM, Z_DIM, act_dim=ACT_DIM, wiring=COND_WIRING[cond], seed=seed); info = {}
    elif cond == "G":
        h = gru_hidden_for(target, OBS_DIM, Z_DIM, act_dim=ACT_DIM); m = GRUBaseline(OBS_DIM, Z_DIM, h, act_dim=ACT_DIM); info = dict(target_params=target, hidden=h)
    elif cond == "T":
        L0 = transformer_layers_for(target, OBS_DIM, Z_DIM, d_model, act_dim=ACT_DIM, max_len=max(64, ctx))
        mk = lambda L: TransformerWM(OBS_DIM, Z_DIM, d_model, L, n_heads=max(4, d_model // 64), act_dim=ACT_DIM, max_len=max(64, ctx))
        L = min((L0, L0 + 1), key=lambda L: abs(mk(L).n_params() - target))   # depth closest to C's count (the floor rule can miss +-5 %)
        m = mk(L)
        info = dict(target_params=target, d_model=d_model, n_layers=L, ctx=ctx)
    else: raise ValueError(cond)
    if "target_params" in info:   # AGENTS.md rule 3: matched within +-5 %
        info["param_ratio"] = m.n_params() / info["target_params"]; assert abs(info["param_ratio"] - 1) <= 0.05, info
    return m, info


def act_prev_of(act, last=None):
    """act [B,T] (action after step t) -> action that led to step t; `last` [B] = previous window's final action."""
    first = torch.zeros_like(act[:, :1]) if last is None else last[:, None]
    return torch.cat([first, act[:, :-1]], 1)


def rollout(model, obs, act_prev, vis, h=None):
    """Recurrent students: obs [B,T,768] float, act_prev [B,T] long, vis [B,T] -> (z_hat [B,T,256], H [B,T,N], h_last)."""
    if isinstance(model, TransformerWM):
        H = model.encode(model.tokens(obs, act_prev, vis)); return model.W_out(H), H, None
    w = model.edge_weights(); h = model.init_state(obs.shape[0], obs.device) if h is None else h; hs = []
    for t in range(obs.shape[1]):
        h = model.step(h, obs[:, t], act_prev[:, t], vis[:, t], w); hs.append(h)
    H = torch.stack(hs, 1); return model.W_out(H), H, h


@torch.no_grad()
def transformer_states(model, obs, act_prev, vis, ctx=48, chunk=4096):
    """State at every step of whole trajectories with a sliding context of `ctx` tokens: [B,T,d]."""
    tok = model.tokens(obs, act_prev, vis); B, T, _ = tok.shape
    out = torch.empty(B, T, model.N, device=tok.device)
    out[:, :ctx] = model.encode(tok[:, :ctx])                              # causal: step t < ctx sees tokens 0..t at positions 0..t
    if T > ctx:
        win = tok.unfold(1, ctx, 1)[:, 1:].permute(0, 1, 3, 2)             # windows ending at t = ctx..T-1: [B, T-ctx, ctx, in]
        flat = win.reshape(-1, ctx, tok.shape[-1]); res = []
        for i in range(0, len(flat), chunk): res.append(model.encode(flat[i:i + chunk])[:, -1])
        out[:, ctx:] = torch.cat(res).view(B, T - ctx, model.N)
    return out
