"""Build / load students from results/runs/<run>.pt, including checkpoints written before the edge-list refactor
(those carry dense M/S buffers and a full-N W_in; see issues #3, #11)."""
from __future__ import annotations
import numpy as np, torch
from wm2cx.student import FlyBrain, GRUBaseline
from wm2cx.connectome import load as load_conn
from wm2cx.data import Latents


def build_model(args, conn, obs_dim, state=None):
    if args["wiring"] == "gru":
        vis_flag = bool(args.get("vis_flag", state is None or state["cell.weight_ih"].shape[1] == obs_dim + 6))
        return GRUBaseline(obs_dim, args["k"], args["gru_hidden"], vis_flag=vis_flag)
    vis_flag = bool(args.get("vis_flag", state is None or "w_vis" in state))
    edges = None
    if state is not None and "M" in state and args["wiring"] != "dense":  # legacy: reuse the exact edge list that was trained
        edges = (state["e_pre"].numpy(), state["e_post"].numpy(), state["e_sign"].numpy())
    return FlyBrain(conn, obs_dim, args["k"], wiring=args["wiring"], n_brains=args["n_brains"], seed=args["seed"],
                    edges=edges, vis_flag=vis_flag, in_rank=int(args.get("in_rank", 0)))


def convert_legacy_state(state, model):
    if "M" not in state: return state
    s = {k: v for k, v in state.items() if k not in ("M", "S", "in_mask")}
    s["vis_idx"] = model.vis_idx.cpu()
    if s["W_in.weight"].shape[0] == model.N:  # keep only the VIS rows; the rest were multiplied by zero
        s["W_in.weight"] = s["W_in.weight"][model.vis_idx]; s["W_in.bias"] = s["W_in.bias"][model.vis_idx]
    for k in ("e_pre", "e_post"): 
        if k in s: s[k] = s[k].long()
    return s


def load_run(run, obs_dim=None, device="cpu"):
    """-> (model, args, conn, lat). obs_dim defaults to the eye size used in collection (8x32x3)."""
    ck = torch.load(f"results/runs/{run}.pt", weights_only=False, map_location="cpu"); args = ck["args"]; conn = load_conn(args["conn"])
    model = build_model(args, conn, obs_dim or 8 * 32 * 3, ck["state"])
    model.load_state_dict(convert_legacy_state(ck["state"], model)); model.to(device).eval()
    lat = Latents.__new__(Latents); lat.mu, lat.P, lat.scale = ck["lat"]["mu"], ck["lat"]["P"], ck["lat"]["scale"]
    return model, args, conn, lat
