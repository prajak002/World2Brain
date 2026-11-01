"""Build / load students from results/runs/<run>.pt, including checkpoints written by earlier implementations
(dense M/S buffers, full-N W_in, no blackout flag; see #3, #4, #11). Every evaluation script goes through `load_run`
so a checkpoint is reconstructed in exactly one place."""
from __future__ import annotations
import torch
from wm2cx.student import FlyBrain, GRUBaseline
from wm2cx.connectome import load as load_conn
from wm2cx.data import Latents

OBS_DIM = 8 * 32 * 3   # compound-eye observation used by collect.py / ingest_video.py (32x8 RGB)


def build_model(args, conn, obs_dim=OBS_DIM, state=None):
    """Model matching a run's recorded args. For old checkpoints the flag and input layout are inferred from the state."""
    if args["wiring"] == "gru":
        vis_flag = bool(args.get("vis_flag", state is None or state["cell.weight_ih"].shape[1] == obs_dim + 6))
        return GRUBaseline(obs_dim, args["k"], args["gru_hidden"], vis_flag=vis_flag)
    vis_flag = bool(args.get("vis_flag", state is None or "w_vis" in state))
    return FlyBrain(conn, obs_dim, args["k"], wiring=args["wiring"], n_brains=args["n_brains"], seed=args["seed"],
                    vis_flag=vis_flag, in_rank=int(args.get("in_rank", 0)))


def latents_from(ck):
    lat = Latents.__new__(Latents); lat.mu, lat.P, lat.scale = ck["lat"]["mu"], ck["lat"]["P"], float(ck["lat"]["scale"])
    lat.explained = float(ck.get("pca_explained", float("nan"))); return lat


def load_run(run, obs_dim=OBS_DIM, device="cpu", conn_path=None):
    """-> (model, args, conn, lat); the model is on `device` and in eval mode."""
    ck = torch.load(f"results/runs/{run}.pt", weights_only=False, map_location="cpu"); args = ck["args"]
    conn = load_conn(conn_path or args["conn"])
    model = build_model(args, conn, obs_dim, ck["state"]); model.load_state_dict(ck["state"]); model.to(device).eval()
    return model, args, conn, latents_from(ck)
