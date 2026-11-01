"""E7 real-video tooling: egomotion sign conventions, ingest -> eval_ckpt round trip on a tiny synthetic checkpoint. CPU only."""
import os, subprocess, sys, json, numpy as np, torch, pytest
from wm2cx.ego import egomotion, quantise
from wm2cx.student import FlyBrain
from wm2cx.data import Latents
from wm2cx.synthetic import synthetic_connectome

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def texture(R=256, seed=0):
    rng = np.random.default_rng(seed); img = rng.integers(0, 255, (R // 8, R // 8, 3), np.uint8)
    return np.kron(img, np.ones((8, 8, 1), np.uint8))                      # blocky random texture, plenty of structure


def test_yaw_sign_scene_shifts_right_means_turn_left():
    f0 = texture(); f1 = np.roll(f0, 12, axis=1)                             # scene moves right by 12 px
    yaw, exp_ = egomotion(f0, f1, grid=4, fov_deg=60.0)
    assert yaw > 2.0 and abs(exp_) < 1.0, (yaw, exp_)                       # 12 px * 60/256 = 2.8 deg, left turn
    yaw2, _ = egomotion(f1, f0, grid=4, fov_deg=60.0)
    assert yaw2 < -2.0


def test_expansion_sign_zoom_in_means_forward():
    f0 = texture(); R = f0.shape[0]; c = R // 2; s = 0.9                     # zoom in: crop the centre 90 % and resize back
    ys = np.clip((np.arange(R) - c) * s + c, 0, R - 1).astype(int)
    f1 = f0[ys][:, ys]
    yaw, exp_ = egomotion(f0, f1, grid=4, fov_deg=60.0)
    assert exp_ > 1.0 and abs(yaw) < 1.5, (yaw, exp_)
    assert quantise(yaw, exp_, turn_deg=6.0, fwd_px=1.0) == 1
    assert quantise(10.0, 0.0) == 2 and quantise(-10.0, 5.0) == 3 and quantise(0.0, -3.0) == 4 and quantise(0.0, 0.0) == 0


def test_eval_ckpt_round_trip(tmp_path):
    """A tiny checkpoint + an ingest-shaped file -> eval_ckpt writes train.py-compatible JSON and the baselines."""
    conn = synthetic_connectome(n_edges=800, regions={"VIS": 40, "CX": 50, "MB": 8, "OUT": 20}, n_epg=4, seed=0)
    np.savez_compressed(tmp_path / "conn.npz", **conn)
    E, T, D, k = 6, 24, 64, 8; rng = np.random.default_rng(0)
    obs = rng.integers(0, 255, (E, T, 8, 32, 3), np.uint8); act = rng.integers(0, 5, (E, T)).astype(np.int8)
    z = rng.normal(size=(E, T, D)).astype(np.float16); lat = Latents(z, k)
    model = FlyBrain(conn, 8 * 32 * 3, k, wiring="connectome", seed=0)
    os.makedirs(tmp_path / "results/runs"); os.makedirs(tmp_path / "data")
    torch.save(dict(state=model.state_dict(), lat=dict(mu=lat.mu, P=lat.P, scale=lat.scale),
                    args=dict(wiring="connectome", n_brains=1, seed=0, conn=str(tmp_path / "conn.npz"), k=k, tag="")), tmp_path / "results/runs/toy_K1_s0.pt")
    np.savez_compressed(tmp_path / "data/realvid.npz", obs=obs, act=act, z=z, heading=np.zeros((E, T), np.float32), pos=np.zeros((E, T, 2), np.float32),
                        clip=np.array([0, 0, 1, 1, 2, 2], np.int16))
    env = dict(os.environ, PYTHONPATH=ROOT)
    for extra, tag in (([], "_realvid"), (["--finetune_epochs", "1"], "_realvid_ft")):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/eval_ckpt.py"), "--run", "toy_K1_s0", "--test", "data/realvid.npz", "--tag", tag, "--device", "cpu"] + extra,
                           cwd=tmp_path, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        j = json.load(open(tmp_path / f"results/runs/toy_K1_s0{tag}.json"))
        assert len(j["test_mse"]) == 8 and len(j["test_mse_per_ep"]) == E and j["n_clips"] == 3
        assert all(np.isfinite(j["test_mse"]))
    b = json.load(open(tmp_path / "results/baselines/copy_last_K0_s0_realvid.json")); assert len(b["test_mse_per_ep"]) == E
