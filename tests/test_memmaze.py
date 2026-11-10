"""Memory Maze loader and metrics on a 3-trajectory synthetic fixture with the released file layout."""
import numpy as np, pytest
from wm2cx.memmaze import load_traj, downsample, causal_clip_index, walls_acc, objects_mse, list_split, T_STEPS


def make_traj(path, seed):
    r = np.random.default_rng(seed); T = T_STEPS
    a = np.zeros((T, 6)); a[np.arange(1, T), r.integers(0, 6, T - 1)] = 1          # action[t] led to frame t; zeros at reset
    lay = (r.random((9, 9)) < 0.5).astype(np.uint8)
    reset = np.zeros(T, bool); reset[0] = True
    np.savez(path, image=r.integers(0, 255, (T, 64, 64, 3), dtype=np.uint8), action=a, agent_pos=r.random((T, 2)) * 9,
             agent_dir=r.random((T, 2)), maze_layout=np.repeat(lay[None], T, 0), targets_vec=r.normal(size=(T, 3, 2)),
             reset=reset, terminal=np.zeros(T, bool))
    return a


@pytest.fixture
def root(tmp_path):
    for sp, n in (("train/part0", 3), ("eval", 3)):
        (tmp_path / sp).mkdir(parents=True)
        for i in range(n): make_traj(tmp_path / sp / f"2022{sp[-1]}{i:03d}-1000.npz", hash((sp, i)) % 1000)
    return tmp_path


def test_split_and_action_shift(root):
    tr = list_split(str(root), "train_all"); ev = list_split(str(root), "eval")
    assert len(tr) == 3 and len(ev) == 3 and not set(tr) & set(ev)
    assert list_split(str(root), "val", n_val=1) == tr[-1:] and list_split(str(root), "train", n_val=1) == tr[:-1]
    d = load_traj(tr[0]); raw = np.load(tr[0])["action"].argmax(-1)
    assert d["act"].shape == (T_STEPS,) and (d["act"][:-1] == raw[1:]).all() and d["act"][-1] == 0
    assert d["obs"].shape == (T_STEPS, 768) and d["maze_layout"].shape == (9, 9)


def test_downsample_is_block_mean():
    img = np.zeros((2, 64, 64, 3), np.uint8); img[:, :4, :4, 0] = 200
    o = downsample(img).reshape(2, 16, 16, 3)
    assert o[0, 0, 0, 0] == 200 and o[0, 0, 1, 0] == 0 and o.sum() == 2 * 200


def test_causal_window_never_sees_future():
    idx = causal_clip_index(10, clip=8)            # into frames padded with 7 copies of frame 0
    assert idx.shape == (10, 8) and (idx[:, -1] - 7 == np.arange(10)).all() and idx.max() - 7 == 9


def test_metrics_constant_and_perfect():
    lay = np.array([[1, 0] * 40 + [1]], float)
    assert walls_acc(lay * 20 - 10, lay)[0] == 100.0 and walls_acc(np.full_like(lay, 0.9), lay, is_logit=False)[0] == pytest.approx(100 * 41 / 81)
    t = np.ones((5, 3, 2)); assert objects_mse(t, t).max() == 0 and objects_mse(t * 0, t)[0] == pytest.approx(2.0)


def test_lesion_groups_and_matched_random():
    from wm2cx.mm_models import lesion_groups, matched_random
    r = np.random.default_rng(0); N = 400
    region = np.array(["VIS"] * 20 + ["CX"] * 30 + ["MB"] * 10 + ["OUT"] * 10 + ["OTH"] * 330); ct = np.array([""] * N, dtype="U40"); ct[70:90] = "KCab"
    conn = dict(region=region, cell_type=ct, pre=r.integers(0, N, 5000), post=r.integers(0, N, 5000))
    g = lesion_groups(conn)
    assert len(g["CX"]) == 30 and len(g["MB"]) == 30 and len(g["POOL"]) == 310 and not set(g["POOL"]) & (set(g["CX"]) | set(g["MB"]))
    m = matched_random(conn, g["CX"], g["POOL"], np.random.default_rng(1))
    assert len(m) == 30 and set(m) <= set(g["POOL"])
    assert not np.array_equal(m, matched_random(conn, g["CX"], g["POOL"], np.random.default_rng(2)))


def test_rollout_keep_silences_neurons():
    import torch
    from wm2cx.mm_models import rollout

    class M:
        N = 5
        def edge_weights(self): return None
        def init_state(self, B, dev): return torch.zeros(B, 5)
        def step(self, h, o, a, v, w): return h + 1
        def W_out(self, H): return H
    keep = torch.ones(2, 5); keep[1, :2] = 0
    _, H, h = rollout(M(), torch.zeros(2, 3, 1), torch.zeros(2, 3).long(), torch.ones(2, 3), keep=keep)
    assert (H[0] == torch.tensor([1., 2., 3.])[:, None]).all() and (H[1, :, :2] == 0).all() and (H[1, -1, 2:] == 3).all()
