"""Edge-list recurrence == dense-mask recurrence; legacy checkpoints load; checkpoints are small. Runs on CPU, no data."""
import io, numpy as np, torch, pytest
from wm2cx.student import FlyBrain, GRUBaseline, build_edges, build_mask
from wm2cx.synthetic import synthetic_connectome

SMALL = {"VIS": 60, "CX": 80, "MB": 10, "OUT": 30}


def small_conn(seed=0):
    return synthetic_connectome(n_edges=1500, regions=SMALL, n_epg=5, seed=seed)


def legacy_dense_step(m, h, obs, act, vis):
    """Reference: the previous implementation (materialised W = M * S * softplus(W_raw), h @ W.T)."""
    W = m.W()
    inp = m.W_in(obs) * m.in_mask * vis[:, None] + m.W_act(torch.nn.functional.one_hot(act, m.act_dim).float()) * m.act_mask
    return (1 - m.alpha) * h + m.alpha * torch.tanh(m.gain * (h @ W.T + inp + m.b))


@pytest.mark.parametrize("wiring", ["connectome", "connectome_nosign", "shuffled", "random", "dense"])
@pytest.mark.parametrize("K", [1, 2, 3])
def test_edge_recurrence_matches_dense(wiring, K):
    if wiring == "dense" and K > 1: pytest.skip("dense multi-brain not a studied configuration")
    conn = small_conn(); torch.manual_seed(0)
    m = FlyBrain(conn, obs_dim=24, z_dim=8, wiring=wiring, n_brains=K, seed=1)
    B = 4; h = torch.randn(B, m.N); obs = torch.randn(B, 24); act = torch.randint(0, 5, (B,)); vis = torch.ones(B)
    ref = legacy_dense_step(m, h, obs, act, vis); out = m.step(h, obs, act, vis)
    assert torch.allclose(ref, out, atol=1e-5), (ref - out).abs().max()
    # whole forward
    T = 6; obs_s = torch.randn(B, T, 24); act_s = torch.randint(0, 5, (B, T)); vis_s = torch.ones(B, T); vis_s[:, 2:4] = 0
    zh, H = m(obs_s, act_s, vis_s)
    hh = m.init_state(B, "cpu"); hs = []
    for t in range(T):
        hh = legacy_dense_step(m, hh, obs_s[:, t], act_s[:, t - 1] if t > 0 else torch.zeros_like(act_s[:, 0]), vis_s[:, t]); hs.append(hh)
    assert torch.allclose(H, torch.stack(hs, 1), atol=1e-5)


def test_gradients_match_dense():
    conn = small_conn(); m = FlyBrain(conn, 24, 8, wiring="connectome", seed=2)
    B = 3; h = torch.randn(B, m.N); obs = torch.randn(B, 24); act = torch.randint(0, 5, (B,)); vis = torch.ones(B)
    g_ref = torch.autograd.grad(legacy_dense_step(m, h, obs, act, vis).pow(2).sum(), m.W_raw)[0]
    g_new = torch.autograd.grad(m.step(h, obs, act, vis).pow(2).sum(), m.W_raw)[0]
    assert torch.allclose(g_ref, g_new, atol=1e-5, rtol=1e-4)


def test_edge_order_matches_nonzero_of_mask():
    """W_raw ordering must equal np.nonzero(M) so old checkpoints line up."""
    conn = small_conn()
    for wiring in ("connectome", "shuffled", "random"):
        for K in (1, 2):
            N, post, pre, sign = build_edges(conn, wiring, K, seed=3); M, S = build_mask(conn, wiring, K, seed=3)
            p2, q2 = np.nonzero(M)
            assert N == M.shape[0] and np.array_equal(post, p2) and np.array_equal(pre, q2)
            assert np.array_equal(sign, S[p2, q2])
            assert len(post) == len(np.unique(post * N + pre))


def test_multibrain_moves_cross_hemisphere_edges():
    conn = small_conn(); N0 = len(conn["region"])
    side = conn["side"]; pre, post = conn["pre"], conn["post"]
    cross = (side[pre] != side[post]) & (side[pre] != "center") & (side[post] != "center")
    N, p, q, _ = build_edges(conn, "connectome", 2, seed=0)
    brain_of = lambda i: i // N0
    inter = brain_of(p) != brain_of(q)
    assert inter.sum() > 0
    assert inter.sum() == 2 * len(np.unique((post * N0 + pre)[cross]))   # every cross edge appears once per brain, in the partner
    assert N == 2 * N0


def test_legacy_checkpoint_with_dense_buffers_loads():
    conn = small_conn(); m = FlyBrain(conn, 24, 8, wiring="connectome", seed=4)
    sd = m.state_dict(); M, S = build_mask(conn, "connectome", 1, seed=4)
    legacy = dict(sd); legacy["M"] = torch.from_numpy(M); legacy["S"] = torch.from_numpy(S)
    m2 = FlyBrain(conn, 24, 8, wiring="connectome", seed=5)
    res = m2.load_state_dict(legacy); assert not res.missing_keys and not res.unexpected_keys
    x = torch.randn(2, 3, 24); a = torch.randint(0, 5, (2, 3)); v = torch.ones(2, 3)
    assert torch.allclose(m(x, a, v)[0], m2(x, a, v)[0])


def test_checkpoint_has_no_nxn_tensors():
    conn = synthetic_connectome(seed=0)                       # full-size 6,935 neurons, 91,663 edges
    m = FlyBrain(conn, 768, 256, wiring="connectome")
    buf = io.BytesIO(); torch.save(m.state_dict(), buf)
    mb = buf.tell() / 1e6
    assert all(v.numel() < m.N * m.N // 4 for v in m.state_dict().values())
    assert mb < 40, f"checkpoint {mb:.0f} MB"             # was ~420 MB with the dense M and S buffers
    assert m.n_edges <= 91_663 and m.flops_per_step() < 2 * (91_663 + 6935 * (768 + 5 + 256)) + 1


def test_step_accepts_precomputed_edge_weights_and_legacy_dense_matrix():
    conn = small_conn(); m = FlyBrain(conn, 24, 8, wiring="connectome")
    h = torch.randn(2, m.N); obs = torch.randn(2, 24); act = torch.zeros(2, dtype=torch.long); vis = torch.ones(2)
    a = m.step(h, obs, act, vis); b = m.step(h, obs, act, vis, m.edge_weights()); c = m.step(h, obs, act, vis, m.W())
    assert torch.allclose(a, b) and torch.allclose(a, c, atol=1e-5)


def test_gru_interface():
    g = GRUBaseline(24, 8, 16); x = torch.randn(2, 5, 24); a = torch.randint(0, 5, (2, 5)); v = torch.ones(2, 5)
    zh, H = g(x, a, v); assert zh.shape == (2, 5, 8) and H.shape == (2, 5, 16) and g.flops_per_step() > 0
