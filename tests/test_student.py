"""Edge-list recurrence == dense matmul; rewiring controls keep exact counts, degrees, blocks and Dale signs (#1, #2);
W_in feeds VIS only and parameters are accounted (#3); GRU matching and the shared blackout flag (#4); legacy checkpoints
load (#11). Runs on CPU with the synthetic connectome, no data."""
import io, numpy as np, torch, pytest
from wm2cx.student import FlyBrain, GRUBaseline, build_edges, build_mask, block_counts, gru_hidden_for, WIRINGS
from wm2cx.rewire import maslov_sneppen, block_rewire, erdos_renyi
from wm2cx.synthetic import synthetic_connectome

SMALL = {"VIS": 60, "CX": 80, "MB": 10, "OUT": 30}
SPARSE = [w for w in WIRINGS if w != "dense"]


def small_conn(seed=0):
    return synthetic_connectome(n_edges=1500, regions=SMALL, n_epg=5, seed=seed)


def dense_step(m, h, obs, act, vis):
    """Reference: the materialised matrix W = M * S * softplus(W_raw), h @ W.T, with the input layer scattered on VIS rows."""
    W = m.W(); inp = torch.zeros_like(h).index_add_(1, m.vis_idx, m.W_in(obs) * vis[:, None])
    inp = inp + m.W_act(torch.nn.functional.one_hot(act, m.act_dim).float()) * m.act_mask
    if m.w_vis is not None: inp = inp + m.w_vis * (1 - vis)[:, None]
    return (1 - m.alpha) * h + m.alpha * torch.tanh(m.gain * (h @ W.T + inp + m.b))


@pytest.mark.parametrize("wiring", WIRINGS)
@pytest.mark.parametrize("K", [1, 2, 3])
def test_edge_recurrence_matches_dense(wiring, K):
    if wiring == "dense" and K > 1: pytest.skip("dense multi-brain not a studied configuration")
    conn = small_conn(); torch.manual_seed(0)
    m = FlyBrain(conn, obs_dim=24, z_dim=8, wiring=wiring, n_brains=K, seed=1)
    B = 4; h = torch.randn(B, m.N); obs = torch.randn(B, 24); act = torch.randint(0, 5, (B,)); vis = torch.ones(B)
    ref = dense_step(m, h, obs, act, vis); out = m.step(h, obs, act, vis)
    assert torch.allclose(ref, out, atol=1e-5), (ref - out).abs().max()
    T = 6; obs_s = torch.randn(B, T, 24); act_s = torch.randint(0, 5, (B, T)); vis_s = torch.ones(B, T); vis_s[:, 2:4] = 0
    zh, H = m(obs_s, act_s, vis_s)
    hh = m.init_state(B, "cpu"); hs = []
    for t in range(T):
        hh = dense_step(m, hh, obs_s[:, t], act_s[:, t - 1] if t > 0 else torch.zeros_like(act_s[:, 0]), vis_s[:, t]); hs.append(hh)
    assert torch.allclose(H, torch.stack(hs, 1), atol=1e-5)


def test_gradients_match_dense():
    conn = small_conn(); m = FlyBrain(conn, 24, 8, wiring="connectome", seed=2)
    B = 3; h = torch.randn(B, m.N); obs = torch.randn(B, 24); act = torch.randint(0, 5, (B,)); vis = torch.ones(B)
    g_ref = torch.autograd.grad(dense_step(m, h, obs, act, vis).pow(2).sum(), m.W_raw)[0]
    g_new = torch.autograd.grad(m.step(h, obs, act, vis).pow(2).sum(), m.W_raw)[0]
    assert torch.allclose(g_ref, g_new, atol=1e-5, rtol=1e-4)


def _simple(pre, post):
    return len(set(zip(pre.tolist(), post.tolist()))) == len(pre) and not (pre == post).any()


def test_rewires_keep_exact_edge_count_degrees_and_blocks():
    conn = synthetic_connectome(n_edges=6000, regions=SMALL, n_epg=5, seed=1); N = len(conn["region"])
    # the synthetic generator may contain duplicates / self-loops: rewire the simple version of it
    key = np.unique(conn["post"] * N + conn["pre"]); post0, pre0 = np.divmod(key, N); keep = post0 != pre0; pre0, post0 = pre0[keep], post0[keep]
    E = len(pre0); din, dout = np.bincount(post0, minlength=N), np.bincount(pre0, minlength=N)
    for pre, post in [maslov_sneppen(pre0, post0, 0, 3), block_rewire(pre0, post0, conn["region"], 0, 3), erdos_renyi(N, E, 0)]:
        assert len(pre) == E and _simple(pre, post)
    p, q = maslov_sneppen(pre0, post0, 0, 3)
    assert np.array_equal(np.bincount(q, minlength=N), din) and np.array_equal(np.bincount(p, minlength=N), dout)
    assert block_counts(q, p, conn["region"]) != block_counts(post0, pre0, conn["region"])          # macro-architecture destroyed
    p, q = block_rewire(pre0, post0, conn["region"], 0, 3)
    assert np.array_equal(np.bincount(q, minlength=N), din) and np.array_equal(np.bincount(p, minlength=N), dout)
    assert block_counts(q, p, conn["region"]) == block_counts(post0, pre0, conn["region"])          # block counts exact
    assert (p != pre0).any() or (q != post0).any()


def test_controls_are_dale_consistent_and_inhibitory_edge_fraction_matched():
    conn = synthetic_connectome(n_edges=6000, regions=SMALL, n_epg=5, seed=2); N = len(conn["region"])
    real = float((conn["nt"][conn["pre"]] < 0).mean())
    for w in ("connectome", "shuffled", "block_shuffled", "random", "connectome_randsign"):
        _, post, pre, sign = build_edges(conn, w, seed=1)
        per_neuron = {}
        for a, s in zip(pre.tolist(), sign.tolist()): assert per_neuron.setdefault(a, s) == s   # one sign per presynaptic neuron
        assert abs(float((sign < 0).mean()) - real) < 0.03, (w, float((sign < 0).mean()), real)
        assert len(post) == len(np.unique(post * N + pre))
    _, _, _, s_rand = build_edges(conn, "connectome_randsign", seed=1); _, _, _, s_real = build_edges(conn, "connectome", seed=1)
    assert (s_rand != s_real).any()
    _, _, _, s_ns = build_edges(conn, "connectome_nosign", seed=1); assert (s_ns == 1).all()


def test_edge_order_matches_nonzero_of_mask():
    conn = small_conn()
    for wiring in SPARSE:
        for K in (1, 2):
            N, post, pre, sign = build_edges(conn, wiring, K, seed=3); M, S = build_mask(conn, wiring, K, seed=3)
            p2, q2 = np.nonzero(M)
            assert N == M.shape[0] and np.array_equal(post, p2) and np.array_equal(pre, q2) and np.array_equal(sign, S[p2, q2])


def test_multibrain_moves_cross_hemisphere_edges():
    conn = small_conn(); N0 = len(conn["region"])
    N, p, q, _ = build_edges(conn, "connectome", 2, seed=0)
    inter = (p // N0) != (q // N0)
    assert inter.sum() > 0 and N == 2 * N0


def test_w_in_feeds_vis_only_and_params_are_reported():
    conn = small_conn(); m = FlyBrain(conn, 24, 8); pb = m.param_breakdown()
    assert m.W_in.weight.shape[0] == SMALL["VIS"] and pb["recurrent"] == m.n_edges
    assert sum(pb[k] for k in ("recurrent", "input", "output", "other")) == pb["total"] == m.n_params()
    m2 = FlyBrain(conn, 24, 8, in_rank=4); assert m2.n_params() < m.n_params() and m2.flops_per_step() < m.flops_per_step()
    m3 = FlyBrain(conn, 24, 8, vis_flag=False); assert m3.w_vis is None and m3.n_params() == m.n_params() - m.N


def test_gru_param_matching_and_flag():
    conn = small_conn(); fb = FlyBrain(conn, 24, 8); h = gru_hidden_for(fb.n_params(), 24, 8); g = GRUBaseline(24, 8, h)
    assert g.n_params() <= fb.n_params() < GRUBaseline(24, 8, h + 1).n_params()
    x = torch.randn(2, 5, 24); a = torch.randint(0, 5, (2, 5)); v = torch.ones(2, 5)
    zh, H = g(x, a, v); assert zh.shape == (2, 5, 8) and H.shape == (2, 5, h) and g.flops_per_step() > 0
    assert GRUBaseline(24, 8, 16, vis_flag=False).cell.input_size == 24 + 5


def test_legacy_checkpoints_load():
    """Dense M/S buffers, an in_mask buffer, a full-N W_in and no blackout flag -> same predictions after conversion."""
    conn = small_conn(); m = FlyBrain(conn, 24, 8, wiring="connectome", seed=4, vis_flag=False)
    sd = dict(m.state_dict()); M, S = build_mask(conn, "connectome", 1, seed=4)
    sd["M"] = torch.from_numpy(M); sd["S"] = torch.from_numpy(S); sd.pop("vis_idx"); sd["in_mask"] = torch.zeros(m.N)
    W_full = torch.randn(m.N, 24); b_full = torch.randn(m.N); W_full[m.vis_idx] = sd["W_in.weight"]; b_full[m.vis_idx] = sd["W_in.bias"]
    sd["W_in.weight"], sd["W_in.bias"] = W_full, b_full
    for flag in (False, True):
        m2 = FlyBrain(conn, 24, 8, wiring="connectome", seed=5, vis_flag=flag)
        res = m2.load_state_dict(sd); assert not res.missing_keys and not res.unexpected_keys
        x = torch.randn(2, 3, 24); a = torch.randint(0, 5, (2, 3)); v = torch.ones(2, 3)
        assert torch.allclose(m(x, a, v)[0], m2(x, a, v)[0], atol=1e-6)


def test_checkpoint_has_no_nxn_tensors():
    conn = synthetic_connectome(seed=0)                       # full-size 6,935 neurons, 91,663 edges
    m = FlyBrain(conn, 768, 256, wiring="connectome")
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); mb = buf.tell() / 1e6
    assert all(v.numel() < m.N * m.N // 4 for v in m.state_dict().values())
    assert mb < 20, f"checkpoint {mb:.0f} MB"             # 29 MB with the full-N W_in, ~420 MB with dense M and S
    assert m.param_breakdown()["input"] == 768 * 2330 + 2330 + 5 * m.N


def test_step_accepts_precomputed_edge_weights_and_dense_matrix():
    conn = small_conn(); m = FlyBrain(conn, 24, 8, wiring="connectome")
    h = torch.randn(2, m.N); obs = torch.randn(2, 24); act = torch.zeros(2, dtype=torch.long); vis = torch.ones(2)
    a = m.step(h, obs, act, vis); b = m.step(h, obs, act, vis, m.edge_weights()); c = m.step(h, obs, act, vis, m.W())
    assert torch.allclose(a, b) and torch.allclose(a, c, atol=1e-5)


@pytest.mark.parametrize("kw", [dict(learn_alpha=True), dict(gate=True), dict(learn_alpha=True, gate=True)])
def test_neuron_model_ablations_run_and_train(kw):
    conn = small_conn(); m = FlyBrain(conn, 24, 8, wiring="connectome", seed=3, **kw)
    x = torch.randn(2, 5, 24); a = torch.randint(0, 5, (2, 5)); v = torch.ones(2, 5); v[:, 2:4] = 0
    zh, H = m(x, a, v); assert zh.shape == (2, 5, 8) and torch.isfinite(H).all()
    zh.pow(2).sum().backward()
    if kw.get("gate"): assert m.Wg_raw.grad is not None and m.Wg_raw.grad.abs().sum() > 0 and m.param_breakdown()["recurrent"] == 2 * m.n_edges
    if kw.get("learn_alpha") and not kw.get("gate"): assert m.alpha_logit.grad is not None
    base = FlyBrain(conn, 24, 8, wiring="connectome", seed=3); assert m.n_edges == base.n_edges   # wiring unchanged
