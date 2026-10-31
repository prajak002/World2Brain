"""Fast invariants for the rewiring controls, the edge-list recurrence and legacy-checkpoint loading."""
import numpy as np, torch, pytest
from wm2cx.connectome import load as load_conn
from wm2cx.rewire import maslov_sneppen, block_rewire, erdos_renyi, block_counts
from wm2cx.student import FlyBrain, GRUBaseline, gru_hidden_for, build_edges, WIRINGS

conn = load_conn(); N = len(conn["region"]); E = len(conn["pre"])

def _simple(pre, post):
    return len(set(zip(pre.tolist(), post.tolist()))) == len(pre) and not (pre == post).any()

def test_rewires_keep_exact_edge_count_and_simple_graph():
    for pre, post in [maslov_sneppen(conn["pre"], conn["post"], 0, 2), block_rewire(conn["pre"], conn["post"], conn["region"], 0, 2), erdos_renyi(N, E, 0)]:
        assert len(pre) == E and _simple(pre, post)

def test_switch_rewires_keep_degrees_and_block_rewire_keeps_blocks():
    din, dout = np.bincount(conn["post"], minlength=N), np.bincount(conn["pre"], minlength=N)
    p, q = maslov_sneppen(conn["pre"], conn["post"], 0, 2)
    assert np.array_equal(np.bincount(q, minlength=N), din) and np.array_equal(np.bincount(p, minlength=N), dout)
    p, q = block_rewire(conn["pre"], conn["post"], conn["region"], 0, 2)
    assert np.array_equal(np.bincount(q, minlength=N), din) and block_counts(p, q, conn["region"]) == block_counts(conn["pre"], conn["post"], conn["region"])
    assert block_counts(*maslov_sneppen(conn["pre"], conn["post"], 0, 2), conn["region"]) != block_counts(conn["pre"], conn["post"], conn["region"])

def test_signs_are_dale_consistent_and_edge_fraction_matched():
    real = float((conn["nt"][conn["pre"]] < 0).mean())
    for w in ("connectome", "shuffled", "block_shuffled", "random", "connectome_randsign"):
        pre, post, sign = build_edges(conn, w, seed=1)
        per_neuron = {}
        for a, s in zip(pre.tolist(), sign.tolist()): assert per_neuron.setdefault(a, s) == s  # one sign per presynaptic neuron
        assert abs(float((sign < 0).mean()) - real) < 0.01

def test_edge_list_recurrence_matches_dense_matmul():
    m = FlyBrain(conn, 16, 8, wiring="connectome"); h = torch.randn(3, m.N)
    assert torch.allclose(m.recur(h, m.weights()), h @ m.W().T, atol=1e-4)

def test_w_in_only_feeds_vis_and_params_are_reported():
    m = FlyBrain(conn, 16, 8); pb = m.param_breakdown()
    assert m.W_in.weight.shape[0] == int((conn["region"] == "VIS").sum()) and pb["recurrent"] == E and sum(pb[k] for k in ("recurrent", "input", "output", "other")) == pb["total"]

def test_gru_param_matching():
    fb = FlyBrain(conn, 768, 256); h = gru_hidden_for(fb.n_params(), 768, 256); g = GRUBaseline(768, 256, h)
    assert g.n_params() <= fb.n_params() < GRUBaseline(768, 256, h + 1).n_params()

def test_multibrain_moves_cross_edges():
    pre, post, _ = build_edges(conn, "connectome", n_brains=2)
    assert len(pre) == 2 * E and ((pre // N) != (post // N)).sum() > 0

@pytest.mark.skipif(not __import__("os").path.exists("results/runs/connectome_K1_s0.pt"), reason="no legacy checkpoint")
def test_legacy_checkpoint_loads():
    from wm2cx.ckpt import load_run
    m, args, _, _ = load_run("connectome_K1_s0"); assert m.w_vis is None and m.n_edges == E
