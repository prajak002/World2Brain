"""Issue #21 auxiliary objectives: targets, masks and the contrastive pairing."""
import math, torch
from wm2cx.aux_losses import blind_window, dead_reckoning_target, dead_reckoning_loss, blackout_contrastive_loss


def vis_with(T, windows):
    v = torch.ones(len(windows), T)
    for b, w in enumerate(windows):
        if w: v[b, w[0]:w[1]] = 0
    return v


def test_dead_reckoning_is_egocentric_and_blind_only():
    T = 10; vis = vis_with(T, [(4, 8), None])
    pos = torch.zeros(2, T, 2); pos[:, :, 1] = torch.arange(T).float()            # walking +y, one unit per step
    heading = torch.full((2, T), math.pi / 2)                                      # facing +y
    tgt, m = dead_reckoning_target(pos, heading, vis)
    assert m[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0] and m[1].sum() == 0
    assert torch.allclose(tgt[0, 4:8], torch.tensor([[1., 0.], [2., 0.], [3., 0.], [4., 0.]]), atol=1e-5)   # forward = +x in the agent frame
    assert dead_reckoning_loss(tgt, pos, heading, vis) == 0 and dead_reckoning_loss(tgt + 1, pos, heading, vis) == 2


def test_contrastive_pairs_state_with_post_blackout_latent():
    T, k = 10, 6; vis = vis_with(T, [(4, 7), (5, 8), (6, 10), None]); torch.manual_seed(0)
    has, t0, t1 = blind_window(vis); assert has.tolist() == [True, True, True, False] and t1.tolist()[:3] == [7, 8, 10]
    z = torch.randn(4, T, k); q = torch.zeros(4, T, k); q[0, 6] = z[0, 7]; q[1, 7] = z[1, 8]     # row 2 ends at T: excluded
    good = blackout_contrastive_loss(q, z, vis); q[0, 6], q[1, 7] = z[1, 8], z[0, 7]
    assert good < 0.01 < blackout_contrastive_loss(q, z, vis)
    assert blackout_contrastive_loss(q[2:], z[2:], vis[2:]) == 0
