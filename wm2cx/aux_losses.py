"""Auxiliary training objectives that use the circuit structure (issue #21). Both leave the wiring, signs and masks
untouched and add only a small training-time head; they are separately labelled conditions, never the default.

dead reckoning : a linear head on the central-complex neurons (all units for free baselines) predicts, at every blind
                 step, the egocentric displacement since the last visible step (rotated into the heading at that step).
contrastive    : the projected state at the last blind step must match the teacher latent of the first frame after the
                 blackout better than the other episodes' post-blackout latents (InfoNCE over the batch).
"""
import torch, torch.nn.functional as F


def blind_window(vis):
    """vis [B,T] with at most one contiguous blind window per row -> (has [B] bool, t0 [B], t1 [B]); blind = t0..t1-1."""
    blind = vis < 0.5; has = blind.any(1); t0 = blind.float().argmax(1); t1 = t0 + blind.sum(1)
    return has, t0, t1


def dead_reckoning_target(pos, heading, vis):
    """pos [B,T,2], heading [B,T] (rad), vis [B,T] -> (target [B,T,2], mask [B,T]). The reference is the last visible step
    before the blackout; mask is 1 on blind steps of rows that have a window with a visible step before it."""
    has, t0, _ = blind_window(vis); has = has & (t0 > 0); r = (t0 - 1).clamp(min=0); b = torch.arange(len(pos), device=pos.device)
    d = pos - pos[b, r][:, None]; c, s = torch.cos(heading[b, r])[:, None], torch.sin(heading[b, r])[:, None]
    tgt = torch.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)
    return tgt, (vis < 0.5).float() * has[:, None].float()


def dead_reckoning_loss(pred, pos, heading, vis):
    tgt, m = dead_reckoning_target(pos, heading, vis)
    return (((pred - tgt) ** 2).sum(-1) * m).sum() / m.sum().clamp(min=1)


def blackout_contrastive_loss(q, z, vis, tau=0.1):
    """q [B,T,k] projected states, z [B,T,k] teacher latents. Rows whose blind window ends before the last step only."""
    has, _, t1 = blind_window(vis); T = vis.shape[1]; ok = has & (t1 < T); b = torch.nonzero(ok)[:, 0]
    if len(b) < 2: return q.sum() * 0
    a = F.normalize(q[b, t1[b] - 1], dim=-1); p = F.normalize(z[b, t1[b]], dim=-1)
    return F.cross_entropy(a @ p.T / tau, torch.arange(len(b), device=q.device))
