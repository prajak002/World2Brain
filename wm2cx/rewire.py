"""Edge-exact rewiring controls for the connectome mask (issues #1, #2).

All rewires keep: neuron count, region labels, exact edge count, no self-loops, no duplicate edges,
and every neuron's in- and out-degree (switch / Maslov-Sneppen rewiring).
  * maslov_sneppen(pre, post)         : degree-preserving, global (macro-architecture NOT preserved)
  * block_rewire(pre, post, region)   : degree-preserving *within each (pre-region, post-region) block*,
                                        so the region-level block structure (stochastic-block model) is kept
  * erdos_renyi(N, E)                 : uniform random simple digraph with exactly E edges
Signs are always assigned per *presynaptic* neuron (Dale's law) by the caller.
"""
from __future__ import annotations
import numpy as np


def _switch(pre, post, rng, n_swaps):
    """In-place switch rewiring on an edge list: (a->b, c->d) -> (a->d, c->b). Keeps degrees exactly."""
    E = len(pre); present = set(zip(pre.tolist(), post.tolist())); done = 0; tries = 0
    while done < n_swaps and tries < 20 * n_swaps:
        tries += 1
        i, j = rng.integers(0, E, 2)
        a, b, c, d = int(pre[i]), int(post[i]), int(pre[j]), int(post[j])
        if a == d or c == b or (a, d) in present or (c, b) in present: continue
        present.discard((a, b)); present.discard((c, d)); present.add((a, d)); present.add((c, b))
        post[i], post[j] = d, b; done += 1
    return done


def maslov_sneppen(pre, post, seed=0, n_swaps_per_edge=10):
    rng = np.random.default_rng(seed); pre, post = pre.copy(), post.copy()
    _switch(pre, post, rng, n_swaps_per_edge * len(pre))
    return pre, post


def block_rewire(pre, post, region, seed=0, n_swaps_per_edge=10):
    """Switch rewiring restricted to edges within the same (region[pre], region[post]) block."""
    rng = np.random.default_rng(seed); pre, post = pre.copy(), post.copy()
    blk = np.char.add(np.char.add(region[pre].astype(str), ">"), region[post].astype(str))
    for b in np.unique(blk):
        idx = np.nonzero(blk == b)[0]
        p, q = pre[idx], post[idx]
        _switch(p, q, rng, n_swaps_per_edge * len(idx))
        post[idx] = q
    return pre, post


def erdos_renyi(N, E, seed=0):
    rng = np.random.default_rng(seed); got = set()
    while len(got) < E:
        a, b = rng.integers(0, N, (2, E - len(got)))
        for x, y in zip(a.tolist(), b.tolist()):
            if x != y: got.add((x, y))
            if len(got) == E: break
    e = np.array(sorted(got)); return e[:, 0], e[:, 1]


def block_counts(pre, post, region):
    """{(pre_region, post_region): n_edges} for reporting."""
    blk = list(zip(region[pre].tolist(), region[post].tolist()))
    u, c = np.unique(np.array([f"{a}>{b}" for a, b in blk]), return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))
