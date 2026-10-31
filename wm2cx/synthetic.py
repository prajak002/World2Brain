"""Synthetic stand-in for data/connectome.npz with the same schema and population sizes as the FlyWire v783 subgraph.

For unit tests and compute benchmarks only (shape / cost realism, no biology): random edges, not the real wiring.
Population sizes, edge count and sign fraction follow the real build (6,935 neurons: VIS 2,330, CX 2,875, MB 427,
OUT 1,303; 91,663 edges with >= 5 synapses; 49.9 % excitatory).
"""
from __future__ import annotations
import json, numpy as np

REGIONS = {"VIS": 2330, "CX": 2875, "MB": 427, "OUT": 1303}


def synthetic_connectome(n_edges=91_663, regions=REGIONS, frac_exc=0.499, n_epg=51, seed=0, n_brains_hint=1):
    rng = np.random.default_rng(seed)
    region = np.concatenate([np.full(n, r, dtype="U4") for r, n in regions.items()]); N = len(region)
    pre = rng.integers(0, N, n_edges); post = rng.integers(0, N, n_edges)
    nt = np.where(rng.random(N) < frac_exc, 1, -1).astype(np.int8); sign = nt[pre]
    side = rng.choice(np.array(["left", "right", "center"], dtype="U8"), N, p=[0.48, 0.48, 0.04])
    cell_type = np.full(N, "", dtype="U40"); cx = np.where(region == "CX")[0]
    cell_type[rng.choice(cx, n_epg, replace=False)] = "EPG"
    meta = dict(N=int(N), n_edges=int(n_edges), regions={r: int(n) for r, n in regions.items()}, synthetic=True)
    return dict(pre=pre, post=post, syn=rng.integers(5, 60, n_edges).astype(np.float32), sign=sign, nt=nt, region=region,
                cell_type=cell_type, side=side, root_id=np.arange(N, dtype=np.int64), meta=json.dumps(meta))
