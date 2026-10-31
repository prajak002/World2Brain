"""Build a signed, sparse, multi-region recurrent wiring mask from the FlyWire (FAFB v783) connectome.

Regions ("brains") used as the student's modules (all neuron-level, real root ids):
  VIS : visual projection neurons feeding the central complex ring (TuBu, MeTu) + LC/LPLC optic-glomeruli types
  CX  : central complex (EB/PB/FB/NO neurons: EPG, PEN, PEG, Delta7, PFN*, PFL*, hDelta, vDelta, FC, FS, ER ring neurons, ExR ...)
  MB  : mushroom-body output neurons + dopaminergic neurons (MBON, PAM, PPL1) -- valence / memory
  OUT : descending neurons (DN) -- motor output
Edges come from proofread_connections_783 (synapse counts >= min_syn). Sign is taken from the
presynaptic neuron's predicted neurotransmitter (ACh +, Glu -, GABA -, others +).
"""
from __future__ import annotations
import numpy as np, pandas as pd, pyarrow.feather as feather, json, os

NT_SIGN = {"acetylcholine": +1, "glutamate": -1, "gaba": -1, "dopamine": +1, "serotonin": +1, "octopamine": +1}
REGION_RULES = {
    "VIS": lambda a: a.cell_type.fillna("").str.match(r"^(TuBu|MeTu|LC10|LPLC)"),
    "CX":  lambda a: a.cell_class.eq("CX"),
    "MB":  lambda a: a.cell_class.isin(["MBON", "DAN"]),
    "OUT": lambda a: a.super_class.eq("descending"),
}


def build(ann_path="data/raw/neuron_annotations.tsv", conn_path="data/raw/proofread_connections_783.feather",
          out="data/connectome.npz", min_syn=5, regions=("VIS", "CX", "MB", "OUT")):
    a = pd.read_csv(ann_path, sep="\t", low_memory=False)
    a = a[a.root_id.notna()].copy(); a["root_id"] = a.root_id.astype(np.int64)
    region = pd.Series("", index=a.index)
    for r in regions:
        m = REGION_RULES[r](a) & region.eq("")
        region[m] = r
    sel = a[region.ne("")].copy(); sel["region"] = region[region.ne("")]
    sel = sel.drop_duplicates("root_id").reset_index(drop=True)
    idx = {rid: i for i, rid in enumerate(sel.root_id.values)}
    N = len(sel)
    conn = feather.read_feather(conn_path)
    conn.columns = [c.lower() for c in conn.columns]
    pre_c = [c for c in conn.columns if c.startswith("pre")][0]; post_c = [c for c in conn.columns if c.startswith("post")][0]
    syn_c = [c for c in conn.columns if "syn" in c and c not in (pre_c, post_c)][0]
    conn = conn[conn[pre_c].isin(idx) & conn[post_c].isin(idx)]
    conn = conn.groupby([pre_c, post_c], as_index=False)[syn_c].sum()
    conn = conn[conn[syn_c] >= min_syn]
    pre = conn[pre_c].map(idx).values; post = conn[post_c].map(idx).values; w = conn[syn_c].values.astype(np.float32)
    nt = sel.top_nt.fillna("acetylcholine").map(NT_SIGN).fillna(1).values.astype(np.int8)
    sign = nt[pre]
    side = sel.side.fillna("center").values
    meta = dict(N=int(N), n_edges=int(len(pre)), regions={r: int((sel.region == r).sum()) for r in regions},
                density=float(len(pre) / (N * N)), min_syn=min_syn,
                nt_counts={k: int(v) for k, v in sel.top_nt.value_counts().items()})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, pre=pre, post=post, syn=w, sign=sign, nt=nt, region=sel.region.values.astype("U4"),
                        cell_type=sel.cell_type.fillna("").values.astype("U40"), side=side.astype("U8"),
                        root_id=sel.root_id.values, meta=json.dumps(meta))
    return meta


def load(path="data/connectome.npz"):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build the FlyWire subgraph; --min_syn / --regions for the ablations in #13.")
    p.add_argument("--ann", default="data/raw/neuron_annotations.tsv"); p.add_argument("--conn", default="data/raw/proofread_connections_783.feather")
    p.add_argument("--out", default="data/connectome.npz"); p.add_argument("--min_syn", type=int, default=5)
    p.add_argument("--regions", nargs="+", default=["VIS", "CX", "MB", "OUT"], choices=list(REGION_RULES))
    a = p.parse_args()
    assert "VIS" in a.regions and "CX" in a.regions, "the student needs VIS (visual input) and CX (action input)"
    print(json.dumps(build(a.ann, a.conn, a.out, a.min_syn, tuple(a.regions)), indent=1))
