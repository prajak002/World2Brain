"""Issue #13: subgraph ablations of the connectome used by the student.
Builds data/connectome_<tag>.npz for min_syn in {1, 5, 10} and for region sets without MB / without OUT / CX only.
Train with `scripts/train.py --conn data/connectome_<tag>.npz --tag _<tag>`."""
import json
from wm2cx.connectome import build

VARIANTS = {"syn1": dict(min_syn=1), "syn5": dict(min_syn=5), "syn10": dict(min_syn=10),
            "noMB": dict(regions=("VIS", "CX", "OUT")), "noOUT": dict(regions=("VIS", "CX", "MB")), "VIS_CX": dict(regions=("VIS", "CX"))}
if __name__ == "__main__":
    for tag, kw in VARIANTS.items():
        meta = build(out=f"data/connectome_{tag}.npz", **kw)
        print(tag, json.dumps(dict(N=meta["N"], edges=meta["n_edges"], regions=meta["regions"], density=round(meta["density"], 5))), flush=True)
