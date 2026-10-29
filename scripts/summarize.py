"""Aggregate results/runs, results/hd, results/plan into results/summary.md (E1 blind prediction, E3 HD probes, E2 planning, E4 scaling)."""
import json, glob, os, numpy as np

def load(pattern):
    return {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(pattern)) if "smoke" not in f}

runs, hd, plan = load("results/runs/*.json"), load("results/hd/*.json"), load("results/plan/*.json")
L = ["# wm2cx results summary", ""]
L += ["## E1 Blind-horizon latent prediction (test, 8 visible then 1..8 blind steps)", "",
      "| run | neurons | params | MSE h1 | MSE h4 | MSE h8 | cos h1 | cos h8 |", "|---|---|---|---|---|---|---|---|"]
for n, r in runs.items():
    m, c = r["test_mse"], r["test_cos"]
    L.append(f"| {n} | {r['neurons']} | {r['params']/1e6:.2f}M | {m[0]:.4f} | {m[3]:.4f} | {m[7]:.4f} | {c[0]:.3f} | {c[7]:.3f} |")
# seed-averaged wiring comparison
by = {}
for n, r in runs.items():
    a = r["args"]
    if a["n_brains"] == 1: by.setdefault(a["wiring"], []).append((np.mean(r["test_mse"]), np.mean(r["test_cos"])))
L += ["", "### Wiring comparison, K=1, mean over horizons (mean ± sd over seeds)", "", "| wiring | seeds | MSE | cos |", "|---|---|---|---|"]
for w, v in by.items():
    v = np.array(v); sd = v.std(0) if len(v) > 1 else np.zeros(2)
    L.append(f"| {w} | {len(v)} | {v[:,0].mean():.4f} ± {sd[0]:.4f} | {v[:,1].mean():.3f} ± {sd[1]:.3f} |")
L += ["", "## E4 Multi-brain scaling (connectome, seed 0)", "", "| K | neurons | params | MSE mean | cos mean |", "|---|---|---|---|---|"]
for n, r in runs.items():
    if r["args"]["wiring"] == "connectome" and r["args"]["seed"] == 0:
        L.append(f"| {r['args']['n_brains']} | {r['neurons']} | {r['params']/1e6:.2f}M | {np.mean(r['test_mse']):.4f} | {np.mean(r['test_cos']):.3f} |")
L += ["", "## E3 Head-direction / position linear probes (median heading error, deg; R²)", "",
      "| run | vis all | blind all | vis EPG | blind EPG | vis CX | blind CX | blind pos R² (all) |", "|---|---|---|---|---|---|---|---|"]
g = lambda d, k: f"{d[k]['heading_err_deg']:.1f} ({d[k]['heading_r2']:.2f})" if k in d else "-"
for n, d in hd.items():
    L.append(f"| {n} | {g(d,'vis_all')} | {g(d,'blind_all')} | {g(d,'vis_EPG')} | {g(d,'blind_EPG')} | {g(d,'vis_CX')} | {g(d,'blind_CX')} | {d['blind_all']['pos_r2']:.2f} |")
L += ["", "## E2 Teacher-free planning (success rate, mean goal distance start -> end)", "",
      "| run | task | policy | success | dist start | dist end | episodes |", "|---|---|---|---|---|---|---|"]
for n, d in plan.items():
    L.append(f"| {d['run']} | {d['task']} | {d['policy']} | {d['success']:.2f} | {d['dist_start']:.2f} | {d['dist_end']:.2f} | {d['episodes']} |")
open("results/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
