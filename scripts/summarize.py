"""Aggregate results/runs, results/hd, results/plan into results/summary.md.
Issue #6: trivial baselines in every prediction table and MSE in the original teacher space.
Issue #8: seed-level mean ± sd and paired bootstrap 95% CIs over the shared test episodes (from *_episodes.npz); second blind window.
Issue #3: recurrent / input / output parameter counts."""
import json, glob, os, numpy as np

def load(pattern):
    return {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(pattern)) if "smoke" not in f}

def boot_ci(x, n=2000, seed=0):
    """95% bootstrap CI of the mean over episodes (x: [E] or [E,H] -> mean over H first)."""
    x = np.asarray(x); x = x.mean(1) if x.ndim == 2 else x; rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def paired_boot(x, y, n=2000, seed=0):
    """95% CI of mean(x - y) over shared episodes (paired)."""
    d = np.asarray(x).mean(1) - np.asarray(y).mean(1); rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]); return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

runs, hd, plan = load("results/runs/*.json"), load("results/hd/*.json"), load("results/plan/*.json")
eps = {n: np.load(f"results/runs/{n}_episodes.npz") for n in runs if os.path.exists(f"results/runs/{n}_episodes.npz")}
L = ["# wm2cx results summary", ""]
L += ["## E1 Blind-horizon latent prediction (test, 8 visible then 1..8 blind steps, PCA-256 whitened space)", "",
      "| run | neurons | params (rec / in / out) | MSE mean [95% CI] | MSE h1 | MSE h8 | cos h1 | cos h8 | raw-space MSE | const-mean | copy-last | alt window MSE |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for n, r in runs.items():
    m, c = r["test_mse"], r["test_cos"]; pb = r.get("param_breakdown"); b = r.get("baselines"); alt = r.get("alt_window")
    ps = f"{pb['recurrent']/1e6:.2f}M / {pb['input']/1e6:.2f}M / {pb['output']/1e6:.2f}M" if pb else f"{r['params']/1e6:.2f}M (legacy, incl. dead W_in rows)"
    ci = " [%.3f, %.3f]" % boot_ci(eps[n]["mse"]) if n in eps else ""
    L.append(f"| {n} | {r['neurons']} | {ps} | {np.mean(m):.4f}{ci} | {m[0]:.4f} | {m[7]:.4f} | {c[0]:.3f} | {c[7]:.3f} | "
             f"{r.get('test_mse_raw', float('nan')):.4f} | {np.mean(b['const_mean']) if b else float('nan'):.4f} | {np.mean(b['copy_last']) if b else float('nan'):.4f} | "
             f"{np.mean(alt['mse']) if alt else float('nan'):.4f} (t0={alt['t0'] if alt else '-'}) |")
L += ["", "Trivial baselines are computed in the same whitened PCA space on the same windows; raw-space MSE is the MSE of the PCA-reconstructed",
      "3,840-d teacher latent (residual variance outside the 256 components is identical for every model and excluded)."]
# seed-averaged wiring comparison, by dataset tag
by = {}
for n, r in runs.items():
    a = r["args"]; tag = a.get("tag", "") or "_arena"
    if a["n_brains"] == 1: by.setdefault((tag, a["wiring"]), []).append(n)
L += ["", "### Wiring comparison, K=1, mean over horizons (mean ± sd over seeds; paired bootstrap 95% CI of the difference vs `connectome`, seed-matched)", "",
      "| data | wiring | seeds | MSE | cos | Δ vs connectome [95% CI] |", "|---|---|---|---|---|---|"]
for (tag, w), names in sorted(by.items()):
    v = np.array([(np.mean(runs[n]["test_mse"]), np.mean(runs[n]["test_cos"])) for n in names]); sd = v.std(0) if len(v) > 1 else np.zeros(2)
    ref = by.get((tag, "connectome"), []); d = ""
    if w != "connectome" and ref:
        pairs = [(n, r) for n in names for r in ref if runs[n]["args"]["seed"] == runs[r]["args"]["seed"] and n in eps and r in eps]
        if pairs:
            ds = np.stack([eps[n]["mse"].mean(1) - eps[r]["mse"].mean(1) for n, r in pairs]).mean(0); rng = np.random.default_rng(0)
            bm = np.array([ds[rng.integers(0, len(ds), len(ds))].mean() for _ in range(2000)]); d = f"{ds.mean():+.4f} [{np.percentile(bm,2.5):+.4f}, {np.percentile(bm,97.5):+.4f}] ({len(pairs)} pairs)"
    L.append(f"| {tag.strip('_')} | {w} | {len(v)} | {v[:,0].mean():.4f} ± {sd[0]:.4f} | {v[:,1].mean():.3f} ± {sd[1]:.3f} | {d} |")
L += ["", "## E4 Multi-brain scaling (connectome, seed 0)", "", "| K | neurons | params | MSE mean | cos mean |", "|---|---|---|---|---|"]
for n, r in runs.items():
    if r["args"]["wiring"] == "connectome" and r["args"]["seed"] == 0 and not r["args"].get("tag"):
        L.append(f"| {r['args']['n_brains']} | {r['neurons']} | {r['params']/1e6:.2f}M | {np.mean(r['test_mse']):.4f} | {np.mean(r['test_cos']):.3f} |")
L += ["", "## E3 Head-direction / position linear probes (median heading error, deg; R²)", "",
      "| run | actions-only (linear) | actions integrated (ceiling) | vis all | blind all | blind matched-762 | vis EPG | blind EPG | vis CX | blind CX | blind pos R² (all) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
g = lambda d, k: f"{d[k]['heading_err_deg']:.1f} ({d[k]['heading_r2']:.2f})" if k in d else "-"
for n, d in hd.items():
    L.append(f"| {n} | {g(d,'blind_actions_only')} | {g(d,'blind_actions_integ')} | {g(d,'vis_all')} | {g(d,'blind_all')} | {g(d,'blind_matched')} | {g(d,'vis_EPG')} | {g(d,'blind_EPG')} | {g(d,'vis_CX')} | {g(d,'blind_CX')} | {d['blind_all']['pos_r2']:.2f} |")
L += ["", "Heading is a deterministic function of the discrete turn actions (the integrated-actions ceiling), so a probe measures whether",
      "the network *represents* the integral linearly, not whether the information exists. `matched-762` uses random 762-neuron subsets (the GRU width)."]
L += ["", "## E2 Teacher-free planning (success rate, SPL, mean goal distance start -> end)", "",
      "| run | task | policy | dynamics | success | SPL | dist start | dist end | steps | episodes |", "|---|---|---|---|---|---|---|---|---|---|"]
for n, d in plan.items():
    L.append(f"| {d['run']} | {d['task']} | {d['policy']} | {d.get('dynamics','student')} | {d['success']:.2f} | {d.get('spl', float('nan')):.2f} | {d['dist_start']:.2f} | {d['dist_end']:.2f} | {d.get('steps_to_goal', float('nan')):.1f} | {d['episodes']} |")
open("results/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
