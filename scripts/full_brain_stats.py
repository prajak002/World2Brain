"""Whole-brain batch A (docs/RUNBOOK_GPU.md A, PREREG H6): blind latent MSE of the full-brain students on the in-house
tasks, with the AGENTS.md rule-4 statistics. Reads results/runs/<wiring>_K1_s<seed>_full[_tn].json and writes
results/full_brain_summary.md + .json.

Per run: horizon-mean blind MSE per test episode (test_mse_per_ep). Per wiring: mean over seeds, sd over seeds, bootstrap
95 % CI over test episodes (10,000 resamples, seeds averaged per episode). Against the connectome: paired difference per
episode (same test episodes, seeds averaged), bootstrap 95 % CI and two-sided bootstrap p, Holm over the five comparisons
of each task. A difference whose CI contains zero is reported as "no difference".
    PYTHONPATH=. python scripts/full_brain_stats.py
"""
import json, glob, os, numpy as np

W = ["connectome", "shuffled", "block_shuffled", "random", "gru", "transformer"]
rng = np.random.default_rng(0); B = 10000
out, lines = {}, ["# Whole-brain students, in-house tasks (batch A)", ""]
for sfx, title in (("_full", "Arena (NAV + ARRANGE)"), ("_full_tn", "Arena, continuous turn angles")):
    per = {}
    for w in W:
        fs = sorted(glob.glob(f"results/runs/{w}_K1_s?{sfx}.json"))
        if not fs: continue
        runs = [json.load(open(f)) for f in fs]
        e = np.stack([np.asarray(r["test_mse_per_ep"], float).reshape(len(r["test_mse_per_ep"]), -1).mean(1) for r in runs])   # [seeds, episodes]
        per[w] = dict(ep=e, params=runs[0]["params"], seeds=len(fs))
    if "connectome" not in per: continue
    n = per["connectome"]["ep"].shape[1]; idx = rng.integers(0, n, (B, n))
    rows, ps = {}, {}
    for w, d in per.items():
        m = d["ep"].mean(0); boot = m[idx].mean(1)
        rows[w] = dict(params=d["params"], seeds=d["seeds"], mse=float(d["ep"].mean()), sd_seeds=float(d["ep"].mean(1).std(ddof=1)) if d["seeds"] > 1 else 0.0,
                       ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
        if w != "connectome":
            k = min(d["seeds"], per["connectome"]["seeds"])
            diff = d["ep"][:k].mean(0) - per["connectome"]["ep"][:k].mean(0); bd = diff[idx].mean(1)
            p = float(min(1.0, 2 * min((bd <= 0).mean(), (bd >= 0).mean()) + 1 / B))
            rows[w].update(diff=float(diff.mean()), diff_ci=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))], p=p); ps[w] = p
    order = sorted(ps, key=ps.get); run_max = 0.0                       # Holm step-down
    for i, w in enumerate(order):
        run_max = max(run_max, min(1.0, (len(order) - i) * ps[w])); rows[w]["p_holm"] = run_max
    out[sfx] = rows
    lines += [f"## {title}", "", f"{n} test episodes; blind window 8:16; MSE in whitened PCA-256 space; lower is better.", "",
              "| wiring | params | seeds | blind MSE (mean over seeds) | sd over seeds | 95 % CI over episodes | diff vs connectome [95 % CI] | p (Holm) | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for w in W:
        if w not in rows: continue
        r = rows[w]
        if w == "connectome": tail = "ref | | |"
        else:
            lo, hi = r["diff_ci"]; verdict = "no difference" if lo <= 0 <= hi or r["p_holm"] >= 0.05 else ("worse than connectome" if r["diff"] > 0 else "better than connectome")
            tail = f"{r['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {r['p_holm']:.4f} | {verdict} |"
        lines.append(f"| {w} | {r['params']:,} | {r['seeds']} | {r['mse']:.4f} | {r['sd_seeds']:.4f} | [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}] | {tail}")
    lines.append("")
os.makedirs("results", exist_ok=True)
open("results/full_brain_summary.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump(out, open("results/full_brain_summary.json", "w"), indent=1)
print("\n".join(lines))
