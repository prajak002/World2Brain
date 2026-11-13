"""Whole-brain Blackout stage (docs/BENCHMARK_BLACKOUT.md T1 Return and T3 Re-entry) with the AGENTS.md rule-4 statistics.
Reads results/homing/<wiring>_K1_s<seed>_full_tn_arena_<readout>_k<k>_tn.json (scripts/eval_homing.py, 12 visible + k
dark steps, 100 episodes, the same episodes for every run) and results/runs/<wiring>_K1_s<seed>_full_tn_reentry.json;
writes results/full_brain_blackout_summary.md + .json. Nothing else is written (results/blackout_table.md is the
circuit-scale table and is left alone).

Return, per readout and k: success (final distance < radius) and final distance to the start, mean over seeds. Against
the connectome: paired difference per episode (seeds averaged per episode), bootstrap 95 % CI (10,000 resamples) and
two-sided bootstrap p, Holm over the five comparisons of each metric, readout and k. A difference whose CI contains
zero is reported as "no difference". Re-entry: per-seed blind MSE (no per-episode values are stored, so seeds only).
    PYTHONPATH=. python scripts/full_brain_blackout_stats.py
"""
import json, glob, os, numpy as np

W = ["connectome", "shuffled", "block_shuffled", "random", "gru", "transformer"]
KS = [8, 16, 24, 30, 40]
rng = np.random.default_rng(0); B = 10000
out, lines = {"return": {}, "reentry": {}}, ["# Whole-brain students, Blackout benchmark (arena, continuous turn angles)", ""]


def holm(ps):
    order = sorted(ps, key=ps.get); adj, run = {}, 0.0
    for i, w in enumerate(order): run = max(run, min(1.0, ps[w] * (len(order) - i))); adj[w] = run
    return adj


for ro in ("state", "policy"):
    for k in KS:
        runs = {w: [json.load(open(f)) for f in sorted(glob.glob(f"results/homing/{w}_K1_s?_full_tn_arena_{ro}_k{k}_tn.json"))] for w in W}
        runs = {w: r for w, r in runs.items() if r}
        if "connectome" not in runs: continue
        radius = runs["connectome"][0]["radius"]; fin = {w: np.stack([np.asarray(j["per_episode_final"], float) for j in r]) for w, r in runs.items()}
        n = fin["connectome"].shape[1]; assert all(e.shape[1] == n for e in fin.values()), "episode counts differ"
        for w, r in runs.items():
            for j, e in zip(r, fin[w]): assert abs((e < radius).mean() - j["success"]) < 0.0101, f"success differs from final < radius by more than one episode (values are stored rounded to 3 decimals) in {j['run']}"
        idx = rng.integers(0, n, (B, n)); cell = {w: dict(seeds=len(r)) for w, r in runs.items()}
        base = [f"results/homing/connectome_K1_s0_full_tn_arena_{b}_k{k}_tn.json" for b in ("oracle", "actions", "random")]
        cell["_baselines"] = {b: json.load(open(f))["success"] for b, f in zip(("oracle", "actions", "random"), base) if os.path.exists(f)}
        for key, ep in (("success", {w: (e < radius).astype(float) for w, e in fin.items()}), ("final", fin)):
            ps = {}
            for w, e in ep.items():
                cell[w][key] = dict(mean=float(e.mean()), sd_seeds=float(e.mean(1).std(ddof=1)) if len(e) > 1 else 0.0, seeds=[float(x) for x in e.mean(1)])
                if w == "connectome": continue
                d = e.mean(0) - ep["connectome"].mean(0); bs = d[idx].mean(1)
                p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / B)); ps[w] = p
                cell[w][key].update(diff=float(d.mean()), ci=[float(x) for x in np.percentile(bs, [2.5, 97.5])], p=p)
            for w, pa in holm(ps).items(): cell[w][key]["p_holm"] = pa
        out["return"][f"{ro}_k{k}"] = cell

for key, label, fmt in (("success", "success (final distance < radius)", "{:.2f}"), ("final", "final distance to the start", "{:.2f}")):
    for ro in ("state", "policy"):
        lines += [f"## T1 Return, {ro} readout: {label}; other minus connectome [95 % CI], Holm p", "", "| wiring | seeds | " + " | ".join(f"k={k}" for k in KS) + " |", "|---|---|" + "---|" * len(KS)]
        if key == "success":
            for b, name in (("oracle", "oracle pose (ceiling)"), ("actions", "dead reckoning of actions"), ("random", "random actions (floor)")):
                lines.append(f"| {name} | - | " + " | ".join(fmt.format(out["return"][f"{ro}_k{k}"]["_baselines"][b]) if b in out["return"].get(f"{ro}_k{k}", {}).get("_baselines", {}) else "-" for k in KS) + " |")
        for w in W:
            cs = []
            for k in KS:
                c = out["return"].get(f"{ro}_k{k}", {}).get(w)
                if not c: cs.append("-"); continue
                m = c[key]; s = fmt.format(m["mean"])
                if "diff" in m:
                    nd = m["ci"][0] <= 0 <= m["ci"][1] or m["p_holm"] >= 0.05
                    s += f" ({m['diff']:+.2f} [{m['ci'][0]:+.2f}, {m['ci'][1]:+.2f}], p {m['p_holm']:.4f}{', no difference' if nd else ''})"
                cs.append(s)
            sd = next((out["return"][f"{ro}_k{k}"][w]["seeds"] for k in KS if w in out["return"].get(f"{ro}_k{k}", {})), 0)
            lines.append(f"| {w} | {sd} | " + " | ".join(cs) + " |")
        lines.append("")

lines += ["## T3 Re-entry: blind latent MSE over the 8 dark steps (PCA space, 200 test episodes); other minus connectome [95 % CI], Holm p", "",
          "| wiring | seeds | mean blind MSE | h1 | h4 | h8 | vs connectome |", "|---|---|---|---|---|---|---|"]
re_runs = {w: [json.load(open(f)) for f in sorted(glob.glob(f"results/runs/{w}_K1_s?_full_tn_reentry.json"))] for w in W}
re_runs = {w: r for w, r in re_runs.items() if r}
if "connectome" in re_runs:
    ep = {w: np.stack([np.asarray(j["test_mse_per_ep"], float).mean(-1) for j in r]) for w, r in re_runs.items()}
    for b in ("const_mean", "copy_last"):
        f = f"results/baselines/{b}_K0_s0_reentry.json"
        if os.path.exists(f): re_runs[b] = [json.load(open(f))]; ep[b] = np.asarray(re_runs[b][0]["test_mse_per_ep"], float).mean(-1)[None]
    n = ep["connectome"].shape[1]; assert all(e.shape[1] == n for e in ep.values()), "episode counts differ"
    idx = rng.integers(0, n, (B, n)); ps = {}
    for w, e in ep.items():
        h = np.mean([j["test_mse"] for j in re_runs[w]], 0)
        out["reentry"][w] = dict(seeds=len(e), mean=float(e.mean()), seeds_mean=[float(x) for x in e.mean(1)], h1=float(h[0]), h4=float(h[3]), h8=float(h[7]))
        if w == "connectome": continue
        d = e.mean(0) - ep["connectome"].mean(0); bs = d[idx].mean(1)
        ps[w] = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / B))
        out["reentry"][w].update(diff=float(d.mean()), ci=[float(x) for x in np.percentile(bs, [2.5, 97.5])], p=ps[w])
    for w, pa in holm(ps).items(): out["reentry"][w]["p_holm"] = pa
    for w, c in out["reentry"].items():
        v = "-"
        if "diff" in c:
            nd = c["ci"][0] <= 0 <= c["ci"][1] or c["p_holm"] >= 0.05
            v = f"{c['diff']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}], p {c['p_holm']:.4f}{', no difference' if nd else ''}"
        lines.append(f"| {w} | {c['seeds']} | {c['mean']:.4f} | {c['h1']:.4f} | {c['h4']:.4f} | {c['h8']:.4f} | {v} |")

json.dump(out, open("results/full_brain_blackout_summary.json", "w"), indent=1)
open("results/full_brain_blackout_summary.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
