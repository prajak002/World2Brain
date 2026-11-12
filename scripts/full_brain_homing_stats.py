"""Whole-brain batch A (PREREG H6, homing part): linear-readout homing of the full-brain students, with the AGENTS.md
rule-4 statistics. Reads results/homing/<wiring>_K1_s<seed>_full_arena_state.json and ..._full_tn_arena_state_tn.json
(scripts/eval_homing.py, 12 visible + 24 dark steps, 50 episodes, the same episodes for every run) and writes
results/full_brain_homing_summary.md + .json.

Per wiring: success, final distance to the start and decoded-position drift at the end of the blackout; mean over
seeds, sd over seeds. Against the connectome: paired difference per episode (seeds averaged per episode), bootstrap
95 % CI (10,000 resamples) and two-sided bootstrap p, Holm over the five comparisons of each metric and task. A
difference whose CI contains zero is reported as "no difference".
    PYTHONPATH=. python scripts/full_brain_homing_stats.py
"""
import json, glob, numpy as np

W = ["connectome", "shuffled", "block_shuffled", "random", "gru", "transformer"]
METRICS = [("per_episode_final", "final distance to start"), ("per_episode_drift", "decoded-position drift at the end of the blackout")]
rng = np.random.default_rng(0); B = 10000
out, lines = {}, ["# Whole-brain students, linear-readout homing (batch A)", ""]
for sfx, title in (("_full_arena_state", "Arena"), ("_full_tn_arena_state_tn", "Arena, continuous turn angles")):
    runs = {w: [json.load(open(f)) for f in sorted(glob.glob(f"results/homing/{w}_K1_s?{sfx}.json"))] for w in W}
    runs = {w: r for w, r in runs.items() if r}
    if "connectome" not in runs: continue
    j0 = runs["connectome"][0]; radius = j0["radius"]
    rows = {w: dict(seeds=len(r), success=float(np.mean([j["success"] for j in r])), success_seeds=[j["success"] for j in r],
                    start=float(np.mean([j["dist_blind_start"] for j in r]))) for w, r in runs.items()}
    for key, label in METRICS:
        ep = {w: np.stack([np.asarray(j[key], float) for j in r]) for w, r in runs.items()}           # [seeds, episodes]
        n = ep["connectome"].shape[1]; assert all(e.shape[1] == n for e in ep.values()), "episode counts differ"
        idx = rng.integers(0, n, (B, n)); ps = {}
        for w, e in ep.items():
            rows[w][key] = dict(mean=float(e.mean()), sd_seeds=float(e.mean(1).std(ddof=1)) if len(e) > 1 else 0.0, seeds=[float(x) for x in e.mean(1)])
            if w == "connectome": continue
            k = min(len(e), len(ep["connectome"])); diff = e[:k].mean(0) - ep["connectome"][:k].mean(0); bd = diff[idx].mean(1)
            p = float(min(1.0, 2 * min((bd <= 0).mean(), (bd >= 0).mean()) + 1 / B)); ps[w] = p
            rows[w][key].update(diff=float(diff.mean()), diff_ci=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))], p=p)
        order = sorted(ps, key=ps.get); run_max = 0.0                                                 # Holm step-down
        for i, w in enumerate(order):
            run_max = max(run_max, min(1.0, (len(order) - i) * ps[w])); rows[w][key]["p_holm"] = run_max
    out[sfx] = rows
    lines += [f"## {title}", "", f"{j0['t_vis']} visible + {j0['t_blind']} dark steps, {j0['episodes']} episodes per run, success radius {radius}, "
              f"mean distance from the start when the dark begins {rows['connectome']['start']:.2f}. Lower distance and drift are better.", "",
              "| wiring | seeds | success (per seed) | final distance (sd) | diff vs connectome [95 % CI], p (Holm), verdict | drift (sd) | diff vs connectome [95 % CI], p (Holm), verdict |", "|---|---|---|---|---|---|---|"]
    for w in W:
        if w not in rows: continue
        r = rows[w]; cells = []
        for key, _ in METRICS:
            m = r[key]; cells.append(f"{m['mean']:.3f} ({m['sd_seeds']:.3f})")
            if w == "connectome": cells.append("ref")
            else:
                lo, hi = m["diff_ci"]; verdict = "no difference" if lo <= 0 <= hi or m["p_holm"] >= 0.05 else ("worse than connectome" if m["diff"] > 0 else "better than connectome")
                cells.append(f"{m['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}], {m['p_holm']:.4f}, {verdict}")
        lines.append(f"| {w} | {r['seeds']} | {r['success']:.3f} ({', '.join(f'{s:.2f}' for s in r['success_seeds'])}) | " + " | ".join(cells) + " |")
    lines.append("")
open("results/full_brain_homing_summary.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump(out, open("results/full_brain_homing_summary.json", "w"), indent=1)
print("\n".join(lines))
