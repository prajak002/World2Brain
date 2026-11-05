"""Blackout Navigation figures (docs/BENCHMARK_BLACKOUT.md section 6) from results/homing/*_k<k>*.json.

Fig. 1  success vs blackout length k, one line per wiring, one panel per world; oracle / dead-reckoning / random-action
        bands. Bootstrap 95% CI over episodes pooled across seeds (1000 resamples).
Fig. 2  end-of-blackout position drift of the decoded pose vs k (the mechanism).
Table   success at every k with CIs -> results/blackout_table.md (also printed).

  PYTHONPATH=. python scripts/make_blackout_figs.py [--out paper_v2/fig] [--readout state] [--tn 1]
"""
import argparse, glob, json, re, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = argparse.ArgumentParser(); p.add_argument("--out", default="paper_v2/fig"); p.add_argument("--readout", default="state")
p.add_argument("--tn", type=int, default=1, help="arena panel: 1 = continuous-turn (_tn) runs, 0 = clean-turn runs"); p.add_argument("--boot", type=int, default=1000)
a = p.parse_args(); rng = np.random.default_rng(0)
NAMES = {"connectome": "connectome", "block_shuffled": "block-shuffled", "degree_shuffled": "degree-shuffled", "random": "random wiring", "dense": "dense",
         "gru": "GRU (param.-matched)", "gru_budget": "GRU (budget-matched)", "lstm": "LSTM", "transformer": "transformer"}
COL = {"connectome": "#c0392b", "block_shuffled": "#e67e22", "degree_shuffled": "#f1c40f", "random": "#7f8c8d", "dense": "#8e44ad",
       "gru": "#2980b9", "gru_budget": "#5dade2", "lstm": "#1abc9c", "transformer": "#27ae60"}

rows = {}   # (env, wiring, readout, k) -> list of per-episode success (pooled over seeds)
for f in glob.glob("results/homing/*_k*.json"):
    d = json.load(open(f)); m = re.match(r"^(.*)_K1_s(\d+)(_tn)?(_nyc)?$", d["run"])
    if not m: continue
    w = m.group(1); env = d["env"]
    if env == "arena" and (d.get("turn_noise", 0) > 0) != bool(a.tn): continue
    ro = d["readout"]; k = d["t_blind"]
    succ = [float(x < d["radius"]) for x in d["per_episode_final"]]
    key = (env, w if ro in ("state", "mlp", "policy") else ro, ro, k); rows.setdefault(key, []).append((succ, d.get("per_episode_drift", [])))

def ci(x):
    x = np.asarray(x, float); b = [rng.choice(x, len(x)).mean() for _ in range(a.boot)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

rows2 = {}  # T2 Continue: (env, wiring/readout, k) -> list of (success, spl) per seed
for f in glob.glob("results/goalnav/*_bumper_k*.json"):
    d = json.load(open(f)); m = re.match(r"^(.*)_K1_s(\d+)(_nyc)?$", d["run"])
    if not m: continue
    w = m.group(1) if d["readout"] in ("state", "mlp", "policy") else d["readout"]
    if d["readout"] in ("state", "mlp", "policy") and d["readout"] != a.readout: continue
    rows2.setdefault((d["env"], w, d["blackout"]), []).append((d["success"], d["spl"]))
envs = [e for e in ("arena", "nyc") if any(k[0] == e for k in rows)]
if not envs: raise SystemExit("no results/homing/*_k*.json yet")
ks = sorted({k[3] for k in rows}); lines = ["| world | row | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|---|" + "---|" * len(ks)]
fig1, ax1 = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 3.4), squeeze=False); fig2, ax2 = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 3.4), squeeze=False)
for j, env in enumerate(envs):
    A1, A2 = ax1[0, j], ax2[0, j]
    for ro, lab, c in (("oracle", "oracle pose (ceiling)", "0.75"), ("actions", "dead reckoning of actions", "0.55"), ("random", "random actions (floor)", "0.35")):
        pts = [(k, ci([s for S, _ in rows[(env, ro, ro, k)] for s in S])) for k in ks if (env, ro, ro, k) in rows]
        if pts: A1.plot([k for k, _ in pts], [m for _, (m, _, _) in pts], "--", color=c, label=lab); lines.append(f"| {env} | {lab} | " + " | ".join(f"{m:.2f}" for _, (m, _, _) in pts) + " |")
    for w in NAMES:
        pts = []
        for k in ks:
            key = (env, w, a.readout, k)
            if key in rows: pts.append((k, ci([s for S, _ in rows[key] for s in S]), ci([v for _, D in rows[key] for v in D]) if any(D for _, D in rows[key]) else None, len(rows[key])))
        if not pts: continue
        kk = [k for k, *_ in pts]; m = [v[0] for _, v, *_ in pts]; lo = [v[1] for _, v, *_ in pts]; hi = [v[2] for _, v, *_ in pts]
        A1.plot(kk, m, "-o", color=COL[w], label=NAMES[w], lw=2 if w == "connectome" else 1.3); A1.fill_between(kk, lo, hi, color=COL[w], alpha=0.15)
        dr = [(k, v) for k, _, v, _ in pts if v]
        if dr: A2.plot([k for k, _ in dr], [v[0] for _, v in dr], "-o", color=COL[w], label=NAMES[w]); A2.fill_between([k for k, _ in dr], [v[1] for _, v in dr], [v[2] for _, v in dr], color=COL[w], alpha=0.15)
        lines.append(f"| {env} | {NAMES[w]} ({max(n for *_, n in pts)} seeds) | " + " | ".join(f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for _, v, *_ in pts) + " |")
    A1.set(xlabel="blackout length k (steps)", ylabel="return success", ylim=(-0.02, 1.02), title=("arena" + (", continuous turns" if a.tn else "")) if env == "arena" else "NYC twin")
    A2.set(xlabel="blackout length k (steps)", ylabel="decoded position drift at end (units)", title=A1.get_title())
    A1.grid(alpha=0.3); A2.grid(alpha=0.3)
ax1[0, -1].legend(fontsize=7, loc="upper right"); ax2[0, -1].legend(fontsize=7)
if rows2:
    fig3, ax3 = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 3.4), squeeze=False); ks2 = sorted({k[2] for k in rows2})
    lines += ["", "T2 Continue (goal seen, then k dark steps; success, mean over seeds)", "| world | row | " + " | ".join(f"k={k}" for k in ks2) + " |", "|---|---|" + "---|" * len(ks2)]
    for j, env in enumerate(envs):
        A = ax3[0, j]
        for w in ["oracle", "random"] + list(NAMES):
            pts = [(k, np.mean([x[0] for x in rows2[(env, w, k)]])) for k in ks2 if (env, w, k) in rows2]
            if not pts: continue
            st = dict(ls="--", color="0.6") if w == "oracle" else dict(ls="--", color="0.35") if w == "random" else dict(ls="-", marker="o", color=COL[w], lw=2 if w == "connectome" else 1.3)
            A.plot([k for k, _ in pts], [v for _, v in pts], label={"oracle": "oracle goal vector", "random": "random actions"}.get(w, NAMES.get(w, w)), **st)
            lines.append(f"| {env} | {NAMES.get(w, w)} | " + " | ".join(f"{v:.2f}" for _, v in pts) + " |")
        A.set(xlabel="blackout length k (steps)", ylabel="goal success", ylim=(-0.02, 1.02), title=("arena" if env == "arena" else "NYC twin") + ", T2 continue"); A.grid(alpha=0.3)
    ax3[0, -1].legend(fontsize=7); fig3.tight_layout(); os.makedirs(a.out, exist_ok=True); fig3.savefig(f"{a.out}/blackout_continue.pdf")
os.makedirs(a.out, exist_ok=True)
fig1.tight_layout(); fig1.savefig(f"{a.out}/blackout_success.pdf"); fig2.tight_layout(); fig2.savefig(f"{a.out}/blackout_drift.pdf")
open("results/blackout_table.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); print(f"wrote {a.out}/blackout_success.pdf, {a.out}/blackout_drift.pdf, results/blackout_table.md")
