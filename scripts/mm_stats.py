"""Memory Maze statistics (PREREG section 5) -> results/memmaze/summary.md + summary.json.

Input: results/memmaze/probe/*.json (per-eval-trajectory scores; label <cond>_s<seed>...; one probe per model seed).
Per condition: mean +- std over seeds of the trajectory-averaged score (Table-4 format). Tests: paired bootstrap over the
1,000 eval trajectories (10,000 resamples; model seeds resampled with replacement inside every resample), difference
oriented so that > 0 means "C better" (accuracy C - X, MSE X - C); one-sided p = fraction of resamples <= 0. Holm over the
family {H1, H2a, H2b, H3, H4a, H4b} at alpha 0.05. PASS requires the Holm test to pass and the 95 % CI to exclude 0;
FAIL = CI excludes 0 in the wrong direction; otherwise INCONCLUSIVE. H5 is descriptive (C objects MSE vs published 6.2).
"""
import argparse, glob, json, os, re, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--probe", default="results/memmaze/probe"); p.add_argument("--out", default="results/memmaze")
p.add_argument("--allow_partial", type=int, default=0, help="smoke tests only: accept fewer than 1000 eval trajectories")
p.add_argument("--n_boot", type=int, default=10000); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
PUB = [("Constant", 80.8, 23.9), ("VAE (no memory)", 84.1, 17.2), ("RSSM", 89.0, 8.3), ("VAE+GRU (TBTT)", 90.0, 7.8), ("RSSM (TBTT)", 94.0, 6.2), ("Supervised oracle", 99.7, 0.3)]
FAMILY = [("H1", "objects", "G"), ("H2a", "objects", "S"), ("H2b", "objects", "B"), ("H3", "walls", "G"), ("H4a", "objects", "N"), ("H4b", "walls", "N")]
KEY = dict(walls="walls_acc", objects="objects_mse")
runs = {}
for f in sorted(glob.glob(f"{a.probe}/*.json")):
    r = json.load(open(f)); runs.setdefault(r["cond"], []).append(r)
S = {c: {m: np.array([r[KEY[m]] for r in rs]) for m in KEY} for c, rs in runs.items()}       # [seeds, trajectories]
for c, d in S.items(): assert d["walls"].shape[1] == 1000 or a.allow_partial, f"{c}: {d['walls'].shape[1]} eval trajectories, expected 1000"
rng = np.random.default_rng(a.seed)


def boot(c, x, m):
    A, B = S[c][m], S[x][m]; n = A.shape[1]; sgn = 1 if m == "walls" else -1
    obs = sgn * (A.mean(0) - B.mean(0)).mean(); out = np.empty(a.n_boot)
    for b in range(a.n_boot):
        t = rng.integers(0, n, n); sa = rng.integers(0, len(A), len(A)); sb = rng.integers(0, len(B), len(B))
        out[b] = sgn * (A[sa][:, t].mean() - B[sb][:, t].mean())
    return obs, np.percentile(out, [2.5, 97.5]), float((out <= 0).mean())


tests = []
for h, m, x in FAMILY:
    if "C" in S and x in S:
        d, ci, pv = boot("C", x, m); tests.append(dict(h=h, metric=m, vs=x, diff=float(d), ci=ci.tolist(), p=pv))
    else: tests.append(dict(h=h, metric=m, vs=x, missing=True))
live = sorted([t for t in tests if not t.get("missing")], key=lambda t: t["p"]); k = len(live); prev = 0
for i, t in enumerate(live):  # Holm step-down (monotone adjusted p)
    prev = max(prev, min(1.0, (k - i) * t["p"])); t["p_holm"] = prev
for t in tests:
    if t.get("missing"): t["verdict"] = "NOT RUN"; continue
    t["verdict"] = "PASS" if (t["p_holm"] < 0.05 and t["ci"][0] > 0) else ("FAIL" if t["ci"][1] < 0 else "INCONCLUSIVE")
verdict = {t["h"]: t["verdict"] for t in tests}
H = {"H1": verdict["H1"], "H2": "PASS" if verdict["H2a"] == verdict["H2b"] == "PASS" else ("FAIL" if "FAIL" in (verdict["H2a"], verdict["H2b"]) else "INCONCLUSIVE"),
     "H3": verdict["H3"], "H4": "PASS" if verdict["H4a"] == verdict["H4b"] == "PASS" else ("FAIL" if "FAIL" in (verdict["H4a"], verdict["H4b"]) else "INCONCLUSIVE")}
if "C" in S: c_obj = float(S["C"]["objects"].mean()); H["H5"] = f"descriptive: C objects MSE {c_obj:.2f} vs published RSSM (TBTT) 6.2 -> {'below' if c_obj < 6.2 else 'not below'}"
L = ["# Memory Maze 9x9 offline probing (PREREG sections 2-5)", "", "| condition | seeds | 9x9 Walls acc % | 9x9 Objects MSE |", "|---|---|---|---|"]
for c in ["C", "S", "B", "R", "G", "T", "N", "K"]:
    if c in S:
        w, o = S[c]["walls"].mean(1), S[c]["objects"].mean(1)
        L.append(f"| {c} | {len(w)} | {w.mean():.1f} +- {w.std():.1f} | {o.mean():.2f} +- {o.std():.2f} |")
L += ["", "Published (Pasukonis et al. 2023, Table 4, 3 seeds):", "", "| model | 9x9 Walls % | 9x9 Objects MSE |", "|---|---|---|"]
L += [f"| {n} | {w} | {o} |" for n, w, o in PUB]
L += ["", "| test | metric | C vs | diff (>0 = C better) | 95 % CI | p | p Holm | verdict |", "|---|---|---|---|---|---|---|---|"]
for t in tests:
    L.append(f"| {t['h']} | {t['metric']} | {t['vs']} | " + ("not run | | | | NOT RUN |" if t.get("missing") else
             f"{t['diff']:.3f} | [{t['ci'][0]:.3f}, {t['ci'][1]:.3f}] | {t['p']:.4f} | {t['p_holm']:.4f} | {t['verdict']} |"))
L += ["", "Hypotheses: " + "; ".join(f"{k} {v}" for k, v in H.items())]
os.makedirs(a.out, exist_ok=True); open(f"{a.out}/summary.md", "w").write("\n".join(L) + "\n")
json.dump(dict(tests=tests, hypotheses=H, conditions={c: dict(seeds=len(d["walls"]), walls=d["walls"].mean(1).tolist(), objects=d["objects"].mean(1).tolist()) for c, d in S.items()}),
          open(f"{a.out}/summary.json", "w"), indent=1)
print("\n".join(L))
