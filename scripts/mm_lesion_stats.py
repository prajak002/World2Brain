"""Lesion statistics (PREREG Amendment 2) -> results/memmaze/lesion_summary.md + lesion_summary.json.

Input: results/memmaze/lesion/*.json (eval_mm_lesion.py) and the constant predictor's probe JSON (K).
deficit D(L, m) per eval trajectory = loss of score under lesion L (Walls: intact - lesioned accuracy; Objects: lesioned -
intact MSE; > 0 = worse), divided by the condition's intact gap to the constant predictor |intact - K| so both metrics are
"fraction of the above-constant performance lost". Specific effect E(L, m) = D(L, m) - mean over the matched random
lesions of D(rand, m). Tests (bootstrap over eval trajectories, 10,000 resamples, model seeds resampled inside; one-sided;
Holm over {H7a, H7b, H7c}):  H7a E_C(CX, objects) > 0;  H7b E_C(CX, objects) - E_C(MB, objects) > 0;
H7c E_C(CX, objects) - E_S(CX, objects) > 0. Everything else is reported descriptively with 95 % CIs.
"""
import argparse, glob, json, os, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--lesion", default="results/memmaze/lesion"); p.add_argument("--k", default="results/memmaze/probe/K_s0.json")
p.add_argument("--out", default="results/memmaze"); p.add_argument("--n_boot", type=int, default=10000); p.add_argument("--seed", type=int, default=0)
a = p.parse_args(); KEY = dict(walls="walls_acc", objects="objects_mse"); SGN = dict(walls=1, objects=-1)
k = json.load(open(a.k)); K = dict(walls=k["mean_walls"], objects=k["mean_objects"])
R = {}
for f in sorted(glob.glob(f"{a.lesion}/*.json")):
    r = json.load(open(f)); R.setdefault(r["cond"], []).append(r)


def effects(c):
    """-> {(lesion, metric): array [seeds, trajectories]} of normalised deficits, and specific effects for CX / MB."""
    out = {}
    for m in KEY:
        I = np.array([r[KEY[m]]["intact"] for r in R[c]]); gap = abs(I.mean() - K[m])
        D = {n: SGN[m] * (I - np.array([r[KEY[m]][n] for r in R[c]])) / gap for n in R[c][0]["lesions"] if n != "intact"}
        for n in ("CX", "MB", "OUT", "CXdeaf"): out[("D", n, m)] = D[n]
        for g in ("CX", "MB"):
            rnd = np.mean([v for n, v in D.items() if n.startswith(f"rand{g}_")], 0); out[("D", f"rand{g}", m)] = rnd; out[("E", g, m)] = D[g] - rnd
    return out


EFF = {c: effects(c) for c in R}; rng = np.random.default_rng(a.seed)


def boot(terms):
    """terms: list of (sign, cond, key); statistic = sum of sign * mean over seeds and trajectories. Trajectories are
    resampled jointly for all terms, seeds independently per condition."""
    n = next(iter(EFF.values()))[("E", "CX", "objects")].shape[1]; obs = sum(s * EFF[c][key].mean() for s, c, key in terms); out = np.empty(a.n_boot)
    for b in range(a.n_boot):
        t = rng.integers(0, n, n); sd = {c: rng.integers(0, len(R[c]), len(R[c])) for c in R}
        out[b] = sum(s * EFF[c][key][sd[c]][:, t].mean() for s, c, key in terms)
    return dict(est=float(obs), ci=np.percentile(out, [2.5, 97.5]).tolist(), p=float((out <= 0).mean()))


TESTS = [("H7a", [(1, "C", ("E", "CX", "objects"))]), ("H7b", [(1, "C", ("E", "CX", "objects")), (-1, "C", ("E", "MB", "objects"))]),
         ("H7c", [(1, "C", ("E", "CX", "objects")), (-1, "S", ("E", "CX", "objects"))])]
tests = []
for h, terms in TESTS:
    tests.append(dict(h=h, **boot(terms)) if all(c in R for _, c, _ in terms) else dict(h=h, missing=True))
live = sorted([t for t in tests if not t.get("missing")], key=lambda t: t["p"]); prev = 0
for i, t in enumerate(live): prev = max(prev, min(1.0, (len(live) - i) * t["p"])); t["p_holm"] = prev
for t in tests: t["verdict"] = "NOT RUN" if t.get("missing") else ("PASS" if t["p_holm"] < 0.05 and t["ci"][0] > 0 else ("FAIL" if t["ci"][1] < 0 else "INCONCLUSIVE"))
L = ["# Memory Maze in-silico lesions (PREREG Amendment 2)", "", "Values = fraction of the above-constant performance lost (> 0 = worse), mean [95 % CI].", "",
     "| condition | seeds | quantity | lesion | 9x9 Walls | 9x9 Objects |", "|---|---|---|---|---|---|"]
desc = {}
for c in ["C", "S", "B"]:
    if c not in R: continue
    for kind, n in [("D", "CX"), ("D", "randCX"), ("E", "CX"), ("D", "MB"), ("D", "randMB"), ("E", "MB"), ("D", "CXdeaf"), ("D", "OUT")]:
        cells = [boot([(1, c, (kind, n, m))]) for m in ("walls", "objects")]; desc[f"{c}/{kind}/{n}"] = cells
        L.append(f"| {c} | {len(R[c])} | {'deficit' if kind == 'D' else 'specific effect'} | {n} | " + " | ".join(f"{x['est']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]" for x in cells) + " |")
L += ["", "| test | estimate | 95 % CI | p | p Holm | verdict |", "|---|---|---|---|---|---|"]
for t in tests: L.append(f"| {t['h']} | " + ("not run | | | | NOT RUN |" if t.get("missing") else f"{t['est']:.3f} | [{t['ci'][0]:.3f}, {t['ci'][1]:.3f}] | {t['p']:.4f} | {t['p_holm']:.4f} | {t['verdict']} |"))
sz = next(iter(R.values()))[0]["sizes"]; L += ["", "Group sizes: " + "; ".join(f"{g} {sz[g]['neurons']} neurons / {sz[g]['out_edges']} outgoing edges" for g in ("CX", "MB", "OUT"))]
os.makedirs(a.out, exist_ok=True); open(f"{a.out}/lesion_summary.md", "w").write("\n".join(L) + "\n")
json.dump(dict(tests=tests, descriptive=desc), open(f"{a.out}/lesion_summary.json", "w"), indent=1); print("\n".join(L))
