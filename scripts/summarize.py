"""Aggregate results/runs, results/baselines, results/hd, results/plan into results/summary.md.

E1 blind prediction per dataset (arena / nyc) with trivial baselines (#6), teacher-space MSE (#6), paired bootstrap
95% CIs over the shared test episodes (#8), seed-aggregated wiring comparison, alternative blind window (#8),
E3 HD probes with action-only / feature-matched controls (#7), E2 planning, E4 scaling.
"""
import json, glob, os, re, numpy as np

RNG = np.random.default_rng(0); NBOOT = 2000


def load(pattern):
    return {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(pattern)) if "smoke" not in f}


def tag_of(name):
    m = re.match(r"^(.*)_K(\d+)_s(\d+)(.*)$", name); return m.group(4) if m else ""


def boot_ci(x, stat=np.mean, n=NBOOT):
    x = np.asarray(x); idx = RNG.integers(0, len(x), (n, len(x))); s = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def paired(a, b):
    """Paired bootstrap over shared test episodes: mean(a-b) with 95% CI and two-sided bootstrap p (a, b: [E] per-episode errors)."""
    d = np.asarray(a) - np.asarray(b); lo, hi = boot_ci(d)
    idx = RNG.integers(0, len(d), (NBOOT, len(d))); s = d[idx].mean(1); p = 2 * min((s <= 0).mean(), (s >= 0).mean())
    return float(d.mean()), lo, hi, float(min(p, 1.0))


def per_ep(r):
    return np.array(r["test_mse_per_ep"]).mean(1) if "test_mse_per_ep" in r else None


runs, base, hd, plan = load("results/runs/*.json"), load("results/baselines/*.json"), load("results/hd/*.json"), load("results/plan/*.json")
L = ["# wm2cx results summary", ""]
tags = sorted({tag_of(n) for n in list(runs) + list(base)}, key=lambda t: (t != "", t))
for tag in tags:
    ds = "arena (NAV + ARRANGE)" if tag == "" else tag.strip("_").upper()
    R = {n: r for n, r in runs.items() if tag_of(n) == tag}; Bz = {n: r for n, r in base.items() if tag_of(n) == tag}
    ref_name = next((n for n in R if n.startswith("connectome_K1_s0")), None); ref = per_ep(R[ref_name]) if ref_name else None
    L += [f"## E1 Blind-horizon latent prediction, {ds} (test; 8 visible, then 1..8 blind steps)", "",
          "MSE in whitened PCA-256 space (students are trained in it) and in the original teacher space (3,840-d ViT-B mean + 2x2 pooled tokens). "
          "95% CI: bootstrap over test episodes of the horizon-mean MSE. Paired: mean difference vs `connectome_K1_s0` over the same episodes "
          "with bootstrap 95% CI and two-sided p. Trivial baselines are in the same space and windows (#6).", "",
          "| run | neurons | params | MSE h1 | MSE h4 | MSE h8 | MSE mean [95% CI] | teacher-space MSE mean | cos h1 | cos h8 | paired Δ vs connectome [95% CI], p |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in list(Bz.items()) + list(R.items()):
        m, c = r["test_mse"], r["test_cos"]; pe = per_ep(r)
        ci = f"{np.mean(m):.4f} [{boot_ci(pe)[0]:.3f}, {boot_ci(pe)[1]:.3f}]" if pe is not None else f"{np.mean(m):.4f}"
        ts = f"{np.mean(r['test_mse_teacher']):.4f}" if "test_mse_teacher" in r else "-"
        if pe is not None and ref is not None and n != ref_name and len(pe) == len(ref):
            d, lo, hi, p = paired(pe, ref); pd = f"{d:+.4f} [{lo:+.3f}, {hi:+.3f}], p={p:.3f}"
        else: pd = "ref" if n == ref_name else "-"
        label = f"**{n}** (trivial)" if n in Bz else n
        L.append(f"| {label} | {r['neurons']} | {r['params']/1e6:.2f}M | {m[0]:.4f} | {m[3]:.4f} | {m[7]:.4f} | {ci} | {ts} | {c[0]:.3f} | {c[7]:.3f} | {pd} |")
    if any("teacher_space_variance" in b for b in Bz.values()):
        v = next(b["teacher_space_variance"] for b in Bz.values() if "teacher_space_variance" in b)
        L += ["", f"Teacher-space per-dimension variance of the test latents (MSE of predicting the test mean): {v:.4f}."]
    # seed-aggregated wiring comparison (K=1)
    by = {}
    for n, r in R.items():
        a = r["args"]
        if a["n_brains"] == 1: by.setdefault(a["wiring"], []).append((np.mean(r["test_mse"]), np.mean(r["test_cos"])))
    L += ["", f"### Wiring comparison, K=1, {ds}: mean over horizons, mean ± sd over seeds (95% t-CI when n ≥ 3)", "",
          "| wiring | seeds | MSE | cos |", "|---|---|---|---|"]
    for w, v in by.items():
        v = np.array(v); sd = v.std(0, ddof=1) if len(v) > 1 else np.zeros(2)
        ci = ""
        if len(v) >= 3:
            from scipy import stats
            h = stats.t.ppf(0.975, len(v) - 1) * sd[0] / np.sqrt(len(v)); ci = f" [{v[:,0].mean()-h:.4f}, {v[:,0].mean()+h:.4f}]"
        L.append(f"| {w} | {len(v)} | {v[:,0].mean():.4f} ± {sd[0]:.4f}{ci} | {v[:,1].mean():.3f} ± {sd[1]:.3f} |")
    # alternative blind windows
    wins = sorted({w for r in list(R.values()) + list(Bz.values()) for w in r.get("test_windows", {})})
    if len(wins) > 1:
        L += ["", f"### Alternative blind windows, {ds} (mean MSE over the window, PCA space / teacher space)", "",
              "| run | " + " | ".join(f"blind {w}" for w in wins) + " |", "|---|" + "---|" * len(wins)]
        for n, r in list(Bz.items()) + list(R.items()):
            if "test_windows" not in r: continue
            cells = [f"{np.mean(r['test_windows'][w]['mse']):.4f} / {np.mean(r['test_windows'][w]['mse_teacher']):.4f}" if w in r["test_windows"] else "-" for w in wins]
            L.append(f"| {n} | " + " | ".join(cells) + " |")
    L.append("")

L += ["## E4 Multi-brain scaling (connectome, seed 0, arena)", "", "| K | neurons | params | MSE mean | cos mean |", "|---|---|---|---|---|"]
for n, r in runs.items():
    if r["args"]["wiring"] == "connectome" and r["args"]["seed"] == 0 and tag_of(n) == "":
        L.append(f"| {r['args']['n_brains']} | {r['neurons']} | {r['params']/1e6:.2f}M | {np.mean(r['test_mse']):.4f} | {np.mean(r['test_cos']):.3f} |")

L += ["", "## E3 Head-direction / position linear probes (median heading error, deg; R² in brackets)", "",
      "`actions only`: ridge on cumulative one-hot actions, no neural state (a leaky integrator of discrete turns exposes relative heading "
      "linearly, so this is the floor any recurrent state must beat, #7). `all@1400`: random 1,400-neuron subset, feature count matched to the GRU.", "",
      "| run | actions only (blind) | vis all | blind all | blind all@1400 | vis EPG | blind EPG | vis CX | blind CX | blind pos R² (all) |", "|---|---|---|---|---|---|---|---|---|---|"]
g = lambda d, k: f"{d[k]['heading_err_deg']:.1f} ({d[k]['heading_r2']:.2f})" if k in d else "-"
for n, d in hd.items():
    L.append(f"| {n} | {g(d,'blind_actions_only')} | {g(d,'vis_all')} | {g(d,'blind_all')} | {g(d,'blind_all@1400')} | {g(d,'vis_EPG')} | {g(d,'blind_EPG')} | {g(d,'vis_CX')} | {g(d,'blind_CX')} | {d['blind_all']['pos_r2']:.2f} |")

L += ["", "## E2 Teacher-free planning (success rate, mean goal distance start -> end)", "",
      "| run | task | policy | dynamics | success | dist start | dist end | episodes |", "|---|---|---|---|---|---|---|---|"]
for n, d in plan.items():
    L.append(f"| {d['run']} | {d['task']} | {d['policy']} | {d.get('dynamics', 'student')} | {d['success']:.2f} | {d['dist_start']:.2f} | {d['dist_end']:.2f} | {d['episodes']} |")
open("results/summary.md", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L).encode("ascii", "replace").decode())
