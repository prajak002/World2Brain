"""LaTeX tables and number macros for the paper, generated from results/ so the manuscript never carries hand-typed numbers.

    PYTHONPATH=. python scripts/make_paper_tables.py            # writes paper/tab/*.tex and paper/num.tex

Run names are  <wiring>_K1_s<seed>[variant][dataset][eval]  with
  variant  _recmatch _attnin _hybrid _big          (same data, different model; extra rows)
  dataset  '' (arena) _tn (continuous turns) _nyc  (training data; columns)
  eval     _realvid _realvid_ft                    (evaluated on the real phone clips)
Probe files (results/hd) add _mlp | _b8_24 | _b4_24 | _rv_transfer | _rv_indomain; homing files (results/homing)
add _<arena|nyc_nyc>_<state|mlp|oracle|actions|random>[_tn].  Missing cells print as --, so the tables can be built
while a run is still in progress and rebuilt when it finishes.
"""
import glob
import json
import os
import re
import statistics as st

OUT = os.path.join("paper", "tab"); os.makedirs(OUT, exist_ok=True)
VARIANTS = ("_recmatch", "_attnin", "_hybrid", "_big")
DATASETS = ("", "_tn", "_nyc")
EVALS = ("_realvid_ft", "_realvid")
HD_EXTRA = ("_rv_transfer", "_rv_indomain", "_b8_24", "_b4_24", "_mlp")
RUN_RE = re.compile(r"^(?P<w>.+?)_K1_s(?P<s>\d)(?P<rest>.*)$")

LABEL = {
    "connectome": r"\textbf{connectome}", "connectome_randsign": "connectome, random signs",
    "connectome_nosign": "connectome, unsigned", "connectome_hybrid": "connectome + cell-type attention",
    "connectome_attnin": "connectome, attention input", "block_shuffled": "block-shuffled",
    "block_shuffled_attnin": "block-shuffled, attention input", "shuffled": "degree-shuffled", "random": "random",
    "dense": "dense", "gru": "GRU, parameter-matched", "gru_recmatch": "GRU, recurrent-budget-matched",
    "gru_attnin": "GRU, attention input", "transformer": "Transformer, parameter-matched",
    "transformer_attnin": "Transformer, attention input",
}
ORDER = ["gru", "transformer", "gru_recmatch", "dense", "connectome_nosign", "connectome_randsign", "connectome",
         "connectome_hybrid", "random", "block_shuffled", "shuffled", "connectome_attnin", "block_shuffled_attnin",
         "gru_attnin", "transformer_attnin"]
NAME = {"connectome": "Conn", "connectome_randsign": "Randsign", "connectome_nosign": "Nosign", "connectome_hybrid": "Hybrid",
        "connectome_attnin": "ConnAttn", "block_shuffled": "Block", "block_shuffled_attnin": "BlockAttn", "shuffled": "Degree",
        "random": "Random", "dense": "Dense", "gru": "Gru", "gru_recmatch": "GruRec", "gru_attnin": "GruAttn",
        "transformer": "Tf", "transformer_attnin": "TfAttn", "copy_last": "CopyLast", "const_mean": "ConstMean",
        "actions_only": "ActOnly", "actions_integrated": "Integrator"}
DNAME = {"": "Arena", "_tn": "Tn", "_nyc": "Nyc", "_realvid": "Rv", "_realvid_ft": "RvFt"}


def parse(name):
    """'gru_K1_s0_attnin_nyc_realvid_ft' -> (label 'gru_attnin', seed 0, dataset '_nyc', eval '_realvid_ft', extra '')"""
    m = RUN_RE.match(name)
    if not m:
        return None
    w, s, rest = m.group("w"), int(m.group("s")), m.group("rest")
    variant = ""
    for v in VARIANTS:
        if v in rest:
            variant, rest = v, rest.replace(v, "", 1)
    extra = ""
    for e in HD_EXTRA:
        if rest.endswith(e):
            extra, rest = e, rest[: -len(e)]
    ev = ""
    for e in EVALS:
        if rest.endswith(e):
            ev, rest = e, rest[: -len(e)]
    if rest not in DATASETS:
        return None                                                       # lr sweeps, n50.., ablation subgraphs: not in the main tables
    return w + variant, s, rest, ev, extra


def msd(v):
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


# ---------------------------------------------------------------- collect
runs, base, hd, homing = {}, {}, {}, {}
for f in glob.glob("results/runs/*.json"):
    k = parse(os.path.basename(f)[:-5])
    if k is None:
        continue
    d = json.load(open(f))
    if "test_mse" not in d:
        continue
    lab, s, ds, ev, _ = k
    runs.setdefault((lab, ds + ev), {})[s] = (st.mean(d["test_mse"][:8]), st.mean(d["test_cos"][:8]))
for f in glob.glob("results/baselines/*_K0_s0*.json"):
    n = os.path.basename(f)[:-5]; w, tag = n.split("_K0_s0")
    d = json.load(open(f)); base[(w, tag)] = (st.mean(d["test_mse"][:8]), st.mean(d["test_cos"][:8]))
for f in glob.glob("results/hd/*.json"):
    k = parse(os.path.basename(f)[:-5])
    if k is None:
        continue
    d = json.load(open(f))
    if "blind_all" not in d:
        continue
    lab, s, ds, ev, extra = k
    key = (lab, ds + ev, extra)
    hd.setdefault(key, {})[s] = d
for f in glob.glob("results/homing/*.json"):
    n = os.path.basename(f)[:-5]
    m = re.match(r"^(.*)_(arena|nyc_nyc)_(state|mlp|oracle|actions|random)(_tn)?$", n)
    if not m:
        continue
    k = parse(m.group(1))
    if k is None:
        continue
    d = json.load(open(f)); lab, s, ds, _, _ = k
    env = {"arena": "", "nyc_nyc": "_nyc"}[m.group(2)] if not m.group(4) else "_tn"
    homing.setdefault((lab, env, m.group(3)), {})[s] = (float(d["success"]), 100 * (1 - float(d["dist_final"]) / float(d["dist_blind_start"])))


# ---------------------------------------------------------------- formatting
def pm(vals, fmt="{:.3f}", sd=True):
    m, s = msd(vals)
    return (fmt + r"\pm" + fmt).format(m, s) if sd and len(vals) > 1 else fmt.format(m)


def best(cells, lo=True):
    """index of the best mean among (mean, text) cells; None if empty"""
    c = [(m, i) for i, (m, _) in enumerate(cells) if m is not None]
    return (min(c) if lo else max(c))[1] if c else None


def table(rows, cols, cell, bold):
    """rows: [(label, key)], cols: [(header, colkey, lower_is_better)], cell(key, colkey) -> (mean, text) or None"""
    grid = [[cell(k, c[1]) for c in cols] for _, k in rows]
    bests = [best([g[j] or (None, "") for g in grid], lo=cols[j][2]) if bold else None for j in range(len(cols))]
    lines = []
    for i, (lab, _) in enumerate(rows):
        cells = []
        for j in range(len(cols)):
            v = grid[i][j]
            t = "--" if v is None else v[1]
            cells.append(r"\textbf{" + t + "}" if v is not None and bests[j] == i else t)
        lines.append("    " + lab + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


macros = []
def macro(name, value):
    macros.append(r"\newcommand{\n" + name + "}{" + value + "}")


def write(fn, body):
    open(os.path.join(OUT, fn), "w").write(body + "\n")


# ---------------------------------------------------------------- 1. prediction
def pred_cell(lab, col):
    ds, met = col
    v = runs.get((lab, ds))
    if not v:
        return None
    vals = [x[0 if met == "mse" else 1] for x in v.values()]
    m = st.mean(vals)
    macro(NAME[lab] + DNAME[ds] + met.capitalize(), f"{m:.3f}" if met == "mse" else f"{m:.2f}")
    return m, pm(vals, "{:.3f}") if met == "mse" else f"{m:.2f}"

def nseeds(ds):
    n = [len(v) for (lab, d), v in runs.items() if d == ds]
    return max(n) if n else 0

cols = [(f"Arena ({nseeds('')} seeds)", ("", "mse"), True), ("", ("", "cos"), False),
        (f"Cont.\\ turns ({nseeds('_tn')} seeds)", ("_tn", "mse"), True), ("", ("_tn", "cos"), False),
        (f"NYC twin ({nseeds('_nyc')} seeds)", ("_nyc", "mse"), True), ("", ("_nyc", "cos"), False)]
rows = [(LABEL[l], l) for l in ORDER if any((l, d) in runs for d in DATASETS)]
body = table(rows, cols, pred_cell, True)
triv = []
for w, lab in [("copy_last", "copy last visible latent"), ("const_mean", "constant mean")]:
    cells = []
    for ds in DATASETS:
        b = base.get((w, ds)); cells += [f"{b[0]:.3f}" if b else "--", "--"]
        if b: macro(NAME[w] + DNAME[ds] + "Mse", f"{b[0]:.3f}")
    triv.append("    " + lab + " & " + " & ".join(cells) + r" \\")
head = (r"\begin{tabular}{lcccccc}" "\n" r"    \toprule" "\n"
        r"    & \multicolumn{2}{c}{" + cols[0][0] + r"} & \multicolumn{2}{c}{" + cols[2][0] + r"} & \multicolumn{2}{c}{" + cols[4][0] + r"} \\" "\n"
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
        r"    wiring & MSE $\downarrow$ & cos $\uparrow$ & MSE $\downarrow$ & cos $\uparrow$ & MSE $\downarrow$ & cos $\uparrow$ \\" "\n" r"    \midrule")
write("pred.tex", head + "\n" + body + "\n" + r"    \midrule" + "\n" + "\n".join(triv) + "\n" + r"    \bottomrule" + "\n" + r"\end{tabular}")

# ---------------------------------------------------------------- 2. heading / position probe (ridge, 8:16)
def hd_vals(lab, ds, extra, field, metric):
    v = hd.get((lab, ds, extra))
    if not v:
        return None
    return [d[field][metric] for d in v.values() if field in d]

def probe_cell(lab, col):
    ds, met = col
    vals = hd_vals(lab, ds, "", "blind_all", met)
    if not vals:
        return None
    m = st.mean(vals)
    macro(NAME[lab] + DNAME[ds] + ("Hd" if met == "heading_err_deg" else "Pos"), f"{m:.1f}" if met == "heading_err_deg" else f"{m:.2f}")
    return m, pm(vals, "{:.1f}" if met == "heading_err_deg" else "{:.2f}")

cols = [("Arena", ("", "heading_err_deg"), True), ("", ("", "pos_r2"), False), ("Cont.\\ turns", ("_tn", "heading_err_deg"), True),
        ("", ("_tn", "pos_r2"), False), ("NYC twin", ("_nyc", "heading_err_deg"), True), ("", ("_nyc", "pos_r2"), False)]
rows = [(LABEL[l], l) for l in ORDER if any((l, d, "") in hd for d in DATASETS)]
body = table(rows, cols, probe_cell, True)
ref = []
for field, lab in [("blind_actions_only", "actions only (linear floor)"), ("blind_actions_integrated", "integrator of the action record")]:
    cells = []
    for ds in DATASETS:
        vals = hd_vals("connectome", ds, "", field, "heading_err_deg"); r2 = hd_vals("connectome", ds, "", field, "pos_r2")
        cells += [f"{st.mean(vals):.1f}" if vals else "--", f"{st.mean(r2):.2f}" if r2 else "--"]
        if vals: macro(NAME[field[6:]] + DNAME[ds] + "Hd", f"{st.mean(vals):.1f}")
    ref.append("    " + lab + " & " + " & ".join(cells) + r" \\")
head = (r"\begin{tabular}{lcccccc}" "\n" r"    \toprule" "\n"
        r"    & \multicolumn{2}{c}{Arena} & \multicolumn{2}{c}{Continuous turns} & \multicolumn{2}{c}{NYC twin} \\" "\n"
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
        r"    wiring & heading err & pos $R^2$ & heading err & pos $R^2$ & heading err & pos $R^2$ \\" "\n" r"    \midrule")
write("probe.tex", head + "\n" + body + "\n" + r"    \midrule" + "\n" + "\n".join(ref) + "\n" + r"    \bottomrule" + "\n" + r"\end{tabular}")

# ---------------------------------------------------------------- 3. MLP probe and dark duration (arena)
def dark_cell(lab, col):
    extra = col
    h = hd_vals(lab, "", extra, "blind_all", "heading_err_deg"); r = hd_vals(lab, "", extra, "blind_all", "pos_r2")
    if not h:
        return None
    tag = {"": "", "_mlp": "Mlp", "_b8_24": "DarkSixteen", "_b4_24": "DarkTwenty"}[extra]
    macro(NAME[lab] + "Arena" + tag + "Hd", f"{st.mean(h):.1f}")
    return st.mean(h), pm(h, "{:.1f}") + ", " + f"{st.mean(r):.2f}"

cols = [("ridge, 8 dark (8:16)", "", True), ("MLP, 8 dark", "_mlp", True), ("ridge, 16 dark (8:24)", "_b8_24", True), ("ridge, 20 dark (4:24)", "_b4_24", True)]
rows = [(LABEL[l], l) for l in ORDER if (l, "", "") in hd]
head = (r"\begin{tabular}{lcccc}" "\n" r"    \toprule" "\n    wiring & " + " & ".join(c[0] for c in cols) + r" \\" "\n" r"    \midrule")
write("dark.tex", head + "\n" + table(rows, cols, dark_cell, True) + "\n" + r"    \bottomrule" + "\n" + r"\end{tabular}")

# ---------------------------------------------------------------- 4. blind homing
def hom_cell(key, col):
    lab, ro = key; env, met = col
    v = homing.get((lab, env, ro))
    if not v:
        return None
    vals = [x[1 if met == "closer" else 0] for x in v.values()]
    m = st.mean(vals)
    if ro == "state": macro(NAME[lab] + DNAME[env] + ("Closer" if met == "closer" else "Succ"), f"{m:.0f}" if met == "closer" else f"{m:.2f}")
    return m, f"{m:.0f}" if met == "closer" else f"{m:.2f}"

cols = [("Arena", ("", "closer"), False), ("", ("", "success"), False), ("Cont.\\ turns", ("_tn", "closer"), False), ("", ("_tn", "success"), False),
        ("NYC twin", ("_nyc", "closer"), False), ("", ("_nyc", "success"), False)]
ctl = [("oracle pose (controller ceiling)", ("connectome", "oracle")), ("dead reckoning of actions", ("connectome", "actions")),
       ("random actions (floor)", ("connectome", "random"))]
rows = [(LABEL[l] + " state", (l, "state")) for l in ORDER if any((l, e, "state") in homing for e in DATASETS)]
head = (r"\begin{tabular}{lcccccc}" "\n" r"    \toprule" "\n"
        r"    & \multicolumn{2}{c}{Arena} & \multicolumn{2}{c}{Continuous turns} & \multicolumn{2}{c}{NYC twin} \\" "\n"
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
        r"    pose readout & \% closer & success & \% closer & success & \% closer & success \\" "\n" r"    \midrule")
write("homing.tex", head + "\n" + table(ctl, cols, hom_cell, False) + "\n" + r"    \midrule" + "\n" + table(rows, cols, hom_cell, True)
      + "\n" + r"    \bottomrule" + "\n" + r"\end{tabular}")

# ---------------------------------------------------------------- 5. real handheld video
def rv_cell(lab, col):
    kind, ds = col
    if kind in ("zs", "ft"):
        v = runs.get((lab, ds + ("_realvid" if kind == "zs" else "_realvid_ft")))
        if not v:
            return None
        vals = [x[0] for x in v.values()]; m = st.mean(vals)
        macro(NAME[lab] + DNAME[ds] + ("Rv" if kind == "zs" else "RvFt") + "Mse", f"{m:.2f}")
        return m, pm(vals, "{:.2f}")
    h = hd_vals(lab, ds, "_rv_" + kind, "blind_all", "heading_err_deg")
    if not h:
        return None
    macro(NAME[lab] + DNAME[ds] + "Rv" + kind.capitalize() + "Hd", f"{st.mean(h):.1f}")
    return st.mean(h), pm(h, "{:.1f}")

cols = [("zero-shot MSE", ("zs", ""), True), ("fine-tuned MSE", ("ft", ""), True), ("heading, sim.\\ readout", ("transfer", ""), True),
        ("heading, clip readout", ("indomain", ""), True), ("NYC model, zero-shot", ("zs", "_nyc"), True), ("NYC model, fine-tuned", ("ft", "_nyc"), True)]
rows = [(LABEL[l], l) for l in ORDER if any((l, d + e) in runs for d in DATASETS for e in EVALS)]
body = table(rows, cols, rv_cell, True)
triv = []
for w, lab in [("copy_last", "copy last visible latent"), ("const_mean", "constant mean")]:
    b = base.get((w, "_realvid"))
    if b: macro(NAME[w] + "RvMse", f"{b[0]:.2f}")
    triv.append("    " + lab + " & " + (f"{b[0]:.2f}" if b else "--") + " & " + (f"{b[0]:.2f}" if b else "--") + r" & -- & -- & " + (f"{b[0]:.2f}" if b else "--") + " & " + (f"{b[0]:.2f}" if b else "--") + r" \\")
act = hd_vals("connectome", "", "_rv_indomain", "blind_actions_only", "heading_err_deg")
if act:
    triv.append(r"    actions only (linear floor) & -- & -- & -- & " + f"{st.mean(act):.1f}" + r" & -- & -- \\"); macro("ActOnlyRvHd", f"{st.mean(act):.1f}")
head = (r"\begin{tabular}{lcccccc}" "\n" r"    \toprule" "\n    wiring & " + " & ".join(c[0] for c in cols) + r" \\" "\n" r"    \midrule")
write("realvid.tex", head + "\n" + body + "\n" + r"    \midrule" + "\n" + "\n".join(triv) + "\n" + r"    \bottomrule" + "\n" + r"\end{tabular}")

# ---------------------------------------------------------------- derived numbers for the prose
def rel(a, b):
    return 100 * (b - a) / b

for ds in DATASETS:
    c = runs.get(("connectome", ds)); g = runs.get(("gru", ds)); t = runs.get(("transformer", ds)); b = runs.get(("block_shuffled", ds))
    if c and g: macro("GruBetter" + DNAME[ds], f"{rel(st.mean(x[0] for x in g.values()), st.mean(x[0] for x in c.values())):.0f}")
    if c and t: macro("TfBetter" + DNAME[ds], f"{rel(st.mean(x[0] for x in t.values()), st.mean(x[0] for x in c.values())):.0f}")
    if c and b: macro("ConnOverBlock" + DNAME[ds], f"{rel(st.mean(x[0] for x in c.values()), st.mean(x[0] for x in b.values())):.1f}")
    for extra, tag in [("", ""), ("_b8_24", "DarkSixteen")]:
        hc = hd_vals("connectome", ds, extra, "blind_all", "heading_err_deg"); hb = hd_vals("block_shuffled", ds, extra, "blind_all", "heading_err_deg")
        if hc and hb: macro("ConnBlockGap" + DNAME[ds] + tag, f"{st.mean(hb) - st.mean(hc):.1f}")
macro("Seeds", str(nseeds(""))); macro("SeedsNyc", str(nseeds("_nyc"))); macro("SeedsTn", str(nseeds("_tn")))
rv = [json.load(open(f)) for f in glob.glob("results/runs/connectome_K1_s0_realvid.json")]
if rv:
    macro("RvClips", str(rv[0]["n_clips"])); macro("RvEpisodes", str(rv[0]["n_test_episodes"]))
macro("Checkpoints", str(len(glob.glob("results/runs/*.pt"))))

seen = set(); out = []
for m in macros:                                                   # keep the first definition of each macro
    k = m.split("}")[0]
    if k not in seen:
        seen.add(k); out.append(m)
open(os.path.join("paper", "num.tex"), "w").write("% generated by scripts/make_paper_tables.py; do not edit\n" + "\n".join(out) + "\n")
print(f"wrote {sorted(os.listdir(OUT))} and paper/num.tex ({len(out)} macros); runs {len(runs)} hd {len(hd)} homing {len(homing)}")
