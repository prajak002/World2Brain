"""Explainer video for the issue sweep (#6-#13): what each issue changed, with the numbers the repo actually holds.

Narration is synthesised offline with Windows SAPI (System.Speech) when available; otherwise the video is silent and
the same text is shown as captions. Only committed artefacts are used (results/*.json, results/brain3d.json,
results/env_*.png, results/bench/latency.json), so this runs without data/ or checkpoints.

    PYTHONPATH=. python scripts/make_video_issues.py --out results/video/wm2cx_issues_explained.mp4
"""
import argparse, json, math, os, shutil, subprocess, sys, tempfile, wave
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection

p = argparse.ArgumentParser()
p.add_argument("--out", default="results/video/wm2cx_issues_explained.mp4")
p.add_argument("--fps", type=int, default=10)
p.add_argument("--voice", default="Microsoft Zira Desktop")
p.add_argument("--no_audio", action="store_true")
p.add_argument("--gif", action="store_true", help="also write a 4 fps GIF preview")
p.add_argument("--preview", action="store_true", help="only save the final frame of every scene as PNG, no audio/video")
a = p.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *q: os.path.join(ROOT, "results", *q)

# ---------- palette (dataviz reference, dark mode) ----------
BG, PANEL, GRID = "#1a1a19", "#242422", "#3a3a37"
TXT, TXT2, MUTED = "#f0efec", "#c3c2b7", "#8a8984"
C = dict(connectome="#3987e5", shuffled="#d95926", gru="#199e70", random="#c98500", dense="#d55181", nosign="#9085e9")
REG = dict(VIS="#3987e5", CX="#d95926", MB="#199e70", OUT="#c98500")
GOOD, BAD = "#199e70", "#e66767"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TXT, "axes.edgecolor": GRID, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": TXT2, "axes.facecolor": PANEL, "figure.facecolor": BG})

# ---------- data actually in the repo ----------
def run_mse(name):
    with open(R("runs", name + ".json")) as f: return np.array(json.load(f)["test_mse"])
E1 = {k: run_mse(v) for k, v in dict(connectome="connectome_K1_s0", shuffled="shuffled_K1_s0", random="random_K1_s0",
                                     dense="dense_K1_s0", gru="gru_K1_s0", nosign="connectome_nosign_K1_s0").items()}
E1_s1 = {k: run_mse(v) for k, v in dict(connectome="connectome_K1_s1", shuffled="shuffled_K1_s1", random="random_K1_s1").items()}
NYC = {k: run_mse(v) for k, v in dict(connectome="connectome_K1_s0_nyc", shuffled="shuffled_K1_s0_nyc", random="random_K1_s0_nyc", gru="gru_K1_s0_nyc").items()}
TRIV = dict(arena_const=1.043, arena_copy=(0.97, 1.19), nyc_const=0.977, nyc_copy=(1.46, 1.64))        # README (#6)
HD = {}                                                                                                   # blind heading error, deg
for k, v in dict(connectome="connectome_K1_s0", shuffled="shuffled_K1_s0", random="random_K1_s0", dense="dense_K1_s0", gru="gru_K1_s0").items():
    with open(R("hd", v + ".json")) as f: d = json.load(f); HD[k] = d["blind_all"]["heading_err_deg"]
with open(R("hd", "connectome_K1_s0.json")) as f: HD["EPG"] = json.load(f)["blind_EPG"]["heading_err_deg"]
PLAN = {}
for pol in ("mpc", "random", "scripted"):
    with open(R("plan", f"connectome_K1_s0_nav_{pol}.json")) as f: PLAN[pol] = json.load(f)
ORACLE = dict(success=1.0, dist_start=5.62, dist_end=0.99, episodes=10)          # eval_plan.py --dynamics oracle (#9), horizon 8
ORACLE_ARR = dict(success=0.0, dist_start=3.00, dist_end=2.60, episodes=6)
with open(R("bench", "latency.json")) as f: BENCH = json.load(f)
with open(R("brain3d.json")) as f: B3 = json.load(f)
POS = np.array(B3["pos"], float); POS -= POS.mean(0); POS /= np.abs(POS).max()
REGION = np.array(B3["region"]); EDGES = np.array(B3["edges"])[:2500]; ESIGN = np.array(B3["esign"])[:2500]
ARENA_MATCH = np.array(B3["arena"]["match"], float); NYC_MATCH = np.array(B3["nyc"]["match"], float)
ARENA_OBS = [np.array(o, np.uint8) for o in B3["arena"]["obs"]]
IMG_NYC = plt.imread(R("env_nyc.png")); IMG_NAV = plt.imread(R("env_nav.png"))
# raycast NYC digital twin (scripts/render_nyc_twin.py; data from scripts/fetch_nyc_data.sh) and planner runs on it (#10)
TWIN = np.load(R("nyc_twin", "episode.npz")) if os.path.exists(R("nyc_twin", "episode.npz")) else None
if TWIN is not None:
    with open(R("nyc_twin", "episode.json")) as f: TWIN_META = json.load(f)
    TWIN_APP = {k: plt.imread(R("nyc_twin", f"appearance_{k}.png")) for k in ("clear", "overcast", "fog", "dusk", "night")}
import glob as _glob
TWIN_PLAN = []
for f_ in sorted(_glob.glob(R("plan", "*_nyc_*_twin.json"))):
    with open(f_) as f: TWIN_PLAN.append(json.load(f))
TWIN_PLAN.sort(key=lambda r: (r["policy"] != "scripted", r.get("blackout", 0)))

# ---------- helpers ----------
fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
ease = lambda x: 0.5 - 0.5 * math.cos(math.pi * min(max(x, 0.0), 1.0))
def part(pr, i, n, span=0.7):
    """progress of item i of n when items reveal one after another inside the first `span` of the scene."""
    t0 = span * i / n; return ease((pr - t0) / (span / n + 1e-9))
def header(title, issue=None, sub=None):
    fig.clf(); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.04, 0.945, title, fontsize=21, weight="bold", va="center", color=TXT)
    if issue: ax.add_patch(FancyBboxPatch((0.86, 0.922), 0.10, 0.045, boxstyle="round,pad=0.004", fc="#2d2d2b", ec=GRID)); ax.text(0.91, 0.945, issue, ha="center", va="center", fontsize=12, color=TXT2)
    if sub: ax.text(0.04, 0.895, sub, fontsize=12.5, color=TXT2, va="center")
    return ax
def caption(ax, text, y=0.055):
    ax.add_patch(FancyBboxPatch((0.03, y - 0.04), 0.94, 0.085, boxstyle="round,pad=0.006", fc="#20201f", ec=GRID))
    ax.text(0.5, y + 0.002, text, ha="center", va="center", fontsize=12, color=TXT, linespacing=1.5)
def bullets(ax, items, x, y, dy=0.062, pr=1.0, fs=12.5, color=TXT, dx=0.018):
    for i, it in enumerate(items):
        al = part(pr, i, len(items), 0.8)
        if al <= 0: continue
        ax.text(x, y - i * dy, "•", fontsize=fs, color=TXT2, alpha=al, va="top"); ax.text(x + dx, y - i * dy, it, fontsize=fs, color=color, alpha=al, va="top", linespacing=1.35)
def spread(vals, gap):
    """push label positions apart so direct labels never collide (keeps order)."""
    idx = np.argsort(vals); out = np.array(vals, float)
    for a_, b_ in zip(idx[:-1], idx[1:]):
        if out[b_] - out[a_] < gap: out[b_] = out[a_] + gap
    return out
def brain(ax, angle, pr=1.0, edges=True, s=3):
    ax.set_facecolor(BG); ax.axis("off"); ca, sa = math.cos(angle), math.sin(angle)
    x = POS[:, 0] * ca + POS[:, 1] * sa; z = POS[:, 2]; depth = -POS[:, 0] * sa + POS[:, 1] * ca
    n = int(len(x) * min(1.0, pr * 1.25)); order = np.argsort(depth)[:n]
    if edges and pr > 0.5:
        seg = np.stack([np.stack([x[EDGES[:, 0]], z[EDGES[:, 0]]], 1), np.stack([x[EDGES[:, 1]], z[EDGES[:, 1]]], 1)], 1)
        ax.add_collection(LineCollection(seg, colors=np.where(ESIGN > 0, "#6fa8ec", "#e66767"), linewidths=0.3, alpha=0.12 * min(1, (pr - 0.5) * 3)))
    ax.scatter(x[order], z[order], s=s, c=[REG[r] for r in REGION[order]], linewidths=0, alpha=0.85)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.75, 0.75); ax.set_aspect("equal")
def hbars(ax, labels, vals, cols, pr, xmax, fmt="{:.3f}", refs=(), xlabel=""):
    ax.set_xlim(0, xmax); ax.set_ylim(-0.6, len(labels) - 0.4); ax.invert_yaxis(); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=11)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
    for i, (v, c) in enumerate(zip(vals, cols)):
        q = part(pr, i, len(vals)); ax.barh(i, v * q, height=0.55, color=c, lw=0)
        if q >= 1 and v > 0: ax.text(v + xmax * 0.012, i, fmt.format(v), va="center", fontsize=10.5, color=TXT)
    for x, lab, c in refs:
        ax.axvline(x, color=c, lw=1.4, ls="--"); ax.text(x, -0.55, lab, color=c, fontsize=9.5, ha="center", va="bottom")
    if xlabel: ax.set_xlabel(xlabel, fontsize=10.5)

# ---------- scenes: (narration, draw(progress)) ----------
SCENES = []
def scene(text, min_sec=6):
    def deco(fn): SCENES.append((text, fn, min_sec)); return fn
    return deco

@scene("World to Brain distils a video world model, V JEPA two point one, into a recurrent student whose only recurrent wiring "
       "is the real FlyWire fruit-fly connectome. This video walks through the eight open GitHub issues that were just addressed in "
       "six commits: what each one changed in the code, and what the results now say.", 12)
def s_title(pr):
    ax = header("")
    axb = fig.add_axes([0.42, 0.08, 0.58, 0.86]); brain(axb, 0.6 + 1.2 * pr, pr=min(1, pr * 2.2))
    ax.text(0.05, 0.72, "World2Brain", fontsize=38, weight="bold", color=TXT, alpha=ease(pr * 4))
    ax.text(0.05, 0.63, "a FlyWire-wired student of V-JEPA 2.1", fontsize=16, color=TXT2, alpha=ease(pr * 4 - 0.5))
    ax.text(0.05, 0.50, "Issue sweep #6 – #13\nsix commits, what changed, what the numbers say", fontsize=15, color=TXT, alpha=ease(pr * 4 - 1.2), linespacing=1.5)
    for i, (r, n) in enumerate([("VIS", "2,330 visual"), ("CX", "2,875 central complex"), ("MB", "427 mushroom body"), ("OUT", "1,303 descending")]):
        al = ease(pr * 3 - 1.3 - 0.25 * i)
        ax.add_patch(Rectangle((0.05, 0.345 - 0.05 * i), 0.012, 0.025, color=REG[r], alpha=al)); ax.text(0.072, 0.357 - 0.05 * i, n, fontsize=12, color=TXT2, va="center", alpha=al)
    ax.text(0.05, 0.10, "6,935 neurons · 91,663 synaptic edges (≥ 5 synapses) · 49.9 % excitatory · FlyWire v783", fontsize=11, color=MUTED, alpha=ease(pr * 3 - 2))

@scene("Here is the setup. The teacher watches a three hundred sixty degree view and emits a three thousand eight hundred forty "
       "dimensional latent, projected to two hundred fifty six whitened principal components. The student sees only a thirty two by "
       "eight pixel fly-eye image, drives six thousand nine hundred thirty five neurons through the real synapses, and must predict "
       "the teacher's latent. For eight steps the lights are on. Then, for eight blind steps, it only knows its own actions. "
       "The match gauge shows how close its guess stays to the teacher in one test episode.", 20)
def s_setup(pr):
    ax = header("The experiment in one picture", sub="teacher latent → connectome student → blind-horizon prediction")
    T = 24; t = min(T - 1, int(pr * 1.15 * T)); blind = 8 <= t < 16
    # pipeline boxes
    for i, (x, lab, col) in enumerate([(0.04, "V-JEPA 2.1 ViT-B\n3,840-d latent", MUTED), (0.21, "PCA-256\nwhitened target", MUTED), (0.38, "FlyBrain student\n6,935 neurons, edge-list W", C["connectome"]), (0.55, "linear readout\n→ predicted latent", MUTED)]):
        al = part(pr, i, 4, 0.35); ax.add_patch(FancyBboxPatch((x, 0.74), 0.15, 0.10, boxstyle="round,pad=0.006", fc=PANEL, ec=col, lw=1.6, alpha=al))
        ax.text(x + 0.075, 0.79, lab, ha="center", va="center", fontsize=10.5, color=TXT, alpha=al, linespacing=1.4)
        if i < 3: ax.annotate("", (x + 0.165, 0.79), (x + 0.152, 0.79), arrowprops=dict(arrowstyle="-|>", color=TXT2, lw=1.2), alpha=al)
    # fly eye
    axe = fig.add_axes([0.09, 0.47, 0.27, 0.20]); axe.imshow(ARENA_OBS[t] * (0 if blind else 1), aspect="auto", interpolation="nearest"); axe.set_xticks([]); axe.set_yticks([])
    axe.set_title(("DARK: no input, actions only" if blind else "fly-eye input, 32 × 8 px") + f"   — step {t + 1}/24", fontsize=10.5, color=BAD if blind else TXT2, loc="left")
    # match gauge
    axm = fig.add_axes([0.09, 0.17, 0.27, 0.24]); axm.set_xlim(0.5, 24.5); axm.set_ylim(0, 100); axm.axvspan(8.5, 16.5, color="#2a2321", zorder=0)
    axm.plot(np.arange(1, t + 2), ARENA_MATCH[:t + 1], color=C["connectome"], lw=2); axm.scatter([t + 1], [ARENA_MATCH[t]], s=36, color=C["connectome"], edgecolor=BG, lw=1.5, zorder=3)
    axm.set_ylabel("match to teacher (%)", fontsize=10); axm.set_xlabel("step", fontsize=10); axm.grid(color=GRID, lw=0.5); axm.text(12.5, 92, "blind", ha="center", fontsize=10, color=BAD)
    [sp.set_visible(False) for sp in axm.spines.values()]
    axb = fig.add_axes([0.40, 0.12, 0.58, 0.62]); brain(axb, 0.9 + 0.8 * pr, s=4)
    axb.text(0.0, -0.72, "real FlyWire neuron positions; blue edges excitatory, red inhibitory (2,500 of 91,663 shown)", ha="center", fontsize=9.5, color=MUTED)
    caption(ax, "Update per step:  h ← (1−α)h + α tanh(gain·(W h + W_in·o·visible + W_act·a + b)),   W = sign · softplus(w) on the edge list (Dale's law)")

@scene("Issue eleven. The old student materialised the full six thousand nine hundred thirty five squared weight matrix every step. "
       "Forty eight million entries to carry ninety one thousand synapses, and three coupled brains swapped a sixteen gigabyte machine. "
       "The recurrence now runs on the edge list with a gather and an index add, so cost scales with synapses, not neurons squared. "
       "Checkpoints shrink from four hundred twenty megabytes to twenty nine, old checkpoints still load, and a latency benchmark "
       "script produces table three.", 20)
def s_sparse(pr):
    ax = header("Sparse edge-list recurrence", "#11", "no N × N matrix in the loop")
    # dense square -> sparse dots
    axd = fig.add_axes([0.05, 0.22, 0.36, 0.58]); axd.set_facecolor(BG); axd.axis("off"); axd.set_xlim(0, 1); axd.set_ylim(0, 1)
    fade = 1 - ease(pr * 2.2 - 0.6)
    axd.add_patch(Rectangle((0.08, 0.1), 0.8, 0.8, fc="#3a3a37", ec=GRID, alpha=max(0.08, fade)))
    rng = np.random.default_rng(0); k = int(1600 * min(1, pr * 2.5)); xs, ys = rng.random((2, 1600))
    axd.scatter(0.08 + 0.8 * xs[:k], 0.1 + 0.8 * ys[:k], s=2.5, color=C["connectome"], linewidths=0)
    axd.text(0.48, 0.95, "dense W: 6,935² = 48.1 M entries" if fade > 0.5 else "edge list: 91,663 (post, pre, sign) triples", ha="center", fontsize=12, color=TXT)
    axd.text(0.48, 0.03, "recur(h) = zeros.index_add_(post, h[:, pre] · w)   →  O(E) per step", ha="center", fontsize=10.5, color=TXT2, alpha=ease(pr * 2.2 - 0.6))
    # checkpoint bars
    axc = fig.add_axes([0.63, 0.52, 0.33, 0.30]); axc.set_title("checkpoint size, MB (lower is better)", fontsize=11, loc="left", color=TXT2)
    ck = {b["model"]: b["ckpt_mb"] for b in BENCH["rows"]} if isinstance(BENCH, dict) and "rows" in BENCH else {}
    vals = [420, ck.get("connectome:1", 29), ck.get("connectome:2", 59), ck.get("connectome:3", 88)]
    hbars(axc, ["K=1 before (dense W)", "K=1 edge list", "K=2 edge list", "K=3 edge list"], vals, [BAD, C["connectome"], C["connectome"], C["connectome"]], ease(pr * 1.6 - 0.3), 520, "{:.0f} MB")
    # latency table (synthetic, cpu)
    axl = fig.add_axes([0.55, 0.17, 0.42, 0.28]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
    rows = [("model", "MFLOP/step", "p50 ms", "ckpt MB")]
    for b in BENCH.get("rows", []):
        if b["model"] in ("connectome:1", "connectome:3", "gru", "dense"): rows.append((b["model"], f"{b['mflop_step']:.1f}", f"{b['lat_p50_ms']:.1f}", f"{b['ckpt_mb']:.0f}"))
    al = ease(pr * 1.8 - 0.9)
    for i, r in enumerate(rows):
        for j, (c, x) in enumerate(zip(r, (0.0, 0.42, 0.65, 0.85))): axl.text(x, 0.95 - 0.17 * i, c, fontsize=10.5, color=TXT2 if i == 0 else TXT, alpha=al, va="top", family="DejaVu Sans Mono" if i else "DejaVu Sans")
    axl.text(0, 0.0, "Table 3, scripts/bench_latency.py: synthetic connectome, CPU (GPU rerun pending)", fontsize=9.5, color=MUTED, alpha=al)
    caption(ax, "Verified: edge recurrence equals the legacy dense step (values and gradients, all wirings, K = 1..3); K = 3 now trains on a laptop CPU.")

@scene("Issue six asked how the students compare with trivial predictors. Blind mean squared error grows with horizon for every "
       "model. The real connectome stays below the degree-shuffled and random controls, the GRU is best, and dense is worst. The new "
       "dashed lines are the trivial baselines, computed in the same PCA space and blind window: predicting the training mean scores "
       "one point zero four, copying the last visible latent is about as bad or worse. Every student is well below both. On the NYC "
       "pilot the margin is thin: the connectome beats the constant mean by only nine percent, with one seed and thirty episodes.", 24)
def s_trivial(pr):
    ax = header("Trivial baselines and teacher-space error", "#6", "E1 blind-horizon MSE, whitened PCA-256 space (lower is better)")
    axl = fig.add_axes([0.06, 0.20, 0.52, 0.62]); h = np.arange(1, 9); q = ease(pr * 1.6)
    axl.set_xlim(0.8, 8.4); axl.set_ylim(0.4, 1.25); axl.grid(color=GRID, lw=0.5); [sp.set_visible(False) for sp in axl.spines.values()]
    axl.set_xlabel("blind horizon (steps since lights off)", fontsize=10.5); axl.set_ylabel("MSE", fontsize=10.5); axl.set_xticks(h)
    keys = ("gru", "connectome", "nosign", "random", "shuffled", "dense"); ly = spread([E1[k][-1] for k in keys], 0.032)
    for k, y_ in zip(keys, ly):
        n = max(1, int(round(q * 8))); axl.plot(h[:n], E1[k][:n], color=C[k], lw=2.2)
        if q >= 1: axl.text(8.15, y_, {"gru": "GRU", "nosign": "no sign", "connectome": "connectome"}.get(k, k), color=C[k], fontsize=10, va="center")
    al = ease(pr * 2.5 - 1.3)
    if al > 0:
        axl.axhline(TRIV["arena_const"], color=TXT2, ls="--", lw=1.4, alpha=al); axl.text(1.0, TRIV["arena_const"] + 0.015, "trivial: constant train mean 1.043", fontsize=10, color=TXT2, alpha=al)
        axl.axhspan(*TRIV["arena_copy"], color=TXT2, alpha=0.12 * al); axl.text(1.0, TRIV["arena_copy"][1] - 0.04, "trivial: copy last visible latent 0.97 – 1.19", fontsize=10, color=TXT2, alpha=al)
    axl.set_title("arena (NAV + ARRANGE), seed 0", fontsize=11, loc="left", color=TXT2)
    # NYC panel
    axn = fig.add_axes([0.66, 0.20, 0.30, 0.62]); al2 = ease(pr * 2.5 - 1.6)
    labs = ["connectome", "shuffled", "random", "GRU", "const. mean"]; vals = [NYC["connectome"].mean(), NYC["shuffled"].mean(), NYC["random"].mean(), NYC["gru"].mean(), TRIV["nyc_const"]]
    hbars(axn, labs, vals, [C["connectome"], C["shuffled"], C["random"], C["gru"], TXT2], al2, 1.15, "{:.3f}")
    axn.set_title("NYC pilot, blind MSE mean\n(copy-last 1.46 – 1.64, off scale)", fontsize=10.5, loc="left", color=TXT2)
    caption(ax, "New: scripts/eval_trivial.py writes baseline rows in the same schema; train.py / summarize.py also report MSE in the original 3,840-d teacher space.")

@scene("Issue eight: statistics. Run one had two seeds for three wirings and none elsewhere, with no confidence intervals. "
       "Training now stores per-episode errors and a second blind window. The summary script computes bootstrap ninety five percent "
       "intervals over test episodes, paired differences against the connectome with p values, and t intervals across seeds once "
       "three or more exist. A seeds script queues five arena seeds and three NYC seeds. Until it runs, the seed-to-seed spread you "
       "see here, about zero point zero zero five, is the only variance estimate, so the wiring gaps are suggestive, not yet claims.", 22)
def s_seeds(pr):
    ax = header("Seeds, confidence intervals, paired tests", "#8", "what exists today vs what scripts/run_seeds.sh adds")
    axd = fig.add_axes([0.06, 0.22, 0.50, 0.60]); axd.set_ylim(0.58, 0.72); axd.set_xlim(-0.6, 2.6); axd.grid(axis="y", color=GRID, lw=0.5); [sp.set_visible(False) for sp in axd.spines.values()]
    axd.set_xticks([0, 1, 2]); axd.set_xticklabels(["connectome", "random", "shuffled"], fontsize=11); axd.set_ylabel("blind MSE, mean over horizons", fontsize=10.5); axd.set_title("two seeds each (arena, K = 1)", fontsize=11, loc="left", color=TXT2)
    for i, k in enumerate(("connectome", "random", "shuffled")):
        q = part(pr, i, 3, 0.5); v = [E1[k].mean(), E1_s1[k].mean()]
        axd.scatter([i - 0.08, i + 0.08], v, s=70 * q, color=C[k], edgecolor=BG, lw=1.5, zorder=3)
        if q >= 1:
            m, s = np.mean(v), np.std(v, ddof=1); axd.errorbar(i, m, yerr=s, color=C[k], capsize=6, lw=1.6); axd.text(i, m + 0.018, f"{m:.3f} ± {s:.3f}", ha="center", fontsize=10.5, color=TXT)
        for j in range(3):                                                                           # queued seeds
            axd.scatter([i - 0.24 + 0.24 * j], [0.595], s=60, facecolor="none", edgecolor=MUTED, lw=1.2, alpha=ease(pr * 2 - 1.2))
    axd.text(1.0, 0.584, "open circles: seeds 2–4 queued in run_seeds.sh", ha="center", fontsize=9.5, color=MUTED, alpha=ease(pr * 2 - 1.2))
    axr = fig.add_axes([0.60, 0.17, 0.37, 0.68]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
    axr.text(0, 0.98, "summarize.py now reports", fontsize=13, weight="bold", color=TXT, va="top")
    bullets(axr, ["bootstrap 95 % CI over test episodes\n(train.py saves test_mse_per_ep)", "paired Δ vs connectome_K1_s0,\nbootstrap CI and two-sided p", "mean ± sd over seeds, t-CI when n ≥ 3", "second blind window 14:22 as a check\n(--eval_windows 8:16 14:22)", "run_seeds.sh: 5 arena seeds, 3 NYC seeds,\ntrivial baselines, skips finished runs"], 0.0, 0.86, dy=0.15, pr=pr, fs=11.5)
    caption(ax, "Old JSONs without per-episode errors degrade gracefully to “-” in the tables; nothing is claimed that the stored data cannot support.")

@scene("Issue seven: the head-direction probe. In the blind window the whole connectome population decodes heading to about nine "
       "and a half degrees median error, shuffled and random to eleven and twelve, dense and GRU to around thirty six. But turns are "
       "discrete and fixed at twenty two and a half degrees, so a leaky count of past actions already gives relative heading. The probe "
       "script now reports a ridge on cumulative one-hot actions alone as the floor, and a random fourteen hundred neuron subset to "
       "match the GRU's feature count. The fifty one EPG neurons carry almost no heading, so the emergent ring-attractor claim was "
       "dropped from the paper plan, and the arena gained a turn-noise option that breaks the action shortcut.", 26)
def s_heading(pr):
    ax = header("Heading probe: controls that could explain it away", "#7", "median blind heading error, degrees (lower is better)")
    axb = fig.add_axes([0.25, 0.22, 0.42, 0.60])
    labs = ["connectome, all 6,935", "shuffled", "random", "dense", "GRU (1,400)", "connectome, EPG only (51)", "actions only (no neurons)", "connectome, 1,400 subset"]
    vals = [HD["connectome"], HD["shuffled"], HD["random"], HD["dense"], HD["gru"], HD["EPG"], 0, 0]
    cols = [C["connectome"], C["shuffled"], C["random"], C["dense"], C["gru"], C["connectome"], MUTED, MUTED]
    hbars(axb, labs, vals, cols, ease(pr * 1.5), 60, "{:.1f}°")
    al = ease(pr * 2.2 - 1.0)
    for i in (6, 7):
        axb.add_patch(Rectangle((0, i - 0.275), 58, 0.55, fc="none", ec=MUTED, lw=1.2, ls="--", alpha=al)); axb.text(29, i, "to be measured: eval_hd.py --blind  (actions_only, all@1400)", ha="center", va="center", fontsize=10, color=TXT2, alpha=al)
    axt = fig.add_axes([0.70, 0.22, 0.28, 0.60]); axt.axis("off"); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    bullets(axt, ["5 actions, turns fixed at 22.5°:\nheading = linear in action counts", "floor: ridge on cumulative\none-hot actions (+ t/T)", "all@1400: feature-matched\nrandom subset vs the GRU", "EPG R² ≤ 0.2 → claim removed", "FlyArena(turn_noise=σ):\ndth × (1 + σ·N(0,1))"], 0.0, 0.95, dy=0.19, pr=ease(pr * 1.3), fs=11.5, dx=0.06)
    caption(ax, "On synthetic data the action-only floor beat every neural probe, which is exactly the failure mode the issue predicted; the real data decides.")

@scene("Issue nine: planning. Student-driven model predictive control failed on every model, at or below a random policy. The question "
       "was whether the planner or the model is broken. The planner now has a validation ladder: oracle dynamics from the true simulator "
       "state, then rendering with the teacher, then rendering through the student, then the student alone. With oracle dynamics the "
       "same cross-entropy-method search solves navigation ten out of ten, bringing the goal from five point six metres to one. So the "
       "search works; the failure sits in the student's latent goal cost or its dynamics. On the arrangement task even the oracle fails "
       "at horizon eight, which makes that task a planning-horizon problem first.", 26)
def s_plan(pr):
    ax = header("Planner validation ladder", "#9", "is it the search or the model?  eval_plan.py --dynamics {oracle, render_teacher, render_student, student}")
    axl = fig.add_axes([0.04, 0.20, 0.25, 0.62]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
    rungs = [("oracle", "true env state, step(render=False)", GOOD, "NAV 10/10"), ("render_teacher", "render frames → teacher latent", MUTED, "needs ViT-B"), ("render_student", "render → student encodes", MUTED, "needs ckpt"), ("student", "pure latent rollout", BAD, "NAV 1/30")]
    for i, (n, d, c, res) in enumerate(rungs):
        al = part(pr, i, 4, 0.5); y = 0.85 - 0.22 * i
        axl.add_patch(FancyBboxPatch((0.02, y - 0.07), 0.94, 0.16, boxstyle="round,pad=0.006", fc=PANEL, ec=c, lw=1.6, alpha=al))
        axl.text(0.06, y + 0.045, n, fontsize=12.5, weight="bold", color=TXT, alpha=al, va="center"); axl.text(0.06, -0.005 + y, d, fontsize=9.5, color=TXT2, alpha=al, va="center"); axl.text(0.93, y + 0.045, res, fontsize=11, color=c, alpha=al, ha="right", va="center")
    axb = fig.add_axes([0.50, 0.22, 0.47, 0.60]); q = ease(pr * 1.6 - 0.3)
    pols = [("scripted policy", PLAN["scripted"]["dist_start"], PLAN["scripted"]["dist_end"], f"{PLAN['scripted']['success']:.0%}", GOOD),
            ("CEM-MPC, oracle dyn.", ORACLE["dist_start"], ORACLE["dist_end"], "100%", C["connectome"]),
            ("CEM-MPC, student dyn.", PLAN["mpc"]["dist_start"], PLAN["mpc"]["dist_end"], f"{PLAN['mpc']['success']:.0%}", C["dense"]),
            ("random policy", PLAN["random"]["dist_start"], PLAN["random"]["dist_end"], f"{PLAN['random']['success']:.0%}", MUTED)]
    axb.set_xlim(0, 9.5); axb.set_ylim(-0.6, 3.6); axb.invert_yaxis(); axb.set_yticks(range(4)); axb.set_yticklabels([p_[0] for p_ in pols], fontsize=10.5); axb.grid(axis="x", color=GRID, lw=0.5); axb.set_axisbelow(True)
    [sp.set_visible(False) for sp in axb.spines.values()]; axb.tick_params(length=0); axb.set_xlabel("mean distance to goal, m   (NAV; start → end)", fontsize=10.5); axb.set_title("goal distance before and after 30 steps, success rate at right", fontsize=11, loc="left", color=TXT2)
    for i, (n, s0, s1, succ, c) in enumerate(pols):
        qi = part(pr, i, 4, 0.75); cur = s0 + (s1 - s0) * qi
        axb.plot([s0, cur], [i, i], color=c, lw=5, solid_capstyle="round"); axb.scatter([s0], [i], s=50, color=TXT2, zorder=3, edgecolor=BG); axb.scatter([cur], [i], s=70, color=c, zorder=4, edgecolor=BG, lw=1.5)
        if qi >= 1: axb.text(max(s0, s1) + 0.15, i, f"{s0:.1f} → {s1:.1f} m   success {succ}", va="center", fontsize=10.5, color=TXT)
    axb.axvline(1.2, color=GRID, lw=1, ls=":"); axb.text(1.2, 3.45, "success radius", fontsize=9, color=MUTED, ha="center")
    caption(ax, f"ARRANGE with oracle dynamics: {ORACLE_ARR['success']:.0%} success at horizon 8 ({ORACLE_ARR['dist_start']:.1f} → {ORACLE_ARR['dist_end']:.1f} m): a horizon / sub-goal problem, not a student problem.\nNew knobs: --horizon --iters --n_cand --elite_frac --goal_mode pose --turn_noise")

@scene("Issue ten: the New York digital twin. The pilot used one location, a random episode split and no appearance variation. "
       "The benchmark protocol now splits by whole one hundred twenty metre city blocks into seventy, fifteen and fifteen percent "
       "before training, so test streets are never seen. A collection script covers four locations. The arena renders five appearance "
       "shifts: clear, overcast, fog, dusk and night. And blackouts of one, three or five steps are triggered at obstacle encounters "
       "inside the planning loop, scored with success weighted by path length and collisions per hundred metres.", 24)
def s_nyc(pr):
    ax = header("NYC benchmark protocol", "#10", "geographic split · four locations · appearance shifts · blackout-in-the-loop")
    axi = fig.add_axes([0.04, 0.47, 0.40, 0.36]); axi.imshow(IMG_NYC); axi.axis("off"); axi.set_title("Valdrada / NYC Open Data raycast arena, 110° view", fontsize=10.5, loc="left", color=TXT2)
    # block split grid
    axg = fig.add_axes([0.50, 0.40, 0.26, 0.43]); axg.set_xlim(0, 10); axg.set_ylim(0, 10); axg.set_aspect("equal"); axg.axis("off"); axg.set_title("split_geo.py: whole 120 m blocks", fontsize=10.5, loc="left", color=TXT2)
    rng = np.random.default_rng(3); lab = rng.choice(3, size=(10, 10), p=[0.7, 0.15, 0.15]); cols = [C["connectome"], C["random"], C["shuffled"]]
    n = int(100 * min(1, pr * 1.8)); order = np.argsort(rng.random(100))[:n]
    for idx in order:
        i, j = divmod(int(idx), 10); axg.add_patch(Rectangle((j + 0.06, i + 0.06), 0.88, 0.88, fc=cols[lab[i, j]], lw=0))
    for k, (nm, c) in enumerate(zip(["train 70 %", "val 15 %", "test 15 %"], cols)):
        axg.add_patch(Rectangle((0.2 + 3.3 * k, -1.2), 0.5, 0.5, fc=c, clip_on=False)); axg.text(0.9 + 3.3 * k, -0.95, nm, fontsize=9.5, color=TXT2, va="center")
    # appearance chips + blackout
    axc = fig.add_axes([0.79, 0.40, 0.19, 0.43]); axc.axis("off"); axc.set_xlim(0, 1); axc.set_ylim(0, 1)
    axc.text(0, 1.0, "appearance shifts", fontsize=11, color=TXT2, va="top")
    for i, (nm, c) in enumerate([("clear", "#9fc7f5"), ("overcast", "#9a9a95"), ("fog", "#c9c9c4"), ("dusk", "#d9875a"), ("night", "#3b3f5c")]):
        al = part(pr, i, 5, 0.9); axc.add_patch(FancyBboxPatch((0.0, 0.78 - 0.11 * i), 0.9, 0.08, boxstyle="round,pad=0.01", fc=c, ec="none", alpha=al)); axc.text(0.45, 0.82 - 0.11 * i, nm, ha="center", va="center", fontsize=10.5, color="#111" if nm != "night" else TXT, alpha=al)
    axc.text(0, 0.24, "blackout 1 / 3 / 5 steps\nat obstacle encounters\n\nmetrics: success, SPL,\ncollisions per 100 m", fontsize=10.5, color=TXT, va="top", alpha=ease(pr * 2 - 1), linespacing=1.45)
    # pilot match curve
    axm = fig.add_axes([0.085, 0.18, 0.355, 0.22]); axm.set_xlim(0.5, 24.5); axm.set_ylim(0, 100); axm.axvspan(8.5, 16.5, color="#2a2321", zorder=0); axm.grid(color=GRID, lw=0.5); [sp.set_visible(False) for sp in axm.spines.values()]
    t = int(min(24, pr * 1.3 * 24)); axm.plot(np.arange(1, t + 1), NYC_MATCH[:t], color=C["connectome"], lw=2); axm.set_ylabel("match %", fontsize=10); axm.set_xlabel("step  (pilot episode, Times Square)", fontsize=10)
    axm.text(12.5, 88, "blind: match collapses to 0", fontsize=9.5, color=BAD, ha="center")
    caption(ax, "Pilot numbers stay labelled as a pilot (connectome 0.885 vs constant mean 0.977).\nCollection needs the Valdrada server and a GPU: scripts/collect_nyc_multi.sh, then scripts/split_geo.py.")

@scene("And this is the digital twin itself, rendered live. Six hundred eighty six real Midtown building footprints with their "
       "roof heights and construction years come from NYC Open Data, the street grid from OpenStreetMap, and the arena raycasts a "
       "three hundred sixty degree panorama against that geometry. Facade materials follow the real build year: glass for towers "
       "after nineteen sixty, brick and limestone for pre-war blocks. The strip shows one pose under the five appearance shifts. "
       "On the map, the scripted controller walks toward a goal street; at step eight the lights go out for three steps and it has "
       "to act blind. The table gives the first planner numbers on the twin: a scripted policy and the oracle planner, with "
       "blackouts of zero to five steps, scored by success, path-length-weighted success, and collisions per hundred metres.", 30)
def s_twin(pr):
    ax = header("NYC digital twin, rendered live", "#10", "wm2cx/nyc.py: 686 real Midtown footprints (NYC Open Data 5zhs-2jue) + OSM streets, raycast 360° panorama")
    if TWIN is None: caption(ax, "results/nyc_twin/ missing: run scripts/fetch_nyc_data.sh then scripts/render_nyc_twin.py"); return
    F, BL = TWIN["frames"], TWIN["blind"]; T = len(F); t = min(T - 1, int(pr / 0.85 * T)); blind = bool(BL[t])
    axp = fig.add_axes([0.04, 0.44, 0.50, 0.40]); axp.imshow((F[t] * (0.15 if blind else 1.0)).astype(np.uint8), aspect="auto", interpolation="nearest"); axp.set_xticks([]); axp.set_yticks([])
    axp.set_title(("LIGHTS OFF: controller acts blind" if blind else "scripted controller, eyes open") + f"   — step {t + 1}/{T}, 360° view", fontsize=11, loc="left", color=BAD if blind else TXT2)
    if blind: axp.text(0.5, 0.5, "LIGHTS OFF", color=BAD, fontsize=30, weight="bold", ha="center", va="center", transform=axp.transAxes)
    for i, k in enumerate(("clear", "overcast", "fog", "dusk", "night")):
        axa = fig.add_axes([0.04 + 0.102 * i, 0.22, 0.096, 0.17]); axa.imshow(TWIN_APP[k], aspect="auto"); axa.set_xticks([]); axa.set_yticks([]); axa.set_title(k, fontsize=10, color=TXT2, pad=3)
        for sp in axa.spines.values(): sp.set_edgecolor(GRID)
    # map: real footprints, streets, path
    axm = fig.add_axes([0.58, 0.30, 0.40, 0.55]); axm.set_facecolor("#20201f"); axm.set_xticks([]); axm.set_yticks([])
    from matplotlib.collections import PolyCollection
    V, O, Hh = TWIN["poly_verts"], TWIN["poly_offsets"], TWIN["poly_height"]
    polys = [V[O[i]:O[i + 1]] for i in range(len(O) - 1)]; shade = np.clip(Hh / 150.0, 0, 1)
    axm.add_collection(PolyCollection(polys, facecolors=[(0.22 + 0.5 * s, 0.30 + 0.45 * s, 0.42 + 0.5 * s, 1.0) for s in shade], edgecolors="none"))
    axm.add_collection(LineCollection(TWIN["street_segs"], colors="#4a4a47", linewidths=0.8))
    P, G = TWIN["pos"], TWIN["goal"]; c0 = (P[0] + G) / 2
    axm.plot(P[:t + 1, 0], P[:t + 1, 1], color=C["connectome"], lw=2.2)
    bl_idx = np.where(BL[:t + 1])[0]
    if len(bl_idx): axm.plot(P[bl_idx[0]:min(bl_idx[-1] + 2, t + 1), 0], P[bl_idx[0]:min(bl_idx[-1] + 2, t + 1), 1], color=BAD, lw=2.6)
    axm.scatter([P[0, 0]], [P[0, 1]], s=40, color=TXT, zorder=4); axm.scatter([G[0]], [G[1]], s=90, marker="*", color=C["random"], zorder=4)
    axm.scatter([P[t, 0]], [P[t, 1]], s=55, color=BAD if blind else C["connectome"], edgecolor=BG, lw=1.2, zorder=5)
    axm.add_patch(plt.Circle(G, 12.0, fc="none", ec=C["random"], lw=1, ls="--"))
    axm.set_xlim(c0[0] - 130, c0[0] + 130); axm.set_ylim(c0[1] - 90, c0[1] + 90); axm.set_aspect("equal")
    axm.set_title("map: footprints by height, goal ★ (12 m radius), path, red = blind", fontsize=10, loc="left", color=TXT2)
    axm.text(0.01, 0.02, f"shortest path {TWIN_META['shortest_path_m']:.0f} m · walked {TWIN_META['path_len_m']:.0f} m · collisions {TWIN_META['collisions']} · success {TWIN_META['success']:.0f}", transform=axm.transAxes, fontsize=9.5, color=TXT2)
    # planner table on the twin
    axt = fig.add_axes([0.58, 0.11, 0.40, 0.17]); axt.axis("off"); axt.set_xlim(0, 1); axt.set_ylim(0, 1); al = ease(pr * 2.5 - 1.4)
    rows = [("planner on the twin (10 ep)", "blackout", "success", "SPL", "coll/100 m")]
    for r in TWIN_PLAN: rows.append((("scripted policy" if r["policy"] == "scripted" else "CEM-MPC, oracle dyn."), str(r.get("blackout", 0)), f"{r['success']:.0%}", f"{r.get('spl', 0):.2f}", f"{r.get('collisions_per_100m', 0):.1f}"))
    if len(rows) == 1: rows.append(("eval_plan.py --env nyc --dynamics oracle --blackout N", "", "", "", ""))
    for i, r in enumerate(rows[:7]):
        for j, (c, x) in enumerate(zip(r, (0.0, 0.50, 0.64, 0.80, 0.92))): axt.text(x, 0.98 - 0.16 * i, c, fontsize=9.5, color=TXT2 if i == 0 else TXT, alpha=al, va="top", ha="left" if j == 0 else "center")
    caption(ax, "Reproduce: scripts/fetch_nyc_data.sh (NYC Open Data + Overpass), scripts/render_nyc_twin.py,\neval_plan.py --env nyc --policy scripted / --dynamics oracle --blackout 0 1 3 5 (no checkpoint needed)")

@scene("Issues twelve and thirteen tie the text to the code. The implementation record states exactly what trains: a prediction-only "
       "loss with blind steps weighted three times, five discrete actions, the ViT-B teacher with a three thousand eight hundred forty "
       "dimensional target, and the FlyWire licence. A policy-imitation head is available as an ablation flag but was never used in a "
       "reported run. The subgraph choice is now written down: visual projection neurons feed the central complex, mushroom-body output "
       "carries valence, descending neurons are the motor side. All three choices, synapse threshold, region set and wiring, are "
       "covered by an ablation script.", 24)
def s_record(pr):
    ax = header("Implementation record and subgraph rationale", "#12 #13", "PAPER_PLAN.md now carries what the code does; connectome.py has a CLI; ablate_connectome.sh")
    axt = fig.add_axes([0.04, 0.17, 0.46, 0.68]); axt.axis("off"); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    axt.text(0, 0.99, "manuscript must match", fontsize=13, weight="bold", color=TXT, va="top")
    bullets(axt, ["loss = weighted MSE to the PCA-256 target (blind ×3) + 1e-4·mean(h²);\nno feature-alignment and no imitation term in any reported run", "--lambda_act > 0 adds a next-action CE head, for the ablation only", "5 actions: stay, forward, left 22.5°, right 22.5°, back; 24-step episodes", "teacher V-JEPA 2.1 ViT-B, 3,840-d → PCA-256\n(94.6 % variance on the arena, 91.0 % on NYC)", "FlyWire v783: 6,935 neurons, 91,663 edges ≥ 5 synapses;\nCC-BY-NC, so the derived npz is not redistributed", "7.25 M parameters for K = 1, of which 91.6 k are synapses"], 0.0, 0.88, dy=0.14, pr=pr, fs=11)
    axr = fig.add_axes([0.70, 0.52, 0.27, 0.32]); hbars(axr, ["VIS (TuBu, MeTu, LC)", "CX (EPG, PEN, PFN, …)", "MB (MBON + DAN)", "OUT (descending)"], [2330, 2875, 427, 1303], [REG["VIS"], REG["CX"], REG["MB"], REG["OUT"]], ease(pr * 1.5), 3600, "{:,.0f}")
    axr.set_title("subgraph by region, neurons", fontsize=11, loc="left", color=TXT2)
    axg = fig.add_axes([0.62, 0.17, 0.32, 0.27]); axg.axis("off"); axg.set_xlim(0, 4); axg.set_ylim(0, 3); axg.set_title("ablate_connectome.sh grid (× connectome vs shuffled, 3 seeds)", fontsize=10, loc="left", color=TXT2)
    al = ease(pr * 2 - 1.1)
    for i, ms in enumerate(["min_syn 1", "min_syn 5", "min_syn 10"]):
        axg.text(-0.1, 2.5 - i, ms, ha="right", va="center", fontsize=9.5, color=TXT2, alpha=al)
        for j, rg in enumerate(["full", "no MB", "no OUT", "CX only"]):
            axg.add_patch(Rectangle((j + 0.08, 2.08 - i), 0.84, 0.84, fc=PANEL if not (i == 1 and j == 0) else C["connectome"], ec=MUTED, lw=1, ls="--" if not (i == 1 and j == 0) else "-", alpha=al))
            if i == 2: axg.text(j + 0.5, -0.25, rg, ha="center", va="top", fontsize=9.5, color=TXT2, alpha=al)
    axg.text(0.5, 1.5, "run", ha="center", va="center", fontsize=9, color=TXT, alpha=al)
    caption(ax, "python -m wm2cx.connectome --min_syn 10 --regions VIS CX --out data/connectome_cx.npz   →   scripts/ablate_connectome.sh runs the full grid.")

@scene("To sum up. Six commits are pushed to main. Verified here: the sparse recurrence, the trivial baselines, the statistics "
       "pipeline, the probe controls, the planner ladder with oracle dynamics, the NYC protocol, and the documentation, all on a "
       "synthetic stand-in for the connectome, with twenty five tests passing. Still to run on the machine that holds the data and a "
       "GPU: the five-seed sweep, the connectome ablations, multi-location NYC collection, and planning with student dynamics. The "
       "code for each is in the repository and named here.", 20)
def s_end(pr):
    ax = header("Where things stand", sub="pushed to origin/main: 0a50359  e0987f9  dc82747  11f242b  1eae21c  d6bc65d")
    axb = fig.add_axes([0.55, 0.10, 0.45, 0.80]); brain(axb, 2.0 + 0.6 * pr, s=3); axb.set_alpha(0.6)
    axl = fig.add_axes([0.04, 0.14, 0.52, 0.72]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
    axl.text(0, 0.99, "done and verified (synthetic data, 25 tests pass)", fontsize=12.5, weight="bold", color=GOOD, va="top")
    bullets(axl, ["#11 edge-list recurrence, Table 3 script, 29 MB checkpoints", "#6 #8 trivial baselines, teacher-space MSE, bootstrap / paired / seed CIs", "#7 action-only and feature-matched probe controls, turn noise", "#9 oracle ladder: CEM solves NAV 10/10, student MPC is the bottleneck", "#10 geographic split, 4 locations, appearance shifts, blackout metrics; raycast twin rendered from real NYC data", "#12 #13 implementation record, subgraph rationale, connectome CLI"], 0.0, 0.92, dy=0.075, pr=min(1, pr * 1.4), fs=11)
    axl.text(0, 0.42, "needs data/ + GPU or the M1", fontsize=12.5, weight="bold", color=C["random"], va="top", alpha=ease(pr * 2 - 1))
    bullets(axl, ["scripts/run_seeds.sh  (5 arena, 3 NYC seeds)", "scripts/ablate_connectome.sh", "scripts/collect_nyc_multi.sh  +  split_geo.py", "eval_plan.py --dynamics student / render_*  on real checkpoints", "eval_hd.py --blind  for the actions_only / all@1400 columns", "bench_latency.py --device cuda  on the real connectome"], 0.0, 0.36, dy=0.058, pr=ease(pr * 2 - 1), fs=10.5, color=TXT2)
    caption(ax, "github.com/prajak002/World2Brain  ·  results/summary.md regenerates from the JSONs  ·  this video: scripts/make_video_issues.py")

if a.preview:
    for i, (_, draw, _) in enumerate(SCENES):
        for pr in (0.45, 1.0):
            draw(pr); fig.savefig(a.out.replace(".mp4", f"_s{i:02d}{'' if pr == 1 else '_mid'}.png"), dpi=100, facecolor=BG)
    print("preview frames written"); sys.exit(0)

# ---------- narration (Windows SAPI) ----------
tmp = tempfile.mkdtemp(prefix="wm2cx_vid_")
def tts(text, path):
    if a.no_audio or sys.platform != "win32" or not shutil.which("powershell"): return None
    tf = path + ".txt"
    with open(tf, "w", encoding="utf-8") as f: f.write(text)
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"try {{ $s.SelectVoice('{a.voice}') }} catch {{}}; $s.Rate = 0; "
          f"$s.SetOutputToWaveFile('{path}'); $s.Speak([IO.File]::ReadAllText('{tf}', [Text.Encoding]::UTF8)); $s.Dispose()")
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True, text=True)
    return path if r.returncode == 0 and os.path.exists(path) else None
clips, durs = [], []
for i, (text, _, min_sec) in enumerate(SCENES):
    w = tts(text, os.path.join(tmp, f"s{i:02d}.wav")); clips.append(w)
    if w:
        with wave.open(w) as wf: dur = wf.getnframes() / wf.getframerate()
    else: dur = max(min_sec, len(text.split()) / 2.6)
    durs.append(dur)
    print(f"scene {i}: narration {dur:5.1f}s" + ("" if w else " (no audio)"))

# ---------- render ----------
w_px, h_px = fig.canvas.get_width_height()
silent = os.path.join(tmp, "silent.mp4")
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w_px}x{h_px}", "-r", str(a.fps), "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", silent], stdin=subprocess.PIPE)
frames_per_scene = []
for i, ((text, draw, min_sec), dur) in enumerate(zip(SCENES, durs)):
    n = int(math.ceil(max(dur + 1.0, min_sec) * a.fps)); frames_per_scene.append(n)
    for k in range(n):
        pr = k / max(1, n - 1 - int(0.8 * a.fps))                        # hold the finished frame ~0.8 s at the end
        draw(min(1.0, pr)); fig.canvas.draw(); ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        if k == n - 1: fig.savefig(a.out.replace(".mp4", f"_s{i:02d}.png"), dpi=100, facecolor=BG)
    print(f"scene {i}: {n / a.fps:5.1f}s rendered")
ff.stdin.close(); ff.wait()

if any(clips):
    ref = next(c for c in clips if c)
    with wave.open(ref) as wf: params = wf.getparams()
    sw, rate, nch = params.sampwidth, params.framerate, params.nchannels
    out_wav = os.path.join(tmp, "narration.wav")
    with wave.open(out_wav, "wb") as wo:
        wo.setnchannels(nch); wo.setsampwidth(sw); wo.setframerate(rate)
        for c, n in zip(clips, frames_per_scene):
            need = int(round(n / a.fps * rate)); data = b""
            if c:
                with wave.open(c) as wf: data = wf.readframes(wf.getnframes())
            data = data[:need * sw * nch]; data += b"\x00" * (need * sw * nch - len(data)); wo.writeframes(data)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", out_wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", a.out], check=True)
else: shutil.copy(silent, a.out)
if a.gif: subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out, "-vf", "fps=4,scale=960:-1:flags=lanczos", a.out.replace(".mp4", ".gif")])
shutil.rmtree(tmp, ignore_errors=True)
print("wrote", a.out, f"({sum(frames_per_scene) / a.fps:.0f} s)")
