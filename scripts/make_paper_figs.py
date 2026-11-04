"""Figures for the CVPR paper (paper/fig/*.pdf), generated from results/hd, results/runs and results/homing.

    PYTHONPATH=. python scripts/make_paper_figs.py
"""
import glob
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join("paper", "fig")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

LABEL = {
    "connectome": "connectome", "connectome_randsign": "connectome, random signs",
    "connectome_nosign": "connectome, unsigned", "block_shuffled": "block-shuffled",
    "shuffled": "degree-shuffled", "random": "random", "dense": "dense",
    "gru": "GRU, parameter-matched", "gru_recmatch": "GRU, recurrent-budget-matched",
    "transformer": "Transformer, parameter-matched", "connectome_attnin": "connectome, attention input",
    "gru_attnin": "GRU, attention input", "transformer_attnin": "Transformer, attention input",
    "block_shuffled_attnin": "block-shuffled, attention input", "connectome_hybrid": "connectome + cell-type attention",
}
COLOR = {
    "connectome": "#c0392b", "connectome_randsign": "#e07b6a", "connectome_nosign": "#f0b8ad",
    "block_shuffled": "#2e86c1", "shuffled": "#5dade2", "random": "#85c1e9",
    "dense": "#7f8c8d", "gru": "#1c2833", "gru_recmatch": "#566573", "transformer": "#6c3483",
    "connectome_attnin": "#e67e22", "gru_attnin": "#34495e", "transformer_attnin": "#9b59b6",
    "block_shuffled_attnin": "#5499c7", "connectome_hybrid": "#b03a2e",
}
MARK = {"connectome": "o", "connectome_randsign": "o", "connectome_nosign": "o", "block_shuffled": "s",
        "shuffled": "s", "random": "s", "dense": "D", "gru": "^", "gru_recmatch": "^", "transformer": "v",
        "connectome_attnin": "o", "gru_attnin": "^", "transformer_attnin": "v", "block_shuffled_attnin": "s", "connectome_hybrid": "P"}


def parse(fn):
    """-> (wiring, seed, env, window_tag) for results/hd/<wiring>_K1_s<seed>[_recmatch][_nyc][_b8_24|_b4_24|_mlp].json"""
    m = re.match(r"(.+?)_K1_s(\d)(_recmatch|_attnin|_hybrid)?(_tn|_nyc)?(_b8_24|_b4_24|_mlp)?$", fn)
    if not m:
        return None
    w = m.group(1) + (m.group(3) or "")
    return w, int(m.group(2)), {None: "arena", "_nyc": "nyc", "_tn": "tn"}[m.group(4)], (m.group(5) or "_b8_16").lstrip("_")


def hd_table():
    rows = {}
    for f in glob.glob("results/hd/*.json"):
        k = parse(os.path.basename(f)[:-5])
        if k is None:
            continue
        d = json.load(open(f))
        if "blind_all" not in d:
            continue
        rows.setdefault((k[0], k[2], k[3]), []).append((d["blind_all"]["heading_err_deg"], d["blind_all"]["pos_r2"]))
    return rows


def mse_table():
    rows = {}
    for f in glob.glob("results/runs/*_K1_s?.json") + glob.glob("results/runs/*_K1_s?_recmatch.json") + glob.glob("results/runs/*_K1_s?_attnin.json") + glob.glob("results/runs/*_K1_s?_hybrid.json"):
        fn = os.path.basename(f)[:-5]
        m = re.match(r"(.+?)_K1_s(\d)(_recmatch|_attnin|_hybrid)?$", fn)
        if not m:
            continue
        w = m.group(1) + (m.group(3) or "")
        d = json.load(open(f))
        if "test_mse" not in d or d.get("args", {}).get("env", "arena") not in ("arena", None):
            continue
        if any(t in fn for t in ("_nyc", "_n50", "_n100", "_n200", "_n500", "_big", "_alpha", "_gate")):
            continue
        rows.setdefault(w, []).append(st.mean(d["test_mse"][:8]))
    return rows


def msd(v):
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


# ---------------------------------------------------------------- Fig: schematic
def fig_schematic():
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f6f7", ec="#5d6d7e", fs=8, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#2c3e50", style="-|>", lw=0.9):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, color=color, lw=lw))

    # teacher row (top)
    box(2, 31, 14, 7, "video frames\n(arena / NYC twin)", fc="#fdfefe")
    box(21, 31, 16, 7, "V-JEPA 2.1 ViT-B\n(frozen teacher)", fc="#eaf2f8")
    box(42, 31, 14, 7, "PCA-256 latent\n$z_{t+1}$ (causal)", fc="#eaf2f8")
    arrow(16, 34.5, 21, 34.5); arrow(37, 34.5, 42, 34.5)
    # inputs (left)
    box(1, 19, 15, 6, "action $a_t$\n(5 discrete)", fc="#fdfefe", fs=7.2)
    box(1, 8, 15, 8, "fly-eye input $o_t$\n32$\\times$8, gated by\nvisibility $v_t$", fc="#fdfefe", fs=7.0)
    # student box
    box(21, 5, 37, 21.5, "", fc="#fdf2e9", ec="#c0392b")
    ax.text(39.5, 28.6, "connectome-wired student (6,935 neurons, 91,663 synapses)", ha="center", va="center",
            fontsize=6.9, color="#c0392b", fontweight="bold")
    for name, n, xx in [("VIS", "2,330", 24.5), ("CX", "2,875", 32.5), ("MB", "427", 40.5), ("OUT", "1,303", 48.5)]:
        box(xx, 13.5, 6.5, 8, f"{name}\n{n}", fc="#fadbd8" if name == "CX" else "#fbeee6", ec="#c0392b", fs=7)
    arrow(16, 12, 24.5, 15.5, color="#c0392b"); arrow(16, 22, 32.5, 21.5, color="#c0392b"); arrow(16, 22, 48.5, 21.5, color="#c0392b", lw=0.6)
    ax.text(40.5, 10.3, "$h_{t+1}=(1-\\alpha)h_t+\\alpha\\tanh(g(Wh_t+W_{in}o_t v_t+W_{act}a_t+b))$", ha="center", va="center", fontsize=6.0, color="#641e16")
    ax.text(40.5, 7.6, "$W_{ij}=s_{ij}\\,\\mathrm{softplus}(w_{ij})$ on edges only; $s_{ij}$ fixed by neurotransmitter", ha="center", va="center", fontsize=5.7, color="#641e16")
    box(62, 14.5, 12, 6, "linear readout\n$\\hat z_{t+1}$", fc="#fdfefe")
    arrow(58, 17.5, 62, 17.5)
    # loss (right, top)
    box(79, 28, 19, 8, "loss: weighted MSE\n$\\|\\hat z_{t+1}-z_{t+1}\\|^2$\n(blind steps $\\times$3)", fc="#e9f7ef", ec="#1e8449", fs=7.0)
    arrow(56, 34.5, 79, 33, color="#1e8449"); arrow(74, 17.5, 88, 28, color="#1e8449")
    # probes (right, bottom)
    box(79, 7, 20, 11, "after training:\nridge probe of $h_t$ for\nheading and position in\nthe dark; blind homing", fc="#fef9e7", ec="#b7950b", fs=6.5)
    arrow(58, 12.5, 79, 12.5, color="#b7950b", style="-|>")
    # blackout note
    ax.text(2, 3.0, "blackout: $v_t{=}0$ on a random window during training; test windows 8:16, 8:24, 4:24 (actions still given)",
            ha="left", va="center", fontsize=6.5, color="#7b241c")
    fig.savefig(os.path.join(OUT, "schematic.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig: dissociation + dark duration
def fig_main(hd, mse):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    order = ["gru", "transformer", "gru_recmatch", "dense", "connectome_nosign", "connectome_randsign", "connectome", "connectome_hybrid",
             "random", "block_shuffled", "shuffled", "connectome_attnin", "gru_attnin", "transformer_attnin"]
    for w in order:
        if w not in mse or (w, "arena", "b8_16") not in hd:
            continue
        mx, sx = msd(mse[w]); my, sy = msd([h for h, _ in hd[(w, "arena", "b8_16")]])
        ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt=MARK[w], color=COLOR[w], ms=5.5, capsize=1.5, lw=0.8, label=LABEL[w], zorder=3)
    ax.axhline(40.1, ls=":", color="#7f8c8d", lw=0.8); ax.text(0.515, 41.2, "linear floor: cumulative actions only", fontsize=6.5, color="#7f8c8d")
    ax.set_xlabel("blind latent prediction MSE (lower = better predictor)")
    ax.set_ylabel("median heading error in the dark (deg)")
    ax.set_xlim(0.40, 0.63); ax.set_ylim(0, 46)
    ax.annotate("better predictors,\nno heading code", xy=(0.436, 31.5), xytext=(0.447, 37.5), fontsize=6.8,
                arrowprops=dict(arrowstyle="-", color="#1c2833", lw=0.6), color="#1c2833")
    ax.annotate("worse predictors,\npath-integrating state", xy=(0.577, 8.4), xytext=(0.537, 2.5), fontsize=6.8,
                arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.6), color="#c0392b")
    handles, labels = ax.get_legend_handles_labels()
    ax.set_title("(a) Prediction vs. heading code (arena)", loc="left")

    ax = axes[1]
    xs = {"b8_16": 8, "b8_24": 16, "b4_24": 20}
    for w in ["connectome", "block_shuffled", "shuffled", "random", "dense", "gru", "transformer"]:
        pts = []
        for tag, x in xs.items():
            if (w, "arena", tag) in hd:
                m, s = msd([h for h, _ in hd[(w, "arena", tag)]]); pts.append((x, m, s))
        if not pts:
            continue
        pts.sort()
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], fmt=MARK[w] + "-", color=COLOR[w], ms=4,
                    capsize=1.5, lw=1.0, label=LABEL[w])
    ax.set_xticks([8, 16, 20]); ax.set_xticklabels(["8 (8:16)", "16 (8:24)", "20 (4:24)"])
    ax.set_xlabel("dark steps (blind window)"); ax.set_ylabel("median heading error (deg)")
    ax.set_ylim(0, 46)
    ax.set_title("(b) Dark duration (arena)", loc="left")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, handletextpad=0.3, columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.5, rect=(0, 0.13, 1, 1))
    fig.savefig(os.path.join(OUT, "main_result.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig: homing
def fig_homing():
    agg = {}
    for f in glob.glob("results/homing/*.json"):
        k = os.path.basename(f)[:-5]
        m = re.match(r"(.+?)_K1_s(\d)(_recmatch|_attnin|_hybrid)?(_tn)?_(arena|nyc_nyc)_(state|mlp|oracle|random|actions)(_tn)?$", k)
        if not m or m.group(7) or m.group(4):
            continue
        d = json.load(open(f))
        w = m.group(1) + (m.group(3) or ""); env = "arena" if m.group(5) == "arena" else "nyc"; ro = m.group(6)
        agg.setdefault((env, w, ro), []).append((d["success"], 100 * (1 - d["dist_final"] / d["dist_blind_start"])))
    rows = [("oracle pose (ceiling)", "connectome", "oracle", "#1e8449"), ("dead reckoning of actions", "connectome", "actions", "#52be80"),
            ("random actions (floor)", "connectome", "random", "#7f8c8d"),
            ("connectome state", "connectome", "state", COLOR["connectome"]), ("connectome, unsigned", "connectome_nosign", "state", COLOR["connectome_nosign"]),
            ("block-shuffled state", "block_shuffled", "state", COLOR["block_shuffled"]), ("degree-shuffled state", "shuffled", "state", COLOR["shuffled"]),
            ("random state", "random", "state", COLOR["random"]), ("dense state", "dense", "state", COLOR["dense"]),
            ("GRU, param.-matched", "gru", "state", COLOR["gru"]), ("GRU, budget-matched", "gru_recmatch", "state", COLOR["gru_recmatch"]),
            ("Transformer, param.-matched", "transformer", "state", COLOR["transformer"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, env, title in [(axes[0], "arena", "(a) Arena: 12 visible, 24 blind steps"), (axes[1], "nyc", "(b) NYC twin: 12 visible, 24 blind steps")]:
        ys, labels = [], []
        for i, (lab, w, ro, c) in enumerate(rows):
            v = agg.get((env, w, ro))
            y = len(rows) - i
            ys.append(y); labels.append(lab)
            if not v:
                ax.text(1, y, "n/a", va="center", fontsize=6.5, color="#aaa"); continue
            succ = 100 * st.mean(x[0] for x in v); closer = st.mean(x[1] for x in v)
            ax.barh(y + 0.18, succ, height=0.34, color=c, alpha=0.95)
            ax.barh(y - 0.18, max(closer, 0), height=0.34, color=c, alpha=0.45)
            ax.text(max(succ, max(closer, 0)) + 1.5, y, f"{succ:.0f}% / {closer:.0f}%", va="center", fontsize=6.5)
        ax.set_yticks(ys); ax.set_yticklabels(labels); ax.set_xlim(0, 122); ax.set_xlabel("percent of episodes")
        ax.set_title(title, loc="left")
        ax.axvline(100, color="#ccc", lw=0.6, ls=":")
    fig.text(0.5, 0.02, "dark bar: success (home within 1 unit / 12 m);  light bar: % of the distance to home recovered during the blind phase",
             ha="center", fontsize=7)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, "homing.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    hd, mse = hd_table(), mse_table()
    for w in sorted(mse):
        if (w, "arena", "b8_16") in hd:
            print(f"{w:24s} mse {msd(mse[w])[0]:.3f}  heading {msd([h for h, _ in hd[(w, 'arena', 'b8_16')]])[0]:.1f}")
    fig_schematic(); fig_main(hd, mse); fig_homing()
    print("wrote", sorted(os.listdir(OUT)))
