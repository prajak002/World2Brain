"""Generate the result plots for paper_v2 from numbers in results/summary.md (hard-coded snapshot, 2026-11-04)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pathlib
out = pathlib.Path(__file__).resolve().parent.parent / "fig"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
SP, FR, CX = "#F4A000", "#4285F4", "#B35C00"

# 1. control ladder: MSE and heading per wiring (arena, 5 seeds)
names = ["GRU (param)", "GRU (recurrent)", "dense", "unsigned", "random signs", "connectome", "random (ER)", "block-shuffled", "degree-shuffled"]
mse   = [0.434, 0.501, 0.562, 0.573, 0.578, 0.579, 0.583, 0.585, 0.610]
mses  = [0.004, 0.001, 0.004, 0.001, 0.001, 0.003, 0.002, 0.005, 0.002]
hd    = [30.6, 35.5, 21.3, 13.8, 9.2, 9.1, 12.9, 10.8, 12.7]
hds   = [0.6, 0.7, 0.7, 0.8, 1.3, 1.2, 0.7, 0.8, 0.6]
col   = [FR, FR, FR, SP, SP, CX, SP, SP, SP]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
y = np.arange(len(names))[::-1]
ax[0].barh(y, mse, xerr=mses, color=col, height=0.7); ax[0].set_xlim(0.40, 0.63); ax[0].set_xlabel("blind latent MSE (lower is better)")
ax[0].axvline(0.579, color=CX, lw=.6, ls=":"); ax[0].set_yticks(y); ax[0].set_yticklabels(names)
ax[1].barh(y, hd, xerr=hds, color=col, height=0.7); ax[1].set_xlabel("heading error in 8 dark steps, deg (lower is better)")
ax[1].axvline(40.1, color="k", lw=.6, ls="--"); ax[1].text(40.6, 8.2, "actions-only\nfloor", fontsize=6.5, va="top")
ax[0].set_title("(a) prediction: free recurrence wins", fontsize=8, loc="left"); ax[1].set_title("(b) integration: sparse wiring wins", fontsize=8, loc="left")
for a in ax: a.tick_params(length=2)
fig.tight_layout(); fig.savefig(out / "ladder.pdf"); plt.close(fig)

# 2. dark-window degradation (ridge heading error)
win = [8, 16, 20]
series = {"connectome": ([9.1, 17.7, 12.6], CX), "block-shuffled": ([10.8, 23.3, 15.9], SP), "random (ER)": ([12.9, 24.9, 17.3], "#E8B84A"),
          "dense": ([21.3, 29.9, 21.3], "#7BAAF7"), "GRU (param)": ([30.6, 37.7, 27.9], FR)}
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1))
for k, (v, c) in series.items(): ax[0].plot(win, v, "o-", color=c, label=k, ms=3, lw=1.2)
ax[0].set_xticks(win); ax[0].set_xticklabels(["8", "16", "20"]); ax[0].set_xlabel("dark steps (windows 8:16, 8:24, 4:24)"); ax[0].set_ylabel("heading error, deg")
ax[0].legend(fontsize=6, frameon=False); ax[0].set_title("(a) longer blackouts", fontsize=8, loc="left")
# 3. data efficiency
n = [50, 100, 200, 500, 1000]
de = {"connectome": ([0.970, 0.899, 0.796, 0.677, 0.579], CX), "block-shuffled": ([0.989, 0.924, 0.818, 0.702, 0.585], SP),
      "degree-shuffled": ([0.955, 0.913, 0.791, 0.713, 0.610], "#E8B84A"), "GRU (param)": ([0.821, 0.762, 0.655, 0.573, 0.434], FR)}
for k, (v, c) in de.items(): ax[1].plot(n, v, "o-", color=c, label=k, ms=3, lw=1.2)
ax[1].set_xscale("log"); ax[1].set_xticks(n); ax[1].set_xticklabels(n); ax[1].set_xlabel("training episodes"); ax[1].set_ylabel("blind MSE")
ax[1].legend(fontsize=6, frameon=False); ax[1].set_title("(b) data efficiency", fontsize=8, loc="left")
# 4. multi-brain
K = [1, 2, 3]; km = [0.579, 0.574, 0.574]; ks = [0.003, 0.002, 0.003]; kh = [9.1, 7.0, 8.7]
ax[2].errorbar(K, km, yerr=ks, fmt="o-", color=CX, ms=4, lw=1.2, label="blind MSE"); ax[2].set_ylim(0.55, 0.60); ax[2].set_xticks(K)
ax[2].set_xticklabels(["K=1\n6.9k", "K=2\n13.9k", "K=3\n20.8k"]); ax[2].set_ylabel("blind MSE", color=CX)
a2 = ax[2].twinx(); a2.plot(K, kh, "s--", color="k", ms=3, lw=1, label="heading err"); a2.set_ylim(0, 15); a2.set_ylabel("heading err, deg"); a2.spines["top"].set_visible(False)
ax[2].axhline(0.434, color=FR, lw=.8, ls=":"); ax[2].text(1.05, 0.5515, "GRU 0.434 (off scale)", color=FR, fontsize=6)
ax[2].set_title("(c) more brains, no gain", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(out / "curves.pdf"); plt.close(fig)

# 5. real video transfer
w = ["connectome", "unsigned", "dense", "GRU", "random", "degree-shuf."]
sim = [30.9, 24.3, 29.5, 84.1, 79.1, 88.7]; sime = [22.8, 6.3, 6.4, 34.2, 14.5, 27.0]
clip = [7.7, 7.7, 9.1, 7.6, 9.9, 7.1]; clipe = [0.5, 0.3, 0.9, 0.7, 1.1, 0.6]
fig, ax = plt.subplots(figsize=(3.4, 2.0)); x = np.arange(len(w))
ax.bar(x - 0.2, sim, 0.4, yerr=sime, color=[CX, SP, "#7BAAF7", FR, SP, SP], label="simulator-fit readout (transfer)")
ax.bar(x + 0.2, clip, 0.4, yerr=clipe, color="#BBBBBB", label="clip-fit readout (in-domain)")
ax.axhline(90, color="k", lw=.5, ls="--"); ax.text(-0.4, 92, "chance (90 deg)", fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(w, rotation=25, ha="right"); ax.set_ylabel("heading error, deg"); ax.set_ylim(0, 125)
ax.legend(fontsize=6, frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(out / "realvid_bars.pdf"); plt.close(fig)
print("wrote", [p.name for p in out.glob("*.pdf")])
