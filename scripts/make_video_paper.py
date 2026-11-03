"""Paper video + figure strip: how the world model sees / thinks / navigates vs. how the distilled connectome student
(and a parameter-matched GRU) does, through a blind window of one arena episode.

Columns: V-JEPA teacher | connectome-wired student | GRU student.
  SEES      panorama the teacher gets (dimmed when blind) | fly-eye input + neuron activity on the four populations | GRU hidden activity
  THINKS    latent trajectory (2-D PCA of the teacher latent, students' guesses overlaid) + cosine match
  NAVIGATES top-down arena with the true path and the pose each student *decodes from its own state* (ridge readout fit on
            training episodes, as in eval_hd.py); the view rendered at the decoded pose = "what the brain thinks it sees"
Outputs: results/video/wm2cx_paper.mp4 (+ .gif), results/video/wm2cx_paper_t??.png, paper/fig/video_strip.pdf

    PYTHONPATH=. python scripts/make_video_paper.py --episode 3 --blind 8:16 --n_fit 400
"""
import argparse, os, subprocess
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from wm2cx.env import FlyArena
from wm2cx.ckpt import load_run
from wm2cx.data import load_split

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", default=["connectome_K1_s0", "gru_K1_s0"])
p.add_argument("--labels", nargs="+", default=["connectome-wired student", "GRU student (parameter-matched)"])
p.add_argument("--episode", type=int, default=3); p.add_argument("--blind", default="8:16"); p.add_argument("--n_fit", type=int, default=400)
p.add_argument("--out", default="results/video/wm2cx_paper.mp4"); p.add_argument("--fps", type=int, default=12); p.add_argument("--hold", type=float, default=1.2)
p.add_argument("--device", default="cpu"); p.add_argument("--lam", type=float, default=100.0); p.add_argument("--strip", default="paper/fig/video_strip.pdf")
a = p.parse_args(); b0, b1 = map(int, a.blind.split(":")); dev = a.device
os.makedirs(os.path.dirname(a.out), exist_ok=True); os.makedirs(os.path.dirname(a.strip), exist_ok=True)
ACTION = {0: "stay", 1: "forward", 2: "turn left", 3: "turn right", 4: "back"}
torch.manual_seed(0); np.random.seed(0)

# ------------------------------------------------------------------ models, episode, states
models, Hs, Zh, lat = [], [], [], None
for r in a.runs:
    m, args, conn, lat = load_run(r, device=dev); models.append(m)
te, _ = load_split(["data/nav_test.npz"], lat); tr, _ = load_split(["data/nav_train.npz"], lat, with_z=False)
raw = np.load("data/nav_test.npz"); ep = max(a.episode, 0); T = te["obs"].shape[1]
obs, act, z = te["obs"][ep:ep + 1].to(dev), te["act"][ep:ep + 1].to(dev), te["z"][ep].numpy()
vis = torch.ones(1, T, device=dev); vis[:, b0:b1] = 0
for m in models:
    with torch.no_grad(): zh, H = m(obs, act, vis)
    Hs.append(H[0].cpu().numpy()); Zh.append(zh[0].cpu().numpy())
cos = [np.sum(zh * z, -1) / (np.linalg.norm(zh, axis=-1) * np.linalg.norm(z, axis=-1) + 1e-8) for zh in Zh]
pos, hd, acts = raw["pos"][ep], raw["heading"][ep], raw["act"][ep]
env = FlyArena(task="nav", res=256, n_boxes=1, seed=1 * 100000 + ep)
panos = [env.render(pos[t], hd[t]) for t in range(T)]

# ------------------------------------------------------------------ pose readout (ridge, as eval_hd.py), fit on training episodes in the dark
def states(m, d, n):
    out = []
    with torch.no_grad():
        for i in range(0, n, 64):
            j = min(i + 64, n); o, ac = d["obs"][i:j].to(dev), d["act"][i:j].to(dev); v = torch.ones(o.shape[:2], device=dev); v[:, b0:b1] = 0
            out.append(m(o, ac, v)[1].cpu().numpy())
    return np.concatenate(out)

def rel(d, n):
    th = d["heading"].numpy()[:n]; th0 = th[:, :1]; dth = th - th0
    dp = d["pos"].numpy()[:n] - d["pos"].numpy()[:n, :1]; c, s_ = np.cos(th0), np.sin(th0)
    dpr = np.stack([c * dp[..., 0] + s_ * dp[..., 1], -s_ * dp[..., 0] + c * dp[..., 1]], -1) / 6.0
    return np.concatenate([np.cos(dth)[..., None], np.sin(dth)[..., None], dpr], -1)          # [n, T, 4]

decoded = []   # per model: (heading[T], pos[T,2]) in world frame, decoded from its own state at every step
Y = rel(tr, a.n_fit).reshape(-1, 4)       # all steps of blind-window episodes: one readout valid with vision on and off
readouts, errs_all = [], []
for m, H in zip(models, Hs):
    X = states(m, tr, a.n_fit).reshape(-1, H.shape[-1]).astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xn = (X - mu) / sd
    W = np.linalg.solve(Xn.T @ Xn + a.lam * np.eye(Xn.shape[1]), Xn.T @ (Y - Y.mean(0))); readouts.append((mu, sd, W))
    # per-episode blind heading error over the whole test split, to report the population and to pick a representative episode
    Hte = states(m, te, len(te["obs"])); Yte = rel(te, len(te["obs"]))
    pr = ((Hte - mu) / sd) @ W + Y.mean(0)
    e = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(pr[..., 1], pr[..., 0]) - np.arctan2(Yte[..., 1], Yte[..., 0]))))))[:, b0:b1].mean(1)
    errs_all.append(e); print(f"{m.wiring:12s} test split: median per-episode blind heading error {np.median(e):.1f} deg (IQR {np.percentile(e, 25):.1f}-{np.percentile(e, 75):.1f})")
if a.episode < 0:   # representative episode: closest to the population medians of both models
    med = np.array([np.median(e) for e in errs_all]); E = np.stack(errs_all, 1)
    ep = int(np.argmin(np.abs(np.log(E + 1) - np.log(med + 1)).sum(1))); print(f"picked episode {ep}: errors {E[ep].round(1)} vs medians {med.round(1)}")
    obs, act, z = te["obs"][ep:ep + 1].to(dev), te["act"][ep:ep + 1].to(dev), te["z"][ep].numpy(); Hs, Zh = [], []
    for m in models:
        with torch.no_grad(): zh, H = m(obs, act, vis)
        Hs.append(H[0].cpu().numpy()); Zh.append(zh[0].cpu().numpy())
    cos = [np.sum(zh * z, -1) / (np.linalg.norm(zh, axis=-1) * np.linalg.norm(z, axis=-1) + 1e-8) for zh in Zh]
    pos, hd, acts = raw["pos"][ep], raw["heading"][ep], raw["act"][ep]
    env = FlyArena(task="nav", res=256, n_boxes=1, seed=1 * 100000 + ep); panos = [env.render(pos[t], hd[t]) for t in range(T)]
for (mu, sd, W), H in zip(readouts, Hs):
    pred = ((H - mu) / sd) @ W + Y.mean(0)                                                   # [T, 4] for the video episode
    th0 = hd[0]; dth = np.arctan2(pred[:, 1], pred[:, 0]); dpr = pred[:, 2:] * 6.0
    c, s_ = np.cos(th0), np.sin(th0); dpw = np.stack([c * dpr[:, 0] - s_ * dpr[:, 1], s_ * dpr[:, 0] + c * dpr[:, 1]], -1)
    decoded.append((th0 + dth, pos[0] + dpw))
    print(f"episode {ep}: blind heading error {np.degrees(np.abs(np.angle(np.exp(1j * (th0 + dth - hd)))))[b0:b1].mean():.1f} deg")
imagined = [[env.render(np.clip(P[t], 0.3, env.size - 0.3), TH[t]) for t in range(T)] for TH, P in decoded]
herr = [np.degrees(np.abs(np.angle(np.exp(1j * (TH - hd))))) for TH, _ in decoded]

# ------------------------------------------------------------------ latent PCA (2-D) of the teacher trajectory
zc = z - z.mean(0); Vt = np.linalg.svd(zc, full_matrices=False)[2][:2]
traj_t = zc @ Vt.T; traj_s = [(zh - z.mean(0)) @ Vt.T for zh in Zh]

# ------------------------------------------------------------------ brain layout (connectome) and GRU grid
conn = load_run(a.runs[0], device=dev)[2]; region = conn["region"]; layers = ["VIS", "CX", "MB", "OUT"]
LCOL = {"VIS": "#4ea8ff", "CX": "#ffb347", "MB": "#3ddc97", "OUT": "#ff5d8f"}
xy = np.zeros((len(region), 2)); rng = np.random.default_rng(0)
cent = {"VIS": (0.0, 0.0), "CX": (3.0, 0.0), "MB": (5.6, 0.0), "OUT": (8.0, 0.0)}
for L in layers:
    idx = np.where(region == L)[0]; n = len(idx); rad = 0.45 + 0.7 * np.sqrt(n / 2875)
    r = rad * np.sqrt(rng.random(n)); th = rng.random(n) * 2 * np.pi
    xy[idx] = np.array(cent[L]) + np.stack([r * np.cos(th) * 1.1, r * np.sin(th)], 1)
pre, post, sign = models[0].e_pre.cpu().numpy(), models[0].e_post.cpu().numpy(), models[0].e_sign.cpu().numpy()
keep = rng.choice(len(pre), 1500, replace=False); segs = np.stack([xy[pre[keep] % len(region)], xy[post[keep] % len(region)]], 1); sg = sign[keep]
amp = [np.percentile(np.abs(H), 99) for H in Hs]
is_gru = [m.wiring == "gru" for m in models]

# ------------------------------------------------------------------ figure
BG, INK, MUTED, RED, TEACH = "#0b0d12", "#f2f2f2", "#9aa3b2", "#ff6b6b", "#7ad7f0"
SCOL = ["#ffb347", "#c77dff"]
fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
colx = [0.035, 0.365, 0.695]; cw = 0.28
AX = {}
for ci in range(3):
    AX[f"see{ci}"] = fig.add_axes([colx[ci], 0.60, cw, 0.26]); AX[f"think{ci}"] = fig.add_axes([colx[ci], 0.30, cw, 0.22]); AX[f"nav{ci}"] = fig.add_axes([colx[ci], 0.04, cw, 0.20])
AX["eye"] = fig.add_axes([colx[0] + cw - 0.075, 0.805, 0.07, 0.05])
for ax in fig.axes: ax.set_facecolor(BG); ax.tick_params(colors=MUTED, labelsize=8); [s.set_color("#333") for s in ax.spines.values()]

def head(ci, text, color=INK):
    fig.text(colx[ci] + cw / 2, 0.955, text, ha="center", va="center", color=color, fontsize=15, fontweight="bold")

def rowlabel(y, text):
    fig.text(0.012, y, text, rotation=90, ha="center", va="center", color=MUTED, fontsize=11, fontweight="bold")

def draw(t):
    for ax in fig.axes: ax.cla(); ax.set_facecolor(BG)
    for txt in list(fig.texts): txt.remove()
    blind = b0 <= t < b1
    head(0, "WORLD MODEL (V-JEPA 2.1 teacher)", TEACH); head(1, "DISTILLED FLY CONNECTOME", SCOL[0]); head(2, "DISTILLED GRU (same loss, same budget)", SCOL[1])
    rowlabel(0.73, "S E E S"); rowlabel(0.41, "T H I N K S"); rowlabel(0.14, "N A V I G A T E S")
    fig.text(0.5, 0.905, f"t = {t:02d} / {T - 1}     action: {ACTION[int(acts[t])]}     " + ("VISION CUT: the students must carry the world forward from the action alone" if blind else "vision on"),
             ha="center", color=RED if blind else INK, fontsize=13)
    # ---- SEES
    ax = AX["see0"]; pan = panos[t].astype(float) * (0.18 if blind else 1.0); ax.imshow(pan.astype(np.uint8), aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("360° panorama in the arena (the teacher's input)" + ("  [cut]" if blind else ""), color=RED if blind else INK, fontsize=10, loc="left")
    for ci in (1, 2):
        ax = AX[f"see{ci}"]; H = Hs[ci - 1][t]
        if not is_gru[ci - 1]:
            ax.add_collection(LineCollection(segs, colors=np.where(sg > 0, "#3a86ff", "#ff006e"), linewidths=0.3, alpha=0.16))
            ax.scatter(xy[:, 0], xy[:, 1], c=H, cmap="coolwarm", vmin=-amp[0], vmax=amp[0], s=5, linewidths=0, zorder=3)
            for L in layers: ax.text(cent[L][0], 1.45, f"{L}  ({(region == L).sum()})", ha="center", color=LCOL[L], fontsize=9)
            ax.set_xlim(-1.6, 9.6); ax.set_ylim(-1.5, 1.8); ax.axis("off")
            ax.set_title("6,935 FlyWire neurons on 91,663 real synapses; colour = activity", color=INK, fontsize=10, loc="left")
        else:
            n = len(H); cols = int(np.ceil(np.sqrt(n * 1.4))); rows = int(np.ceil(n / cols)); grid = np.full(rows * cols, np.nan); grid[:n] = H
            ax.imshow(grid.reshape(rows, cols), cmap="coolwarm", vmin=-amp[1], vmax=amp[1], aspect="auto", interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{n} GRU hidden units (dense {n}×{n} recurrence); colour = activity", color=INK, fontsize=10, loc="left")
    ax = AX["eye"]; eye = raw["obs"][ep, t] if not blind else np.zeros_like(raw["obs"][ep, t]); ax.imshow(eye, aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("what the students get: fly-eye 32×8", color=MUTED, fontsize=7, pad=2)
    # ---- THINKS
    ax = AX["think0"]; ax.plot(traj_t[:, 0], traj_t[:, 1], color=TEACH, lw=1.2, alpha=0.5); ax.plot(traj_t[:t + 1, 0], traj_t[:t + 1, 1], color=TEACH, lw=2.2)
    ax.scatter(traj_t[t, 0], traj_t[t, 1], s=80, color=TEACH, zorder=5, edgecolors=INK)
    ax.scatter(traj_t[b0:b1, 0], traj_t[b0:b1, 1], s=14, color=RED, zorder=4)
    ax.set_title("teacher latent through the episode (2-D PCA of the 256-d code; red = steps the students are blind)", color=INK, fontsize=9, loc="left"); ax.set_xticks([]); ax.set_yticks([])
    for ci in (1, 2):
        ax = AX[f"think{ci}"]; i = ci - 1
        ax.plot(traj_t[:, 0], traj_t[:, 1], color=TEACH, lw=1.0, alpha=0.35)
        ax.plot(traj_s[i][:t + 1, 0], traj_s[i][:t + 1, 1], color=SCOL[i], lw=2.0); ax.scatter(traj_s[i][t, 0], traj_s[i][t, 1], s=80, color=SCOL[i], zorder=5, edgecolors=INK)
        ax.scatter(traj_t[t, 0], traj_t[t, 1], s=40, color=TEACH, zorder=4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"its guess of the teacher latent (cosine to teacher now: {cos[i][t]:.2f})", color=INK, fontsize=9, loc="left")
        ins = ax.inset_axes([0.62, 0.05, 0.36, 0.38]); ins.set_facecolor("#14161e"); ins.tick_params(colors=MUTED, labelsize=6)
        ins.axvspan(b0 - 0.5, b1 - 0.5, color=RED, alpha=0.15); ins.plot(np.arange(t + 1), cos[i][:t + 1], color=SCOL[i], lw=1.4); ins.set_xlim(0, T - 1); ins.set_ylim(0, 1)
        ins.set_title("cosine match", color=MUTED, fontsize=7, pad=2); [s.set_color("#333") for s in ins.spines.values()]
    # ---- NAVIGATES
    env.pos, env.heading = pos[t], hd[t]; top = env.topdown(); s = 256 / env.size
    ax = AX["nav0"]; ax.imshow(top, origin="lower"); ax.plot(pos[:t + 1, 0] * s, pos[:t + 1, 1] * s, color="#ffd166", lw=2)
    ax.scatter(pos[b0:min(t + 1, b1), 0] * s, pos[b0:min(t + 1, b1), 1] * s, s=18, color=RED, zorder=4)
    ax.arrow(pos[t, 0] * s, pos[t, 1] * s, 12 * np.cos(hd[t]), 12 * np.sin(hd[t]), color="#ffd166", width=1.5, head_width=6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("true path (yellow) and heading; red = blind steps", color=INK, fontsize=9, loc="left")
    for ci in (1, 2):
        i = ci - 1; TH, P = decoded[i]; ax = AX[f"nav{ci}"]
        ax.imshow(imagined[i][t], aspect="auto", extent=[0, 256, 0, 90]); ax.set_xticks([]); ax.set_yticks([])
        ins = ax.inset_axes([0.0, 0.0, 0.42, 1.0]); ins.imshow(top, origin="lower"); ins.set_xticks([]); ins.set_yticks([])
        Pc = np.clip(P, 0.2, env.size - 0.2)
        ins.plot(pos[:t + 1, 0] * s, pos[:t + 1, 1] * s, color="#ffd166", lw=1.5, alpha=0.8)
        ins.plot(Pc[:t + 1, 0] * s, Pc[:t + 1, 1] * s, color=SCOL[i], lw=1.5, ls="--")
        ins.arrow(Pc[t, 0] * s, Pc[t, 1] * s, 12 * np.cos(TH[t]), 12 * np.sin(TH[t]), color=SCOL[i], width=1.5, head_width=6)
        ins.set_xlim(0, 256); ins.set_ylim(0, 256)
        ax.set_title(f"where it thinks it is (dashed, read out of its state) and the view there;  heading error now {herr[i][t]:4.0f}°", color=INK, fontsize=9, loc="left")
    fig.text(0.5, 0.005, "Readouts are ridge regressions from the state to pose, fit on training episodes in the dark (no pose supervision during training). "
                         "Dashed path and rendered view = the pose decoded from the network's own activity.", ha="center", color=MUTED, fontsize=9)

w, h = fig.canvas.get_width_height()
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(a.fps), "-i", "-",
                       "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], stdin=subprocess.PIPE)
stills = sorted({0, b0 - 1, b0 + 3, b1 - 1, T - 1})
for t in range(T):
    draw(t); fig.canvas.draw(); buf = np.asarray(fig.canvas.buffer_rgba()).tobytes()
    for _ in range(int(a.fps * a.hold * (2.0 if t in (b0 - 1, b1 - 1) else 1.0))): ff.stdin.write(buf)
    if t in stills: fig.savefig(a.out.replace(".mp4", f"_t{t:02d}.png"), dpi=100, facecolor=BG)
ff.stdin.close(); ff.wait()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out, "-vf", "fps=3,scale=1100:-1:flags=lanczos", a.out.replace(".mp4", ".gif")])
print("wrote", a.out)

# ------------------------------------------------------------------ paper strip (light theme, vector)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "pdf.fonttype": 42})
cols_t = [b0 - 2, b0 - 1, b0 + 1, b0 + 4, b1 - 1, b1 + 1]
fig2 = plt.figure(figsize=(7.2, 5.0))
gs = fig2.add_gridspec(4, len(cols_t), height_ratios=[1, 1, 1, 1.5], hspace=0.28, wspace=0.08)
rows = [("true view", lambda t: panos[t]), ("fly connectome:\nview at its\ndecoded pose", lambda t: imagined[0][t]), ("GRU:\nview at its\ndecoded pose", lambda t: imagined[1][t])]
for ri, (lab, fn) in enumerate(rows):
    for ci, t in enumerate(cols_t):
        ax = fig2.add_subplot(gs[ri, ci]); img = fn(t)
        if ri == 0 and b0 <= t < b1: img = (img * 0.25).astype(np.uint8)
        ax.imshow(img, aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
        if ri == 0: ax.set_title(f"t={t}" + ("  (dark)" if b0 <= t < b1 else ""), fontsize=7.5, color="#c0392b" if b0 <= t < b1 else "black")
        if ci == 0: ax.set_ylabel(lab, fontsize=7.5)
        for sp in ax.spines.values(): sp.set_edgecolor("#c0392b" if (b0 <= t < b1 and ri > 0) else "#999"); sp.set_linewidth(1.2 if (b0 <= t < b1 and ri > 0) else 0.5)
ax = fig2.add_subplot(gs[3, :4])
for i, (lab, c) in enumerate([("fly connectome", "#c0392b"), ("GRU", "#1c2833")]):
    ax.plot(np.arange(T), herr[i], color=c, lw=1.4, marker="o", ms=2.5, label=lab)
ax.axvspan(b0 - 0.5, b1 - 0.5, color="#c0392b", alpha=0.08); ax.text((b0 + b1) / 2 - 0.5, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 50, "vision cut", ha="center", fontsize=7, color="#c0392b")
ax.set_xlabel("time step"); ax.set_ylabel("heading error of the\ndecoded pose (deg)"); ax.set_xlim(0, T - 1); ax.legend(frameon=False, fontsize=7, loc="upper left")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax = fig2.add_subplot(gs[3, 4:]); env.pos, env.heading = pos[T - 1], hd[T - 1]; top = env.topdown(); s = 256 / env.size
ax.imshow(top, origin="lower"); ax.plot(pos[:, 0] * s, pos[:, 1] * s, color="#f1c40f", lw=2, label="true path")
ax.scatter(pos[b0:b1, 0] * s, pos[b0:b1, 1] * s, s=10, color="#c0392b", zorder=4)
for i, (lab, c) in enumerate([("fly connectome", "#c0392b"), ("GRU", "#1c2833")]):
    P = np.clip(decoded[i][1], 0.2, env.size - 0.2); ax.plot(P[:, 0] * s, P[:, 1] * s, color=c, lw=1.2, ls="--", label=f"{lab}: decoded")
lo = np.clip(np.concatenate([pos, np.clip(decoded[0][1], 0.2, env.size - 0.2)]).min(0) * s - 25, 0, 256); hi = np.clip(np.concatenate([pos, np.clip(decoded[0][1], 0.2, env.size - 0.2)]).max(0) * s + 25, 0, 256)
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("top-down: true vs decoded path", fontsize=8); ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1)
fig2.savefig(a.strip, bbox_inches="tight"); print("wrote", a.strip)
