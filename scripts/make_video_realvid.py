"""Real-video version of the paper video: how the world model sees / thinks vs. how the distilled connectome (and a GRU)
sees / thinks / navigates on the user's own handheld phone clips (data/realvid_all_z.npz: frames + egomotion + V-JEPA z).

Leave-one-clip-out: both students are fine-tuned on the other clips (as in eval_ckpt.py --finetune_epochs) and shown on
one held-out episode through a blind window. "What it imagines" = the training frame (other clips) whose teacher latent is
nearest to the student's guess; "where it thinks it is" = heading / position read out of its own state by a ridge
readout fit on the training clips (pose targets come from the clip's own egomotion estimate, not from a simulator).

    PYTHONPATH=. python scripts/make_video_realvid.py --data data/realvid_all_z.npz --clip 6 --episode -1
Outputs results/video/wm2cx_realvid.{mp4,gif,_t??.png} and paper/fig/realvid_strip.pdf
"""
import argparse, os, subprocess
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
from wm2cx.ckpt import load_run
from wm2cx.data import load_split, blackout

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", default=["connectome_K1_s0", "gru_K1_s0"]); p.add_argument("--data", default="data/realvid_all_z.npz")
p.add_argument("--clip", type=int, default=-1, help="held-out clip (default: the clip with most episodes)"); p.add_argument("--episode", type=int, default=-1, help="episode index inside the held-out clip (-1: middle one)")
p.add_argument("--blind", default="8:16"); p.add_argument("--finetune_epochs", type=int, default=10); p.add_argument("--finetune_lr", type=float, default=5e-4)
p.add_argument("--out", default="results/video/wm2cx_realvid.mp4"); p.add_argument("--strip", default="paper/fig/realvid_strip.pdf")
p.add_argument("--fps", type=int, default=12); p.add_argument("--hold", type=float, default=1.2); p.add_argument("--device", default="cpu"); p.add_argument("--lam", type=float, default=100.0)
a = p.parse_args(); b0, b1 = map(int, a.blind.split(":")); dev = a.device; torch.manual_seed(0); rng = np.random.default_rng(0)
os.makedirs(os.path.dirname(a.out), exist_ok=True); os.makedirs(os.path.dirname(a.strip), exist_ok=True)
ACTION = {0: "stay", 1: "forward", 2: "turn left", 3: "turn right", 4: "back"}

# ------------------------------------------------------------------ data
raw = np.load(a.data); clips = raw["clip"]; src = raw["src"]
hold = a.clip if a.clip >= 0 else int(np.bincount(clips).argmax())
tr_idx, te_idx = np.where(clips != hold)[0], np.where(clips == hold)[0]
ep = te_idx[len(te_idx) // 2] if a.episode < 0 else te_idx[a.episode]
print(f"held-out clip {hold} ({src[ep]}): {len(te_idx)} episodes; fine-tune on {len(tr_idx)}; showing episode {ep}")
models, Hs, Zh, lat = [], [], [], None
for r in a.runs:
    m, args, conn, lat = load_run(r, device=dev); models.append(m)
d, _ = load_split([a.data], lat)                                     # z projected into the arena PCA basis of the checkpoint
E, T = d["obs"].shape[:2]
vis_ep = torch.ones(1, T, device=dev); vis_ep[:, b0:b1] = 0

# ------------------------------------------------------------------ leave-one-clip-out fine-tuning (eval_ckpt.py recipe)
def finetune(model, idx, epochs):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=a.finetune_lr, weight_decay=1e-4); idx = torch.from_numpy(idx)
    for e in range(epochs):
        perm = idx[torch.randperm(len(idx))]
        for i in range(0, len(perm), 16):
            j = perm[i:i + 16]; obs, act, z = d["obs"][j].to(dev), d["act"][j].to(dev), d["z"][j].to(dev)
            vis = blackout(len(j), T, 8, rng, dev); zh, H = model(obs, act, vis)
            loss = ((((zh - z) ** 2).mean(-1)) * (1.0 + 2.0 * (1 - vis))).mean() + 1e-4 * (H ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model.eval()

for m in models:
    finetune(m, tr_idx, a.finetune_epochs)
obs, act, z = d["obs"][ep:ep + 1].to(dev), d["act"][ep:ep + 1].to(dev), d["z"][ep].numpy()
for m in models:
    with torch.no_grad(): zh, H = m(obs, act, vis_ep)
    Hs.append(H[0].cpu().numpy()); Zh.append(zh[0].cpu().numpy())
cos = [np.sum(zh * z, -1) / (np.linalg.norm(zh, axis=-1) * np.linalg.norm(z, axis=-1) + 1e-8) for zh in Zh]
# copy-last-visible baseline on this episode (the trivial predictor the paper reports)
zc_last = z.copy(); zc_last[b0:b1] = z[b0 - 1]; cos_copy = np.sum(zc_last * z, -1) / (np.linalg.norm(zc_last, axis=-1) * np.linalg.norm(z, axis=-1) + 1e-8)
frames = raw["frames_full"][ep]; eye = raw["obs"][ep]; acts = raw["act"][ep]; pos, hd = raw["pos"][ep], raw["heading"][ep]

# ------------------------------------------------------------------ "what it imagines": nearest training frame by latent
bank_z = d["z"][torch.from_numpy(tr_idx)].numpy().reshape(-1, z.shape[-1]); bank_f = raw["frames"][tr_idx].reshape(-1, 64, 64, 3)
bank_n = bank_z / (np.linalg.norm(bank_z, axis=1, keepdims=True) + 1e-8)
def nearest(q): return bank_f[int(np.argmax(bank_n @ (q / (np.linalg.norm(q) + 1e-8))))]
imagined = [[nearest(zh[t]) for t in range(T)] for zh in Zh]; teacher_nn = [nearest(z[t]) for t in range(T)]

# ------------------------------------------------------------------ pose readout from the state (ridge on training clips, all steps, blind-window states)
def states(m, idx):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 32):
            j = torch.from_numpy(idx[i:i + 32]); o, ac = d["obs"][j].to(dev), d["act"][j].to(dev); v = torch.ones(o.shape[:2], device=dev); v[:, b0:b1] = 0
            out.append(m(o, ac, v)[1].cpu().numpy())
    return np.concatenate(out)
def rel(idx):
    th = raw["heading"][idx]; th0 = th[:, :1]; dth = th - th0; dp = raw["pos"][idx] - raw["pos"][idx][:, :1]; c, s_ = np.cos(th0), np.sin(th0)
    dpr = np.stack([c * dp[..., 0] + s_ * dp[..., 1], -s_ * dp[..., 0] + c * dp[..., 1]], -1)
    return np.concatenate([np.cos(dth)[..., None], np.sin(dth)[..., None], dpr], -1)
Y = rel(tr_idx).reshape(-1, 4); decoded, herr = [], []
for m, H in zip(models, Hs):
    X = states(m, tr_idx).reshape(-1, H.shape[-1]).astype(np.float64); mu, sd = X.mean(0), X.std(0) + 1e-6; Xn = (X - mu) / sd
    W = np.linalg.solve(Xn.T @ Xn + a.lam * np.eye(Xn.shape[1]), Xn.T @ (Y - Y.mean(0))); pred = ((H - mu) / sd) @ W + Y.mean(0)
    th0 = hd[0]; dth = np.arctan2(pred[:, 1], pred[:, 0]); c, s_ = np.cos(th0), np.sin(th0); dpr = pred[:, 2:]
    decoded.append((th0 + dth, pos[0] + np.stack([c * dpr[:, 0] - s_ * dpr[:, 1], s_ * dpr[:, 0] + c * dpr[:, 1]], -1)))
    herr.append(np.degrees(np.abs(np.angle(np.exp(1j * (th0 + dth - hd))))))
    print(f"{m.wiring:12s} cosine in the dark {cos[len(decoded)-1][b0:b1].mean():.3f} (copy-last {cos_copy[b0:b1].mean():.3f}); decoded heading err in the dark {herr[-1][b0:b1].mean():.1f} deg")
# action-only floor for heading on this episode: integrate the quantised action record
th_act = np.zeros(T)
for t in range(1, T): th_act[t] = th_act[t - 1] + (np.pi / 8) * ((acts[t - 1] == 2) - (acts[t - 1] == 3))
herr_act = np.degrees(np.abs(np.angle(np.exp(1j * (hd[0] + th_act - hd)))))

# ------------------------------------------------------------------ teacher latent 2-D trajectory
zc = z - z.mean(0); Vt = np.linalg.svd(zc, full_matrices=False)[2][:2]; traj_t = zc @ Vt.T; traj_s = [(zh - z.mean(0)) @ Vt.T for zh in Zh]

# ------------------------------------------------------------------ brain layout
conn = load_run(a.runs[0], device=dev)[2]; region = conn["region"]; layers = ["VIS", "CX", "MB", "OUT"]
LCOL = {"VIS": "#4ea8ff", "CX": "#ffb347", "MB": "#3ddc97", "OUT": "#ff5d8f"}; cent = {"VIS": (0.0, 0.0), "CX": (3.0, 0.0), "MB": (5.6, 0.0), "OUT": (8.0, 0.0)}
xy = np.zeros((len(region), 2)); r0 = np.random.default_rng(0)
for L in layers:
    idx = np.where(region == L)[0]; n = len(idx); rad = 0.45 + 0.7 * np.sqrt(n / 2875); rr = rad * np.sqrt(r0.random(n)); th = r0.random(n) * 2 * np.pi
    xy[idx] = np.array(cent[L]) + np.stack([rr * np.cos(th) * 1.1, rr * np.sin(th)], 1)
pre, post, sign = models[0].e_pre.cpu().numpy(), models[0].e_post.cpu().numpy(), models[0].e_sign.cpu().numpy()
keep = r0.choice(len(pre), 1500, replace=False); segs = np.stack([xy[pre[keep] % len(region)], xy[post[keep] % len(region)]], 1); sg = sign[keep]
amp = [np.percentile(np.abs(H), 99) for H in Hs]; is_gru = [m.wiring == "gru" for m in models]

# ------------------------------------------------------------------ video
BG, INK, MUTED, RED, TEACH = "#0b0d12", "#f2f2f2", "#9aa3b2", "#ff6b6b", "#7ad7f0"; SCOL = ["#ffb347", "#c77dff"]
fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG); colx = [0.035, 0.365, 0.695]; cw = 0.28; AX = {}
for ci in range(3):
    AX[f"see{ci}"] = fig.add_axes([colx[ci], 0.60, cw, 0.26]); AX[f"think{ci}"] = fig.add_axes([colx[ci], 0.30, cw, 0.22]); AX[f"nav{ci}"] = fig.add_axes([colx[ci], 0.04, cw, 0.20])
AX["eye"] = fig.add_axes([colx[0] + cw - 0.075, 0.805, 0.07, 0.05])
def head(ci, text, color=INK): fig.text(colx[ci] + cw / 2, 0.955, text, ha="center", va="center", color=color, fontsize=15, fontweight="bold")
def rowlabel(y, text): fig.text(0.012, y, text, rotation=90, ha="center", va="center", color=MUTED, fontsize=11, fontweight="bold")
sname = os.path.basename(str(src[ep]))[:40]

def draw(t):
    for ax in fig.axes: ax.cla(); ax.set_facecolor(BG); ax.tick_params(colors=MUTED, labelsize=8); [s.set_color("#333") for s in ax.spines.values()]
    for txt in list(fig.texts): txt.remove()
    blind = b0 <= t < b1
    head(0, "WORLD MODEL (V-JEPA 2.1 teacher)", TEACH); head(1, "DISTILLED FLY CONNECTOME", SCOL[0]); head(2, "DISTILLED GRU (same loss, same budget)", SCOL[1])
    rowlabel(0.73, "S E E S"); rowlabel(0.41, "T H I N K S"); rowlabel(0.14, "N A V I G A T E S")
    fig.text(0.5, 0.905, f"real phone clip: {sname}     t = {t:02d} / {T - 1}     estimated action: {ACTION[int(acts[t])]}     " +
             ("VISION CUT: the students carry the world forward from the action alone" if blind else "vision on"), ha="center", color=RED if blind else INK, fontsize=12.5)
    ax = AX["see0"]; fr = frames[t].astype(float) * (0.18 if blind else 1.0); ax.imshow(fr.astype(np.uint8), aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("the real frame (teacher input, 256 px, 2 Hz)" + ("  [cut]" if blind else ""), color=RED if blind else INK, fontsize=10, loc="left")
    for ci in (1, 2):
        ax = AX[f"see{ci}"]; H = Hs[ci - 1][t]
        if not is_gru[ci - 1]:
            ax.add_collection(LineCollection(segs, colors=np.where(sg > 0, "#3a86ff", "#ff006e"), linewidths=0.3, alpha=0.16))
            ax.scatter(xy[:, 0], xy[:, 1], c=H, cmap="coolwarm", vmin=-amp[0], vmax=amp[0], s=5, linewidths=0, zorder=3)
            for L in layers: ax.text(cent[L][0], 1.45, f"{L}  ({(region == L).sum()})", ha="center", color=LCOL[L], fontsize=9)
            ax.set_xlim(-1.6, 9.6); ax.set_ylim(-1.5, 1.8); ax.axis("off"); ax.set_title("6,935 FlyWire neurons on 91,663 real synapses; colour = activity", color=INK, fontsize=10, loc="left")
        else:
            n = len(H); cols = int(np.ceil(np.sqrt(n * 1.4))); rows = int(np.ceil(n / cols)); grid = np.full(rows * cols, np.nan); grid[:n] = H
            ax.imshow(grid.reshape(rows, cols), cmap="coolwarm", vmin=-amp[1], vmax=amp[1], aspect="auto", interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{n} GRU hidden units (dense {n}×{n} recurrence); colour = activity", color=INK, fontsize=10, loc="left")
    ax = AX["eye"]; ax.imshow(eye[t] if not blind else np.zeros_like(eye[t]), aspect="auto"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("what the students get: fly-eye 32×8", color=MUTED, fontsize=7, pad=2)
    ax = AX["think0"]; ax.plot(traj_t[:, 0], traj_t[:, 1], color=TEACH, lw=1.2, alpha=0.5); ax.plot(traj_t[:t + 1, 0], traj_t[:t + 1, 1], color=TEACH, lw=2.2)
    ax.scatter(traj_t[t, 0], traj_t[t, 1], s=80, color=TEACH, zorder=5, edgecolors=INK); ax.scatter(traj_t[b0:b1, 0], traj_t[b0:b1, 1], s=14, color=RED, zorder=4)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("teacher latent through the clip (2-D PCA; red = steps the students are blind)", color=INK, fontsize=9, loc="left")
    ins = ax.inset_axes([0.70, 0.62, 0.29, 0.36]); ins.imshow(teacher_nn[t]); ins.set_xticks([]); ins.set_yticks([]); ins.set_title("nearest training frame", color=MUTED, fontsize=7, pad=2)
    for ci in (1, 2):
        ax = AX[f"think{ci}"]; i = ci - 1
        ax.plot(traj_t[:, 0], traj_t[:, 1], color=TEACH, lw=1.0, alpha=0.35); ax.plot(traj_s[i][:t + 1, 0], traj_s[i][:t + 1, 1], color=SCOL[i], lw=2.0)
        ax.scatter(traj_s[i][t, 0], traj_s[i][t, 1], s=80, color=SCOL[i], zorder=5, edgecolors=INK); ax.scatter(traj_t[t, 0], traj_t[t, 1], s=40, color=TEACH, zorder=4)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"its guess of the teacher latent (cosine now {cos[i][t]:.2f}; copy-last baseline {cos_copy[t]:.2f})", color=INK, fontsize=9, loc="left")
        ins = ax.inset_axes([0.70, 0.62, 0.29, 0.36]); ins.imshow(imagined[i][t]); ins.set_xticks([]); ins.set_yticks([]); ins.set_title("what it imagines (nearest frame)", color=MUTED, fontsize=7, pad=2)
        ins = ax.inset_axes([0.62, 0.05, 0.36, 0.38]); ins.set_facecolor("#14161e"); ins.tick_params(colors=MUTED, labelsize=6); ins.axvspan(b0 - 0.5, b1 - 0.5, color=RED, alpha=0.15)
        ins.plot(np.arange(t + 1), cos[i][:t + 1], color=SCOL[i], lw=1.4); ins.plot(np.arange(t + 1), cos_copy[:t + 1], color=MUTED, lw=1.0, ls=":"); ins.set_xlim(0, T - 1); ins.set_ylim(0, 1)
        ins.set_title("cosine match (dotted: copy-last)", color=MUTED, fontsize=7, pad=2); [s.set_color("#333") for s in ins.spines.values()]
    # navigates: egomotion path from the clip vs decoded
    allp = np.concatenate([pos] + [np.clip(P, pos.min(0) - 5, pos.max(0) + 5) for _, P in decoded]); lo, hi = allp.min(0) - 1, allp.max(0) + 1
    ax = AX["nav0"]; ax.plot(pos[:t + 1, 0], pos[:t + 1, 1], color="#ffd166", lw=2); ax.scatter(pos[b0:min(t + 1, b1), 0], pos[b0:min(t + 1, b1), 1], s=18, color=RED, zorder=4)
    ax.arrow(pos[t, 0], pos[t, 1], 0.6 * np.cos(hd[t]), 0.6 * np.sin(hd[t]), color="#ffd166", width=0.05, head_width=0.25); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("camera path and heading estimated from the clip's own egomotion (yellow); red = blind steps", color=INK, fontsize=9, loc="left")
    for ci in (1, 2):
        i = ci - 1; TH, P = decoded[i]; ax = AX[f"nav{ci}"]; Pc = np.clip(P, pos.min(0) - 5, pos.max(0) + 5)
        ax.plot(pos[:t + 1, 0], pos[:t + 1, 1], color="#ffd166", lw=1.5, alpha=0.8); ax.plot(Pc[:t + 1, 0], Pc[:t + 1, 1], color=SCOL[i], lw=1.5, ls="--")
        ax.arrow(Pc[t, 0], Pc[t, 1], 0.6 * np.cos(TH[t]), 0.6 * np.sin(TH[t]), color=SCOL[i], width=0.05, head_width=0.25); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"where it thinks it is (dashed, read out of its state);  heading error now {herr[i][t]:4.0f}°  (action-count floor {herr_act[t]:3.0f}°)", color=INK, fontsize=9, loc="left")
    fig.text(0.5, 0.005, "Students fine-tuned on the other clips (leave-one-clip-out). Pose readouts are ridge regressions from the state to the clip's egomotion estimate, fit on the training clips. "
                         "Imagined frames are nearest-neighbour training frames by latent, not decoded images.", ha="center", color=MUTED, fontsize=9)

w, h = fig.canvas.get_width_height()
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(a.fps), "-i", "-",
                       "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], stdin=subprocess.PIPE)
stills = sorted({0, b0 - 1, b0 + 3, b1 - 1, T - 1})
for t in range(T):
    draw(t); fig.canvas.draw(); buf = np.asarray(fig.canvas.buffer_rgba()).tobytes()
    for _ in range(int(a.fps * a.hold * (2.0 if t in (b0 - 1, b1 - 1) else 1.0))): ff.stdin.write(buf)
    if t in stills: fig.savefig(a.out.replace(".mp4", f"_t{t:02d}.png"), dpi=100, facecolor=BG)
ff.stdin.close(); ff.wait()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out, "-vf", "fps=3,scale=1100:-1:flags=lanczos", a.out.replace(".mp4", ".gif")]); print("wrote", a.out)

# ------------------------------------------------------------------ paper strip
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "pdf.fonttype": 42})
cols_t = [b0 - 2, b0 - 1, b0 + 1, b0 + 4, b1 - 1, b1 + 1]
fig2 = plt.figure(figsize=(7.2, 5.0)); gs = fig2.add_gridspec(4, len(cols_t), height_ratios=[1, 1, 1, 1.5], hspace=0.28, wspace=0.08)
rows = [("real frame", lambda t: frames[t]), ("fly connectome\nimagines", lambda t: imagined[0][t]), ("GRU\nimagines", lambda t: imagined[1][t])]
for ri, (lab, fn) in enumerate(rows):
    for ci, t in enumerate(cols_t):
        ax = fig2.add_subplot(gs[ri, ci]); img = fn(t)
        if ri == 0 and b0 <= t < b1: img = (img * 0.25).astype(np.uint8)
        ax.imshow(img, aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
        if ri == 0: ax.set_title(f"t={t}" + ("  (dark)" if b0 <= t < b1 else ""), fontsize=7.5, color="#c0392b" if b0 <= t < b1 else "black")
        if ci == 0: ax.set_ylabel(lab, fontsize=7.5)
        for sp in ax.spines.values(): sp.set_edgecolor("#c0392b" if (b0 <= t < b1 and ri > 0) else "#999"); sp.set_linewidth(1.2 if (b0 <= t < b1 and ri > 0) else 0.5)
ax = fig2.add_subplot(gs[3, :3])
for i, (lab, c) in enumerate([("fly connectome", "#c0392b"), ("GRU", "#1c2833")]): ax.plot(np.arange(T), cos[i], color=c, lw=1.4, marker="o", ms=2.5, label=lab)
ax.plot(np.arange(T), cos_copy, color="#7f8c8d", lw=1.2, ls=":", label="copy last visible")
ax.axvspan(b0 - 0.5, b1 - 0.5, color="#c0392b", alpha=0.08); ax.set_ylim(0, 1); ax.set_xlim(0, T - 1); ax.set_xlabel("time step"); ax.set_ylabel("cosine to teacher latent"); ax.legend(frameon=False, fontsize=6.5, loc="lower left")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax = fig2.add_subplot(gs[3, 3:])
for i, (lab, c) in enumerate([("fly connectome", "#c0392b"), ("GRU", "#1c2833")]): ax.plot(np.arange(T), herr[i], color=c, lw=1.4, marker="o", ms=2.5, label=lab)
ax.plot(np.arange(T), herr_act, color="#7f8c8d", lw=1.2, ls=":", label="action-count integrator")
ax.axvspan(b0 - 0.5, b1 - 0.5, color="#c0392b", alpha=0.08); ax.set_xlim(0, T - 1); ax.set_xlabel("time step"); ax.set_ylabel("heading error vs. egomotion (deg)"); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig2.savefig(a.strip, bbox_inches="tight"); print("wrote", a.strip)
