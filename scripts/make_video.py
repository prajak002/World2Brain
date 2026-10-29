"""Video: world-model side (arena panorama the teacher sees, top-down path, fly-eye input, teacher vs student latent)
next to the layered connectome student (VIS -> CX -> MB -> OUT populations lit by activity), through a blind window.
Optionally a second student (e.g. degree-shuffled control) is shown below for structural comparison."""
import argparse, subprocess, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from wm2cx.env import FlyArena
from wm2cx.student import FlyBrain
from wm2cx.data import load_split, Latents
from wm2cx.connectome import load as load_conn

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", default=["connectome_K1_s0", "shuffled_K1_s0"])
p.add_argument("--episode", type=int, default=3); p.add_argument("--blind", default="8:16")
p.add_argument("--out", default="results/video/wm2cx_connectome.mp4"); p.add_argument("--fps", type=int, default=2)
p.add_argument("--device", default="cpu"); p.add_argument("--n_edges", type=int, default=2500)
a = p.parse_args()
b0, b1 = map(int, a.blind.split(":")); dev = a.device
import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)

# ---------- data + models ----------
cks = [torch.load(f"results/runs/{r}.pt", weights_only=False) for r in a.runs]
args = cks[0]["args"]; conn = load_conn(args["conn"])
lat = Latents.__new__(Latents); lat.mu, lat.P, lat.scale = cks[0]["lat"]["mu"], cks[0]["lat"]["P"], cks[0]["lat"]["scale"]
te, _ = load_split(["data/nav_test.npz"], lat)
raw = np.load("data/nav_test.npz"); ep = a.episode; T = te["obs"].shape[1]
obs, act, z = te["obs"][ep:ep + 1].to(dev), te["act"][ep:ep + 1].to(dev), te["z"][ep].numpy()
vis = torch.ones(1, T, device=dev); vis[:, b0:b1] = 0
models, Hs, Zh = [], [], []
for ck, r in zip(cks, a.runs):
    ar = ck["args"]; m = FlyBrain(conn, obs.shape[-1], ar["k"], wiring=ar["wiring"], n_brains=ar["n_brains"], seed=ar["seed"])
    m.load_state_dict(ck["state"]); m.to(dev).eval()
    with torch.no_grad(): zh, H = m(obs, act, vis)
    models.append(m); Hs.append(H[0].cpu().numpy()); Zh.append(zh[0].cpu().numpy())
cos = [np.sum(zh * z, -1) / (np.linalg.norm(zh, axis=-1) * np.linalg.norm(z, axis=-1) + 1e-8) for zh in Zh]

# ---------- arena replay ----------
env = FlyArena(task="nav", res=256, n_boxes=1, seed=1 * 100000 + ep)
pos, hd = raw["pos"][ep], raw["heading"][ep]
panos = [env.render(pos[t], hd[t]) for t in range(T)]

# ---------- layered layout of the real neurons ----------
region, ctype, side = conn["region"], conn["cell_type"], conn["side"]
layers = ["VIS", "CX", "MB", "OUT"]; xy = np.zeros((len(region), 2))
for li, L in enumerate(layers):
    idx = np.where(region == L)[0]; order = idx[np.lexsort((side[idx], ctype[idx]))]
    n = len(order); cols = max(6, int(np.sqrt(n * 1.5))); rows = int(np.ceil(n / cols))
    r, c = np.arange(n) // cols, np.arange(n) % cols
    xy[order, 0] = li * 3.0 + (c - cols / 2) / cols * 2.0
    xy[order, 1] = (rows / 2 - r) / max(rows, 1) * 2.2 * min(1.0, rows / 45)
rng = np.random.default_rng(0)
def edge_segments(m):
    pre, post, sign = m.e_pre.cpu().numpy(), m.e_post.cpu().numpy(), m.e_sign.cpu().numpy()
    N = len(region) if m.N == len(region) else None
    keep = rng.choice(len(pre), min(a.n_edges, len(pre)), replace=False)
    pre, post, sign = pre[keep] % len(region), post[keep] % len(region), sign[keep]
    return np.stack([xy[pre], xy[post]], 1), sign
segs = [edge_segments(m) for m in models]
amp = [np.abs(H).max() for H in Hs]

# ---------- figure ----------
nR = len(models); fig = plt.figure(figsize=(16, 4.2 + 3.2 * nR), dpi=100, facecolor="#0e0f14")
gs = fig.add_gridspec(2 + nR, 6, height_ratios=[2.2, 1.3] + [2.6] * nR, hspace=0.45, wspace=0.35)
ax_pan = fig.add_subplot(gs[0, :3]); ax_map = fig.add_subplot(gs[0, 3:5]); ax_eye = fig.add_subplot(gs[0, 5])
ax_lat = fig.add_subplot(gs[1, :4]); ax_cos = fig.add_subplot(gs[1, 4:])
ax_net = [fig.add_subplot(gs[2 + i, :4]) for i in range(nR)]; ax_ras = [fig.add_subplot(gs[2 + i, 4:]) for i in range(nR)]
for ax in fig.axes: ax.set_facecolor("#0e0f14"); ax.tick_params(colors="#aaa", labelsize=7); [s.set_color("#444") for s in ax.spines.values()]
W = "#e8e8e8"
reg_sorted = np.concatenate([np.where(region == L)[0] for L in layers]); bounds = np.cumsum([(region == L).sum() for L in layers])

def draw(t):
    for ax in fig.axes: ax.cla(); ax.set_facecolor("#0e0f14")
    blind = b0 <= t < b1
    pan = panos[t].astype(float) * (0.25 if blind else 1.0)
    ax_pan.imshow(pan.astype(np.uint8), aspect="auto"); ax_pan.set_xticks([]); ax_pan.set_yticks([])
    ax_pan.set_title(f"World model view (360° panorama, teacher input)   t={t:02d}/{T-1}   " + ("BLIND: student imagines" if blind else "visible"), color="#ff6b6b" if blind else W, fontsize=11, loc="left")
    env.pos, env.heading = pos[t], hd[t]; ax_map.imshow(env.topdown(), origin="lower"); s = 256 / env.size
    ax_map.plot(pos[: t + 1, 0] * s, pos[: t + 1, 1] * s, color="#ffd166", lw=1.5); ax_map.scatter(pos[b0:min(t + 1, b1), 0] * s, pos[b0:min(t + 1, b1), 1] * s, s=12, color="#ff6b6b")
    ax_map.set_title("top-down path (red = blind steps)", color=W, fontsize=9); ax_map.set_xticks([]); ax_map.set_yticks([])
    eye = raw["obs"][ep, t] if not blind else np.zeros_like(raw["obs"][ep, t]); ax_eye.imshow(eye, aspect="auto"); ax_eye.set_title("fly-eye input 32×8", color=W, fontsize=9); ax_eye.set_xticks([]); ax_eye.set_yticks([])
    k = 96; ax_lat.plot(z[t, :k], color="#7ad7f0", lw=1.4, label="teacher latent (V-JEPA 2.1, PCA)")
    for zh, r, c in zip(Zh, a.runs, ["#ffd166", "#c77dff", "#06d6a0"]): ax_lat.plot(zh[t, :k], color=c, lw=1.0, alpha=0.9, label=f"student {r.split('_')[0]}")
    ax_lat.set_ylim(-3.5, 3.5); ax_lat.legend(fontsize=7, loc="upper right", facecolor="#1a1b22", labelcolor=W, ncol=3); ax_lat.set_title("latent prediction (first 96 PCA dims)", color=W, fontsize=9)
    for cs, r, c in zip(cos, a.runs, ["#ffd166", "#c77dff", "#06d6a0"]): ax_cos.plot(np.arange(t + 1), cs[: t + 1], color=c, lw=1.5, label=r.split('_')[0])
    ax_cos.axvspan(b0 - 0.5, b1 - 0.5, color="#ff6b6b", alpha=0.15); ax_cos.set_xlim(0, T - 1); ax_cos.set_ylim(0, 1); ax_cos.set_title("cosine(student, teacher)", color=W, fontsize=9); ax_cos.legend(fontsize=7, facecolor="#1a1b22", labelcolor=W)
    for i, (m, H, r) in enumerate(zip(models, Hs, a.runs)):
        ax = ax_net[i]; (sg, sign) = segs[i]
        ax.add_collection(LineCollection(sg, colors=np.where(sign > 0, "#3a86ff", "#ff006e"), linewidths=0.25, alpha=0.18))
        h = H[t] if H.shape[1] == len(region) else H[t][: len(region)]
        ax.scatter(xy[:, 0], xy[:, 1], c=h, cmap="coolwarm", vmin=-amp[i], vmax=amp[i], s=4, linewidths=0, zorder=3)
        for li, L in enumerate(layers): ax.text(li * 3.0, 1.25, f"{L}\n{(region == L).sum()} neurons", ha="center", color=W, fontsize=8)
        ax.set_xlim(-1.5, 10.5); ax.set_ylim(-1.3, 1.6); ax.axis("off")
        ax.set_title(f"{r}: {m.wiring} wiring, {len(m.e_pre)} synapse pairs (blue excitatory / red inhibitory sample), colour = activity", color=W, fontsize=9, loc="left")
        ras = ax_ras[i]; ras.imshow(H[:, reg_sorted].T, aspect="auto", cmap="coolwarm", vmin=-amp[i], vmax=amp[i], interpolation="nearest")
        for b in bounds[:-1]: ras.axhline(b, color=W, lw=0.5, alpha=0.6)
        for li, L in enumerate(layers): ras.text(-0.6, (bounds[li] + (bounds[li - 1] if li else 0)) / 2, L, color=W, fontsize=7, ha="right", va="center")
        ras.axvspan(b0 - 0.5, b1 - 0.5, color="#ff6b6b", alpha=0.12); ras.axvline(t, color="#ffd166", lw=1.5); ras.set_yticks([]); ras.set_xlabel("time step", color=W, fontsize=8)
        ras.set_title("activity raster (neurons by layer)", color=W, fontsize=9)
    fig.suptitle("World model → connectome distillation: V-JEPA 2.1 teacher, FlyWire-wired recurrent students", color=W, fontsize=13, y=0.995)

w, h = fig.canvas.get_width_height()
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(a.fps), "-i", "-",
                       "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], stdin=subprocess.PIPE)
for t in range(T):
    draw(t); fig.canvas.draw(); buf = np.asarray(fig.canvas.buffer_rgba())
    for _ in range(1 if t not in (0, T - 1) else a.fps): ff.stdin.write(buf.tobytes())
    if t in (b0 - 1, b1 - 1): fig.savefig(a.out.replace(".mp4", f"_t{t:02d}.png"), dpi=100, facecolor=fig.get_facecolor())
ff.stdin.close(); ff.wait()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out, "-vf", "fps=2,scale=1100:-1:flags=lanczos", a.out.replace(".mp4", ".gif")])
print("wrote", a.out, "and", a.out.replace(".mp4", ".gif"))
