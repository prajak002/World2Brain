"""Two synchronized videos in one frame: LEFT = the world model (V-JEPA 2.1) watching the NYC digital twin,
with its patch-feature map (PCA of the ViT tokens -> RGB) and its latent trajectory; RIGHT = the distilled student
in the fly CNS (FlyWire wiring): neuron activity on real soma positions, the latent it reconstructs, and the
training frame whose teacher latent is nearest to the student's guess ("what the brain imagines").
Needs: results/hq/nyc_ep4_t*.png, data/nyc_{train,test}.npz, results/runs/connectome_K1_s0_nyc.pt, results/brain3d.json."""
import argparse, json, os, subprocess, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from wm2cx.teacher import VJEPA21Encoder
from wm2cx.ckpt import build_model
from wm2cx.data import load_split, Latents
from wm2cx.connectome import load as load_conn

p = argparse.ArgumentParser()
p.add_argument("--run", default="connectome_K1_s0_nyc"); p.add_argument("--episode", type=int, default=4); p.add_argument("--blind", default="8:16")
p.add_argument("--frames", default="results/hq/nyc_ep4_t{t:02d}.png"); p.add_argument("--out", default="results/video/wm2cx_nyc_duel.mp4")
p.add_argument("--fps", type=int, default=12); p.add_argument("--hold", type=float, default=1.5); p.add_argument("--device", default="cpu")
a = p.parse_args(); b0, b1 = map(int, a.blind.split(":")); dev = a.device; os.makedirs(os.path.dirname(a.out), exist_ok=True)
ACTION = {0: "stay", 1: "step forward", 2: "turn left", 3: "turn right", 4: "step back"}

# ---------- student on the stored episode ----------
ck = torch.load(f"results/runs/{a.run}.pt", weights_only=False); ar = ck["args"]; conn = load_conn(ar["conn"])
lat = Latents.__new__(Latents); lat.mu, lat.P, lat.scale = ck["lat"]["mu"], ck["lat"]["P"], ck["lat"]["scale"]
te, _ = load_split(ar["test"], lat); raw = np.load(ar["test"][0]); tr = np.load(ar["train"][0]); ep = a.episode; T = te["obs"].shape[1]
obs, act = te["obs"][ep:ep + 1].to(dev), te["act"][ep:ep + 1].to(dev); z = te["z"][ep].numpy()
vis = torch.ones(1, T, device=dev); vis[:, b0:b1] = 0
m = build_model(ar, conn, obs.shape[-1], ck["state"]); m.load_state_dict(ck["state"]); m.to(dev).eval()
with torch.no_grad(): zh, H = m(obs, act, vis)
zh, H = zh[0].cpu().numpy(), H[0].cpu().numpy(); region = conn["region"]
match = 100 * np.clip(np.sum(zh * z, -1) / (np.linalg.norm(zh, axis=-1) * np.linalg.norm(z, axis=-1) + 1e-8), 0, 1)
acts, heading = raw["act"][ep], raw["heading"][ep]
# retrieval bank: training frames by teacher latent (cosine), "what the brain imagines"
bank_z = lat(tr["z"].reshape(-1, tr["z"].shape[-1])); bank_f = tr["frames"].reshape(-1, *tr["frames"].shape[2:])
bank_n = bank_z / (np.linalg.norm(bank_z, axis=1, keepdims=True) + 1e-8)
def nearest(q): return bank_f[np.argmax(bank_n @ (q / (np.linalg.norm(q) + 1e-8)))]
imagined = [nearest(zh[t]) for t in range(T)]; teacher_nn = [nearest(z[t]) for t in range(T)]

# ---------- teacher on the HQ frames: patch-feature map + trajectory ----------
hq = [np.asarray(Image.open(a.frames.format(t=t)).convert("RGB")) for t in range(T)]
small = np.stack([np.asarray(Image.fromarray(f).resize((256, 256), Image.LANCZOS)) for f in hq])
teacher = VJEPA21Encoder("vitb", crop=256, frames=16).to(dev); clip = 8
clips = torch.from_numpy(small.reshape(T // clip, clip, 256, 256, 3).repeat(2, axis=1)).to(dev)
with torch.no_grad(): tok = teacher(clips).cpu().numpy()                    # [3, 8*16*16, 768]
tok = tok.reshape(T, 16, 16, -1); flat = tok.reshape(-1, tok.shape[-1]); mu = flat.mean(0)
U, S, Vt = np.linalg.svd(flat - mu, full_matrices=False); pc = (flat - mu) @ Vt[:3].T
lo, hi = np.percentile(pc, 2, axis=0), np.percentile(pc, 98, axis=0); fmap = np.clip((pc - lo) / (hi - lo), 0, 1).reshape(T, 16, 16, 3)
zt = lat(teacher.per_step_latents(clips).cpu().numpy().reshape(T, -1)); traj = (zt - zt.mean(0)) @ np.linalg.svd(zt - zt.mean(0), full_matrices=False)[2][:2].T
trajs = (zh - zt.mean(0)) @ np.linalg.svd(zt - zt.mean(0), full_matrices=False)[2][:2].T

# ---------- brain layout ----------
D = json.load(open("results/brain3d.json")); pos = np.array(D["pos"], np.float32); reg3 = np.array(D["region"])
REG = {"VIS": ("#4ea8ff", "Eyes (visual)"), "CX": ("#ffb347", "Compass & map"), "MB": ("#3ddc97", "Memory"), "OUT": ("#ff5d8f", "Motor output")}
cols = np.array([matplotlib.colors.to_rgb(REG[r][0]) for r in reg3]); amp99 = np.percentile(np.abs(H), 99)
BG, INK, MUTED, TEACH, STUD = "#0b0d12", "#f2f2f2", "#9aa3b2", "#4ea8ff", "#ffb347"
fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
L = dict(main=[0.035, 0.40, 0.44, 0.50], a=[0.035, 0.10, 0.20, 0.27], b=[0.26, 0.10, 0.215, 0.27])
R = dict(main=[0.525, 0.40, 0.44, 0.50], a=[0.525, 0.10, 0.20, 0.27], b=[0.75, 0.10, 0.215, 0.27])
AX = {k: fig.add_axes(v) for k, v in {**{"L" + k: v for k, v in L.items()}, **{"R" + k: v for k, v in R.items()}}.items()}
fig.text(0.255, 0.955, "WORLD MODEL", color=TEACH, fontsize=24, weight="bold", ha="center"); fig.text(0.255, 0.922, "V-JEPA 2.1 (ViT-B) watching the NYC digital twin", color=MUTED, fontsize=13, ha="center")
fig.text(0.745, 0.955, "DISTILLED INTO A FLY CNS", color=STUD, fontsize=24, weight="bold", ha="center"); fig.text(0.745, 0.922, "FlyWire connectome student, 6,935 neurons, 91,663 real synapses", color=MUTED, fontsize=13, ha="center")
fig.add_artist(plt.Line2D([0.5, 0.5], [0.08, 0.90], color="#2a2f3a", lw=1.5)); cap = fig.text(0.5, 0.035, "", color=INK, fontsize=14, ha="center")

def caption(t):
    if t == 0: return "Same moment, two systems. The world model turns the view into a 256-number summary; the fly brain must reproduce that summary from a 64×64 glimpse and its own motion."
    if t < b0: return "Eyes open. Left: the teacher's internal feature map and where this moment sits in its latent space. Right: the brain's activity, its guess of the summary, and the scene it is closest to."
    if t == b0: return "Lights off for 8 steps. The teacher still sees; the brain does not. Watch the two trajectories diverge."
    if t < b1: return "Blind. The brain's visual neurons go quiet and the central complex carries on. Its 'imagined' scene drifts but stays on the same street."
    if t == b1: return "Lights on. The brain's guess snaps back towards the teacher."
    return "The wiring is the fly's; only synapse strengths were learned, by copying the teacher. Blind-window error: real wiring 0.885 vs scrambled 0.906, random 0.917."

def draw(t, ang, fade):
    for ax in AX.values(): ax.clear(); ax.set_facecolor(BG); ax.axis("off")
    blind = b0 <= t < b1
    # LEFT main: HQ frame
    ax = AX["Lmain"]; ax.imshow(hq[t]); ax.text(0.02, 0.97, f"what the teacher sees · step {t + 1}/{T}", color=INK, fontsize=12, va="top", transform=ax.transAxes, bbox=dict(facecolor="#000000aa", edgecolor="none", pad=5))
    ax.text(0.02, 0.03, f"{ACTION[int(acts[t])]} · heading {heading[t]:.0f}°", color=INK, fontsize=11, transform=ax.transAxes, bbox=dict(facecolor="#000000aa", edgecolor="none", pad=5))
    # LEFT a: feature map
    ax = AX["La"]; ax.imshow(fmap[t], interpolation="nearest"); ax.set_title("teacher's feature map (ViT tokens)", color=MUTED, fontsize=11, loc="left")
    # LEFT b: trajectory
    ax = AX["Lb"]; ax.set_title("latent space, 2-D view", color=MUTED, fontsize=11, loc="left"); allp = np.concatenate([traj, trajs]); pad = 0.1 * np.ptp(allp, 0).max()
    ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad); ax.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    ax.plot(traj[: t + 1, 0], traj[: t + 1, 1], color=TEACH, lw=2); ax.plot(trajs[: t + 1, 0], trajs[: t + 1, 1], color=STUD, lw=2, alpha=0.9)
    ax.plot([traj[t, 0], trajs[t, 0]], [traj[t, 1], trajs[t, 1]], color=MUTED, lw=1, ls=":"); ax.scatter([traj[t, 0]], [traj[t, 1]], s=90, c=TEACH, zorder=5); ax.scatter([trajs[t, 0]], [trajs[t, 1]], s=90, c=STUD, zorder=5)
    ax.text(0.98, 0.04, "teacher", color=TEACH, fontsize=10, ha="right", transform=ax.transAxes); ax.text(0.98, 0.12, "brain", color=STUD, fontsize=10, ha="right", transform=ax.transAxes)
    # RIGHT main: brain
    ax = AX["Rmain"]; c, s = np.cos(ang), np.sin(ang); x = pos[:, 0] * c - pos[:, 2] * s; zc = pos[:, 0] * s + pos[:, 2] * c; y = pos[:, 1]
    amp = np.clip(np.abs(H[t]) / amp99, 0, 1); rgba = np.concatenate([cols, (0.12 + 0.88 * amp)[:, None]], 1); o = np.argsort(zc)
    ax.scatter(x[o], y[o], s=(3 + 26 * amp ** 1.5)[o], c=rgba[o], linewidths=0, rasterized=True); ax.set_xlim(-330, 330); ax.set_ylim(-230, 230); ax.set_aspect("equal")
    ax.text(0.02, 0.97, ("EYES CLOSED · imagining" if blind else "eyes open") + f" · step {t + 1}/{T}", color=("#ffd166" if blind else INK), fontsize=12, va="top", transform=ax.transAxes, weight="bold" if blind else "normal")
    for i, (k, (cc, lab)) in enumerate(REG.items()):
        ax.scatter([0.03], [0.12 - 0.04 * i], s=50, c=[cc], transform=ax.transAxes, clip_on=False); ax.text(0.055, 0.12 - 0.04 * i, f"{lab} {np.mean(np.abs(H[t][region == k])) / amp99:.2f}", color=INK, fontsize=10, va="center", transform=ax.transAxes)
    # RIGHT a: imagined scene
    ax = AX["Ra"]; im = np.asarray(Image.fromarray(imagined[t]).resize((256, 256), Image.BICUBIC)); ax.imshow(im if not blind else im, interpolation="bicubic")
    ax.set_title("brain 'imagines' (nearest known scene)", color=MUTED, fontsize=11, loc="left")
    if blind: ax.add_patch(Rectangle((0, 0), 255, 255, fill=False, ec="#ffd166", lw=3))
    # RIGHT b: bars + match
    ax = AX["Rb"]; n = 48; xs = np.arange(n); ax.bar(xs - 0.2, z[t, :n], 0.4, color=TEACH); ax.bar(xs + 0.2, zh[t, :n], 0.4, color=STUD); lim = max(1, np.abs(z[:, :n]).max() * 1.05); ax.set_ylim(-lim * 1.6, lim); ax.set_xlim(-1, n)
    ax.set_title("teacher (blue) vs brain's guess (orange)", color=MUTED, fontsize=11, loc="left"); mm = match[t]
    ax.add_patch(Rectangle((-1, -lim * 1.55), n + 1, lim * 0.35, color="#1f2430")); ax.add_patch(Rectangle((-1, -lim * 1.55), (n + 1) * mm / 100, lim * 0.35, color="#3ddc97" if mm > 50 else ("#ffd166" if mm > 25 else "#ff5d8f")))
    ax.text(n / 2, -lim * 1.375, f"match {mm:.0f} %", color=INK, fontsize=12, weight="bold", ha="center", va="center")
    cap.set_text(caption(t))

nper = max(1, int(round(a.hold * a.fps))); tmp = a.out + ".frames"; os.makedirs(tmp, exist_ok=True); k = 0; ang = 0.0
for t in range(T):
    for j in range(nper):
        draw(t, ang, 0); ang += 0.004; fig.savefig(f"{tmp}/f{k:05d}.png", facecolor=BG); k += 1
    if t in (0, b0 - 1, b1 - 1, T - 1): fig.savefig(a.out.replace(".mp4", f"_t{t:02d}.png"), facecolor=BG)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/f%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out, "-vf", "fps=6,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer", a.out.replace(".mp4", ".gif")], check=True)
for f in os.listdir(tmp): os.remove(f"{tmp}/{f}")
os.rmdir(tmp); print("wrote", a.out, "frames", k)
