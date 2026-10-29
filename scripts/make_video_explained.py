"""Explainable video: plain-English narrated story of one episode.
Teacher (V-JEPA 2.1) watches the world; a fly brain (real FlyWire wiring) must guess the teacher's summary of the scene,
first with eyes open, then in the dark. A second brain with the same neurons but scrambled wiring is shown for comparison."""
import argparse, subprocess, os, textwrap, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from wm2cx.env import FlyArena
from wm2cx.student import FlyBrain
from wm2cx.data import load_split, Latents
from wm2cx.connectome import load as load_conn

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", default=["connectome_K1_s0", "shuffled_K1_s0"])
p.add_argument("--labels", nargs="+", default=["Real fly wiring", "Same neurons, scrambled wires"])
p.add_argument("--episode", type=int, default=3); p.add_argument("--blind", default="8:16")
p.add_argument("--out", default="results/video/wm2cx_explained.mp4"); p.add_argument("--fps", type=int, default=10)
p.add_argument("--hold", type=float, default=1.6, help="seconds per time step"); p.add_argument("--device", default="cpu")
p.add_argument("--data", default="data/nav_test.npz"); p.add_argument("--world", default="a simple arena", help="name of the world for the captions")
a = p.parse_args(); b0, b1 = map(int, a.blind.split(":")); dev = a.device
os.makedirs(os.path.dirname(a.out), exist_ok=True)
ACTION = {0: "stay still", 1: "step forward", 2: "turn left", 3: "turn right", 4: "step back"}

# ---------- models + episode ----------
cks = [torch.load(f"results/runs/{r}.pt", weights_only=False) for r in a.runs]; conn = load_conn(cks[0]["args"]["conn"])
lat = Latents.__new__(Latents); lat.mu, lat.P, lat.scale = cks[0]["lat"]["mu"], cks[0]["lat"]["P"], cks[0]["lat"]["scale"]
te, _ = load_split([a.data], lat); raw = np.load(a.data); ep = a.episode; T = te["obs"].shape[1]
obs, act, z = te["obs"][ep:ep + 1].to(dev), te["act"][ep:ep + 1].to(dev), te["z"][ep].numpy()
vis = torch.ones(1, T, device=dev); vis[:, b0:b1] = 0
Hs, Zh, models = [], [], []
for ck in cks:
    ar = ck["args"]; m = FlyBrain(conn, obs.shape[-1], ar["k"], wiring=ar["wiring"], n_brains=ar["n_brains"], seed=ar["seed"])
    m.load_state_dict(ck["state"]); m.to(dev).eval()
    with torch.no_grad(): zh, H = m(obs, act, vis)
    models.append(m); Hs.append(H[0].cpu().numpy()); Zh.append(zh[0].cpu().numpy())
match = [100 * np.clip(np.sum(zh * z, -1) / (np.linalg.norm(zh, axis=-1) * np.linalg.norm(z, axis=-1) + 1e-8), 0, 1) for zh in Zh]
pos, hd, acts = raw["pos"][ep], raw["heading"][ep], raw["act"][ep]
if "frames" in raw.files: panos = [raw["frames"][ep, t] for t in range(T)]          # stored renders (NYC digital twin)
else:
    env = FlyArena(task="nav", res=256, n_boxes=1, seed=1 * 100000 + ep); panos = [env.render(pos[t], hd[t]) for t in range(T)]

# ---------- brain layout: 4 named areas ----------
region = conn["region"]; AREAS = [("VIS", "Eyes\n(visual neurons)"), ("CX", "Compass & map\n(central complex)"), ("MB", "Memory\n(mushroom body)"), ("OUT", "Motor output\n(descending)")]
centers = {"VIS": (0.0, 0.0), "CX": (3.2, 0.0), "MB": (6.0, 0.0), "OUT": (8.6, 0.0)}; xy = np.zeros((len(region), 2)); rng = np.random.default_rng(0)
for L, _ in AREAS:
    idx = np.where(region == L)[0]; n = len(idx); rad = 0.35 + 0.55 * np.sqrt(n / 2875)
    r = rad * np.sqrt(rng.random(n)); th = rng.random(n) * 2 * np.pi
    xy[idx] = np.array(centers[L]) + np.stack([r * np.cos(th) * 1.15, r * np.sin(th)], 1)
amp = [np.percentile(np.abs(H), 99) for H in Hs]
pre_r, post_r = region[conn["pre"]], region[conn["post"]]
flow = {(A, B): int(((pre_r == A) & (post_r == B)).sum()) for A, _ in AREAS for B, _ in AREAS if A != B}

BG, W, Y, R_, G_ = "#101218", "#f2f2f2", "#ffd166", "#ff6b6b", "#06d6a0"
fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
def box(ax, text, size=13, color=W, y=0.5):
    ax.axis("off"); ax.text(0.5, y, text, ha="center", va="center", color=color, fontsize=size, wrap=True, transform=ax.transAxes, linespacing=1.5)

def caption(t):
    if t == 0: return ("A teacher AI (V-JEPA 2.1, a video world model) watches this 360° view and writes a short numeric summary of the scene.\n"
                       "The fly brain on the right sees only a tiny 32×8 picture and must guess that summary.")
    if t < b0: return ("Eyes open. Light enters the 'Eyes' area, flows to the 'Compass & map' area and on to memory and motor output.\n"
                       "Colour = how active each neuron is. The 'match' bar says how close the brain's guess is to the teacher's summary.")
    if t == b0: return ("LIGHTS OFF. From now on the brain gets no picture at all. It only knows which move it just made.\n"
                        "To keep guessing well it must imagine how the scene changes as it moves: this is path integration, what a real fly does in the dark.")
    if t < b1: return (f"Still dark (step {t - b0 + 1} of {b1 - b0}). Watch the match bars: the real wiring keeps its guess closer to the truth\n"
                       "than the same neurons with scrambled connections. The wiring diagram itself is doing work.")
    if t == b1: return ("LIGHTS ON. The picture is back and both brains snap back to a high match.\n"
                        "The gap that opened in the dark is the evidence that the real connectome is a better prior for imagining the world.")
    return ("Eyes open again. The brain's guess tracks the teacher closely once more.\nEnd of episode next.")

def draw_brain(ax, H_t, label, i):
    ax.cla(); ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(-1.4, 10.0); ax.set_ylim(-1.6, 1.9)
    for (A, _), (B, _) in [(AREAS[0], AREAS[1]), (AREAS[1], AREAS[2]), (AREAS[1], AREAS[3]), (AREAS[2], AREAS[3])]:
        n = flow[(A, B)]; ax.add_patch(FancyArrowPatch(centers[A], centers[B], arrowstyle="-|>", mutation_scale=14 + n / 400, lw=0.8 + n / 1500, color="#6c7a99", alpha=0.7, shrinkA=38, shrinkB=38, connectionstyle="arc3,rad=-0.25" if (A, B) == ("CX", "OUT") else "arc3,rad=0"))
    ax.scatter(xy[:, 0], xy[:, 1], c=H_t, cmap="coolwarm", vmin=-amp[i], vmax=amp[i], s=5, linewidths=0, zorder=3)
    for L, name in AREAS: ax.text(centers[L][0], -1.05, f"{name}\n{(region == L).sum():,} neurons", ha="center", va="top", color=W, fontsize=8.5)
    ax.text(-1.3, 1.6, label, color=Y if i == 0 else "#c77dff", fontsize=12, weight="bold", va="top")

def frame(t):
    fig.clf(); gs = fig.add_gridspec(4, 5, height_ratios=[0.5, 3.0, 2.6, 1.2], width_ratios=[1.1, 1.1, 0.9, 1.3, 1.3], hspace=0.35, wspace=0.3, left=0.03, right=0.98, top=0.97, bottom=0.03)
    blind = b0 <= t < b1
    axT = fig.add_subplot(gs[0, :]); box(axT, "Can a real fly brain learn to imagine the world like a video AI?", 20, W)
    ax = fig.add_subplot(gs[1, :3]); ax.imshow((panos[t] * (0.2 if blind else 1.0)).astype(np.uint8), aspect="auto", interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(("DARK  (brain sees nothing)" if blind else ("What the world looks like right now (360° view)" if "frames" not in raw.files else "What the world looks like right now (street view)")) + f"      step {t + 1} of {T}", color=R_ if blind else W, fontsize=12, loc="left")
    if blind: ax.text(0.5, 0.5, "LIGHTS OFF", color=R_, fontsize=34, weight="bold", ha="center", va="center", transform=ax.transAxes)
    ax = fig.add_subplot(gs[1, 3]); eye = np.zeros_like(raw["obs"][ep, t]) if blind else raw["obs"][ep, t]; ax.imshow(eye, aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("What the fly brain sees\n(32×8 pixels)", color=W, fontsize=11)
    ax = fig.add_subplot(gs[1, 4]); ax.axis("off"); a_prev = ACTION[int(acts[t - 1])] if t > 0 else "—"
    ax.text(0.5, 0.78, "Last move", color="#aaa", fontsize=11, ha="center", transform=ax.transAxes); ax.text(0.5, 0.58, a_prev, color=Y, fontsize=18, weight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.36, "How close is each brain's guess\nto the teacher's summary?", color="#aaa", fontsize=10, ha="center", transform=ax.transAxes)
    for i, (lab, mt) in enumerate(zip(a.labels, match)):
        y = 0.16 - 0.19 * i; ax.add_patch(FancyBboxPatch((0.05, y), 0.9, 0.07, boxstyle="round,pad=0.005", fc="#2a2d3a", ec="none", transform=ax.transAxes, clip_on=False))
        ax.add_patch(FancyBboxPatch((0.05, y), 0.9 * mt[t] / 100, 0.07, boxstyle="round,pad=0.005", fc=Y if i == 0 else "#c77dff", ec="none", transform=ax.transAxes, clip_on=False))
        ax.text(0.05, y + 0.09, f"{lab}: {mt[t]:.0f}%", color=W, fontsize=9.5, transform=ax.transAxes)
    for i in range(len(models)):
        axb = fig.add_subplot(gs[2, 3 * (i % 2) // 2 * 3:(3 if i == 0 else 5)] if False else gs[2, :3] if i == 0 else gs[2, 3:])
        draw_brain(axb, Hs[i][t], a.labels[i], i)
    axc = fig.add_subplot(gs[3, :]); axc.axis("off")
    axc.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.01", fc="#1b1e2a", ec="#333", transform=axc.transAxes))
    axc.text(0.5, 0.5, caption(t), color=W, fontsize=12.5, ha="center", va="center", transform=axc.transAxes, linespacing=1.6)

def card(lines, title):
    fig.clf(); ax = fig.add_subplot(111); ax.axis("off"); ax.set_facecolor(BG)
    ax.text(0.5, 0.86, title, color=W, fontsize=24, weight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.45, "\n\n".join(lines), color=W, fontsize=15, ha="center", va="center", transform=ax.transAxes, linespacing=1.5)

w, h = fig.canvas.get_width_height()
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(a.fps), "-i", "-",
                       "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], stdin=subprocess.PIPE)
def emit(seconds):
    fig.canvas.draw(); buf = np.asarray(fig.canvas.buffer_rgba()).tobytes()
    for _ in range(int(seconds * a.fps)): ff.stdin.write(buf)

card([f"1. A video AI (the teacher, V-JEPA 2.1) watches {a.world} and summarises each moment as numbers.",
      "2. A student brain is built from 6,935 REAL fruit-fly neurons, connected exactly as in the FlyWire connectome.",
      "3. The student only gets a tiny 32×8 picture and must guess the teacher's summary, including when the lights go off.",
      "4. We compare it with the SAME neurons wired at random, to test whether the real wiring diagram matters."],
     "The experiment in one minute"); emit(7)
for t in range(T):
    frame(t); emit(a.hold * (2.2 if t in (0, b0, b1) else 1.0))
    if t in (b0 - 1, b1 - 1, 0): fig.savefig(a.out.replace(".mp4", f"_t{t:02d}.png"), dpi=100, facecolor=BG)
gap = np.mean(match[0][b0:b1]) - np.mean(match[1][b0:b1]) if len(match) > 1 else 0
card([f"In the dark, the real wiring stayed {gap:+.0f} points closer to the teacher than the scrambled wiring (this episode).",
      "Across the whole test set: real wiring beats scrambled and random wiring at every blind horizon.",
      "The whole brain can report heading in the dark to within about 10°, a form of path integration it was never trained for.",
      "Open problems: a plain GRU still predicts better, and using the brain to plan routes does not work yet."],
     "What we found"); emit(8)
ff.stdin.close(); ff.wait()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out, "-vf", "fps=4,scale=960:-1:flags=lanczos", a.out.replace(".mp4", ".gif")])
print("wrote", a.out)
