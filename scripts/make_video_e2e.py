"""End-to-end demo video: high-quality NYC digital-twin frames (left) next to the trained connectome student's
activity drawn on real FlyWire soma positions (right), with teacher latent vs student reconstruction and a match
gauge below. Input: results/hq/nyc_ep4_t*.png (768 px renders from Valdrada) and results/brain3d.json (export
of neuron positions / regions and per-step activity, int8 quantised to ±127)."""
import argparse, json, os, subprocess, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--data", default="results/brain3d.json"); p.add_argument("--world", default="nyc")
p.add_argument("--frames", default="results/hq/nyc_ep4_t{t:02d}.png"); p.add_argument("--out", default="results/video/wm2cx_nyc_e2e.mp4")
p.add_argument("--fps", type=int, default=12); p.add_argument("--hold", type=float, default=1.5); p.add_argument("--blind", default="8:16")
a = p.parse_args(); b0, b1 = map(int, a.blind.split(":")); os.makedirs(os.path.dirname(a.out), exist_ok=True)
D = json.load(open(a.data)); Wd = D[a.world]; T = len(Wd["act"])
pos = np.array(D["pos"], np.float32); region = np.array(D["region"]); act = np.array(Wd["act"], np.float32) / 127.0
z, zh, match, actions, heading = (np.array(Wd[k]) for k in ("z", "zh", "match", "actions", "heading"))
ACTION = {0: "stay", 1: "step forward", 2: "turn left", 3: "turn right", 4: "step back"}
REG = {"VIS": ("#4ea8ff", "Eyes (visual)"), "CX": ("#ffb347", "Compass & map (central complex)"),
       "MB": ("#3ddc97", "Memory (mushroom body)"), "OUT": ("#ff5d8f", "Motor output (descending)")}
BG, INK, MUTED = "#0b0d12", "#f2f2f2", "#9aa3b2"
order = np.argsort(pos[:, 2])  # fixed draw order, far first
frames = [np.asarray(Image.open(a.frames.format(t=t)).convert("RGB")) for t in range(T)]

def project(xyz, ang):
    c, s = np.cos(ang), np.sin(ang); x = xyz[:, 0] * c - xyz[:, 2] * s; zc = xyz[:, 0] * s + xyz[:, 2] * c
    return x, xyz[:, 1], zc

fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
axW = fig.add_axes([0.03, 0.25, 0.42, 0.63]); axB = fig.add_axes([0.47, 0.25, 0.51, 0.63])
axL = fig.add_axes([0.03, 0.07, 0.55, 0.14]); axG = fig.add_axes([0.63, 0.07, 0.34, 0.14])
title = fig.text(0.03, 0.955, "", color=INK, fontsize=22, weight="bold", va="center")
sub = fig.text(0.03, 0.925, "", color=MUTED, fontsize=13, va="center")
cap = fig.text(0.5, 0.025, "", color=INK, fontsize=14, ha="center", va="center")

def caption(t):
    if t == 0: return "V-JEPA 2.1 (the teacher) watches this real-NYC view and writes a 256-number summary. The fly brain must reproduce it from a 64×64 glimpse and its own motion."
    if t < b0: return "Eyes open: visual neurons (blue) light up, the central complex (orange) integrates heading, the student tracks the teacher's summary."
    if t == b0: return "Eyes closed for 8 steps. The student still knows which way it is turning and must imagine what the teacher would say."
    if t < b1: return "Blind: no new picture reaches the eyes. Activity now lives in the compass/memory areas; the match slowly decays."
    if t == b1: return "Eyes open again: the match recovers as fresh input arrives."
    return "Same brain, real FlyWire wiring, 91,663 synapses with their true signs. Only the synapse strengths were learned."

def draw(t, ang, blind_fade):
    for ax in (axW, axB, axL, axG): ax.clear(); ax.set_facecolor(BG); ax.axis("off")
    blind = b0 <= t < b1
    img = frames[t].astype(np.float32); f = 1 - 0.78 * blind_fade; axW.imshow((img * f).astype(np.uint8)); axW.set_xlim(0, img.shape[1]); axW.set_ylim(img.shape[0], 0)
    axW.text(0.02, 0.97, f"NYC digital twin · Times Square · step {t + 1}/{T}", color=INK, fontsize=12, transform=axW.transAxes, va="top",
             bbox=dict(facecolor="#000000aa", edgecolor="none", pad=6))
    axW.text(0.02, 0.04, f"action: {ACTION[int(actions[t])]}   heading {heading[t]:.0f}°", color=INK, fontsize=12, transform=axW.transAxes,
             bbox=dict(facecolor="#000000aa", edgecolor="none", pad=6))
    if blind_fade > 0.5: axW.text(0.5, 0.5, "EYES CLOSED\nstudent imagines", color="#ffd166", fontsize=26, weight="bold", ha="center", va="center", transform=axW.transAxes)
    # brain
    x, y, zc = project(pos, ang); amp = np.abs(act[t]); amp = np.clip(amp / 0.6, 0, 1)
    cols = np.array([matplotlib.colors.to_rgb(REG[r][0]) for r in region]); rgba = np.concatenate([cols, (0.12 + 0.88 * amp)[:, None]], 1)
    o = np.argsort(zc); size = 3 + 26 * amp ** 1.5
    axB.scatter(x[o], y[o], s=size[o], c=rgba[o], linewidths=0, rasterized=True)
    axB.set_xlim(-330, 330); axB.set_ylim(-230, 230); axB.set_aspect("equal")
    axB.text(0.5, 0.98, "Fly brain student · 6,935 FlyWire neurons at their real positions · brightness = activity", color=INK, fontsize=12, ha="center", va="top", transform=axB.transAxes)
    for i, (k, (c, lab)) in enumerate(REG.items()):
        m = (region == k); r = float(np.mean(np.abs(act[t][m])))
        axB.scatter([0.04], [0.14 - 0.045 * i], s=70, c=[c], transform=axB.transAxes, clip_on=False)
        axB.text(0.07, 0.14 - 0.045 * i, f"{lab}  ·  mean |activity| {r:.2f}", color=INK, fontsize=11, va="center", transform=axB.transAxes)
    # latents
    n = z.shape[1]; xs = np.arange(n)
    axL.bar(xs - 0.2, z[t], 0.4, color="#4ea8ff", label="teacher (V-JEPA 2.1)"); axL.bar(xs + 0.2, zh[t], 0.4, color="#ffb347", label="student reconstruction")
    axL.set_xlim(-1, n); lim = max(1.0, np.abs(z).max() * 1.05); axL.set_ylim(-lim, lim); axL.axhline(0, color=MUTED, lw=0.6)
    axL.text(0, 1.08, f"Teacher summary vs student guess (first {n} of 256 numbers)", color=INK, fontsize=12, transform=axL.transAxes, va="bottom")
    axL.legend(loc="upper right", frameon=False, labelcolor=INK, fontsize=10, ncol=2, bbox_to_anchor=(1, 1.3))
    # gauge
    m = float(match[t]); axG.set_xlim(0, 100); axG.set_ylim(0, 1)
    axG.add_patch(Rectangle((0, 0.35), 100, 0.3, color="#1f2430")); axG.add_patch(Rectangle((0, 0.35), m, 0.3, color="#3ddc97" if m > 50 else ("#ffd166" if m > 25 else "#ff5d8f")))
    axG.text(0, 0.95, "Match between student and teacher", color=INK, fontsize=12, va="top"); axG.text(m + 1, 0.5, f"{m:.0f} %", color=INK, fontsize=16, weight="bold", va="center")
    axG.plot(np.linspace(0, 100, T)[: t + 1], 0.08 + 0.18 * match[: t + 1] / 100, color=MUTED, lw=1.2); axG.text(0, 0.0, "history", color=MUTED, fontsize=9)
    title.set_text("World model → fly connectome: one episode, end to end"); sub.set_text("Teacher: V-JEPA 2.1 (ViT-B)  ·  World: Valdrada NYC digital twin, 768 px  ·  Student: FlyWire v783 wiring, leaky RNN")
    cap.set_text(caption(t))

nper = max(1, int(round(a.hold * a.fps))); tmp = a.out + ".frames"; os.makedirs(tmp, exist_ok=True); k = 0; ang = 0.0
for t in range(T):
    for j in range(nper):
        fade = 1.0 if b0 <= t < b1 else 0.0
        if t == b0: fade = min(1, j / (nper * 0.5))
        draw(t, ang, fade); ang += 0.004
        fig.savefig(f"{tmp}/f{k:05d}.png", facecolor=BG); k += 1
    if t in (0, b0 - 1, b1 - 1, T - 1): fig.savefig(a.out.replace(".mp4", f"_t{t:02d}.png"), facecolor=BG, dpi=100)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/f%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out, "-vf", "fps=6,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer", a.out.replace(".mp4", ".gif")], check=True)
for f in os.listdir(tmp): os.remove(f"{tmp}/{f}")
os.rmdir(tmp); print("wrote", a.out, "frames", k)
