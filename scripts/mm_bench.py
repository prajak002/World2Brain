"""B0 throughput for the Memory Maze batch (docs/RUNBOOK_GPU.md): teacher clips/s at a given crop and precision, and
student training steps/s (C = full-brain connectome, G = parameter-matched GRU) on TBTT windows of 48 steps.
Numbers go into PREREG Amendment 1; nothing here touches the eval split beyond reading frames for timing."""
import argparse, glob, json, time, numpy as np, torch
from wm2cx.teacher import VJEPA21Encoder
from wm2cx.student import FlyBrain, GRUBaseline, gru_hidden_for
from wm2cx.connectome import load as load_conn

p = argparse.ArgumentParser()
p.add_argument("--frames", default="/root/mm9/eval"); p.add_argument("--crops", type=int, nargs="+", default=[256, 128])
p.add_argument("--bs", type=int, nargs="+", default=[32, 64]); p.add_argument("--student_bs", type=int, default=32)
p.add_argument("--conn", default="data/connectome_full.npz"); p.add_argument("--out", default="results/memmaze/b0_bench.json")
a = p.parse_args(); dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
img = np.load(sorted(glob.glob(a.frames + "/*.npz"))[0])["image"][:64]                     # [64,64,64,3]
res = {"gpu": torch.cuda.get_device_name(0), "teacher": [], "student": []}
for crop in a.crops:
    T = VJEPA21Encoder("vitb", crop=crop, frames=16).to(dev)
    for bs in a.bs:
        for prec in ("fp32", "bf16"):
            x = torch.from_numpy(img[:8]).to(dev).permute(0, 3, 1, 2).float()
            x = torch.nn.functional.interpolate(x, size=(crop, crop), mode="bilinear", align_corners=False).round().clamp(0, 255).byte().permute(0, 2, 3, 1)
            clip = x.repeat_interleave(2, 0)[None].repeat(bs, 1, 1, 1, 1).contiguous()        # [bs,16,crop,crop,3]
            ctx = torch.autocast("cuda", torch.bfloat16) if prec == "bf16" else torch.autocast("cuda", enabled=False)
            try:
                with ctx:
                    for _ in range(2): T.per_step_latents(clip)
                    torch.cuda.synchronize(); t0 = time.time(); n = 5
                    for _ in range(n): z = T.per_step_latents(clip)
                    torch.cuda.synchronize(); cps = n * bs / (time.time() - t0)
                r = dict(crop=crop, bs=bs, prec=prec, clips_per_s=round(cps, 1), mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
            except torch.cuda.OutOfMemoryError: r = dict(crop=crop, bs=bs, prec=prec, oom=True)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); print(r, flush=True); res["teacher"].append(r)
    del T; torch.cuda.empty_cache()
conn = load_conn(a.conn); B, W = a.student_bs, 48
fly = FlyBrain(conn, 768, 256, act_dim=6).to(dev)
models = {"C": fly, "G": GRUBaseline(768, 256, gru_hidden_for(fly.n_params(), 768, 256, act_dim=6), act_dim=6).to(dev)}
for name, m in models.items():
    opt = torch.optim.AdamW(m.parameters(), 1e-3); obs = torch.rand(B, W, 768, device=dev); act = torch.randint(0, 6, (B, W), device=dev)
    vis = torch.ones(B, W, device=dev); z = torch.randn(B, W, 256, device=dev); torch.cuda.reset_peak_memory_stats()
    for i in range(6):
        if i == 2: torch.cuda.synchronize(); t0 = time.time()
        zh, H = m(obs, act, vis); loss = ((zh - z) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); dt = (time.time() - t0) / 4
    r = dict(model=name, params=m.n_params(), N=m.N, bs=B, window=W, s_per_update=round(dt, 3), traj_steps_per_s=round(B * W / dt), mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
    print(r, flush=True); res["student"].append(r)
import os; os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
