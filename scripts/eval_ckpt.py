"""Evaluate a trained checkpoint on another episode set, zero-shot or after few-shot fine-tuning (E7: real video).

Reuses the checkpoint's own PCA (fit on its training set) so the numbers are in the same whitened space as the training
tables, writes the same JSON keys as `scripts/train.py` (so `summarize.py` picks the run up under its tag), and writes the
two trivial predictors for the same episodes to results/baselines/ (same keys as `scripts/eval_trivial.py`).

  # zero-shot transfer of an arena-trained student to real handheld video
  python scripts/eval_ckpt.py --run connectome_K1_s0 --test data/realvid_all.npz --tag _realvid --device cuda
  # leave-one-clip-out few-shot fine-tuning: for each clip, fine-tune on the other clips' episodes and test on it
  python scripts/eval_ckpt.py --run connectome_K1_s0 --test data/realvid_all.npz --tag _realvid_ft --finetune_epochs 10 --device cuda
"""
import argparse, json, os, copy, time, numpy as np, torch
from wm2cx.ckpt import build_model, latents_from
from wm2cx.data import load_split, blackout, parse_windows, window_errors
from wm2cx.connectome import load as load_conn

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="checkpoint name under results/runs (without .pt)")
p.add_argument("--test", nargs="+", required=True); p.add_argument("--tag", default="_realvid")
p.add_argument("--conn", default=None, help="connectome npz (default: the one recorded in the checkpoint)")
p.add_argument("--device", default="cuda"); p.add_argument("--eval_windows", nargs="+", default=["8:16", "14:22"])
p.add_argument("--finetune_epochs", type=int, default=0, help="> 0: leave-one-clip-out fine-tuning on the test file's other clips")
p.add_argument("--finetune_lr", type=float, default=5e-4); p.add_argument("--H", type=int, default=8); p.add_argument("--seed", type=int, default=0)
p.add_argument("--baselines", type=int, default=1, help="also write const_mean / copy_last for these episodes")
a = p.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = a.device

ck = torch.load(f"results/runs/{a.run}.pt", map_location="cpu", weights_only=False); args = ck["args"]; lat = latents_from(ck)
conn = load_conn(a.conn or args["conn"])
te, _ = load_split(a.test, lat, keep_raw=True)
clip = torch.from_numpy(np.concatenate([np.load(f)["clip"] if "clip" in np.load(f) else np.full(len(np.load(f)["obs"]), i) for i, f in enumerate(a.test)]).astype(np.int64))
E, T = te["obs"].shape[:2]; windows = parse_windows(a.eval_windows, T); W0 = f"{windows[0][0]}:{windows[0][1]}"
k = te["z"].shape[-1]


def build():
    m = build_model(args, conn, te["obs"].shape[-1], ck["state"]); m.load_state_dict(ck["state"]); return m.to(dev)


def evaluate(model, d, idx):
    """Per-episode, per-horizon errors on episodes idx of d for every window -> {window: {mse [n,H], cos, mse_teacher}}."""
    model.eval(); out = {}
    with torch.no_grad():
        for t0, t1 in windows:
            acc = dict(mse=[], cos=[], mse_teacher=[])
            for i in range(0, len(idx), 64):
                j = idx[i:i + 64]
                obs, act, z = d["obs"][j].to(dev), d["act"][j].to(dev), d["z"][j].to(dev)
                vis = torch.ones(obs.shape[:2], device=dev); vis[:, t0:t1] = 0
                zh, _ = model(obs, act, vis); e = window_errors(zh, z, t0, t1, lat, d["z_raw"][j].to(dev))
                for k_, v in e.items(): acc[k_].append(v.cpu())
            out[f"{t0}:{t1}"] = {k_: torch.cat(v) for k_, v in acc.items()}
    return out


def finetune(model, d, idx, epochs):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=a.finetune_lr, weight_decay=1e-4)
    for ep in range(epochs):
        perm = idx[torch.randperm(len(idx))]
        for i in range(0, len(perm), 16):
            j = perm[i:i + 16]; obs, act, z = d["obs"][j].to(dev), d["act"][j].to(dev), d["z"][j].to(dev)
            vis = blackout(len(j), T, a.H, rng, dev); zh, Hs = model(obs, act, vis)
            loss = ((((zh - z) ** 2).mean(-1)) * (1.0 + 2.0 * (1 - vis))).mean() + 1e-4 * (Hs ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


t0_ = time.time(); all_idx = torch.arange(E)
if a.finetune_epochs > 0:                                                  # leave-one-clip-out; errors are collected on held-out clips only
    parts = {w: dict(mse=[], cos=[], mse_teacher=[]) for w in [f"{t0}:{t1}" for t0, t1 in windows]}; order = []
    for c in torch.unique(clip).tolist():
        tr_idx, te_idx = all_idx[clip != c], all_idx[clip == c]
        if len(tr_idx) == 0: continue
        m = finetune(build(), te, tr_idx, a.finetune_epochs); ev = evaluate(m, te, te_idx); order.append(te_idx)
        for w in parts:
            for k_ in parts[w]: parts[w][k_].append(ev[w][k_])
        print(f"fold clip {c}: fine-tuned on {len(tr_idx)} eps, tested on {len(te_idx)}: blind MSE mean {ev[W0]['mse'].mean():.4f}", flush=True)
    order = torch.cat(order); inv = torch.argsort(order)
    ev = {w: {k_: torch.cat(v)[inv] for k_, v in parts[w].items()} for w in parts}; n_train = int(E - len(te_idx))
else:
    model = build(); ev = evaluate(model, te, all_idx); n_train = 0


def pack(ev):
    return {w: dict(mse=v["mse"].mean(0).tolist(), cos=v["cos"].mean(0).tolist(), mse_teacher=v["mse_teacher"].mean(0).tolist(),
                    per_ep_mse=v["mse"].numpy().round(5).tolist()) for w, v in ev.items()}


name = f"{a.run}{a.tag}"; os.makedirs("results/runs", exist_ok=True); pe = pack(ev)
model_ = build()
json.dump(dict(args=dict(args, source_run=a.run, test=a.test, tag=args.get("tag", "") + a.tag, finetune_epochs=a.finetune_epochs, eval_windows=a.eval_windows),
               neurons=model_.N, params=model_.n_params(), test_mse=pe[W0]["mse"], test_cos=pe[W0]["cos"], test_mse_teacher=pe[W0]["mse_teacher"],
               test_mse_per_ep=pe[W0]["per_ep_mse"], test_windows=pe, n_test_episodes=int(E), n_train_episodes=n_train, n_clips=int(clip.max()) + 1,
               per_episode_clip=clip.tolist(), log=[], pca_explained=lat.explained), open(f"results/runs/{name}.json", "w"), indent=1)
m = pe[W0]["mse"]; print(f"{name}: blind MSE h1 {m[0]:.4f} h4 {m[3]:.4f} h8 {m[-1]:.4f} mean {np.mean(m):.4f} cos h1 {pe[W0]['cos'][0]:.3f} "
                         f"teacher-space {np.mean(pe[W0]['mse_teacher']):.4f} on {E} episodes / {int(clip.max()) + 1} clips, {time.time()-t0_:.0f}s")

if a.baselines:                                                            # same episodes, same PCA, same windows (#6)
    os.makedirs("results/baselines", exist_ok=True); z, z_raw = te["z"], te["z_raw"]
    var_teacher = float(z_raw.float().reshape(-1, z_raw.shape[-1]).var(0).mean())
    for bname in ("const_mean", "copy_last"):
        evb = {}
        for t0, t1 in windows:
            zh = torch.zeros_like(z) if bname == "const_mean" else z.clone()   # train mean is 0 in the checkpoint's whitened PCA space
            if bname == "copy_last": zh[:, t0:t1] = z[:, t0 - 1:t0]
            e = window_errors(zh, z, t0, t1, lat, z_raw)
            evb[f"{t0}:{t1}"] = dict(mse=e["mse"].mean(0).tolist(), cos=e["cos"].mean(0).tolist(), mse_teacher=e["mse_teacher"].mean(0).tolist(), per_ep_mse=e["mse"].numpy().round(5).tolist())
        fn = f"results/baselines/{bname}_K0_s0{a.tag}.json"
        json.dump(dict(args=dict(wiring=bname, n_brains=0, seed=0, tag=a.tag, k=k, test=a.test, eval_windows=a.eval_windows), neurons=0, params=0,
                       test_mse=evb[W0]["mse"], test_cos=evb[W0]["cos"], test_mse_teacher=evb[W0]["mse_teacher"], test_mse_per_ep=evb[W0]["per_ep_mse"],
                       test_windows=evb, n_test_episodes=int(E), pca_explained=lat.explained, teacher_space_variance=var_teacher), open(fn, "w"), indent=1)
        print(f"{bname:11s}: PCA-space blind MSE mean {np.mean(evb[W0]['mse']):.4f} | teacher-space {np.mean(evb[W0]['mse_teacher']):.4f} -> {fn}")
