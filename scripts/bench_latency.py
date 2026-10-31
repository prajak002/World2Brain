"""Table 3: compute cost of every student on identical hardware (#11).

Per model: parameters, synapses (edges), analytic multiply-adds per step, p50 / p95 single-sample action latency
(one recurrent step, B=1, what a closed-loop controller pays), training-step time (B=32, T=24, forward+backward),
peak memory, checkpoint size. Uses the real connectome if present, otherwise the synthetic stand-in with matched sizes.

  PYTHONPATH=. python scripts/bench_latency.py --device cuda --models connectome:1 connectome:2 connectome:3 gru dense
"""
import argparse, io, json, os, time, numpy as np, torch
from wm2cx.student import FlyBrain, GRUBaseline

p = argparse.ArgumentParser()
p.add_argument("--conn", default="data/connectome.npz"); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--models", nargs="+", default=["connectome:1", "connectome:2", "connectome:3", "shuffled:1", "random:1", "gru", "dense"])
p.add_argument("--obs_dim", type=int, default=8 * 32 * 3); p.add_argument("--k", type=int, default=256); p.add_argument("--gru_hidden", type=int, default=1400)
p.add_argument("--n_lat", type=int, default=200); p.add_argument("--n_train", type=int, default=5); p.add_argument("--bs", type=int, default=32); p.add_argument("--T", type=int, default=24)
p.add_argument("--out", default="results/bench/latency")
a = p.parse_args(); dev = a.device

if os.path.exists(a.conn):
    from wm2cx.connectome import load; conn = load(a.conn); src = a.conn
else:
    from wm2cx.synthetic import synthetic_connectome; conn = synthetic_connectome(); src = "synthetic (matched sizes)"
    print("WARNING: data/connectome.npz not found; benchmarking the synthetic stand-in with the same N / E", flush=True)


def sync():
    if dev.startswith("cuda"): torch.cuda.synchronize()


def peak_mem_reset():
    if dev.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
    import psutil; return psutil.Process().memory_info().rss


def peak_mem(rss0):
    if dev.startswith("cuda"): return torch.cuda.max_memory_allocated() / 2**20
    return float("nan")   # no reliable per-model peak on CPU (RSS reuses freed pages); use a CUDA device for this column


def build(spec):
    if spec == "gru": return GRUBaseline(a.obs_dim, a.k, a.gru_hidden), "gru", 1
    w, _, K = spec.partition(":"); K = int(K or 1)
    return FlyBrain(conn, a.obs_dim, a.k, wiring=w, n_brains=K), w, K


rows = []
for spec in a.models:
    try:
        rss0 = peak_mem_reset(); m, w, K = build(spec); m.to(dev)
        buf = io.BytesIO(); torch.save(m.state_dict(), buf); ckpt_mb = buf.tell() / 2**20
        n_edges = m.N * m.N if getattr(m, "dense", False) else getattr(m, "n_edges", 3 * m.N * m.N)   # GRU: recurrent weights
        # action latency: one step, B=1 (edge weights precomputed once per rollout, as a controller would)
        m.eval(); lat = []
        with torch.no_grad():
            h = m.init_state(1, dev); obs = torch.randn(1, a.obs_dim, device=dev); act = torch.zeros(1, dtype=torch.long, device=dev); vis = torch.ones(1, device=dev)
            we = m.edge_weights()
            for i in range(20): h = m.step(h, obs, act, vis, we)
            sync()
            for i in range(a.n_lat):
                t0 = time.perf_counter(); h = m.step(h, obs, act, vis, we); sync(); lat.append((time.perf_counter() - t0) * 1e3)
        # training step: B x T forward + backward
        m.train(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); tt = []
        O = torch.randn(a.bs, a.T, a.obs_dim, device=dev); A = torch.randint(0, 5, (a.bs, a.T), device=dev); V = torch.ones(a.bs, a.T, device=dev); Z = torch.randn(a.bs, a.T, a.k, device=dev)
        for i in range(a.n_train + 1):
            sync(); t0 = time.perf_counter()
            zh, H = m(O, A, V); loss = ((zh - Z) ** 2).mean() + 1e-4 * (H ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sync()
            if i: tt.append(time.perf_counter() - t0)
        mem = peak_mem(rss0)
        r = dict(model=spec, wiring=w, K=K, neurons=m.N, params=m.n_params(), synapses=n_edges, mflop_step=m.flops_per_step(a.obs_dim) / 1e6,
                 lat_p50_ms=float(np.percentile(lat, 50)), lat_p95_ms=float(np.percentile(lat, 95)), train_step_s=float(np.mean(tt)),
                 peak_mem_mb=float(mem), ckpt_mb=float(ckpt_mb), status="ok")
        del m, opt, O, A, V, Z, zh, H
        if dev.startswith("cuda"): torch.cuda.empty_cache()
    except (RuntimeError, MemoryError) as e:
        r = dict(model=spec, status=f"failed: {str(e).splitlines()[0][:80]}")
    rows.append(r); print(json.dumps(r), flush=True)

hw = torch.cuda.get_device_name(0) if dev.startswith("cuda") else f"cpu x{os.cpu_count()}"
L = [f"# Table 3: compute cost (device: {dev}, {hw}; connectome: {src}; torch {torch.__version__})", "",
     f"Action latency = one recurrent step at B=1 (p50 / p95 over {a.n_lat} steps). Train step = forward+backward, B={a.bs}, T={a.T}. "
     "MFLOP = analytic multiply-adds per sample per step. Peak memory = CUDA max allocated (GPU) or RSS growth (CPU).", "",
     "| model | neurons | params | synapses | MFLOP/step | action p50 ms | p95 ms | train step s | peak MB | ckpt MB |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if r["status"] != "ok": L.append(f"| {r['model']} | {r['status']} |||||||||"); continue
    L.append(f"| {r['model']} | {r['neurons']} | {r['params']/1e6:.2f}M | {r['synapses']:,} | {r['mflop_step']:.1f} | {r['lat_p50_ms']:.2f} | {r['lat_p95_ms']:.2f} | {r['train_step_s']:.2f} | {'n/a' if np.isnan(r['peak_mem_mb']) else f'{r['peak_mem_mb']:.0f}'} | {r['ckpt_mb']:.0f} |")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out + ".md", "w").write("\n".join(L) + "\n"); json.dump(dict(device=dev, hardware=hw, connectome=src, rows=rows), open(a.out + ".json", "w"), indent=1)
print("\n".join(L))
