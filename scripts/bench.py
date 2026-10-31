"""Issue #11: per-step latency (p50 / p95), analytic FLOPs and peak memory for every wiring on identical hardware -> results/bench.json + table."""
import argparse, json, sys, time, resource, numpy as np, torch
from wm2cx.student import FlyBrain, GRUBaseline, gru_hidden_for, WIRINGS
from wm2cx.connectome import load as load_conn

p = argparse.ArgumentParser(); p.add_argument("--device", default="cpu"); p.add_argument("--batch", type=int, default=1); p.add_argument("--reps", type=int, default=200)
p.add_argument("--n_brains", nargs="+", type=int, default=[1, 2, 3]); a = p.parse_args()
conn = load_conn(); obs_dim, k = 8 * 32 * 3, 256; dev = a.device; rows = []

def flops(m):
    pb = m.param_breakdown(); return 2 * (pb["edges"] * 3 if m.wiring == "gru" else pb["edges"]) + 2 * (pb["input"] + pb["output"])  # MACs*2 per step, batch 1

def bench(m, name):
    m.to(dev).eval(); h = m.init_state(a.batch, dev); obs = torch.rand(a.batch, obs_dim, device=dev); act = torch.zeros(a.batch, dtype=torch.long, device=dev); vis = torch.ones(a.batch, device=dev)
    w = m.weights(); ts = []
    with torch.no_grad():
        for i in range(a.reps + 20):
            t = time.perf_counter(); h = m.step(h, obs, act, vis, w)
            if dev == "mps": torch.mps.synchronize()
            if i >= 20: ts.append((time.perf_counter() - t) * 1e3)
    mem = torch.mps.driver_allocated_memory() / 2**20 if dev == "mps" else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == 'darwin' else 2**10)
    r = dict(model=name, neurons=m.N, params=m.n_params(), edges=m.n_edges, p50_ms=float(np.percentile(ts, 50)), p95_ms=float(np.percentile(ts, 95)), mflops_per_step=flops(m) / 1e6, peak_mem_mb=float(mem))
    rows.append(r); print(r, flush=True); m.cpu()

for K in a.n_brains:
    for w in WIRINGS:
        if w == "dense" and K > 1: continue
        bench(FlyBrain(conn, obs_dim, k, wiring=w, n_brains=K), f"{w} K={K}")
    fb = FlyBrain(conn, obs_dim, k, n_brains=K); hdim = gru_hidden_for(fb.n_params(), obs_dim, k); del fb
    bench(GRUBaseline(obs_dim, k, hdim), f"gru(h={hdim}) K={K}-matched")
json.dump(dict(device=dev, batch=a.batch, torch=torch.__version__, rows=rows), open(f"results/bench_{dev}.json", "w"), indent=1)
print("| model | neurons | params | edges | p50 ms | p95 ms | MFLOPs/step | process peak mem MB (cumulative over rows) |"); print("|---|---|---|---|---|---|---|---|")
for r in rows: print(f"| {r['model']} | {r['neurons']} | {r['params']/1e6:.2f}M | {r['edges']} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['mflops_per_step']:.1f} | {r['peak_mem_mb']:.0f} |")
