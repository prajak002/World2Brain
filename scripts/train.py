import argparse, json, time, numpy as np, torch, os
from wm2cx.aux_losses import dead_reckoning_loss, blackout_contrastive_loss
from wm2cx.student import FlyBrain, FlyGNN, GRUBaseline, LSTMBaseline, gru_hidden_for, lstm_hidden_for, WIRINGS
from wm2cx.transformer import TransformerWM, transformer_layers_for
from wm2cx.data import load_split, blackout, parse_windows, window_errors
from wm2cx.connectome import load as load_conn

p = argparse.ArgumentParser()
p.add_argument("--wiring", default="connectome", choices=WIRINGS + ("gru", "lstm", "transformer", "gnn")); p.add_argument("--n_brains", type=int, default=1)
p.add_argument("--d_model", type=int, default=256, help="transformer baseline width")
p.add_argument("--gnn_graph", default="connectome", choices=["connectome", "block_shuffled", "shuffled", "random"], help="#23: graph the GNN student runs on (--wiring gnn)")
p.add_argument("--gnn_d", type=int, default=8, help="#23: state channels per neuron"); p.add_argument("--gnn_group", default="region", choices=["cell_type", "region"], help="#23: sharing unit for phi / update / readout")
p.add_argument("--gnn_update", default="gru", choices=["gru", "leaky"]); p.add_argument("--gnn_phi", default="mlp", choices=["mlp", "identity"])
p.add_argument("--n_layers", type=int, default=0, help="transformer baseline depth; 0 = deepest model at --d_model with <= the FlyBrain parameter count of this config")
p.add_argument("--in_mode", default="linear", choices=["linear", "attn"], help="attn: cross-attention input encoder instead of the linear W_in (GRU / transformer budgets follow)")
p.add_argument("--hybrid", type=int, default=0, help="connectome recurrence plus one shared attention layer over cell-type-pooled activity per step (ablation)")
p.add_argument("--epochs", type=int, default=30); p.add_argument("--bs", type=int, default=32); p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--H", type=int, default=8); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="mps")
p.add_argument("--k", type=int, default=256)
p.add_argument("--gru_hidden", type=int, default=0, help="0 = largest hidden size with <= the FlyBrain parameter count of this config (#4)")
p.add_argument("--vis_flag", type=int, default=1, help="feed the blackout flag to every unit of both FlyBrain and GRU (#4)")
p.add_argument("--in_rank", type=int, default=0, help="low-rank W_in bottleneck (0 = full); shrinks the shared input layer (#3)")
p.add_argument("--learn_alpha", type=int, default=0, help="neuron-model ablation: per-neuron learnable time constant (wiring fixed)")
p.add_argument("--gate", type=int, default=0, help="neuron-model ablation: GRU-style update gate on a second weight per connectome edge (wiring fixed)")
p.add_argument("--val", type=float, default=0.0, help="fraction of train episodes held out as a validation split for hyper-parameter selection (never test)")
p.add_argument("--train", nargs="+", default=["data/nav_train.npz", "data/arrange_train.npz"])
p.add_argument("--test", nargs="+", default=["data/nav_test.npz", "data/arrange_test.npz"])
p.add_argument("--conn", default="data/connectome.npz"); p.add_argument("--tag", default="")
p.add_argument("--eval_windows", nargs="+", default=["8:16", "14:22"], help="blind windows t0:t1 evaluated on test; the first is primary (#8)")
p.add_argument("--lambda_act", type=float, default=0.0, help="weight of the optional policy-imitation loss (predict the next action from h); 0 = prediction-only (#12)")
p.add_argument("--lambda_dr", type=float, default=0.0, help="#21: weight of the dead-reckoning loss (CX neurons -> egocentric displacement during blind steps); 0 = off")
p.add_argument("--lambda_con", type=float, default=0.0, help="#21: weight of the across-blackout contrastive loss; 0 = off")
p.add_argument("--n_train", type=int, default=0, help="data-efficiency axis: train on a random subset of this many episodes (seeded, stratified over the train files); 0 = all. PCA stays fit on the full train set so targets are comparable across n_train.")
a = p.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = a.device
tr, lat = load_split(a.train, k=a.k); te, _ = load_split(a.test, lat, keep_raw=True)
if a.n_train and a.n_train < len(tr["obs"]):
    task = tr["task"].numpy(); keep = []
    for i in np.unique(task):                                                        # same fraction from every train file (NAV / ARRANGE)
        idx = np.where(task == i)[0]; n_i = max(1, int(round(a.n_train * len(idx) / len(task)))); keep.append(rng.choice(idx, n_i, replace=False))
    keep = torch.from_numpy(np.sort(np.concatenate(keep))); tr = {k: v[keep] for k, v in tr.items()}
va = None
if a.val > 0:  # validation split carved from the training episodes (fixed by seed), never from test
    E0 = len(tr["obs"]); perm = np.random.default_rng(1000 + a.seed).permutation(E0); nv = int(a.val * E0)
    vi, ti = torch.from_numpy(perm[:nv]), torch.from_numpy(perm[nv:])
    va = {k: v[vi] for k, v in tr.items()}; tr = {k: v[ti] for k, v in tr.items()}
    va["z_raw"] = te["z_raw"][:0]  # validation is scored in PCA space only
E, T = tr["obs"].shape[:2]; windows = parse_windows(a.eval_windows, T); obs_dim = tr["obs"].shape[-1]
print(f"train eps {E} val eps {0 if va is None else len(va['obs'])} test eps {len(te['obs'])} PCA{a.k} explained {lat.explained:.3f}")
conn = load_conn(a.conn)
fly = FlyBrain(conn, obs_dim, a.k, wiring=a.wiring if a.wiring not in ("gru", "lstm", "transformer", "gnn") else "connectome", n_brains=a.n_brains, seed=a.seed, vis_flag=bool(a.vis_flag), in_rank=a.in_rank,
               learn_alpha=bool(a.learn_alpha), gate=bool(a.gate), in_mode=a.in_mode, hybrid=bool(a.hybrid))
if a.wiring == "gru":
    if a.gru_hidden <= 0: a.gru_hidden = gru_hidden_for(fly.n_params(), obs_dim, a.k, vis_flag=bool(a.vis_flag))
    model = GRUBaseline(obs_dim, a.k, a.gru_hidden, vis_flag=bool(a.vis_flag)); del fly
elif a.wiring == "lstm":   # #24: unidirectional LSTM at the same parameter budget; --gru_hidden holds its hidden size
    if a.gru_hidden <= 0: a.gru_hidden = lstm_hidden_for(fly.n_params(), obs_dim, a.k, vis_flag=bool(a.vis_flag))
    model = LSTMBaseline(obs_dim, a.k, a.gru_hidden, vis_flag=bool(a.vis_flag)); del fly
elif a.wiring == "gnn":   # #23: vector state on the same edge list; must stay within +-5 % of the FlyBrain parameter count (AGENTS.md rule 3)
    assert a.n_brains == 1
    model = FlyGNN(conn, obs_dim, a.k, wiring=a.gnn_graph, d=a.gnn_d, group=a.gnn_group, update=a.gnn_update, phi=a.gnn_phi, seed=a.seed, vis_flag=bool(a.vis_flag))
    ratio = model.n_params() / fly.n_params(); print(f"gnn params / FlyBrain params = {ratio:.4f}"); assert abs(ratio - 1) <= 0.05, ratio; del fly
elif a.wiring == "transformer":
    if a.n_layers <= 0: a.n_layers = transformer_layers_for(fly.n_params(), obs_dim, a.k, a.d_model, vis_flag=bool(a.vis_flag))
    model = TransformerWM(obs_dim, a.k, a.d_model, a.n_layers, vis_flag=bool(a.vis_flag)); del fly
else:
    model = fly
model.to(dev); pb = model.param_breakdown(); print("neurons", model.N, "params", pb)
act_head = torch.nn.Linear(model.N, model.act_dim).to(dev) if a.lambda_act > 0 else None
dr_head = con_head = None
if a.lambda_dr > 0:   # #21: read the central complex only (region labels are kept by every rewiring control); free baselines read every unit
    is_graph = a.wiring not in ("gru", "lstm", "transformer")
    reg = np.repeat(conn["region"], a.gnn_d) if a.wiring == "gnn" else np.tile(conn["region"], a.n_brains)   # gnn state is neuron-major
    dr_idx = torch.from_numpy(np.nonzero(reg == "CX")[0] if is_graph else np.arange(model.N)).to(dev)
    dr_head = torch.nn.Linear(len(dr_idx), 2).to(dev)
if a.lambda_con > 0: con_head = torch.nn.Linear(model.N, a.k).to(dev)
params = list(model.parameters()) + [q for hd in (act_head, dr_head, con_head) if hd is not None for q in hd.parameters()]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)


def run_eval(d):
    """Blind-window prediction error on test episodes, for every window in --eval_windows.
    Returns {"t0:t1": {mse: [H], cos: [H], mse_teacher: [H], per_ep_mse: [E,H]}} (mse in whitened PCA space,
    mse_teacher in the original teacher space). Per-episode errors are kept for paired bootstrap CIs (#8)."""
    model.eval(); acc = {f"{t0}:{t1}": dict(mse=[], cos=[], mse_teacher=[]) for t0, t1 in windows}
    with torch.no_grad():
        for t0, t1 in windows:
            for i in range(0, len(d["obs"]), 64):
                obs, act, z = d["obs"][i:i + 64].to(dev), d["act"][i:i + 64].to(dev), d["z"][i:i + 64].to(dev)
                vis = torch.ones(obs.shape[:2], device=dev); vis[:, t0:t1] = 0
                zh, _ = model(obs, act, vis)
                e = window_errors(zh, z, t0, t1, lat, d["z_raw"][i:i + 64].to(dev) if len(d["z_raw"]) else None)
                for k_, v in e.items(): acc[f"{t0}:{t1}"][k_].append(v.cpu())
    model.train(); out = {}
    for w, v in acc.items():
        mse = torch.cat(v["mse"]); out[w] = dict(mse=mse.mean(0).tolist(), cos=torch.cat(v["cos"]).mean(0).tolist(), per_ep_mse=mse.numpy().round(5).tolist(),
                                                mse_teacher=torch.cat(v["mse_teacher"]).mean(0).tolist() if v["mse_teacher"] else [])
    return out


t0 = time.time(); log = []; W0 = f"{windows[0][0]}:{windows[0][1]}"
for ep in range(a.epochs):
    perm = torch.randperm(E); tot = 0; n = 0
    for i in range(0, E, a.bs):
        idx = perm[i:i + a.bs]
        obs, act, z = tr["obs"][idx].to(dev), tr["act"][idx].to(dev), tr["z"][idx].to(dev)
        vis = blackout(len(idx), T, a.H, rng, dev)
        zh, Hs = model(obs, act, vis)
        w = 1.0 + 2.0 * (1 - vis)  # emphasise blind (imagination) steps
        loss = (((zh - z) ** 2).mean(-1) * w).mean() + 1e-4 * (Hs ** 2).mean()
        if act_head is not None:   # policy imitation: h_t -> action taken after step t (act[:, t])
            loss = loss + a.lambda_act * torch.nn.functional.cross_entropy(act_head(Hs).flatten(0, 1), act.flatten())
        if dr_head is not None: loss = loss + a.lambda_dr * dead_reckoning_loss(dr_head(Hs[..., dr_idx]), tr["pos"][idx].to(dev), tr["heading"][idx].to(dev), vis)
        if con_head is not None: loss = loss + a.lambda_con * blackout_contrastive_loss(con_head(Hs), z, vis)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        tot += loss.item(); n += 1
    sched.step()
    if ep % 5 == 4 or ep == a.epochs - 1:
        ev = run_eval(te); mse, cs = ev[W0]["mse"], ev[W0]["cos"]; rec = dict(ep=ep, loss=tot / n, mse=mse, cos=cs)
        if va is not None: rec["val_mse"] = float(np.mean(run_eval(va)[W0]["mse"]))
        print(f"ep {ep} loss {tot/n:.4f} test blind-MSE h1 {mse[0]:.4f} h4 {mse[3]:.4f} h8 {mse[-1]:.4f} cos h1 {cs[0]:.3f} h8 {cs[-1]:.3f} "
              f"teacher-space {np.mean(ev[W0]['mse_teacher']):.4f}" + (f" val {rec['val_mse']:.4f}" if va is not None else "") + f" {time.time()-t0:.0f}s", flush=True)
        log.append(rec)
ev = run_eval(te)
name = f"{a.wiring}_K{a.n_brains}_s{a.seed}{a.tag}"
os.makedirs("results/runs", exist_ok=True)
json.dump(dict(args=vars(a), neurons=model.N, params=model.n_params(), param_breakdown=pb, val_mse=(log[-1].get("val_mse") if va is not None else None),
               test_mse=ev[W0]["mse"], test_cos=ev[W0]["cos"],
               test_mse_teacher=ev[W0]["mse_teacher"], test_mse_per_ep=ev[W0]["per_ep_mse"], test_windows=ev,
               n_test_episodes=len(te["obs"]), n_train_episodes=int(E), log=log, pca_explained=lat.explained),
          open(f"results/runs/{name}.json", "w"), indent=1)
state = dict(state=model.state_dict(), lat=dict(mu=lat.mu, P=lat.P, scale=lat.scale), args=vars(a))
if act_head is not None: state["act_head"] = act_head.state_dict()
torch.save(state, f"results/runs/{name}.pt")
print("saved", name)
