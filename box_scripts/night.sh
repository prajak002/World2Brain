#!/usr/bin/env bash
# Overnight finisher: goal-vector navigation sweep, corrected dead-reckoning controls, integrator fix, summary, Kaggle versions.
cd /root/World2Brain; export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; PY="uv run --no-sync python"
kaggle_version() { rm -rf /root/kaggle_up/results; mkdir -p /root/kaggle_up/results
  cp -r results/runs results/hd results/homing results/plan results/bench results/baselines results/goalnav results/summary.md /root/kaggle_up/results/ 2>/dev/null
  cp docs/results_draft_path_integration.md /root/kaggle_up/ 2>/dev/null
  $PY -m kaggle datasets version -p /root/kaggle_up -m "$1" --dir-mode zip > /root/kaggle_version_$(date +%H%M).log 2>&1; tail -n 1 /root/kaggle_version_*.log | tail -1; }
G() { local out=$1; shift; [ -f "results/goalnav/$out.json" ] && return 0; $PY scripts/eval_goalnav.py "$@" --device cuda 2>&1 | grep -vi warn | grep -E "readout:|^\{" | cut -c1-200 || echo "goalnav failed: $out"; }
echo "### goalnav $(date +%T)"
for b in 0 3 5; do G connectome_K1_s0_arena_oracle_b$b --run connectome_K1_s0 --readout oracle --blackout $b; G connectome_K1_s0_arena_random_b$b --run connectome_K1_s0 --readout random --blackout $b; done
for s in 0 1 2; do for w in connectome block_shuffled shuffled random dense gru connectome_nosign gru_recmatch; do
  r=${w%_recmatch}_K1_s$s; [ "$w" = gru_recmatch ] && r=gru_K1_s${s}_recmatch; [ -f results/runs/$r.pt ] || continue
  for b in 0 3 5; do G ${r}_arena_state_b$b --run $r --readout state --blackout $b; done
done; done
G connectome_K1_s0_arena_mlp_b0 --run connectome_K1_s0 --readout mlp --blackout 0; G gru_K1_s0_arena_mlp_b0 --run gru_K1_s0 --readout mlp --blackout 0
for b in 0 5; do G connectome_K1_s0_nyc_nyc_oracle_b$b --run connectome_K1_s0_nyc --env nyc --readout oracle --blackout $b; G connectome_K1_s0_nyc_nyc_random_b$b --run connectome_K1_s0_nyc --env nyc --readout random --blackout $b
  for w in connectome block_shuffled gru; do for s in 0 1 2; do r=${w}_K1_s${s}_nyc; [ -f results/runs/$r.pt ] || continue; G ${r}_nyc_state_b$b --run $r --env nyc --readout state --blackout $b; done; done; done
echo "### wait for probe sweeps $(date +%T)"
until ! tmux has-session -t hdx 2>/dev/null && ! tmux has-session -t hdrv 2>/dev/null && ! tmux has-session -t hrm 2>/dev/null; do sleep 60; done
echo "### corrected dead-reckoning controls $(date +%T)"
mkdir -p results/homing_oldscale; for f in connectome_K1_s0_arena_actions connectome_K1_s0_arena_actions_tn; do [ -f results/homing/$f.json ] && mv results/homing/$f.json results/homing_oldscale/; done
$PY scripts/eval_homing.py --run connectome_K1_s0 --readout actions --device cuda 2>&1 | grep -vi warn | tail -1 | cut -c1-200
$PY scripts/eval_homing.py --run connectome_K1_s0 --readout actions --turn_noise 0.3 --tag _tn --device cuda 2>&1 | grep -vi warn | tail -1 | cut -c1-200
echo "### integrator fix $(date +%T)"; $PY scripts/fix_hd_integrator.py 2>&1 | grep -vi warn | tail -3
$PY scripts/summarize.py > /dev/null && echo "summary ok"
echo "### kaggle final $(date +%T)"; kaggle_version "final: probes (MLP, dark windows, real video), capacity-matched GRU, goal navigation, ablation"
echo "NIGHT_DONE $(date +%T)"
