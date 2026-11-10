#!/usr/bin/env bash
# Full-brain run (2026-11-06, professor's instruction): the student is the WHOLE proofread FlyWire v783 brain,
# 139,248 neurons and 2,700,429 signed edges (>= 5 synapses), not the 6,935-neuron navigation subcircuit.
# Same protocol as run_l4.sh; every run carries the tag _full (connectome_K1_s0_full, *_full_tn, *_full_nyc) and
# trains with --conn data/connectome_full.npz. No dense control (139k^2 is not a matrix anyone can train); GRU and
# transformer are re-matched automatically to the full-brain parameter count (41.3 M, 35.6 M of it the readout).
# Restartable: every stage skips work whose output exists.
#
#   cd ~/World2Brain && STAGES="graph arena tn nyc hd homing blackout" nohup bash scripts/run_l4_full.sh > logs/l4_full_$(date +%F).log 2>&1 &
# Big GPU (H200 141 GB / H100 80 GB): JOBS=N trains N runs at once (one full-brain run needs ~14 GB), each into
# logs/train_<run>.log; P=N parallelises the blackout evals. Same protocol and batch size, so results stay comparable.
#   JOBS=6 P=6 OMP=8 STAGES="arena tn nyc hd homing blackout" nohup bash scripts/run_l4_full.sh > logs/h200_full_$(date +%F).log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."; mkdir -p logs results/runs results/hd results/homing
STAGES=${STAGES:-"graph arena tn nyc hd homing blackout"}
[ -x /opt/pytorch/bin/python ] && DEF_PY=/opt/pytorch/bin/python || DEF_PY="uv run --no-sync python"   # AWS DLAMI venv, else the project's uv env
export PYTHONPATH=. DEVICE=${DEVICE:-cuda} PY=${PY:-$DEF_PY} TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
OMP=${OMP:-4}; export OMP_NUM_THREADS=$OMP OPENBLAS_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP
JOBS=${JOBS:-1}; P=${P:-2}
stage() { case " $STAGES " in *" $1 "*) echo; echo "##### $1  $(date +%T)"; return 0;; *) return 1;; esac; }
SEEDS=${SEEDS:-"0 1 2"}; EPOCHS=${EPOCHS:-20}; TN=${TN:-0.3}; BS=${BS:-16}
LR_CONN=${LR_CONN:-1e-3}; LR_GRU=${LR_GRU:-2e-3}; LR_TF=${LR_TF:-1e-3}
WIRINGS=${WIRINGS:-"connectome shuffled block_shuffled random"}      # sparse rows on the full graph; shuffled = degree-preserving
CONN=data/connectome_full.npz; MIN_SYN=${MIN_SYN:-5}

if stage graph; then
  [ -f $CONN ] || $PY -m wm2cx.connectome --regions VIS CX MB OUT OTH --min_syn $MIN_SYN --out $CONN
fi
[ -f $CONN ] || { echo "missing $CONN (run STAGES=graph; needs data/raw/neuron_annotations.tsv and proofread_connections_783.feather)"; exit 1; }
AR="--train data/nav_train.npz data/arrange_train.npz --test data/nav_test.npz data/arrange_test.npz"
TND="--train data/nav_tn_train.npz data/arrange_tn_train.npz --test data/nav_tn_test.npz data/arrange_tn_test.npz"
NYD="--train data/nyc_twin_train.npz --test data/nyc_twin_test.npz"

done_run() { [ -f "results/runs/$1.json" ] && grep -q test_mse_per_ep "results/runs/$1.json"; }
T() {
  local name=$1; shift
  if done_run "$name"; then echo "skip $name"; return 0; fi
  if [ "$JOBS" -le 1 ]; then
    echo "=== $name  $(date +%T)"; $PY scripts/train.py --device "$DEVICE" --epochs "$EPOCHS" --bs $BS --conn $CONN "$@" 2>&1 | grep -v -i -E "warn|redirect|AttentionInterface|pynvml"
    return 0
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 5; done   # at most JOBS trainings on the GPU at once
  echo "=== $name  $(date +%T)  (bg, logs/train_$name.log)"
  ( if $PY scripts/train.py --device "$DEVICE" --epochs "$EPOCHS" --bs $BS --conn $CONN "$@" > "logs/train_$name.log" 2>&1 && done_run "$name"
    then echo "done $name  $(date +%T)"; else echo "FAILED $name  $(date +%T)  (see logs/train_$name.log)"; fi ) &
}
train_set() {   # $1 = tag suffix ("" | _tn | _nyc), $2.. = data args
  local sfx=$1; shift
  for s in $SEEDS; do
    for w in $WIRINGS; do T "${w}_K1_s${s}_full${sfx}" --wiring $w --seed $s --lr $LR_CONN --tag _full${sfx} "$@"; done
    T "gru_K1_s${s}_full${sfx}"         --wiring gru         --seed $s --lr $LR_GRU --tag _full${sfx} "$@"
    T "transformer_K1_s${s}_full${sfx}" --wiring transformer --seed $s --lr $LR_TF  --tag _full${sfx} "$@"
  done
}
stage arena && train_set ""    $AR
stage tn    && [ -f data/nav_tn_train.npz ]   && train_set _tn  $TND
stage nyc   && [ -f data/nyc_twin_train.npz ] && train_set _nyc --epochs 30 $NYD
wait   # background trainings (JOBS > 1) must finish before the evals read their checkpoints

if stage hd; then
  GH=$($PY -c "import json; print(json.load(open('results/runs/gru_K1_s0_full.json'))['args']['gru_hidden'])" 2>/dev/null || echo 3600)
  for ck in results/runs/*_K1_s[0-4]_full*.pt; do
    r=$(basename "$ck" .pt)
    [ -f results/hd/$r.json ]         || $PY scripts/eval_hd.py --run "$r" --device "$DEVICE" --match_n "$GH" 2>&1 | grep -v -i pynvml || echo "hd failed: $r"
    [ -f results/hd/${r}_mlp.json ]   || $PY scripts/eval_hd.py --run "$r" --device "$DEVICE" --match_n "$GH" --probe mlp --tag _mlp 2>&1 | grep -v -i pynvml || echo "hd mlp failed: $r"
    [ -f results/hd/${r}_b8_24.json ] || $PY scripts/eval_hd.py --run "$r" --device "$DEVICE" --match_n "$GH" --blind 8:24 --tag _b8_24 2>&1 | grep -v -i pynvml || echo "hd 8:24 failed: $r"
  done
fi

if stage homing; then     # linear-readout homing, as in run_l4.sh, for the homing table
  H() { local out=$1; shift; [ -f "results/homing/$out.json" ] && return 0; $PY scripts/eval_homing.py "$@" --device "$DEVICE" 2>&1 | grep -v -i pynvml || echo "homing failed: $out"; }
  for s in $SEEDS; do for w in $WIRINGS gru transformer; do
    [ -f results/runs/${w}_K1_s${s}_full.pt ]     && H ${w}_K1_s${s}_full_arena_state --run ${w}_K1_s${s}_full --readout state
    [ -f results/runs/${w}_K1_s${s}_full_tn.pt ]  && H ${w}_K1_s${s}_full_tn_arena_state_tn --run ${w}_K1_s${s}_full_tn --readout state --turn_noise $TN --tag _tn
    [ -f results/runs/${w}_K1_s${s}_full_nyc.pt ] && H ${w}_K1_s${s}_full_nyc_nyc_state --run ${w}_K1_s${s}_full_nyc --env nyc --readout state --t_blind 30
  done; done
fi

# Blackout Navigation benchmark (docs/BENCHMARK_BLACKOUT.md) with the policy head on the full-brain checkpoints
stage blackout && P=$P SFX=_full WIRINGS="$WIRINGS gru transformer" READOUTS="policy state" SEEDS="$SEEDS" DEVICE=$DEVICE PY=$PY bash scripts/run_blackout.sh
echo; echo "FULL-BRAIN STAGES DONE $(date +%T)"
