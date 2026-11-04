#!/usr/bin/env bash
# Second GPU run (2026-11-04, one NVIDIA L4, AWS g6.xlarge): the transformer baseline, the attention input encoder, the
# hybrid routing ablation and the continuous-turn (turn-noise) data, on top of the 2026-11-02 protocol.
# Uses the box's prebuilt PyTorch venv (no uv). Restartable: every stage skips work whose output JSON already exists.
#
#   cd ~/World2Brain && STAGES="data trivial lr arena tn nyc hd homing summary" nohup bash scripts/run_l4.sh > logs/l4_$(date +%F).log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."; mkdir -p logs results/runs results/hd results/homing
STAGES=${STAGES:-"data trivial lr arena tn nyc hd homing summary"}
export PYTHONPATH=. DEVICE=${DEVICE:-cuda} PY=${PY:-/opt/pytorch/bin/python} TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
stage() { case " $STAGES " in *" $1 "*) echo; echo "##### $1  $(date +%T)"; return 0;; *) return 1;; esac; }
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1
SEEDS=${SEEDS:-"0 1 2"}; EPOCHS=${EPOCHS:-20}; TN=${TN:-0.3}
LR_CONN=${LR_CONN:-1e-3}; LR_GRU=${LR_GRU:-2e-3}                         # selected on the validation split in the 2026-11-02 run

# ---- data ----------------------------------------------------------------------------------------------------
[ -f data/connectome.npz ] || { echo "missing data/connectome.npz"; exit 1; }
if stage data; then
  for t in nav arrange; do
    [ -f data/${t}_train.npz ] || $PY scripts/collect.py --task $t --episodes 1000 --seed 0 --device "$DEVICE" --causal 1 --out data/${t}_train.npz
    [ -f data/${t}_test.npz ]  || $PY scripts/collect.py --task $t --episodes 200 --seed 1 --device "$DEVICE" --causal 1 --out data/${t}_test.npz
    # continuous turn angles: pi/8 * (1 + N(0, TN)); relative heading is no longer a linear function of the action counts
    [ -f data/${t}_tn_train.npz ] || $PY scripts/collect.py --task $t --episodes 1000 --seed 0 --device "$DEVICE" --causal 1 --turn_noise $TN --out data/${t}_tn_train.npz
    [ -f data/${t}_tn_test.npz ]  || $PY scripts/collect.py --task $t --episodes 200 --seed 1 --device "$DEVICE" --causal 1 --turn_noise $TN --out data/${t}_tn_test.npz
  done
  if [ ! -f data/nyc_twin_train.npz ]; then
    [ -f data/nyc/midtown_buildings.json ] || bash scripts/fetch_nyc_data.sh
    $PY scripts/collect.py --env nyc --episodes ${NYC_EPISODES:-600} --seed 0 --device "$DEVICE" --causal 1 --out data/nyc_twin_all.npz
    $PY scripts/split_geo.py data/nyc_twin_all.npz --out data/nyc_twin --block 120 --margin 40
  fi
fi
for f in data/nav_train.npz data/arrange_train.npz data/nav_test.npz data/arrange_test.npz; do [ -f "$f" ] || { echo "missing $f (run STAGES=data)"; exit 1; }; done
AR="--train data/nav_train.npz data/arrange_train.npz --test data/nav_test.npz data/arrange_test.npz"
TND="--train data/nav_tn_train.npz data/arrange_tn_train.npz --test data/nav_tn_test.npz data/arrange_tn_test.npz"
NYD="--train data/nyc_twin_train.npz --test data/nyc_twin_test.npz"

done_run() { [ -f "results/runs/$1.json" ] && grep -q test_mse_per_ep "results/runs/$1.json"; }
T() { local name=$1; shift; if done_run "$name"; then echo "skip $name"; else echo "=== $name  $(date +%T)"; $PY scripts/train.py --device "$DEVICE" --epochs "$EPOCHS" "$@" 2>&1 | grep -v -i -E "warn|redirect|AttentionInterface|pynvml"; fi; }

if stage trivial; then
  $PY scripts/eval_trivial.py $AR
  [ -f data/nav_tn_train.npz ] && $PY scripts/eval_trivial.py $TND --tag _tn
  [ -f data/nyc_twin_train.npz ] && $PY scripts/eval_trivial.py $NYD --tag _nyc
fi

if stage lr; then                                                               # transformer lr on a validation split (never test)
  for lr in 5e-4 1e-3 2e-3; do
    done_run transformer_K1_s0_lr${lr} || $PY scripts/train.py --wiring transformer --lr $lr --val 0.15 --epochs 10 --device "$DEVICE" $AR --tag _lr${lr} 2>&1 | grep -v -i -E "warn|pynvml"
  done
fi
pick_lr() { $PY -c "import json,glob; r=[(json.load(open(f))['val_mse'],f.split('_lr')[-1][:-5]) for f in glob.glob('results/runs/$1_K1_s0_lr*.json')]; print(min(r)[1] if r else '$2')" 2>/dev/null || echo "$2"; }
LR_TF=${LR_TF:-$(pick_lr transformer 1e-3)}; echo "lr: connectome $LR_CONN gru $LR_GRU transformer $LR_TF"

if stage arena; then
  for s in $SEEDS; do
    for w in connectome block_shuffled random; do T "${w}_K1_s${s}" --wiring $w --seed $s --lr $LR_CONN $AR; done
    T "gru_K1_s${s}"         --wiring gru         --seed $s --lr $LR_GRU  $AR
    T "transformer_K1_s${s}" --wiring transformer --seed $s --lr $LR_TF   $AR
    # attention input encoder: the sparse students lose 1.7 M dead input parameters; GRU and transformer are re-matched to the smaller budget
    for w in connectome block_shuffled; do T "${w}_K1_s${s}_attnin" --wiring $w --seed $s --lr $LR_CONN --in_mode attn --tag _attnin $AR; done
    T "gru_K1_s${s}_attnin"         --wiring gru         --seed $s --lr $LR_GRU --in_mode attn --tag _attnin $AR
    T "transformer_K1_s${s}_attnin" --wiring transformer --seed $s --lr $LR_TF  --in_mode attn --tag _attnin $AR
    # hybrid: connectome recurrence + one attention layer over cell-type tokens per step
    T "connectome_K1_s${s}_hybrid" --wiring connectome --seed $s --lr $LR_CONN --hybrid 1 --tag _hybrid $AR
  done
fi

if stage tn && [ -f data/nav_tn_train.npz ]; then
  for s in $SEEDS; do
    for w in connectome block_shuffled random; do T "${w}_K1_s${s}_tn" --wiring $w --seed $s --lr $LR_CONN --tag _tn $TND; done
    T "gru_K1_s${s}_tn"         --wiring gru         --seed $s --lr $LR_GRU --tag _tn $TND
    T "transformer_K1_s${s}_tn" --wiring transformer --seed $s --lr $LR_TF  --tag _tn $TND
  done
fi

if stage nyc && [ -f data/nyc_twin_train.npz ]; then
  for s in $SEEDS; do
    for w in connectome block_shuffled random; do T "${w}_K1_s${s}_nyc" --wiring $w --seed $s --lr $LR_CONN --tag _nyc --epochs 30 $NYD; done
    T "gru_K1_s${s}_nyc"         --wiring gru         --seed $s --lr $LR_GRU --tag _nyc --epochs 30 $NYD
    T "transformer_K1_s${s}_nyc" --wiring transformer --seed $s --lr $LR_TF  --tag _nyc --epochs 30 $NYD
  done
fi

if stage hd; then                                                               # ridge 8:16, MLP 8:16, ridge 8:24 for every checkpoint of this run
  GH=$($PY -c "import json; print(json.load(open('results/runs/gru_K1_s0.json'))['args']['gru_hidden'])" 2>/dev/null || echo 762)
  for ck in results/runs/*_K1_s[0-4]*.pt; do
    r=$(basename "$ck" .pt); case "$r" in *_lr*) continue;; esac
    [ -f results/hd/$r.json ]       || $PY scripts/eval_hd.py --run "$r" --device "$DEVICE" --match_n "$GH" 2>&1 | grep -v -i pynvml || echo "hd failed: $r"
    [ -f results/hd/${r}_mlp.json ] || $PY scripts/eval_hd.py --run "$r" --device "$DEVICE" --match_n "$GH" --probe mlp --tag _mlp 2>&1 | grep -v -i pynvml || echo "hd mlp failed: $r"
    [ -f results/hd/${r}_b8_24.json ] || $PY scripts/eval_hd.py --run "$r" --device "$DEVICE" --match_n "$GH" --blind 8:24 --tag _b8_24 2>&1 | grep -v -i pynvml || echo "hd 8:24 failed: $r"
  done
fi

if stage homing; then
  H() { local out=$1; shift; [ -f "results/homing/$out.json" ] && return 0; $PY scripts/eval_homing.py "$@" --device "$DEVICE" 2>&1 | grep -v -i pynvml || echo "homing failed: $out"; }
  H connectome_K1_s0_arena_oracle --run connectome_K1_s0 --readout oracle; H connectome_K1_s0_arena_actions --run connectome_K1_s0 --readout actions; H connectome_K1_s0_arena_random --run connectome_K1_s0 --readout random
  for s in $SEEDS; do for w in connectome block_shuffled random gru transformer connectome_attnin block_shuffled_attnin gru_attnin transformer_attnin connectome_hybrid; do
    r=${w%_attnin}; r=${r%_hybrid}; sfx=""; case "$w" in *_attnin) sfx=_attnin;; *_hybrid) sfx=_hybrid;; esac
    [ -f results/runs/${r}_K1_s${s}${sfx}.pt ] && H ${r}_K1_s${s}${sfx}_arena_state --run ${r}_K1_s${s}${sfx} --readout state
  done; done
  for w in connectome block_shuffled gru transformer; do H ${w}_K1_s0_arena_mlp --run ${w}_K1_s0 --readout mlp; done
  if [ -f results/runs/connectome_K1_s0_tn.pt ]; then                            # continuous turns: dead reckoning is no longer exact
    H connectome_K1_s0_tn_arena_actions_tn --run connectome_K1_s0_tn --readout actions --turn_noise $TN --tag _tn
    H connectome_K1_s0_tn_arena_oracle_tn --run connectome_K1_s0_tn --readout oracle --turn_noise $TN --tag _tn
    for s in $SEEDS; do for w in connectome block_shuffled random gru transformer; do
      [ -f results/runs/${w}_K1_s${s}_tn.pt ] && H ${w}_K1_s${s}_tn_arena_state_tn --run ${w}_K1_s${s}_tn --readout state --turn_noise $TN --tag _tn
    done; done
  fi
  if [ -f results/runs/connectome_K1_s0_nyc.pt ]; then
    H connectome_K1_s0_nyc_nyc_oracle --run connectome_K1_s0_nyc --env nyc --readout oracle --t_blind 30; H connectome_K1_s0_nyc_nyc_actions --run connectome_K1_s0_nyc --env nyc --readout actions --t_blind 30
    for s in $SEEDS; do for w in connectome block_shuffled gru transformer; do
      [ -f results/runs/${w}_K1_s${s}_nyc.pt ] && H ${w}_K1_s${s}_nyc_nyc_state --run ${w}_K1_s${s}_nyc --env nyc --readout state --t_blind 30
    done; done
  fi
fi

stage summary && $PY scripts/summarize.py > /dev/null && echo "summary -> results/summary.md"
echo; echo "ALL STAGES DONE $(date +%T)"
