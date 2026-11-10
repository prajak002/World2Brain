#!/usr/bin/env bash
# Small-model batch for a Kaggle GPU (issues #24 LSTM baseline, #21 circuit-structured objectives). Subcircuit students
# only (6,935 neurons, 3.71 M parameters); full-brain work stays on the H200. Self-contained: regenerates the causal
# arena data with the run-2 commands of scripts/run_l4.sh (same seeds), then trains and evaluates.
#   STAGES="data lr anchor lstm obj hd homing" bash scripts/run_kaggle_small.sh
#   STAGES="data gnnlr gnn anchor hd homing" TN_DATA=0 bash scripts/run_kaggle_small.sh     # issue #23, graph-network student
# GNN settings fixed before the run: d = 8 with region sharing (0.997 x the FlyBrain parameter count), GRU update, learned phi;
# lr from {5e-4, 1e-3, 2e-3} on a validation split with the real graph, seed 0, 10 epochs; then connectome / block_shuffled /
# random graphs x seeds 0-2, plus d = 1 with learned phi on the real graph (cell-type sharing, 1.000 x).
# Restartable: a run whose JSON exists is skipped. Settings fixed before the run (no search except the LSTM lr on a
# validation split): 20 epochs, seeds 0-2, connectome lr 1e-3, GRU lr 2e-3 (run-2 values), lambda_dr = lambda_con = 0.1.
set -u; cd "$(dirname "$0")/.."; export PYTHONPATH=. TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
PY=${PY:-python}; DEVICE=${DEVICE:-cuda}; STAGES=${STAGES:-"data lr anchor lstm obj hd homing"}; SEEDS=${SEEDS:-"0 1 2"}; EPOCHS=${EPOCHS:-20}; TN=0.3
mkdir -p data logs results/runs results/hd results/homing
stage() { case " $STAGES " in *" $1 "*) echo; echo "##### $1  $(date +%T)"; return 0;; *) return 1;; esac; }
AR="--train data/nav_train.npz data/arrange_train.npz --test data/nav_test.npz data/arrange_test.npz"
TND="--train data/nav_tn_train.npz data/arrange_tn_train.npz --test data/nav_tn_test.npz data/arrange_tn_test.npz"
T() { local name=$1; shift; if [ -f "results/runs/$name.json" ]; then echo "skip $name"; else echo "=== $name  $(date +%T)"; $PY scripts/train.py --device "$DEVICE" --epochs "$EPOCHS" "$@" 2>&1 | grep -v -i -E "warn|pynvml" | tail -3; fi; }
if stage data; then
  for t in nav arrange; do
    [ -f data/${t}_train.npz ] || $PY scripts/collect.py --task $t --episodes 1000 --seed 0 --device "$DEVICE" --causal 1 --out data/${t}_train.npz
    [ -f data/${t}_test.npz ]  || $PY scripts/collect.py --task $t --episodes 200 --seed 1 --device "$DEVICE" --causal 1 --out data/${t}_test.npz
    [ "${TN_DATA:-1}" = 1 ] || continue
    [ -f data/${t}_tn_train.npz ] || $PY scripts/collect.py --task $t --episodes 1000 --seed 0 --device "$DEVICE" --causal 1 --turn_noise $TN --out data/${t}_tn_train.npz
    [ -f data/${t}_tn_test.npz ]  || $PY scripts/collect.py --task $t --episodes 200 --seed 1 --device "$DEVICE" --causal 1 --turn_noise $TN --out data/${t}_tn_test.npz
  done
fi
if stage lr; then                                                               # LSTM lr on a validation split (never test), as the transformer in run 2
  for lr in 5e-4 1e-3 2e-3; do T "lstm_K1_s0_lr${lr}" --wiring lstm --lr $lr --val 0.15 --epochs 10 --tag _lr${lr} $AR; done
fi
LR_LSTM=$($PY -c "import json,glob; r=[(json.load(open(f))['val_mse'],f.split('_lr')[-1][:-5]) for f in glob.glob('results/runs/lstm_K1_s0_lr*.json')]; print(min(r)[1] if r else '2e-3')" 2>/dev/null || echo 2e-3)
echo "lr: lstm $LR_LSTM"
if stage anchor; then                                                           # plain-loss rows on this regenerated data: the reference for everything below
  for s in $SEEDS; do
    for w in connectome block_shuffled; do T "${w}_K1_s${s}" --wiring $w --seed $s --lr 1e-3 $AR; done
    T "gru_K1_s${s}" --wiring gru --seed $s --lr 2e-3 $AR
  done
fi
if stage lstm; then                                                             # issue #24
  for s in $SEEDS; do
    T "lstm_K1_s${s}" --wiring lstm --seed $s --lr $LR_LSTM $AR
    T "lstm_K1_s${s}_tn" --wiring lstm --seed $s --lr $LR_LSTM --tag _tn $TND
  done
fi
if stage obj; then                                                              # issue #21: plain (anchor) / dead reckoning / contrastive
  for s in $SEEDS; do
    for w in connectome block_shuffled gru; do
      lr=1e-3; [ $w = gru ] && lr=2e-3
      T "${w}_K1_s${s}_dr"  --wiring $w --seed $s --lr $lr --lambda_dr 0.1  --tag _dr  $AR
      T "${w}_K1_s${s}_con" --wiring $w --seed $s --lr $lr --lambda_con 0.1 --tag _con $AR
    done
  done
fi
if stage gnnlr; then                                                            # issue #23: GNN lr on a validation split (never test)
  for lr in 5e-4 1e-3 2e-3; do T "gnn_K1_s0_lr${lr}" --wiring gnn --gnn_d 8 --gnn_group region --lr $lr --val 0.15 --epochs 10 --tag _lr${lr} $AR; done
fi
if stage gnn; then
  LR_GNN=$($PY -c "import json,glob; r=[(json.load(open(f))['val_mse'],f.split('_lr')[-1][:-5]) for f in glob.glob('results/runs/gnn_K1_s0_lr*.json')]; print(min(r)[1] if r else '1e-3')" 2>/dev/null || echo 1e-3)
  echo "lr: gnn $LR_GNN"
  for s in $SEEDS; do
    T "gnn_K1_s${s}_c8"  --wiring gnn --gnn_graph connectome     --gnn_d 8 --gnn_group region --seed $s --lr $LR_GNN --tag _c8  $AR
    T "gnn_K1_s${s}_bs8" --wiring gnn --gnn_graph block_shuffled --gnn_d 8 --gnn_group region --seed $s --lr $LR_GNN --tag _bs8 $AR
    T "gnn_K1_s${s}_r8"  --wiring gnn --gnn_graph random         --gnn_d 8 --gnn_group region --seed $s --lr $LR_GNN --tag _r8  $AR
  done
  for s in $SEEDS; do T "gnn_K1_s${s}_c1" --wiring gnn --gnn_graph connectome --gnn_d 1 --gnn_group cell_type --seed $s --lr $LR_GNN --tag _c1 $AR; done
fi
if stage hd; then
  GH=$($PY -c "import json; print(json.load(open('results/runs/gru_K1_s0.json'))['args']['gru_hidden'])" 2>/dev/null || echo 762)
  for ck in results/runs/*_K1_s[0-4]*.pt; do
    r=$(basename "$ck" .pt); case "$r" in *_lr*|gnn_*8) continue;; esac   # d = 8 GNN state (55,480 columns) does not fit the ridge probes yet
    [ -f results/hd/$r.json ] || $PY scripts/eval_hd.py --run "$r" --device "$DEVICE" --match_n "$GH" 2>&1 | grep -v -i pynvml | tail -2 || echo "hd failed: $r"
  done
fi
if stage homing; then
  for ck in results/runs/*_K1_s[0-4]*.pt; do
    r=$(basename "$ck" .pt); case "$r" in *_lr*|*_tn|gnn_*8) continue;; esac
    [ -f results/homing/${r}_arena_state.json ] || $PY scripts/eval_homing.py --run "$r" --readout state --device "$DEVICE" 2>&1 | grep -v -i pynvml | tail -2 || echo "homing failed: $r"
  done
fi
echo "KAGGLE_SMALL_DONE $(date -u)"; ls results/runs/*.json | wc -l
