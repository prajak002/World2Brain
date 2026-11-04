#!/usr/bin/env bash
# Real handheld video (8 phone clips, 79 episodes, data/realvid_all.npz) as a test world for every trained student.
# Waits for run_l4.sh to finish, then for each arena / attention-input / NYC-twin model and seed:
#   zero-shot blind prediction on real video            -> results/runs/<run>_realvid.json
#   leave-one-clip-out fine-tuning (10 epochs)          -> results/runs/<run>_realvid_ft.json
#   heading / position probe, simulator-fit readout     -> results/hd/<run>_rv_transfer.json
#   heading / position probe, fit on the other clips    -> results/hd/<run>_rv_indomain.json
#   cd ~/World2Brain && nohup bash scripts/run_l4_realvid.sh > logs/l4_realvid_$(date +%F).log 2>&1 &
set -u; cd "$(dirname "$0")/.."; export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=${PY:-/opt/pytorch/bin/python}; DEVICE=${DEVICE:-cuda}; SEEDS=${SEEDS:-"0 1 2"}; RV=data/realvid_all.npz
MAIN_LOG=$(ls -t logs/l4_20*.log 2>/dev/null | grep -v realvid | head -1)
if [ -n "$MAIN_LOG" ]; then until grep -q "ALL STAGES DONE" "$MAIN_LOG"; do sleep 120; done; fi
echo "##### realvid stage $(date +%T)"; $PY -c "import numpy as np; d=np.load('$RV'); print('realvid', d['z'].shape, 'clips', int(d['clip'].max())+1)"
GH=$($PY -c "import json; print(json.load(open('results/runs/gru_K1_s0.json'))['args']['gru_hidden'])" 2>/dev/null || echo 762)
RUNS=""
for s in $SEEDS; do
  for w in connectome block_shuffled random gru transformer; do RUNS="$RUNS ${w}_K1_s$s"; done
  for w in connectome block_shuffled gru transformer; do RUNS="$RUNS ${w}_K1_s${s}_attnin"; done
  RUNS="$RUNS connectome_K1_s${s}_hybrid"
  for w in connectome block_shuffled random gru transformer; do RUNS="$RUNS ${w}_K1_s${s}_nyc"; done
done
for r in $RUNS; do
  [ -f results/runs/$r.pt ] || { echo "skip $r (no checkpoint)"; continue; }
  echo "=== $r realvid $(date +%T)"
  [ -f results/runs/${r}_realvid.json ]    || $PY scripts/eval_ckpt.py --run $r --test $RV --tag _realvid --device $DEVICE 2>&1 | grep -v -i -E "warn|pynvml" || echo "failed: $r realvid"
  [ -f results/runs/${r}_realvid_ft.json ] || $PY scripts/eval_ckpt.py --run $r --test $RV --tag _realvid_ft --finetune_epochs 10 --baselines 0 --device $DEVICE 2>&1 | grep -v -i -E "warn|pynvml" || echo "failed: $r realvid_ft"
  HD_EXTRA="--match_n $GH"
  [ -f results/hd/${r}_rv_transfer.json ] || $PY scripts/eval_hd.py --run $r --device $DEVICE $HD_EXTRA --test $RV --tag _rv_transfer 2>&1 | grep -v -i -E "warn|pynvml" | grep -E "^blind|split|Error" || echo "hd failed: $r rv_transfer"
  [ -f results/hd/${r}_rv_indomain.json ] || $PY scripts/eval_hd.py --run $r --device $DEVICE $HD_EXTRA --train $RV --test $RV --split_by_clip 2 --tag _rv_indomain 2>&1 | grep -v -i -E "warn|pynvml" | grep -E "^blind|split|Error" || echo "hd failed: $r rv_indomain"
done
$PY scripts/summarize.py > /dev/null 2>&1 || echo "summarize failed"
echo; echo "REALVID DONE $(date +%T)"
