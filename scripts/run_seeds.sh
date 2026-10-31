#!/usr/bin/env bash
# Seeds and confidence intervals (#8): 5 seeds per condition on the arena, 3 on NYC, plus the trivial baselines (#6).
# Every run stores per-episode test errors and two blind windows (8:16 primary, 14:22 alternative), so
# scripts/summarize.py can report bootstrap 95% CIs and paired tests over the shared test episodes.
#   PYTHONPATH=. DEVICE=cuda bash scripts/run_seeds.sh
set -u
cd "$(dirname "$0")/.."; export PYTHONPATH=.
DEVICE=${DEVICE:-mps}; EPOCHS=${EPOCHS:-20}; SEEDS_ARENA=${SEEDS_ARENA:-"0 1 2 3 4"}; SEEDS_NYC=${SEEDS_NYC:-"0 1 2"}
PY=${PY:-"uv run python"}
done_run() { [ -f "results/runs/$1.json" ] && grep -q test_mse_per_ep "results/runs/$1.json"; }
T() { local name=$1; shift; if done_run "$name"; then echo "skip $name"; else echo "=== $name"; $PY scripts/train.py --device "$DEVICE" --epochs "$EPOCHS" "$@" 2>&1 | grep -v -i -E "warn|redirect|AttentionInterface"; fi; }

echo "=== trivial baselines"
$PY scripts/eval_trivial.py
[ -f data/nyc_train.npz ] && $PY scripts/eval_trivial.py --train data/nyc_train.npz --test data/nyc_test.npz --tag _nyc

for s in $SEEDS_ARENA; do
  for w in connectome shuffled random connectome_nosign; do T "${w}_K1_s${s}" --wiring $w --seed $s; done
  T "gru_K1_s${s}" --wiring gru --gru_hidden 1400 --seed $s
  T "dense_K1_s${s}" --wiring dense --seed $s --lr 1e-3
  T "connectome_K2_s${s}" --wiring connectome --n_brains 2 --seed $s
  T "connectome_K3_s${s}" --wiring connectome --n_brains 3 --seed $s
done
# Data-efficiency axis (the experiment that can beat the GRU where a prior should matter: few episodes).
# N_TRAIN episodes out of the 1,000 arena episodes, 3 seeds, connectome vs shuffled vs GRU; PCA is still fit on the
# full train set (train.py --n_train). Set N_TRAIN="" to skip.
N_TRAIN=${N_TRAIN:-"50 100 200 500"}; SEEDS_EFF=${SEEDS_EFF:-"0 1 2"}
for n in $N_TRAIN; do for s in $SEEDS_EFF; do
  for w in connectome shuffled; do T "${w}_K1_s${s}_n${n}" --wiring $w --seed $s --n_train $n --tag "_n${n}"; done
  T "gru_K1_s${s}_n${n}" --wiring gru --gru_hidden 1400 --seed $s --n_train $n --tag "_n${n}"
done; done
if [ -f data/nyc_train.npz ]; then
  D="--train data/nyc_train.npz --test data/nyc_test.npz --tag _nyc --epochs 30"
  for s in $SEEDS_NYC; do
    for w in connectome shuffled random; do T "${w}_K1_s${s}_nyc" --wiring $w --seed $s $D; done
    T "gru_K1_s${s}_nyc" --wiring gru --gru_hidden 1400 --seed $s $D
  done
fi
$PY scripts/summarize.py > /dev/null && echo SEEDS_DONE
