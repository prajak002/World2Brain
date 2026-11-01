#!/usr/bin/env bash
# Seeds and confidence intervals (#8): 5 seeds per condition on the arena, 3 on NYC, plus the trivial baselines (#6).
# Controls (#1, #2): shuffled (Maslov-Sneppen), block_shuffled, random, connectome_nosign, connectome_randsign, dense.
# GRU (#4): `gru` is parameter-matched to the FlyBrain (gru_hidden auto); `gru_big` (1,400 hidden, 9.5M) is the over-budget reference.
# Every run stores per-episode test errors and two blind windows (8:16 primary, 14:22 alternative), so
# scripts/summarize.py can report bootstrap 95% CIs and paired tests over the shared test episodes.
#   PYTHONPATH=. DEVICE=cuda bash scripts/run_seeds.sh
set -u
cd "$(dirname "$0")/.."; export PYTHONPATH=.
DEVICE=${DEVICE:-mps}; EPOCHS=${EPOCHS:-20}; SEEDS_ARENA=${SEEDS_ARENA:-"0 1 2 3 4"}; SEEDS_NYC=${SEEDS_NYC:-"0 1 2"}
PY=${PY:-"uv run python"}
TR=${TRAIN:-"data/nav_train.npz data/arrange_train.npz"}; TE=${TEST:-"data/nav_test.npz data/arrange_test.npz"}   # causal data by default (collect.py --causal 1)
DATA="--train $TR --test $TE"
# per-architecture learning rates selected on the validation split by the lr stage of aws_h100.sh (#4); defaults = run-1 values
LR_CONN=${LR_CONN:-2e-3}; LR_DENSE=${LR_DENSE:-1e-3}; LR_GRU=${LR_GRU:-2e-3}; echo "lr: FlyBrain $LR_CONN dense $LR_DENSE gru $LR_GRU"
done_run() { [ -f "results/runs/$1.json" ] && grep -q test_mse_per_ep "results/runs/$1.json"; }
T() { local name=$1; shift; if done_run "$name"; then echo "skip $name"; else echo "=== $name"; $PY scripts/train.py --device "$DEVICE" --epochs "$EPOCHS" $DATA "$@" 2>&1 | grep -v -i -E "warn|redirect|AttentionInterface"; fi; }

echo "=== trivial baselines"
$PY scripts/eval_trivial.py --train $TR --test $TE
[ -f data/nyc_train.npz ] && $PY scripts/eval_trivial.py --train data/nyc_train.npz --test data/nyc_test.npz --tag _nyc

for s in $SEEDS_ARENA; do
  for w in connectome shuffled block_shuffled random connectome_nosign connectome_randsign; do T "${w}_K1_s${s}" --wiring $w --seed $s --lr $LR_CONN; done
  T "gru_K1_s${s}" --wiring gru --seed $s --lr $LR_GRU                                   # parameter-matched (#4)
  T "gru_K1_s${s}_big" --wiring gru --gru_hidden 1400 --seed $s --tag _big --lr $LR_GRU    # over-budget reference (old Table 1 GRU)
  T "dense_K1_s${s}" --wiring dense --seed $s --lr $LR_DENSE
  T "connectome_K2_s${s}" --wiring connectome --n_brains 2 --seed $s --lr $LR_CONN
  T "connectome_K3_s${s}" --wiring connectome --n_brains 3 --seed $s --lr $LR_CONN
done
# Data-efficiency axis (the experiment that can beat the GRU where a prior should matter: few episodes).
# N_TRAIN episodes out of the 1,000 arena episodes, 3 seeds, connectome vs shuffled vs GRU; PCA is still fit on the
# full train set (train.py --n_train). Set N_TRAIN="" to skip.
N_TRAIN=${N_TRAIN:-"50 100 200 500"}; SEEDS_EFF=${SEEDS_EFF:-"0 1 2"}
for n in $N_TRAIN; do for s in $SEEDS_EFF; do
  for w in connectome shuffled block_shuffled; do T "${w}_K1_s${s}_n${n}" --wiring $w --seed $s --n_train $n --tag "_n${n}" --lr $LR_CONN; done
  T "gru_K1_s${s}_n${n}" --wiring gru --seed $s --n_train $n --tag "_n${n}" --lr $LR_GRU
done; done
NYC_TR=${NYC_TRAIN:-data/nyc_train.npz}; NYC_TE=${NYC_TEST:-data/nyc_test.npz}      # Valdrada episodes (legacy, non-causal) or data/nyc_twin_*.npz (raycast twin, causal)
if [ -f "$NYC_TR" ]; then
  DATA=""; D="--train $NYC_TR --test $NYC_TE --tag _nyc --epochs 30"
  for s in $SEEDS_NYC; do
    for w in connectome shuffled block_shuffled random; do T "${w}_K1_s${s}_nyc" --wiring $w --seed $s $D --lr $LR_CONN; done
    T "gru_K1_s${s}_nyc" --wiring gru --seed $s $D --lr $LR_GRU
  done
fi
$PY scripts/summarize.py > /dev/null && echo SEEDS_DONE
