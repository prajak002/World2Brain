#!/bin/zsh
# Train + probe students on the NYC digital-twin data; waits for the training set file to exist.
cd /Users/prajaksen/worldmodel2connectom; export PYTHONPATH=.
until [[ -f data/nyc_train.npz ]]; do sleep 30; done; sleep 20
F() { grep --line-buffered -v -i -E "warn|redirect|AttentionInterface"; }
D=(--train data/nyc_train.npz --test data/nyc_test.npz --tag _nyc --epochs 30)
for w in connectome shuffled random; do echo "=== train $w nyc"; uv run python -u scripts/train.py --wiring $w --seed 0 $D 2>&1 | F; done
echo "=== train gru nyc"; uv run python -u scripts/train.py --wiring gru --gru_hidden 1400 --seed 0 $D 2>&1 | F
echo "=== hd probes"
for n in connectome shuffled random gru; do echo "--- hd ${n}_K1_s0_nyc"; uv run python -u scripts/eval_hd.py --run ${n}_K1_s0_nyc 2>&1 | F; done
echo NYC_PIPELINE_DONE
