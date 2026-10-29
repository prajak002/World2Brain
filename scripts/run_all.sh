#!/bin/zsh
cd /Users/prajaksen/worldmodel2connectom
export PYTHONPATH=.
until grep -q ALLDONE results/collect.log; do sleep 20; done
F="grep -v -i -E warn|redirect|AttentionInterface"
T() { echo "=== train $@"; uv run python scripts/train.py --epochs 20 "$@" 2>&1 | grep -v -i -E "warn|redirect|AttentionInterface"; }
for w in connectome shuffled random connectome_nosign; do T --wiring $w --seed 0; done
T --wiring gru --gru_hidden 1400 --seed 0
T --wiring dense --seed 0 --lr 1e-3
T --wiring connectome --n_brains 2 --seed 0
T --wiring connectome --n_brains 3 --seed 0
for w in connectome shuffled random; do T --wiring $w --seed 1; done
echo "=== hd probes"
for r in results/runs/*_s0.json results/runs/*_s1.json; do n=$(basename $r .json); uv run python scripts/eval_hd.py --run $n 2>&1 | grep -v -i -E "warn|redirect|AttentionInterface"; done
echo "=== planning"
P() { uv run python scripts/eval_plan.py "$@" 2>&1 | grep -v -i -E "warn|redirect|AttentionInterface" | tail -1; }
for pol in random scripted; do P --run connectome_K1_s0 --task nav --episodes 30 --steps 30 --policy $pol; P --run connectome_K1_s0 --task arrange --episodes 30 --steps 60 --policy $pol; done
for n in connectome_K1_s0 shuffled_K1_s0 random_K1_s0 gru_K1_s0 dense_K1_s0 connectome_K2_s0; do
  P --run $n --task nav --episodes 30 --steps 30 --n_cand 64
  P --run $n --task arrange --episodes 30 --steps 60 --n_cand 64
done
echo PIPELINE_DONE
