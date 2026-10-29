#!/bin/zsh
# Resume run_all.sh: skip any training run whose results/runs/<name>.json already exists.
cd /Users/prajaksen/worldmodel2connectom
export PYTHONPATH=.
FLT() { grep -v -i -E "warn|redirect|AttentionInterface"; }
T() { # usage: T <name> <train args...>
  n=$1; shift
  if [[ -f results/runs/$n.json ]]; then echo "=== skip $n (done)"; return; fi
  echo "=== train $n: $@"; uv run python scripts/train.py --epochs 20 "$@" 2>&1 | FLT
}
T gru_K1_s0 --wiring gru --gru_hidden 1400 --seed 0
T dense_K1_s0 --wiring dense --seed 0 --lr 1e-3
T connectome_K2_s0 --wiring connectome --n_brains 2 --seed 0
T connectome_K3_s0 --wiring connectome --n_brains 3 --seed 0
for w in connectome shuffled random; do T ${w}_K1_s1 --wiring $w --seed 1; done
echo "=== hd probes"
for r in results/runs/*_s0.json results/runs/*_s1.json; do n=$(basename $r .json)
  [[ -f results/hd/$n.json ]] && { echo "skip hd $n"; continue; }
  echo "--- hd $n"; uv run python scripts/eval_hd.py --run $n 2>&1 | FLT; done
echo "=== planning"
P() { uv run python scripts/eval_plan.py "$@" 2>&1 | FLT | tail -1; }
for pol in random scripted; do P --run connectome_K1_s0 --task nav --episodes 30 --steps 30 --policy $pol; P --run connectome_K1_s0 --task arrange --episodes 30 --steps 60 --policy $pol; done
for n in connectome_K1_s0 shuffled_K1_s0 random_K1_s0 gru_K1_s0 dense_K1_s0 connectome_K2_s0; do
  [[ -f results/plan/${n}_nav_mpc.json ]] || P --run $n --task nav --episodes 30 --steps 30 --n_cand 64
  [[ -f results/plan/${n}_arrange_mpc.json ]] || P --run $n --task arrange --episodes 30 --steps 60 --n_cand 64
done
echo PIPELINE_DONE
