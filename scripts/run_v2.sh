#!/bin/zsh
# Rerun of every experiment with the issue fixes (#1-#9, #11): causal teacher targets, exact edge-matched Dale-consistent
# controls, VIS-only W_in, parameter-matched GRU with a shared blackout flag, 5 seeds, trivial baselines, bootstrap CIs,
# action-only heading baselines, planner oracles, latency table. Resumable: finished runs are skipped.
# Expected wall-clock on an M1 16 GB: data ~12 h (teacher, 8x more clips than before), training ~5 min/run x 45 runs.
cd "$(dirname "$0")/.." || exit 1; export PYTHONPATH=.
F() { grep --line-buffered -v -i -E "warn|redirect|AttentionInterface"; }
SEEDS=(0 1 2 3 4); WIRINGS=(connectome connectome_nosign connectome_randsign shuffled block_shuffled random dense gru)
echo "=== 1. causal data (issue #5)"
for t in nav arrange; do
  [[ -f data/${t}_train_causal.npz ]] || uv run python -u scripts/collect.py --task $t --episodes 1000 --seed 0 --out data/${t}_train_causal.npz 2>&1 | F
  [[ -f data/${t}_test_causal.npz ]]  || uv run python -u scripts/collect.py --task $t --episodes 200 --seed 1 --out data/${t}_test_causal.npz 2>&1 | F
done
D=(--train data/nav_train_causal.npz data/arrange_train_causal.npz --test data/nav_test_causal.npz data/arrange_test_causal.npz --epochs 20 --tag _v2)
# every v2 run carries the _v2 tag so legacy (non-causal) results in results/runs are never mistaken for finished v2 runs
echo "=== 2. lr sweep on a validation split (issue #4), seed 0, 10 epochs"
for w in connectome gru dense; do for lr in 5e-4 1e-3 2e-3 4e-3; do
  [[ -f results/runs/${w}_K1_s0_v2_lr${lr}.json ]] || uv run python -u scripts/train.py --wiring $w --lr $lr --val 0.15 --epochs 10 ${D[@]:0:6} --tag _v2_lr${lr} 2>&1 | F
done; done
echo "=== 3. main runs, 5 seeds (issues #1, #2, #3, #8)"
for s in $SEEDS; do for w in $WIRINGS; do
  [[ -f results/runs/${w}_K1_s${s}_v2.json ]] || uv run python -u scripts/train.py --wiring $w --seed $s $D 2>&1 | F
done; done
echo "=== 4. multi-brain K=2,3 + fixed-budget single-brain control (issue #11)"
for K in 2 3; do [[ -f results/runs/connectome_K${K}_s0_v2.json ]] || uv run python -u scripts/train.py --wiring connectome --n_brains $K --seed 0 $D 2>&1 | F; done
[[ -f results/runs/gru_K1_s0_v2_budgetK2.json ]] || uv run python -u scripts/train.py --wiring gru --gru_hidden 1200 --seed 0 ${D[@]:0:7} --tag _v2_budgetK2 2>&1 | F
echo "=== 5. heading probes with action-only baselines (issue #7)"
for r in results/runs/*_K?_s?_v2.json; do n=$(basename $r .json); [[ -f results/hd/$n.json ]] || uv run python -u scripts/eval_hd.py --run $n 2>&1 | F; done
echo "=== 6. planner: oracles first, then students (issue #9)"
P() { uv run python -u scripts/eval_plan.py "$@" 2>&1 | F | tail -1; }
for task in nav arrange; do
  P --run connectome_K1_s0_v2 --task $task --episodes 30 --policy random; P --run connectome_K1_s0_v2 --task $task --episodes 30 --policy scripted
  P --run connectome_K1_s0_v2 --task $task --episodes 30 --dynamics oracle_state --n_cand 64 --iters 1
  P --run connectome_K1_s0_v2 --task $task --episodes 30 --dynamics oracle_render --n_cand 32
  for w in connectome shuffled block_shuffled random gru; do P --run ${w}_K1_s0_v2 --task $task --episodes 30 --n_cand 64 --iters 3; done
done
echo "=== 7. latency / FLOPs / memory (issue #11)"
uv run python scripts/bench.py --device cpu 2>&1 | F; uv run python scripts/bench.py --device mps 2>&1 | F
uv run python scripts/summarize.py > /dev/null; echo V2_PIPELINE_DONE
