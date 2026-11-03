#!/usr/bin/env bash
cd /root/World2Brain; export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
rm -f results/homing/*_dbg*.json
for r in connectome_K1_s0 gru_K1_s0 block_shuffled_K1_s0; do uv run --no-sync python scripts/eval_homing.py --run $r --readout state --device cuda --episodes 30 --tag _dbgbwd; done
uv run --no-sync python scripts/eval_homing.py --run connectome_K1_s0 --readout state --device cuda --episodes 30 --backward 0 --tag _dbgfwd
uv run --no-sync python scripts/eval_homing.py --run connectome_K1_s0 --readout mlp --device cuda --episodes 30 --tag _dbgbwd
for r in oracle actions random; do uv run --no-sync python scripts/eval_homing.py --run connectome_K1_s0 --readout $r --device cuda --episodes 30 --tag _dbgbwd; done
for j in results/homing/*_dbg*.json; do python3 -c "import json; d=json.load(open('$j')); print('$j'.split('/')[-1], 'success', d['success'], 'start', round(d['dist_blind_start'],2), 'final', round(d['dist_final'],2), 'eff', round(d['path_efficiency'],2))"; done
echo DBGDONE
