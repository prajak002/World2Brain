#!/usr/bin/env bash
cd /root/World2Brain; export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; PY="uv run --no-sync python"
for s in 0 1 2; do r=gru_K1_s${s}_recmatch; [ -f results/runs/$r.pt ] || continue
  [ -f results/homing/${r}_arena_state.json ] || $PY scripts/eval_homing.py --run $r --readout state --device cuda 2>&1 | grep -vi warn | tail -1
done
[ -f results/homing/gru_K1_s0_recmatch_arena_mlp.json ] || $PY scripts/eval_homing.py --run gru_K1_s0_recmatch --readout mlp --device cuda 2>&1 | grep -vi warn | tail -1
echo HOMING_RECMATCH_DONE
