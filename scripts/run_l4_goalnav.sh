#!/usr/bin/env bash
# Closed-loop goal-vector navigation on the NYC twin for every run-2 NYC checkpoint (#9, #10), bumper controller.
# Each job fits the state readout (training episodes + closed-loop rollouts), then runs EPISODES test episodes.
#   cd ~/World2Brain && nohup bash scripts/run_l4_goalnav.sh > logs/l4_goalnav_$(date +%F).log 2>&1 &
set -u; cd "$(dirname "$0")/.."; mkdir -p results/goalnav logs
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=${PY:-/opt/pytorch/bin/python}; DEVICE=${DEVICE:-cuda}; SEEDS=${SEEDS:-"0 1 2"}; EP=${EPISODES:-30}; FR=${FIT_ROLLOUTS:-100}; PAR=${PAR:-2}
C="--env nyc --controller bumper --episodes $EP --fit_rollouts $FR --device $DEVICE"
jobs=()
J() { local out=$1; shift; [ -f "results/goalnav/$out.json" ] && return 0; jobs+=("$out|$*"); }
# ceiling and floor (no model)
for b in 0 3 5; do J connectome_K1_s0_nyc_nyc_oracle_b${b}_bumper --run connectome_K1_s0_nyc --readout oracle --blackout $b $C
                  J connectome_K1_s0_nyc_nyc_random_b${b}_bumper --run connectome_K1_s0_nyc --readout random --blackout $b $C; done
# state readout, five cores x seeds x blackouts
for s in $SEEDS; do for w in connectome block_shuffled random gru transformer; do r=${w}_K1_s${s}_nyc; [ -f results/runs/$r.pt ] || continue
  for b in 0 3 5; do J ${r}_nyc_state_b${b}_bumper --run $r --readout state --blackout $b $C; done
done; done
# appearance shifts, seed 0, no blackout
for w in connectome block_shuffled gru transformer; do r=${w}_K1_s0_nyc; for app in overcast fog dusk night; do
  J ${r}_nyc_state_b0_bumper_${app} --run $r --readout state --blackout 0 --appearance $app $C; done; done
# non-linear readout, seed 0
for w in connectome gru; do J ${w}_K1_s0_nyc_nyc_mlp_b0_bumper --run ${w}_K1_s0_nyc --readout mlp --blackout 0 $C; done
echo "${#jobs[@]} jobs, $PAR parallel  $(date +%T)"
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -I{} bash -c 'o="${1%%|*}"; args="${1#*|}"; echo "=== $o $(date +%T)"; '"$PY"' scripts/eval_goalnav.py $args > logs/gn_$o.log 2>&1 || echo "failed: $o"; grep "^{" logs/gn_$o.log | tail -1 | cut -c1-200' _ {}
echo "GOALNAV DONE $(date +%T)"
