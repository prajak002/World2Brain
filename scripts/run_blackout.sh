#!/usr/bin/env bash
# Blackout Navigation, task T1 Return (docs/BENCHMARK_BLACKOUT.md, part A): success vs blackout length for every wiring.
# Every job writes results/homing/<run>_<env>_<readout>_k<k>[_tn].json and is skipped if that file exists (restartable).
#   DEVICE=cuda SEEDS="0 1 2" bash scripts/run_blackout.sh            # L4: all rows, both worlds, P=2 jobs in parallel
#   DEVICE=mps  SEEDS="0" ENVS=arena WIRINGS="connectome gru random dense" TN=0 bash scripts/run_blackout.sh   # Mac pilot
set -u; cd "$(dirname "$0")/.."; export PYTHONPATH=. OMP_NUM_THREADS=4
PY=${PY:-python}; DEVICE=${DEVICE:-cuda}; SEEDS=${SEEDS:-"0 1 2"}; ENVS=${ENVS:-"arena nyc"}; P=${P:-2}
KS=${KS:-"8 16 24 30 40"}                                   # pre-registered blackout lengths; never change after seeing results
WIRINGS=${WIRINGS:-"connectome block_shuffled random dense gru transformer"}
READOUTS=${READOUTS:-"state"}; EPISODES=${EPISODES:-100}; FIT=${FIT:-300}
TN=${TN:-0.3}; TN_CKPT=${TN_CKPT:-1}                   # continuous turns on the arena; TN_CKPT=1 uses the _tn checkpoints, 0 tests clean-trained ones with turn noise
mkdir -p logs results/homing; LOG=logs/blackout_$(date +%F).log
jobs=()
for env in $ENVS; do
  tn_args=""; sfx=""; ck=""; if [ "$env" = arena ] && [ "$TN" != 0 ]; then tn_args="--turn_noise $TN"; sfx=_tn; [ "$TN_CKPT" = 1 ] && ck=_tn; fi
  nyc=""; [ "$env" = nyc ] && nyc=_nyc
  for k in $KS; do
    for ro in oracle actions random; do                       # ceiling, dead-reckoning floor, random-action floor (no network)
      out="connectome_K1_s0${ck}${nyc}_${env}_${ro}_k${k}${sfx}"
      jobs+=("$out|--run connectome_K1_s0${ck}${nyc} --env $env --readout $ro --t_blind $k $tn_args --episodes $EPISODES --tag _k${k}${sfx}")
    done
    for s in $SEEDS; do for w in $WIRINGS; do for ro in $READOUTS; do
      r="${w}_K1_s${s}${ck}${nyc}"; [ -f results/runs/$r.pt ] || continue
      out="${r}_${env}_${ro}_k${k}${sfx}"
      jobs+=("$out|--run $r --env $env --readout $ro --t_blind $k $tn_args --episodes $EPISODES --fit_rollouts $FIT --tag _k${k}${sfx}")
    done; done; done
  done
done
echo "##### blackout T1: ${#jobs[@]} jobs, device $DEVICE, P=$P  $(date +%T)" | tee -a $LOG
printf '%s\n' "${jobs[@]}" | xargs -P $P -I{} bash -c '
  out="${1%%|*}"; args="${1#*|}"; [ -f results/homing/$out.json ] && exit 0
  echo "=== $out $(date +%T)" >> '"$LOG"'
  '"$PY"' scripts/eval_homing.py $args --device '"$DEVICE"' > logs/bk_$out.log 2>&1 || echo "failed: $out" >> '"$LOG"'
' _ {}
echo "BLACKOUT DONE $(date +%T)" | tee -a $LOG
