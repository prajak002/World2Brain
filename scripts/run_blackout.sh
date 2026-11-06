#!/usr/bin/env bash
# Blackout Navigation (docs/BENCHMARK_BLACKOUT.md): success vs blackout length k for every wiring.
#   T1 Return   (TASKS has t1): eval_homing.py, 12 visible steps then k dark steps back to the start   -> results/homing/*_k<k>*.json
#   T2 Continue (TASKS has t2): eval_goalnav.py, goal visible for 12 steps then k dark steps to reach it -> results/goalnav/*_b<k>_bumper_k<k>*.json
#   T3 Re-entry (TASKS has t3): eval_ckpt.py, open-loop latent error over a 12:12+k blind window (k <= 12 with 24-step episodes) -> results/runs/*_reentry.json
# Note: at k = 8 the oracle itself cannot walk back in time (ceiling ~0.3); the memory-limited region is k >= 16. Always plot the oracle.
# Every job writes results/homing/<run>_<env>_<readout>_k<k>[_tn].json and is skipped if that file exists (restartable).
#   DEVICE=cuda SEEDS="0 1 2" bash scripts/run_blackout.sh            # L4: all rows, both worlds, P=2 jobs in parallel
#   DEVICE=mps  SEEDS="0" ENVS=arena WIRINGS="connectome gru random dense" TN=0 bash scripts/run_blackout.sh   # Mac pilot
set -u; cd "$(dirname "$0")/.."; export PYTHONPATH=. OMP_NUM_THREADS=4
PY=${PY:-python}; DEVICE=${DEVICE:-cuda}; SEEDS=${SEEDS:-"0 1 2"}; ENVS=${ENVS:-"arena nyc"}; P=${P:-2}
KS=${KS:-"8 16 24 30 40"}                                   # pre-registered blackout lengths; never change after seeing results
WIRINGS=${WIRINGS:-"connectome block_shuffled random dense gru transformer"}
READOUTS=${READOUTS:-"state mlp"}; EPISODES=${EPISODES:-100}; FIT=${FIT:-300}
RADIUS_ARENA=${RADIUS_ARENA:-2.0}                           # arena success radius = 2 units (3.3 steps of 0.6), same radius/step ratio as NYC (12 m / 3 m); fixed 2026-11-05 before the L4 sweep
TASKS=${TASKS:-"t1 t2 t3"}
SFX=${SFX:-}                                               # run-name suffix before _tn/_nyc, e.g. _full for the full-brain checkpoints (run_l4_full.sh)
TN=${TN:-0.3}; TN_CKPT=${TN_CKPT:-1}                   # continuous turns on the arena; TN_CKPT=1 uses the _tn checkpoints, 0 tests clean-trained ones with turn noise
mkdir -p logs results/homing; LOG=logs/blackout_$(date +%F).log
jobs=()
for env in $ENVS; do
  tn_args=""; sfx=""; ck=""; if [ "$env" = arena ]; then tn_args="--radius $RADIUS_ARENA"; fi; if [ "$env" = arena ] && [ "$TN" != 0 ]; then tn_args="$tn_args --turn_noise $TN"; sfx=_tn; [ "$TN_CKPT" = 1 ] && ck=_tn; fi
  nyc=""; [ "$env" = nyc ] && nyc=_nyc
  for k in $KS; do
    case " $TASKS " in *" t2 "*)
      T2="--env $env --blackout_at 12 --blackout $k --steps $((12 + k)) --controller bumper --episodes $EPISODES --tag _k${k}"
      for ro in oracle random; do jobs+=("goalnav/connectome_K1_s0${SFX}${nyc}_${env}_${ro}_b${k}_bumper_k${k}|GN|--run connectome_K1_s0${SFX}${nyc} --readout $ro $T2"); done
      for s in $SEEDS; do for w in $WIRINGS; do for ro in $READOUTS; do
        r="${w}_K1_s${s}${SFX}${nyc}"; [ -f results/runs/$r.pt ] || continue
        jobs+=("goalnav/${r}_${env}_${ro}_b${k}_bumper_k${k}|GN|--run $r --readout $ro --fit_rollouts $FIT $T2")
      done; done; done;;
    esac
    case " $TASKS " in *" t1 "*) ;; *) continue;; esac
    for ro in oracle actions random; do                       # ceiling, dead-reckoning floor, random-action floor (no network)
      out="connectome_K1_s0${SFX}${ck}${nyc}_${env}_${ro}_k${k}${sfx}"
      jobs+=("homing/$out|HM|--run connectome_K1_s0${SFX}${ck}${nyc} --env $env --readout $ro --t_blind $k $tn_args --episodes $EPISODES --tag _k${k}${sfx}")
    done
    for s in $SEEDS; do for w in $WIRINGS; do for ro in $READOUTS; do
      r="${w}_K1_s${s}${SFX}${ck}${nyc}"; [ -f results/runs/$r.pt ] || continue
      out="${r}_${env}_${ro}_k${k}${sfx}"
      jobs+=("homing/$out|HM|--run $r --env $env --readout $ro --t_blind $k $tn_args --episodes $EPISODES --fit_rollouts $FIT --tag _k${k}${sfx}")
    done; done; done
  done
  case " $TASKS " in *" t3 "*)                                 # open-loop re-entry: blind windows 12:20 and 12:24 on the test episodes
    test=$([ "$env" = nyc ] && echo data/nyc_test.npz || echo data/nav_test.npz)
    for s in $SEEDS; do for w in $WIRINGS; do
      r="${w}_K1_s${s}${SFX}${ck}${nyc}"; [ -f results/runs/$r.pt ] || continue
      jobs+=("runs/${r}_reentry|CK|--run $r --test $test --eval_windows 12:20 12:24 --baselines 1 --tag _reentry")
    done; done;;
  esac
done
echo "##### blackout [$TASKS]: ${#jobs[@]} jobs, device $DEVICE, P=$P  $(date +%T)" | tee -a $LOG
printf '%s\n' "${jobs[@]}" | xargs -P $P -I{} bash -c '
  out="${1%%|*}"; rest="${1#*|}"; kind="${rest%%|*}"; args="${rest#*|}"; [ -f results/$out.json ] && exit 0
  case $kind in HM) sc=scripts/eval_homing.py;; GN) sc=scripts/eval_goalnav.py;; CK) sc=scripts/eval_ckpt.py;; esac
  echo "=== $out $(date +%T)" >> '"$LOG"'
  '"$PY"' $sc $args --device '"$DEVICE"' > logs/bk_$(basename $out).log 2>&1 || echo "failed: $out" >> '"$LOG"'
' _ {}
echo "BLACKOUT DONE $(date +%T)" | tee -a $LOG
