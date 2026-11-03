  H() { local out=$1; shift; [ -f "results/homing/$out.json" ] && return 0; $PY scripts/eval_homing.py "$@" --device "$DEVICE" || echo "homing failed: $out"; }
  H connectome_K1_s0_arena_oracle --run connectome_K1_s0 --readout oracle;  H connectome_K1_s0_arena_actions --run connectome_K1_s0 --readout actions;  H connectome_K1_s0_arena_random --run connectome_K1_s0 --readout random
  for s in 0 1 2; do for w in connectome block_shuffled shuffled random connectome_nosign dense gru; do
    [ -f results/runs/${w}_K1_s${s}.pt ] && H ${w}_K1_s${s}_arena_state --run ${w}_K1_s${s} --readout state
  done; done
  for w in connectome block_shuffled gru; do H ${w}_K1_s0_arena_mlp --run ${w}_K1_s0 --readout mlp; done
  H connectome_K1_s0_arena_state_tn --run connectome_K1_s0 --readout state --turn_noise 0.3 --tag _tn; H connectome_K1_s0_arena_actions_tn --run connectome_K1_s0 --readout actions --turn_noise 0.3 --tag _tn
  H gru_K1_s0_arena_state_tn --run gru_K1_s0 --readout state --turn_noise 0.3 --tag _tn
  if [ -f results/runs/connectome_K1_s0_nyc.pt ]; then
    H connectome_K1_s0_nyc_nyc_oracle --run connectome_K1_s0_nyc --env nyc --readout oracle --t_blind 30; H connectome_K1_s0_nyc_nyc_actions --run connectome_K1_s0_nyc --env nyc --readout actions --t_blind 30
    for w in connectome block_shuffled gru; do for s in 0 1 2; do [ -f results/runs/${w}_K1_s${s}_nyc.pt ] && H ${w}_K1_s${s}_nyc_nyc_state --run ${w}_K1_s${s}_nyc --env nyc --readout state --t_blind 30; done; done
  fi
  for s in 0 1 2; do [ -f results/runs/gru_K1_s${s}_recmatch.json ] || $PY scripts/train.py --wiring gru --gru_hidden 174 --seed $s --epochs 20 --device "$DEVICE" --lr $LR_GRU --tag _recmatch; done
