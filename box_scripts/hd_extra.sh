#!/usr/bin/env bash
# Extended E3 probes: new groups on every checkpoint, MLP probe and longer dark windows on seeds 0-2. Waits for the homing/extra
# session (capacity-matched GRU) before probing the _recmatch checkpoints.
cd /root/World2Brain; export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; PY="uv run --no-sync python"
rm -f results/hd/*_test*.json
GH=$($PY -c "import json; print(json.load(open('results/runs/gru_K1_s0.json'))['args']['gru_hidden'])")
P() { local r=$1 tag=$2; shift 2; [ -f results/hd/$r$tag.json ] && grep -q actions_integrated results/hd/$r$tag.json && return 0
      echo "=== $r $tag $(date +%T)"; $PY scripts/eval_hd.py --run $r --device cuda --match_n $GH --tag "$tag" "$@" 2>&1 | grep -vi warn | grep -E "^blind (all|actions)" || echo "hd failed: $r $tag"; }
run_all() {
  for ck in results/runs/*_K1_s[0-4].pt results/runs/*_K1_s[0-2]_nyc.pt results/runs/*_K1_s[0-2]_big.pt results/runs/*_K1_s[0-2]_recmatch.pt; do
    [ -f "$ck" ] || continue; r=$(basename "$ck" .pt); case "$r" in *_lr*|*_n[0-9]*|*_alpha*|*_gate*|*_syn*|*_no*|*_cxonly*|*realvid*) continue;; esac
    P $r ""
    case "$r" in *_s[0-2]|*_s[0-2]_nyc|*_s[0-2]_big|*_s[0-2]_recmatch) P $r _mlp --probe mlp; P $r _b8_24 --blind 8:24; P $r _b4_24 --blind 4:24;; esac
  done
}
run_all
until ! tmux has-session -t homing 2>/dev/null; do sleep 60; done     # capacity-matched GRU checkpoints land at the end of that session
run_all
$PY scripts/summarize.py > /dev/null && echo "HD_EXTRA_DONE $(date +%T)"
