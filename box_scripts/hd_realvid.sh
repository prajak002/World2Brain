#!/usr/bin/env bash
# Real-video heading / position probes: (a) transfer, readout fit in the simulator and read out on the phone clips;
# (b) in-domain, fit on the first six clips and tested on the last two. Runs after the hdx sweep.
cd /root/World2Brain; export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; PY="uv run --no-sync python"
until ! tmux has-session -t hdx 2>/dev/null; do sleep 60; done
GH=$($PY -c "import json; print(json.load(open('results/runs/gru_K1_s0.json'))['args']['gru_hidden'])")
RV=data/realvid_all.npz
for w in connectome block_shuffled shuffled random dense gru connectome_nosign; do for s in 0 1 2; do
  r=${w}_K1_s$s; [ -f results/runs/$r.pt ] || continue
  for cfg in "_rv_transfer --test $RV" "_rv_indomain --test $RV --train $RV --split_by_clip 2"; do
    set -- $cfg; tag=$1; shift
    [ -f results/hd/$r$tag.json ] && continue
    echo "=== $r $tag $(date +%T)"; $PY scripts/eval_hd.py --run $r --device cuda --match_n $GH --tag $tag "$@" 2>&1 | grep -vi warn | grep -E "^blind (all|actions)|split" || echo "hd failed: $r $tag"
  done
done; done
$PY scripts/summarize.py > /dev/null && echo "HD_REALVID_DONE $(date +%T)"
