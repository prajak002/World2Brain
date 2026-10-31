#!/usr/bin/env bash
# Connectome subgraph ablations (#13): synapse threshold min_syn in {1, 5, 10}; region sets with / without MB and OUT.
# Each ablation rebuilds data/connectome_<tag>.npz and trains the signed connectome student (plus the degree-shuffled
# control, so the wiring-identity gap is measured under every subgraph choice). Seeds via SEEDS.
#   PYTHONPATH=. DEVICE=cuda bash scripts/ablate_connectome.sh
set -u
cd "$(dirname "$0")/.."; export PYTHONPATH=.
DEVICE=${DEVICE:-mps}; EPOCHS=${EPOCHS:-20}; SEEDS=${SEEDS:-"0 1 2"}; PY=${PY:-"uv run python"}
B() { local tag=$1; shift; [ -f "data/connectome_$tag.npz" ] || $PY -m wm2cx.connectome --out "data/connectome_$tag.npz" "$@"; }
T() { local name=$1; shift; [ -f "results/runs/$name.json" ] && { echo "skip $name"; return; }; echo "=== $name"
      $PY scripts/train.py --device "$DEVICE" --epochs "$EPOCHS" "$@" 2>&1 | grep -v -i -E "warn|redirect|AttentionInterface"; }

B syn1  --min_syn 1;   B syn5 --min_syn 5;   B syn10 --min_syn 10
B noMB  --min_syn 5 --regions VIS CX OUT
B noOUT --min_syn 5 --regions VIS CX MB
B cxonly --min_syn 5 --regions VIS CX
for s in $SEEDS; do
  for tag in syn1 syn5 syn10 noMB noOUT cxonly; do
    for w in connectome shuffled; do T "${w}_K1_s${s}_${tag}" --wiring $w --seed $s --conn "data/connectome_$tag.npz" --tag "_$tag"; done
  done
done
$PY scripts/summarize.py > /dev/null && echo ABLATE_DONE
