#!/usr/bin/env bash
# NYC benchmark collection at >= 3 locations with a geographic train/val/test split defined before training (#10).
# For each location: (re)start the Valdrada server there, collect EPISODES episodes with teacher latents, stop the
# server. Then scripts/split_geo.py assigns whole 120 m blocks to train / val / test (70/15/15) across all locations.
#   PYTHONPATH=. DEVICE=cuda EPISODES=400 bash scripts/collect_nyc_multi.sh
set -u
cd "$(dirname "$0")/.."; export PYTHONPATH=.
DEVICE=${DEVICE:-mps}; EPISODES=${EPISODES:-300}; QUALITY=${QUALITY:-medium}; PY=${PY:-"uv run python"}
# name lat lon  (Times Square, Flatiron / Madison Sq, Lower Manhattan / FiDi, Upper West Side)
LOCS=${LOCS:-"timessq:40.7580:-73.9855 flatiron:40.7411:-73.9897 fidi:40.7075:-74.0113 uws:40.7870:-73.9754"}
F() { grep --line-buffered -v -i -E "warn|redirect|AttentionInterface"; }
i=0; RAW=()
for L in $LOCS; do
  IFS=: read -r name lat lon <<< "$L"; out="data/nyc_raw_${name}.npz"; RAW+=("$out")
  if [ -f "$out" ]; then echo "skip $name"; i=$((i+1)); continue; fi
  echo "=== $name ($lat, $lon)"
  bash scripts/valdrada_server.sh "$lat" "$lon" 256x256 "$QUALITY" > /dev/null 2>&1 & SRV=$!
  until (echo > /dev/tcp/127.0.0.1/2000) 2>/dev/null; do sleep 5; done; sleep 40
  $PY scripts/collect.py --env valdrada --task nav --episodes "$EPISODES" --seed "$i" --lat "$lat" --lon "$lon" \
      --save_frames 64 --device "$DEVICE" --out "$out" 2>&1 | F
  $PY - "$out" "$i" <<'EOF'
import sys, numpy as np; f, i = sys.argv[1], int(sys.argv[2]); d = dict(np.load(f)); d["loc"] = np.full(len(d["obs"]), i); np.savez_compressed(f, **d)
EOF
  kill $SRV 2>/dev/null; pkill -f "electron . --dev-url" 2>/dev/null; pkill -f "vite --port 5219" 2>/dev/null; sleep 5
  i=$((i+1))
done
$PY scripts/split_geo.py "${RAW[@]}" --out data/nyc --block 120 --margin 40
echo NYC_COLLECT_DONE
