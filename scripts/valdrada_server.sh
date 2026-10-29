#!/bin/zsh
# Start the Valdrada NYC digital twin as a headless simulation server (Vite client + Electron host, TCP API on :2000).
# usage: scripts/valdrada_server.sh [lat lon] [res]   (default: Times Square, 256x256)
set -e
ROOT=/Users/prajaksen/worldmodel2connectom; V=$ROOT/data/raw/valdrada; C=$ROOT/data/raw/valdrada_content/Content
LAT=${1:-40.7580}; LON=${2:--73.9855}; RES=${3:-256x256}; QUALITY=${4:-medium}; PEDS=${5:-procedural}
for b in tiles models textures; do [[ -e $V/client/public/$b ]] || ln -s $C/$b $V/client/public/$b; done
mkdir -p $ROOT/results/valdrada
(cd $V/client && npx vite --port 5219 --strictPort > $ROOT/results/valdrada/vite.log 2>&1 &)
until curl -s http://127.0.0.1:5219 >/dev/null; do sleep 1; done
cd $V/server
while true; do   # restart the simulator if it ever exits (GPU/Chromium crashes leave no log line)
  echo "$(date -u +%FT%TZ) [launcher] starting electron" >> $ROOT/results/valdrada/server.log
  npx electron . --dev-url=http://127.0.0.1:5219 --headless --res $RES --port 2000 \
    --start-lat $LAT --start-lon $LON --quality $QUALITY --pedestrians $PEDS --log $ROOT/results/valdrada/server.log
  echo "$(date -u +%FT%TZ) [launcher] electron exited with $?" >> $ROOT/results/valdrada/server.log; sleep 5
done
