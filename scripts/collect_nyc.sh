#!/bin/zsh
# Collect NYC digital-twin (Valdrada) navigation episodes with V-JEPA 2.1 latents. Needs scripts/valdrada_server.sh running.
cd /Users/prajaksen/worldmodel2connectom; export PYTHONPATH=.
until lsof -iTCP:2000 -sTCP:LISTEN >/dev/null; do sleep 5; done; sleep 40   # wait for the simulator
F() { grep -v -i -E "warn|redirect|AttentionInterface"; }
uv run python scripts/collect.py --env valdrada --task nav --episodes 120 --seed 0 --save_frames 64 --out data/nyc_train.npz 2>&1 | F
uv run python scripts/collect.py --env valdrada --task nav --episodes 30 --seed 1 --save_frames 64 --out data/nyc_test.npz 2>&1 | F
echo NYC_COLLECT_DONE
