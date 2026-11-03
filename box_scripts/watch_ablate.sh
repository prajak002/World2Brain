#!/usr/bin/env bash
# On-box watcher: when the main pipeline reaches the ablate stage, restart it with one seed followed by the 79-episode real-video stage.
cd /root/World2Brain
f=$(ls -t logs/vast_*.log | head -1)
until grep -q "##### ablate" "$f" || ! tmux has-session -t wm2cx 2>/dev/null; do sleep 30; done
tmux kill-session -t wm2cx 2>/dev/null; pkill -f "[s]cripts/train.py --wiring" ; sleep 3
for j in results/runs/*syn*_K1_s0.json results/runs/*noMB*_K1_s0.json results/runs/*noOUT*_K1_s0.json results/runs/*cxonly*_K1_s0.json; do [ -f "$j" ] && ! grep -q test_mse_per_ep "$j" && rm -f "$j"; done
git fetch -q origin main && git checkout -q main 2>/dev/null; git pull -q origin main
tmux new -d -s wm2cx "SKIP_SYNC=1 PY=\"uv run --no-sync python\" DEVICE=cuda SEEDS=0 STAGES=\"ablate realvid\" bash scripts/aws_h100.sh 2>&1 | tee logs/vast_\$(date +%F_%H%M).log"
echo "relaunched $(date)" >> /root/watch_ablate.log
