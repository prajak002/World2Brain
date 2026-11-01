#!/usr/bin/env bash
# Rent one GPU on Vast.ai, push data/, run scripts/aws_h100.sh in tmux, pull results/ back. Budget-minded defaults:
# one RTX 4090 (24 GB) at <= $0.45/h is enough for every stage (student 7-22 M params, teacher ViT-B); the full seed
# sweep + data-efficiency axis + ablations is roughly 10-15 GPU-hours, so $17 of credit covers it with margin.
#
#   pip install --upgrade vastai && vastai set api-key $VAST_API_KEY
#   bash scripts/vast_launch.sh search                      # list the cheapest matching offers
#   bash scripts/vast_launch.sh create <OFFER_ID>           # rent it (prints the instance id)
#   bash scripts/vast_launch.sh push <INSTANCE_ID>          # data/*.npz (+ raw FlyWire files, realvid) -> instance
#   bash scripts/vast_launch.sh run <INSTANCE_ID> "data lr seeds hd plan bench ablate realvid"
#   bash scripts/vast_launch.sh log <INSTANCE_ID>           # tail the log
#   bash scripts/vast_launch.sh pull <INSTANCE_ID>          # results/ (JSON, md, png; checkpoints too) -> laptop
#   bash scripts/vast_launch.sh destroy <INSTANCE_ID>       # stop paying
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-"pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"}   # any CUDA 12 image works: uv sync installs the project's own torch
DISK=${DISK:-100}; MAX_DPH=${MAX_DPH:-0.45}
GPUS=${GPUS:-"RTX_4090 RTX_5090 RTX_A6000 L40S A100_PCIE A100_SXM4"}
QUERY="reliability > 0.98 num_gpus=1 gpu_ram >= 20 disk_space >= $DISK inet_down > 200 dph <= $MAX_DPH rentable=true cuda_vers >= 12.1 direct_port_count > 2 gpu_name in [$(echo $GPUS | sed 's/ /, /g')]"
cmd=${1:-search}; id=${2:-}

ssh_target() {  # host port
  vastai ssh-url "$1" | sed -E 's#ssh://root@([^:]+):([0-9]+)#\1 \2#'
}

case "$cmd" in
  search)
    vastai search offers "$QUERY" -o 'dph' | head -15 ;;
  create)
    [ -n "$id" ] || { echo "offer id required"; exit 1; }
    vastai create instance "$id" --image "$IMAGE" --disk "$DISK" --ssh --direct --label wm2cx \
      --onstart-cmd 'apt-get update -qq && apt-get install -y -qq git tmux ffmpeg rsync > /dev/null; curl -LsSf https://astral.sh/uv/install.sh | sh; cd /root && [ -d World2Brain ] || git clone https://github.com/prajak002/World2Brain.git; touch /root/READY'
    echo "then: vastai show instances   (wait for status running), and push" ;;
  push)
    read -r host port < <(ssh_target "$id"); echo "-> root@$host:$port"
    ssh -o StrictHostKeyChecking=no -p "$port" root@"$host" 'until [ -f /root/READY ]; do sleep 5; done; mkdir -p /root/World2Brain/data/raw /root/World2Brain/results/runs'
    # episodes, connectome and (optionally) FlyWire raw files, real-video ingest, existing checkpoints
    scp -o StrictHostKeyChecking=no -P "$port" data/*.npz root@"$host":/root/World2Brain/data/ 2>/dev/null || true
    ls data/raw/* >/dev/null 2>&1 && scp -o StrictHostKeyChecking=no -P "$port" data/raw/* root@"$host":/root/World2Brain/data/raw/
    ls results/runs/*.pt >/dev/null 2>&1 && scp -o StrictHostKeyChecking=no -P "$port" results/runs/*.pt root@"$host":/root/World2Brain/results/runs/
    ssh -p "$port" root@"$host" 'cd /root/World2Brain && git pull -q && ls -la data | head -30' ;;
  run)
    read -r host port < <(ssh_target "$id"); stages=${3:-"data lr seeds hd plan bench ablate realvid"}
    ssh -p "$port" root@"$host" "cd /root/World2Brain && git pull -q && mkdir -p logs && tmux new -d -s wm2cx \"STAGES='$stages' PY='\$HOME/.local/bin/uv run python' bash scripts/aws_h100.sh 2>&1 | tee logs/vast_\$(date +%F_%H%M).log\" && echo started; tmux ls" ;;
  log)
    read -r host port < <(ssh_target "$id"); ssh -p "$port" root@"$host" 'tail -n 40 /root/World2Brain/logs/vast_*.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' ;;
  pull)
    read -r host port < <(ssh_target "$id"); mkdir -p results
    scp -r -P "$port" root@"$host":/root/World2Brain/results/runs root@"$host":/root/World2Brain/results/hd root@"$host":/root/World2Brain/results/plan \
        root@"$host":/root/World2Brain/results/baselines root@"$host":/root/World2Brain/results/bench root@"$host":/root/World2Brain/results/summary.md results/ 2>/dev/null || true
    scp -P "$port" root@"$host":/root/World2Brain/logs/*.log logs/ 2>/dev/null || true
    echo "pulled; now: PYTHONPATH=. python scripts/summarize.py" ;;
  destroy)
    vastai destroy instance "$id" ;;
  *) echo "usage: $0 search|create <offer>|push <inst>|run <inst> [stages]|log <inst>|pull <inst>|destroy <inst>"; exit 1 ;;
esac
