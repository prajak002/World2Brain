#!/usr/bin/env bash
# Drive run 2 (scripts/run_l4.sh + scripts/run_l4_realvid.sh) on the AWS g6.xlarge L4 box from the laptop.
# Same shape as scripts/vast_launch.sh. The box is addressed by L4_HOST (public IP or DNS) and L4_KEY (the .pem);
# L4_USER defaults to ubuntu. Every stage on the box is restartable (skips work whose output JSON exists).
#
#   export L4_HOST=<ip> L4_KEY=~/.ssh/<key>.pem
#   bash scripts/aws_l4_launch.sh push                       # data/*.npz, connectome, realvid -> box (rsync, resumable)
#   bash scripts/aws_l4_launch.sh run "data trivial lr arena tn nyc hd homing summary"   # main run in tmux, realvid queued behind it
#   bash scripts/aws_l4_launch.sh log                        # tail both logs + GPU utilisation
#   bash scripts/aws_l4_launch.sh pull                       # results/{runs,hd,homing} JSON + summary.md -> laptop (no .pt)
#   bash scripts/aws_l4_launch.sh pull-ckpt                  # checkpoints too
#   bash scripts/aws_l4_launch.sh stop                       # stop the instance (aws cli, profile $AWS_PROFILE)
# Full brain on any other ssh GPU box (e.g. a borrowed H200). The causal arena/arrange, *_tn and nyc_twin_* episodes
# exist only on the L4 (the laptop's data/*.npz are the legacy non-causal run-1 files), so stage them via the laptop:
#   bash scripts/aws_l4_launch.sh pull-data                  # (L4_HOST = L4) box data/*.npz -> data/box/, *_full runs -> results/runs/
#   L4_HOST=<h200> L4_USER=<user> L4_KEY=<key> bash scripts/aws_l4_launch.sh push-full   # data/box + full graph (+ *_full runs) -> box
#   L4_HOST=<h200> ... JOBS=6 P=6 OMP=8 bash scripts/aws_l4_launch.sh run-full "arena tn nyc hd homing blackout"
#   L4_HOST=<h200> ... bash scripts/aws_l4_launch.sh log-full      # then pull / pull-ckpt as usual
set -euo pipefail
cd "$(dirname "$0")/.."
: "${L4_HOST:?set L4_HOST}"; : "${L4_KEY:?set L4_KEY}"; U=${L4_USER:-ubuntu}
SSH="ssh -i $L4_KEY -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
R="$U@$L4_HOST"; RD=/home/$U/World2Brain
case "${1:-}" in
  push)
    $SSH "$R" "[ -d $RD ] || git clone -q https://github.com/prajak002/World2Brain.git $RD; mkdir -p $RD/data/raw $RD/results/runs $RD/logs; cd $RD && git pull -q"
    rsync -av --progress -e "$SSH" data/connectome.npz data/*_train.npz data/*_test.npz "$R:$RD/data/"
    ls data/realvid_all.npz >/dev/null 2>&1 && rsync -av --progress -e "$SSH" data/realvid_all.npz "$R:$RD/data/"
    [ -d data/nyc ] && rsync -av -e "$SSH" data/nyc/ "$R:$RD/data/nyc/"
    $SSH "$R" "cd $RD && ls -la data | head -30 && nvidia-smi -L" ;;
  run)
    stages=${2:-"data trivial lr arena tn nyc hd homing summary"}
    $SSH "$R" "cd $RD && git pull -q && mkdir -p logs && tmux has-session -t l4 2>/dev/null && { echo 'already running (tmux l4)'; exit 0; }
      tmux new -d -s l4 \"STAGES='$stages' bash scripts/run_l4.sh > logs/l4_\$(date +%F).log 2>&1; bash scripts/run_l4_realvid.sh > logs/l4_realvid_\$(date +%F).log 2>&1\"
      echo started; tmux ls" ;;
  log)
    $SSH "$R" "cd $RD && tail -n 30 logs/l4_*.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; ls results/runs/*.json | wc -l" ;;
  pull)
    rsync -av -e "$SSH" --include='*/' --include='*.json' --include='*.md' --include='*.png' --exclude='*' "$R:$RD/results/" results/
    rsync -av -e "$SSH" "$R:$RD/logs/" logs/ ;;
  pull-data)
    mkdir -p data/box results/runs
    rsync -av --progress -e "$SSH" --include='*.npz' --exclude='*' "$R:$RD/data/" data/box/
    rsync -av --progress -e "$SSH" --include='*_full*' --exclude='*' "$R:$RD/results/runs/" results/runs/ ;;
  push-full)
    for f in data/box/nav_train.npz data/box/nyc_twin_train.npz data/connectome_full.npz; do
      [ -f "$f" ] || { echo "missing $f (run pull-data against the L4 first; build the graph with STAGES=graph)"; exit 1; }; done
    $SSH "$R" "[ -d $RD ] || git clone -q https://github.com/prajak002/World2Brain.git $RD; mkdir -p $RD/data $RD/results/runs $RD/logs; cd $RD && git pull -q"
    rsync -av --progress -e "$SSH" data/box/*.npz data/connectome_full.npz "$R:$RD/data/"
    [ -d data/nyc ] && rsync -av -e "$SSH" data/nyc/ "$R:$RD/data/nyc/"
    if ls results/runs/*_full* >/dev/null 2>&1; then rsync -av -e "$SSH" results/runs/*_full* "$R:$RD/results/runs/"; fi
    $SSH "$R" "cd $RD && ls -la data && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | head -2" ;;
  run-full)   # DLAMI venv if present, else the project's uv env (uv installed with pip, torch from uv.lock)
    stages=${2:-"arena tn nyc hd homing blackout"}
    $SSH "$R" "cd $RD && git pull -q && mkdir -p logs && tmux has-session -t full 2>/dev/null && { echo 'already running (tmux full)'; exit 0; }
      export PATH=\$HOME/.local/bin:\$PATH
      if [ ! -x /opt/pytorch/bin/python ]; then command -v uv >/dev/null || python3 -m pip install --user -q uv; uv sync --quiet || exit 1
        uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))' || exit 1; fi
      tmux new -d -s full \"export PATH=\$HOME/.local/bin:\$PATH; JOBS=${JOBS:-6} P=${P:-6} OMP=${OMP:-8} STAGES='$stages' bash scripts/run_l4_full.sh > logs/full_\$(date +%F).log 2>&1\"
      echo started; tmux ls" ;;
  log-full)
    $SSH "$R" "cd $RD && tail -n 30 logs/full_*.log; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; ls results/runs/*_full*.json 2>/dev/null | wc -l" ;;
  pull-ckpt)
    rsync -av -e "$SSH" "$R:$RD/results/runs/" results/runs/ ;;
  stop)
    id=$(aws ec2 describe-instances --filters Name=ip-address,Values="$L4_HOST" --query 'Reservations[].Instances[].InstanceId' --output text)
    aws ec2 stop-instances --instance-ids "$id" ;;
  *) sed -n 2,18p "$0"; exit 1 ;;
esac
