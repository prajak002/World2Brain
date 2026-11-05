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
  pull-ckpt)
    rsync -av -e "$SSH" "$R:$RD/results/runs/" results/runs/ ;;
  stop)
    id=$(aws ec2 describe-instances --filters Name=ip-address,Values="$L4_HOST" --query 'Reservations[].Instances[].InstanceId' --output text)
    aws ec2 stop-instances --instance-ids "$id" ;;
  *) sed -n 2,13p "$0"; exit 1 ;;
esac
