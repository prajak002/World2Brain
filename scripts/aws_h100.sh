#!/usr/bin/env bash
# One-shot run order for a fresh GPU box (tested target: Ubuntu 22.04 + CUDA 12, one H100/A100/L4; everything below
# fits on one GPU, the student is 7-22 M parameters). Run it inside tmux/nohup; it is restartable: every stage skips
# work whose output JSON already exists.
#
#   git clone https://github.com/prajak002/World2Brain.git && cd World2Brain
#   # copy data/ from the M1 (connectome.npz, nav_*/arrange_*/nyc_* npz; checkpoints optional):
#   #   rsync -av m1:~/worldmodel2connectom/data/ data/
#   STAGES="seeds hd plan bench ablate" bash scripts/aws_h100.sh 2>&1 | tee logs/aws_$(date +%F).log
#
# Rough budget (H100, 20 epochs, 1,000 arena episodes): one K=1 run is minutes, not hours; the full seed sweep is
# 35 arena runs + 12 NYC runs + 36 data-efficiency runs. The slow parts are NYC *collection* (Valdrada, not in this
# script) and the connectome ablation (18 runs incl. min_syn=1 graphs with ~5x more edges).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p logs results/runs
STAGES=${STAGES:-"seeds hd plan bench ablate"}
export PYTHONPATH=. DEVICE=${DEVICE:-cuda} PY=${PY:-"uv run python"}

# ---- environment -----------------------------------------------------------------------------------------------
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --quiet
$PY - <<'EOF'
import torch, sys
ok = torch.cuda.is_available(); print("torch", torch.__version__, "cuda", ok, torch.cuda.get_device_name(0) if ok else "")
sys.exit(0 if ok else 1)
EOF
# fast kernels: TF32 matmuls are fine for the student (edge-list index_add is fp32 either way)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# ---- data -----------------------------------------------------------------------------------------------------
for f in data/connectome.npz data/nav_train.npz data/arrange_train.npz data/nav_test.npz data/arrange_test.npz; do
  [ -f "$f" ] || { echo "missing $f: copy data/ from the M1, or collect the arena here:"; echo "  bash scripts/collect_all.sh  (needs the V-JEPA 2.1 ViT-B weights; ~1,200 episodes)"; exit 1; }
done
[ -f data/nyc/midtown_buildings.json ] || bash scripts/fetch_nyc_data.sh         # raycast twin inputs (public, ~1 MB)
$PY -m pytest tests -q

# ---- stages ---------------------------------------------------------------------------------------------------
stage() { case " $STAGES " in *" $1 "*) echo; echo "##### $1  $(date +%T)"; return 0;; *) return 1;; esac; }

stage seeds && EPOCHS=${EPOCHS:-20} bash scripts/run_seeds.sh                    # #8 + data-efficiency axis (N_TRAIN)

if stage hd; then                                                               # #7: actions-only floor + feature-matched subset
  for ck in results/runs/*_K1_s[0-4].pt results/runs/*_K1_s[0-2]_nyc.pt; do
    [ -f "$ck" ] || continue; r=$(basename "$ck" .pt); d=""; case "$r" in *_nyc) d="--data data/nyc_test.npz";; esac
    for b in "" "--blind"; do $PY scripts/eval_hd.py --run "$r" --device "$DEVICE" $d $b --match_n 1400; done
  done
fi

if stage plan; then                                                             # #9: the full ladder on the arena, then the twin
  for dyn in oracle render_teacher render_student student; do
    for task in nav arrange; do
      $PY scripts/eval_plan.py --run connectome_K1_s0 --task $task --dynamics $dyn --device "$DEVICE" --episodes 30 --horizon 8 --iters 3 --n_cand 64 --goal_mode pose
    done
  done
  for w in connectome shuffled gru; do $PY scripts/eval_plan.py --run ${w}_K1_s0 --task nav --device "$DEVICE" --episodes 30 --horizon 8 --iters 3 --n_cand 64 --goal_mode pose; done
  if [ -f results/runs/connectome_K1_s0_nyc.pt ]; then
    for B in 0 1 3 5; do $PY scripts/eval_plan.py --run connectome_K1_s0_nyc --env nyc --dynamics student --device "$DEVICE" --episodes 30 --blackout $B --blackout_trigger fixed --tag _twin; done
    for app in overcast fog dusk night; do $PY scripts/eval_plan.py --run connectome_K1_s0_nyc --env nyc --dynamics student --device "$DEVICE" --episodes 30 --appearance $app --tag _twin; done
  fi
fi

stage bench && $PY scripts/bench_latency.py --device "$DEVICE" --conn data/connectome.npz   # Table 3 on the real graph, GPU

stage ablate && bash scripts/ablate_connectome.sh                                # #13: min_syn x region set x wiring, 3 seeds

$PY scripts/summarize.py > /dev/null; echo; echo "ALL STAGES DONE $(date +%T): results/summary.md"
echo "then on the laptop: git pull && PYTHONPATH=. python scripts/make_video_issues.py"
