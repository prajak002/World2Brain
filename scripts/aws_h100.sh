#!/usr/bin/env bash
# One-shot run order for a fresh GPU box (tested target: Ubuntu 22.04 + CUDA 12, one H100/A100/L4; everything below
# fits on one GPU, the student is 7-22 M parameters). Run it inside tmux/nohup; it is restartable: every stage skips
# work whose output JSON already exists.
#
#   git clone https://github.com/prajak002/World2Brain.git && cd World2Brain
#   # copy data/ from the M1 (connectome.npz, nav_*/arrange_*/nyc_* npz; checkpoints optional):
#   #   rsync -av m1:~/worldmodel2connectom/data/ data/
#   STAGES="data lr seeds hd plan bench ablate realvid" bash scripts/aws_h100.sh 2>&1 | tee logs/aws_$(date +%F).log
#   (or from the laptop: scripts/vast_launch.sh, which rents a 4090 on Vast.ai and drives this script over ssh)
#
# Rough budget (H100, 20 epochs, 1,000 arena episodes): one K=1 run is minutes, not hours; the full seed sweep is
# 35 arena runs + 12 NYC runs + 36 data-efficiency runs. The slow parts are NYC *collection* (Valdrada, not in this
# script) and the connectome ablation (18 runs incl. min_syn=1 graphs with ~5x more edges).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p logs results/runs
STAGES=${STAGES:-"data lr seeds hd plan bench arch ablate realvid"}
export PYTHONPATH=. DEVICE=${DEVICE:-cuda} PY=${PY:-"uv run python"}
stage() { case " $STAGES " in *" $1 "*) echo; echo "##### $1  $(date +%T)"; return 0;; *) return 1;; esac; }

# ---- environment -----------------------------------------------------------------------------------------------
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ "${SKIP_SYNC:-0}" = 1 ] || uv sync --quiet
cuda_ok() { $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.zeros(1).cuda().item()==0 else 1)" > /dev/null 2>&1; }
if ! cuda_ok; then   # PyPI torch may be built for a newer CUDA than the host driver: fall back to the cu128 wheels (driver >= 570)
  echo "torch cannot init CUDA on this driver; installing cu128 wheels"; uv pip install --quiet --reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision
  export PY="${PY/uv run/uv run --no-sync}"
fi
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
cuda_ok || { echo "no usable CUDA"; exit 1; }
# fast kernels: TF32 matmuls are fine for the student (edge-list index_add is fp32 either way)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
# BLAS threads = real CPU quota (cgroup), not the host core count: oversubscription made numpy SVD hang on Vast
Q=$(cut -d" " -f1 /sys/fs/cgroup/cpu.max 2>/dev/null || true); [ -z "$Q" ] && Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || true) || true   # cgroup v2, then v1
if [ -n "$Q" ] && [ "$Q" != max ] && [ "$Q" -gt 0 ]; then CPUS=$((Q / 100000)); else CPUS=$(nproc); fi; CPUS=$(( CPUS < 1 ? 1 : (CPUS > 16 ? 16 : CPUS) ))
export OMP_NUM_THREADS=$CPUS OPENBLAS_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS; echo "BLAS threads $CPUS"

# ---- data -----------------------------------------------------------------------------------------------------
[ -f data/connectome.npz ] || { [ -f data/raw/proofread_connections_783.feather ] && $PY -m wm2cx.connectome --out data/connectome.npz; }   # rebuild from the FlyWire raw files if only those were copied
[ -f data/connectome.npz ] || { echo "missing data/connectome.npz (or data/raw/*.feather to rebuild it)"; exit 1; }
if stage data; then                                                             # #5: causal teacher targets; the Kaggle npz are the legacy non-causal run-1 data
  for t in nav arrange; do
    if [ -f data/${t}_train.npz ] && ! $PY -c "import numpy as np,json,sys; d=np.load('data/${t}_train.npz'); sys.exit(0 if 'meta' in d and json.loads(str(d['meta'])).get('causal') else 1)" 2>/dev/null; then
      mkdir -p data/legacy; mv data/${t}_train.npz data/${t}_test.npz data/legacy/ 2>/dev/null; echo "moved non-causal ${t} data to data/legacy/"; fi
    [ -f data/${t}_train.npz ] || $PY scripts/collect.py --task $t --episodes 1000 --seed 0 --device "$DEVICE" --causal 1 --out data/${t}_train.npz
    [ -f data/${t}_test.npz ]  || $PY scripts/collect.py --task $t --episodes 200 --seed 1 --device "$DEVICE" --causal 1 --out data/${t}_test.npz
  done
  if [ ! -f data/nyc_twin_train.npz ]; then                                     # #10: raycast NYC twin, geographic block split
    $PY scripts/collect.py --env nyc --episodes ${NYC_EPISODES:-600} --seed 0 --device "$DEVICE" --causal 1 --out data/nyc_twin_all.npz
    $PY scripts/split_geo.py data/nyc_twin_all.npz --out data/nyc_twin --block 120 --margin 40
  fi
fi
for f in data/nav_train.npz data/arrange_train.npz data/nav_test.npz data/arrange_test.npz; do [ -f "$f" ] || { echo "missing $f (run STAGES=data)"; exit 1; }; done
[ -f data/nyc/midtown_buildings.json ] || bash scripts/fetch_nyc_data.sh         # raycast twin inputs (public, ~1 MB)
uv pip install --quiet pytest; $PY -m pytest tests -q

# ---- stages ---------------------------------------------------------------------------------------------------

if stage lr; then                                                               # #4: per-architecture lr on a validation split, never on test
  for w in connectome gru dense; do for lr in 5e-4 1e-3 2e-3 4e-3; do
    [ -f results/runs/${w}_K1_s0_lr${lr}.json ] || $PY scripts/train.py --wiring $w --lr $lr --val 0.15 --epochs 10 --device "$DEVICE" --tag _lr${lr}
  done; done
fi
# pick each architecture's lr by validation MSE from the lr stage (falls back to the run-1 defaults when the sweep is absent)
pick_lr() { $PY -c "import json,glob; r=[(json.load(open(f))['val_mse'],f.split('_lr')[-1][:-5]) for f in glob.glob('results/runs/$1_K1_s0_lr*.json')]; print(min(r)[1] if r else '$2')" 2>/dev/null || echo "$2"; }
export LR_CONN=$(pick_lr connectome 2e-3) LR_DENSE=$(pick_lr dense 1e-3) LR_GRU=$(pick_lr gru 2e-3)
stage seeds && EPOCHS=${EPOCHS:-20} NYC_TRAIN=${NYC_TRAIN:-data/nyc_twin_train.npz} NYC_TEST=${NYC_TEST:-data/nyc_twin_test.npz} bash scripts/run_seeds.sh   # #8 + data-efficiency axis

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

if stage realvid; then                                                          # E7: real handheld video (scripts/ingest_video.py), zero-shot + leave-one-clip-out fine-tuning
  RV=${RV:-data/realvid_all.npz}
  if [ -f "$RV" ]; then
    $PY -c "import numpy as np,sys; sys.exit(0 if 'z' in np.load('$RV') else 1)" || $PY scripts/ingest_video.py --from_npz "$RV" --device "$DEVICE"
    for ck in results/runs/{connectome,connectome_nosign,shuffled,random,gru,dense}_K1_s[0-4].pt; do
      [ -f "$ck" ] || continue; r=$(basename "$ck" .pt)
      [ -f results/runs/${r}_realvid.json ] || $PY scripts/eval_ckpt.py --run "$r" --test "$RV" --tag _realvid --device "$DEVICE"
      [ -f results/runs/${r}_realvid_ft.json ] || $PY scripts/eval_ckpt.py --run "$r" --test "$RV" --tag _realvid_ft --finetune_epochs 10 --device "$DEVICE" --baselines 0
    done
    for ck in results/runs/{connectome,shuffled,random,gru}_K1_s[0-2]_nyc.pt; do                                   # NYC-trained students on real streets
      [ -f "$ck" ] || continue; r=$(basename "$ck" .pt)
      [ -f results/runs/${r}_realvid.json ] || $PY scripts/eval_ckpt.py --run "$r" --test "$RV" --tag _realvid --device "$DEVICE" --baselines 0
    done
  else echo "no $RV: run scripts/ingest_video.py on the laptop and copy it"; fi
fi

if stage arch; then                                                             # neuron-model ablation, wiring fixed: is the GRU gap the unit model or the graph?
  for s in 0 1 2; do for w in connectome block_shuffled random; do
    [ -f results/runs/${w}_K1_s${s}_alpha.json ]     || $PY scripts/train.py --wiring $w --seed $s --epochs 20 --device "$DEVICE" --lr $LR_CONN --learn_alpha 1 --tag _alpha
    [ -f results/runs/${w}_K1_s${s}_gate.json ]      || $PY scripts/train.py --wiring $w --seed $s --epochs 20 --device "$DEVICE" --lr $LR_CONN --gate 1 --tag _gate
    [ -f results/runs/${w}_K1_s${s}_alphagate.json ] || $PY scripts/train.py --wiring $w --seed $s --epochs 20 --device "$DEVICE" --lr $LR_CONN --learn_alpha 1 --gate 1 --tag _alphagate
  done
  [ -f results/runs/gru_K1_s${s}_gate.json ] || $PY scripts/train.py --wiring gru --seed $s --epochs 20 --device "$DEVICE" --lr $LR_GRU --gate 1 --tag _gate   # GRU matched to the gated FlyBrain's parameter count
  done
fi

stage ablate && bash scripts/ablate_connectome.sh                                # #13: min_syn x region set x wiring, 3 seeds

$PY scripts/summarize.py > /dev/null; echo; echo "ALL STAGES DONE $(date +%T): results/summary.md"
echo "then on the laptop: git pull && PYTHONPATH=. python scripts/make_video_issues.py"
