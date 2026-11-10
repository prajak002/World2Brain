#!/usr/bin/env bash
# Memory Maze B4 (PREREG Amendment 1): PCA + projection, then the training waves, JOBS runs at a time on one GPU.
#   JOBS=3 bash scripts/run_mm_b4.sh            # all waves in the registered order
#   WAVES="1" JOBS=3 bash scripts/run_mm_b4.sh  # wave 1 only (C and G, seeds 0-4)
# Restartable: every stage skips finished work (train_mm.py resumes from <name>_resume.pt). Do not edit while running.
set -u; cd "$(dirname "$0")/.."; export PYTHONPATH=.; mkdir -p logs results/memmaze/runs
PY="${PY:-uv run python}"; JOBS="${JOBS:-3}"; U="${U:-12000}"; WAVES="${WAVES:-1 2 3}"
[ -f data/mm9/pca.npz ] || $PY scripts/mm_encode.py --stage pca --pool 0 1 2 3 2>&1 | tee logs/mm_pca.log
[ -f data/mm9/pca.npz ] || { echo "PCA failed"; exit 1; }
$PY scripts/mm_encode.py --stage project --splits eval train val 2>&1 | tee -a logs/mm_project.log
runs() { case "$1" in
  1) for s in 0 1 2 3 4; do echo "C $s"; echo "G $s"; done ;;
  2) for c in S B R; do for s in 0 1 2; do echo "$c $s"; done; done ;;
  3) for s in 0 1 2; do echo "T $s"; done ;;
esac; }
for w in $WAVES; do
  echo "== wave $w $(date -u)"
  runs "$w" | xargs -P "$JOBS" -L 1 bash -c '$0 scripts/train_mm.py --cond "$2" --seed "$3" --updates "$1" > "logs/b4_${2}_s${3}.log" 2>&1 || echo "FAILED $2 s$3"' "$PY" "$U"
done
echo "== B4 done $(date -u)"
