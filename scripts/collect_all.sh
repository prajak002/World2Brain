#!/bin/zsh
cd /Users/prajaksen/worldmodel2connectom
export PYTHONPATH=.
uv run python scripts/collect.py --task nav --episodes 500 --seed 0 --out data/nav_train.npz
uv run python scripts/collect.py --task arrange --episodes 500 --seed 0 --out data/arrange_train.npz
uv run python scripts/collect.py --task nav --episodes 100 --seed 1 --out data/nav_test.npz
uv run python scripts/collect.py --task arrange --episodes 100 --seed 1 --out data/arrange_test.npz
echo ALLDONE
