#!/usr/bin/env bash
# Inputs for the raycast NYC digital twin (wm2cx/nyc.py): Midtown building footprints with roof heights and
# construction years from NYC Open Data (dataset 5zhs-2jue, public domain) and the street network from OpenStreetMap
# (Overpass API, ODbL). Default box is ~0.9 km x 0.8 km around Times Square; pass another "S,W,N,E" box as $1.
set -euo pipefail
BOX="${1:-40.7545,-73.9910,40.7620,-73.9800}"; OUT="${2:-data/nyc}"
IFS=, read -r S W N E <<< "$BOX"; mkdir -p "$OUT"
curl -sS -m 180 -G "https://data.cityofnewyork.us/resource/5zhs-2jue.json" \
  --data-urlencode "\$where=intersects(the_geom, \"POLYGON(($W $S, $E $S, $E $N, $W $N, $W $S))\")" \
  --data-urlencode '$limit=20000' -o "$OUT/midtown_buildings.json"
# Overpass answers 406 without a User-Agent; mirrors: overpass.kumi.systems, overpass.private.coffee
curl -sS -m 300 -A "wm2cx-nyc-twin/1.0 (research; github prajak002/World2Brain)" -H "Accept: application/json" \
  -X POST "https://overpass-api.de/api/interpreter" \
  --data-urlencode "data=[out:json][timeout:120];way[\"highway\"]($S,$W,$N,$E);(._;>;);out body;" -o "$OUT/midtown_streets.json"
python - "$OUT" <<'EOF'
import json, sys
o = sys.argv[1]; B = json.load(open(f"{o}/midtown_buildings.json", encoding="utf-8")); S = json.load(open(f"{o}/midtown_streets.json", encoding="utf-8"))
print(f"{len(B)} buildings, {sum(e['type']=='way' for e in S['elements'])} ways, {sum(e['type']=='node' for e in S['elements'])} nodes -> {o}/")
EOF
