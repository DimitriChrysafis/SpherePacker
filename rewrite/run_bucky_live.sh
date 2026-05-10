#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  /opt/homebrew/bin/python3.12 -m venv .venv
fi

.venv/bin/python -m pip install -q -r requirements.txt

mkdir -p outputs logs

.venv/bin/python -u pack_bucky_live.py \
  --mesh /Users/dofa/Desktop/high_poly_bucky.obj \
  --output outputs/high_poly_bucky_spheres.json \
  --max-spheres 0 \
  --sample-count 140000 \
  --replenish-count 35000 \
  --voxel-resolution 170 \
  --target-triangles 70000 \
  --sphere-resolution 10 \
  --save-every 10 \
  --display-every 1 \
  "$@" 2>&1 | tee logs/bucky_live.log
