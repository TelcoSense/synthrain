#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"

echo "[1/4] Generating baseline scenario outputs..."
"${PYTHON_BIN}" run_scenario.py \
  --config configs/config.ini \
  --out poster_outputs/01_baseline \
  --seed 0 \
  --n-sites 50 \
  --wet-target 0.2 \
  --idw-power 2 \
  --idw-near 8 \
  --idw-dist-m 10000 \
  --interp-style pycomlink \
  --no-export-csv

echo "[2/4] Generating wet sweep outputs..."
"${PYTHON_BIN}" run_wet_sweep.py \
  --base-config configs/config.ini \
  --out-root poster_outputs/02_wet_sweep \
  --wet-targets 0.05,0.1,0.2,0.35,0.5,0.8 \
  --seed 0 \
  --n-sites 50 \
  --no-log-to-file

echo "[3/4] Generating IDW sweep outputs for 3x4 poster grid..."
"${PYTHON_BIN}" run_idw_sweep.py \
  --base-config configs/config.ini \
  --out-root poster_outputs/03_idw_sweep_3x4 \
  --powers 1,2.5,4 \
  --nears 6,12 \
  --dists 10000,30000 \
  --n-sites-list 50 \
  --seeds 0 \
  --wet-targets 0.2 \
  --no-log-to-file \
  --max-per-page 12

echo "[4/4] Generating fault-case scenario outputs..."
"${PYTHON_BIN}" run_scenario.py \
  --config configs/config.ini \
  --out poster_outputs/04_fault_case \
  --seed 0 \
  --n-sites 50 \
  --wet-target 0.2 \
  --idw-power 2 \
  --idw-near 8 \
  --idw-dist-m 10000 \
  --interp-style pycomlink \
  --outage-fraction 0.15 \
  --clustered-outage-fraction 0.10 \
  --stuck-zero-fraction 0.05 \
  --bias-fraction 0.10 \
  --extra-noise-fraction 0.10 \
  --no-export-csv

echo "Poster outputs are ready under poster_outputs/."

