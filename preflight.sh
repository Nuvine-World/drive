#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${ECCO_PYTHON:-${ADVERCITY_PYTHON:-${COINFRA_PYTHON:-$PROJECT_DIR/.venv/bin/python}}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf '%s\n' 'Python environment is missing. Run: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt' >&2
  exit 127
fi

printf '%s\n' 'Checking shell scripts...'
bash -n run.sh preflight.sh
if [[ -f run_coinfra_at_11pm.sh ]]; then
  bash -n run_coinfra_at_11pm.sh
fi

printf '%s\n' 'Running automated tests...'
"$PYTHON_BIN" -m pytest -q

printf '%s\n' 'Checking CoInfra configuration, dependencies, CUDA, and prepared data...'
"$PYTHON_BIN" main.py --config configs/coinfra.yaml --stage check --require-data

printf '%s\n' 'Checking Adver-City configuration, dependencies, CUDA, and paths...'
"$PYTHON_BIN" main.py --config configs/advercity.yaml --stage check

printf '%s\n' 'Preflight passed. The sequential run is ready to start.'
