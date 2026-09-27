#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${ECCO_PYTHON:-${ADVERCITY_PYTHON:-${COINFRA_PYTHON:-$PROJECT_DIR/.venv/bin/python}}}"
RUN_LOG_DIR="$PROJECT_DIR/logs"
RUN_LOG_DIR="${ECCO_LOG_DIR:-$RUN_LOG_DIR}"
mkdir -p "$RUN_LOG_DIR"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_LOG_DIR/matplotlib_cache}"
mkdir -p "$MPLCONFIGDIR"
RUN_TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')_$$"
RUN_LOG="$RUN_LOG_DIR/run_${RUN_TIMESTAMP}.log"
RUN_STATUS="$RUN_LOG_DIR/run_${RUN_TIMESTAMP}_status.txt"

PIPELINE_ARGS=()
SMALL_RUN=false
for argument in "$@"; do
  if [[ "$argument" == "--small" ]]; then
    SMALL_RUN=true
  else
    PIPELINE_ARGS+=("$argument")
  fi
done
if [[ "$SMALL_RUN" == true ]]; then
  PIPELINE_ARGS+=(--smoke --overwrite --skip-prepare --require-data)
fi

MAX_ATTEMPTS="${ECCO_MAX_ATTEMPTS:-3}"
if [[ ! "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' 'ECCO_MAX_ATTEMPTS must be a positive integer' >&2
  exit 2
fi

set +e
{
  printf '%s status=started command=' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf '%q ' "$0" "$@"
  printf '\n'
  if [[ ! -x "$PYTHON_BIN" ]]; then
    printf '%s\n' 'Python environment is missing. Run: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt' >&2
    exit 127
  fi
  if [[ "$SMALL_RUN" == true ]]; then
    printf '%s\n' 'Small mode: existing-data check + fresh smoke end-to-end run (no download).'
  fi
} 2>&1 | tee -a "$RUN_LOG"
status=${PIPESTATUS[0]}

if [[ $status -eq 0 ]]; then
  attempt=1
  while true; do
    printf '%s attempt=%s/%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$attempt" "$MAX_ATTEMPTS" | tee -a "$RUN_LOG"
    "$PYTHON_BIN" main.py --stage all "${PIPELINE_ARGS[@]}" 2>&1 | tee -a "$RUN_LOG"
    status=${PIPESTATUS[0]}
    if [[ $status -ne 75 || $attempt -ge $MAX_ATTEMPTS ]]; then
      break
    fi
    delay=$((attempt * 60))
    printf '%s transient_failure=1 retry_in_seconds=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$delay" | tee -a "$RUN_LOG"
    sleep "$delay"
    attempt=$((attempt + 1))
  done
fi
set -e

ended="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
if [[ $status -eq 0 ]]; then
  printf '%s status=completed exit_code=0 log=%s\n' "$ended" "$RUN_LOG" | tee "$RUN_STATUS"
else
  printf '%s status=failed exit_code=%s log=%s\n' "$ended" "$status" "$RUN_LOG" | tee "$RUN_STATUS" >&2
fi
exit "$status"
