#!/usr/bin/env bash
# Start the app (local only): http://127.0.0.1:8000   — stop with Ctrl+C
# Optional: LIPSYNC_DEVICE=cpu|mps|auto (default auto = MPS only if tools/compare_devices.py verified it)
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate
( for _ in $(seq 1 120); do
    if curl -s http://127.0.0.1:${LIPSYNC_PORT:-8000}/api/health | grep -q '"worker":"ready"'; then
      open "http://127.0.0.1:${LIPSYNC_PORT:-8000}"; break
    fi
    sleep 1
  done ) &
exec python server.py
