#!/usr/bin/env bash
# One-time setup on a new Mac (Apple Silicon). Safe to re-run.
#   1. Python 3.12 virtual environment with the pinned packages
#   2. checks that the detector code still matches the notebook and that the model files are present
#   3. creates the slim AV-HuBERT checkpoint (the original file is not modified)
#   4. (optional, BUILD_FRONTEND=1) rebuilds the web UI with npm; a built copy is already in frontend/dist
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3.12}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3.12 not found. Install it with:  brew install python@3.12"; exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
  echo "This setup targets Apple Silicon Macs (PyTorch 2.15 has no Intel-Mac build)."; exit 1
fi

if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps facenet-pytorch==2.6.0
python -m pip freeze > requirements.lock
echo "installed packages recorded in requirements.lock"

python - <<'PY'
import sys
sys.path.insert(0, ".")
import config
from worker.core_import import verify_core
verify_core()
print("core/ matches the notebook (checksums OK)")
missing = [p for p in (config.BUNDLE_PATH,) if not p.exists()]
if not config.SLIM_CKPT.exists() and not config.ORIGINAL_CKPT.exists():
    missing.append(config.ORIGINAL_CKPT)
if missing:
    print("\nMISSING model files — copy them into", config.MODELS_DIR)
    for m in missing:
        print("   ", m.name)
    sys.exit(1)
print("model files present in", config.MODELS_DIR)
PY

if [ ! -f models/avhubert_slim.pt ]; then
  python tools/make_slim_ckpt.py
fi

if [ "${BUILD_FRONTEND:-0}" = "1" ]; then
  (cd frontend && npm ci && npm run build)
fi

python tests/test_units.py
echo
echo "Setup complete. Start the app with:  ./run.sh"
