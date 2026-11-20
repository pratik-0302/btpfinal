"""Application settings. Every value can be overridden with an environment variable.

Detector settings (MAX_SECONDS and everything inside core/) are fixed to the notebook's
values on purpose: changing them would change the verdicts.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- paths
CORE_DIR = ROOT / "core"
MODELS_DIR = _env_path("LIPSYNC_MODELS_DIR", ROOT / "models")
ORIGINAL_CKPT = MODELS_DIR / "large_vox_iter5.pt"          # original AV-HuBERT checkpoint (never modified)
SLIM_CKPT = MODELS_DIR / "avhubert_slim.pt"                # derived by tools/make_slim_ckpt.py
BUNDLE_PATH = MODELS_DIR / "lipsync_detector_v4.pkl"       # trained classifiers + thresholds
DEVICE_CHECK = MODELS_DIR / "device_check.json"            # written by tools/compare_devices.py
DATA_DIR = _env_path("LIPSYNC_DATA_DIR", ROOT / "data")
JOBS_DIR = DATA_DIR / "jobs"
FRONTEND_DIST = ROOT / "frontend" / "dist"

# ---------------------------------------------------------------- server (local only)
HOST = "127.0.0.1"                                         # never 0.0.0.0: other machines must not connect
PORT = int(os.environ.get("LIPSYNC_PORT", "8000"))
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
CLIENT_HEADER = "X-LipSync-Client"                         # required on POST/DELETE (blocks cross-site form posts)

# ---------------------------------------------------------------- uploads and storage
MAX_UPLOAD_BYTES = int(float(os.environ.get("LIPSYNC_MAX_UPLOAD_MB", "2048")) * 1024 * 1024)
KEEP_UPLOADS = _env_bool("LIPSYNC_KEEP_UPLOADS", False)    # default: delete each video right after analysis
MAX_RESULTS_IN_MEMORY = 20
# same list as the notebook's VIDEO_EXT
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp", ".mpeg", ".mpg", ".wmv", ".flv"}

# ---------------------------------------------------------------- detector (fixed: same as the notebook)
MAX_SECONDS = 30                                           # LipSyncDetectorV3.analyze default
MIN_PREFLIGHT_SECONDS = 0.9                                # clearly too short -> rejected before the worker

# ---------------------------------------------------------------- device
# auto: MPS only if tools/compare_devices.py recorded that MPS matches the CPU on this machine; else CPU.
DEVICE = os.environ.get("LIPSYNC_DEVICE", "auto").strip().lower()
CROPPER_DEVICE = os.environ.get("LIPSYNC_CROPPER_DEVICE", "").strip().lower()   # optional override (tools only)
LOW_MEMORY_WARNING_GB = float(os.environ.get("LIPSYNC_LOW_MEMORY_GB", "2.0"))
