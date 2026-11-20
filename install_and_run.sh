#!/usr/bin/env bash
# One command to install everything (first time) and start the app.
#   ./install_and_run.sh
# Safe to run again: steps that are already done are skipped, so later runs just start the app.
set -euo pipefail
cd "$(dirname "$0")"

say()  { printf "\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n" "$*"; }
fail() { printf "\n\033[1;31mError:\033[0m %s\n" "$*"; exit 1; }

AVHUBERT_URL="https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt"
AVHUBERT_SIZE=3905065977
BUNDLE_URL="https://drive.usercontent.google.com/download?id=1tZHuCFPRO9gZrIFIq-ooaP-Vnjl_b2dU&export=download&confirm=t"
BUNDLE_SIZE=7001

# ---------------------------------------------------------------- 1. this Mac
say "Checking this Mac"
[ "$(uname -s)" = "Darwin" ] || fail "this app runs on macOS."
[ "$(uname -m)" = "arm64" ]  || fail "this app needs an Apple Silicon Mac (M1/M2/M3/M4)."
free_gb=$(df -g . | awk 'NR==2 {print $4}')
[ "${free_gb:-0}" -ge 8 ] || fail "about 10 GB of free disk space is needed (found ${free_gb} GB)."
echo "Apple Silicon Mac, ${free_gb} GB free — OK"

# ---------------------------------------------------------------- 2. Python 3.12
if ! command -v python3.12 >/dev/null 2>&1; then
  say "Installing Python 3.12"
  command -v brew >/dev/null 2>&1 || fail "Homebrew is needed first. Install it with:
  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"
then run ./install_and_run.sh again."
  brew install python@3.12
fi
echo "Python: $(python3.12 --version)"

# ---------------------------------------------------------------- 3. model files
size_of() { stat -f%z "$1" 2>/dev/null || echo 0; }

download() {  # url  file  expected-size  description
  local url=$1 file=$2 want=$3 what=$4
  if [ -f "$file" ] && [ "$(size_of "$file")" = "$want" ]; then
    echo "$what already downloaded — OK"; return
  fi
  if [ "$file" = "models/large_vox_iter5.pt" ] && [ -f models/avhubert_slim.pt ]; then
    echo "$what not needed (the slim copy already exists) — OK"; return
  fi
  say "Downloading $what"
  curl -L --fail --retry 3 -C - -o "$file.part" "$url" || rm -f "$file.part"
  [ -f "$file.part" ] || fail "download of $what failed. Check your internet connection and run again."
  if [ "$(size_of "$file.part")" != "$want" ]; then
    rm -f "$file.part"
    fail "$what has the wrong size (expected $want bytes). Run ./install_and_run.sh again."
  fi
  mv "$file.part" "$file"
}

mkdir -p models
download "$AVHUBERT_URL" models/large_vox_iter5.pt      "$AVHUBERT_SIZE" "AV-HuBERT checkpoint (3.9 GB, from Meta)"
download "$BUNDLE_URL"   models/lipsync_detector_v4.pkl "$BUNDLE_SIZE"   "trained classifiers (7 KB)"

# ---------------------------------------------------------------- 4. one-time setup
if [ ! -f .venv/.setup_done ] || [ ! -f models/avhubert_slim.pt ]; then
  say "Setting up (first time only, about 5 minutes)"
  ./setup.sh
  touch .venv/.setup_done
else
  echo "Setup already done — OK"
fi

# ---------------------------------------------------------------- 5. start
say "Starting the app — your browser opens http://127.0.0.1:${LIPSYNC_PORT:-8000} when the model is ready"
echo "To stop the app, press Ctrl + C in this window."
exec ./run.sh
