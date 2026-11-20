# Lip-Sync Deepfake Detector — local web app

B.Tech project: **Audiovisual Deepfake Detection Using Lip Synchronicity**.
A locally running web application around the v4 detector from `LipSync_Deepfake_Detector_v4.ipynb`.
The detector code is used **unchanged**; the app adds a FastAPI server, an inference worker process and a React UI.

```
Browser (React)  ──HTTP / SSE──►  FastAPI (127.0.0.1:8000, no torch)  ──pipe──►  Inference worker (1 process)
  <video> plays your file locally         uploads · jobs · progress · security          model loaded once
                                          data/jobs/<id> (deleted after analysis)      core/ = notebook code
```

## Requirements
* Apple Silicon Mac (M1/M2/M3/M4), macOS 14+. **16 GB RAM recommended**; 8 GB works with the slim checkpoint (default).
* Python **3.12** (`brew install python@3.12`). Node.js is only needed to rebuild the UI (a built copy is included).
* ~6 GB free disk (models 5.2 GB + packages).
* Optional: `brew install ffmpeg` — lets the detector read unusual video formats PyAV cannot open.

## Setup on a new Mac
1. Copy this whole `lipsync_app` folder to the Mac.
2. Copy the two model files into `lipsync_app/models/` (see `models/README.txt`):
   * `large_vox_iter5.pt` (3.91 GB, AV-HuBERT checkpoint)
   * `lipsync_detector_v4.pkl` (7 KB, trained classifiers)
3. Run `./setup.sh`. It creates `.venv`, installs the pinned packages, checks the detector code against the
   notebook (checksums), creates `models/avhubert_slim.pt` (one time, ~1 min, needs ~5.5 GB RAM) and runs the unit tests.
4. Run `./run.sh` and the browser opens **http://127.0.0.1:8000** when the model is ready.

## Verify the installation (recommended, in this order)
```bash
source .venv/bin/activate
export LIPSYNC_MANIFEST=/path/to/manifest.csv          # LipSyncTIMIT manifest (btp/ics_av_deepfake/manifest.csv)
export LIPSYNC_DATASET=/path/to/LipSyncTimitDataset

python tools/verify_reference.py   # 6 notebook demo videos: same verdicts? engine == original analyze()?
python tools/compare_devices.py    # optional: may this Mac use the GPU (MPS)? writes models/device_check.json
./run.sh                           # in another terminal, then:
python tools/e2e_check.py          # 47 end-to-end checks against the running server
```
Without the dataset, `verify_reference.py` cannot run; `e2e_check.py` still runs all error, security, crash and
cancel checks with generated videos.

## Device (CPU / Apple GPU)
`LIPSYNC_DEVICE=auto` (default) uses the CPU unless `tools/compare_devices.py` has verified on **this** Mac, with
**this** PyTorch version, that MPS gives the same verdicts (fake score within 0.5 points, raw outputs within 1e-3).
Force a device with `LIPSYNC_DEVICE=cpu ./run.sh` or `LIPSYNC_DEVICE=mps ./run.sh`.

## What the numbers mean
Shown in the app (ⓘ icons and *About the model*). In short: the **fake score** (0–100) is a display scale whose
35 and 65 points are the notebook's thresholds — **not a probability, confidence or accuracy**. Evaluation figures
are copied from the notebook with their test sets; the 27/29 genuine phone clips called REAL measure false alarms
on a small set and are **not** an overall accuracy. Visuals marked *display only* (waveform, filterbank image,
offset heatmap, offset drift) are computed for the UI and are not used by the model.

## Settings (environment variables)
| variable | default | meaning |
|---|---|---|
| `LIPSYNC_DEVICE` | `auto` | `auto`, `cpu`, `mps` |
| `LIPSYNC_PORT` | `8000` | port on 127.0.0.1 |
| `LIPSYNC_MODELS_DIR` | `./models` | where the model files are |
| `LIPSYNC_DATA_DIR` | `./data` | temporary uploads (`data/jobs`) |
| `LIPSYNC_MAX_UPLOAD_MB` | `2048` | upload limit |
| `LIPSYNC_KEEP_UPLOADS` | `0` | `1` keeps uploaded videos instead of deleting them after analysis |

Fixed on purpose (changing them changes verdicts): first 30 s analysed, frames ≤ 640 px, MTCNN every 2nd frame,
AV-HuBERT layers 24/18/12, the saved thresholds.

## Security and privacy
* Listens on **127.0.0.1 only**: other computers cannot connect.
* Host-header check (blocks DNS rebinding) and a required custom header on uploads/cancels (blocks other websites
  from posting to the local server). No CORS is enabled.
* Videos are stored only in `data/jobs/<id>/` while being analysed and deleted right after; leftovers are removed at
  start-up. Results live in memory until the server stops. No cloud services, no external APIs.

## API
| method | path | |
|---|---|---|
| GET | `/api/health` | worker state, device, model info |
| GET | `/api/model` | score definitions, evaluation results, limits |
| POST | `/api/jobs` | multipart upload (`file`), header `X-LipSync-Client: 1` → `202 {job}` · 409 busy · 413/415/422 errors |
| GET | `/api/jobs/{id}/events` | Server-Sent Events: `stage`, `warning`, `result`, `error`, `cancelled` |
| GET | `/api/jobs/{id}` | job state + `AnalysisResult` |
| DELETE | `/api/jobs/{id}` | cancel (running) or delete (finished) |
| POST | `/api/worker/restart` | restart the inference worker |
Interactive docs: http://127.0.0.1:8000/api/docs

## Project layout
```
core/            avhubert_lite.py, lipsync_detector.py — byte-exact copies of notebook cell A3 (+ CHECKSUMS.sha256)
worker/          loader (model once, slim/original checkpoint, device), stages (real progress), engine, result, main
api/             FastAPI app, jobs + storage, worker bridge, preflight (PyAV file check)
common/          error codes, score definitions + notebook evaluation results
frontend/        React + TypeScript + Vite + Tailwind + Recharts + Lucide (built copy in frontend/dist)
tools/           extract_core, make_slim_ckpt, verify_reference, compare_devices, e2e_check
tests/           test_units.py
models/          model files (not included)
```

## Development
* UI with hot reload: `./run.sh` in one terminal, `cd frontend && npm ci && npm run dev` in another →
  http://127.0.0.1:5173 (forwards `/api` to the server).
* Re-extract the detector from the notebook: `python tools/extract_core.py "/path/to/notebook.ipynb"`
  (refuses to overwrite if the code differs).

## Troubleshooting
| symptom | fix |
|---|---|
| "Model failed to load: model files were not found" | put both files in `models/` (or set `LIPSYNC_MODELS_DIR`) |
| `InconsistentVersionWarning` from scikit-learn | the venv must have scikit-learn **1.6.1** (`pip show scikit-learn`) |
| `pip check` reports facenet-pytorch requirements | expected: it is installed with `--no-deps` on purpose |
| "This browser cannot play this video format" | only the preview; the analysis still runs |
| `FILE_UNREADABLE` for an unusual format | `brew install ffmpeg`, or convert to MP4 (H.264 + AAC) |
| `OUT_OF_MEMORY` / worker stopped | close other apps; the worker restarts automatically |
