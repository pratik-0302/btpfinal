# Handoff — state of the project (2026-11-20)

For the next Claude Code session (and for Pratik). Read this, then `README.md`, before changing anything.

## Project
B.Tech project **Audiovisual Deepfake Detection Using Lip Synchronicity**. This repo is a **local web app**
(React UI + FastAPI + one inference worker process) around the detector from the Colab notebook
`LipSync_Deepfake_Detector_v4 (2).ipynb` (not in this repo; it lived in `~/Downloads` on the 8 GB M1).

## Rules the owner set (keep following them)
1. Never modify the original notebook, the detector code in `core/`, or the model checkpoints.
   `core/` = byte-exact copies of notebook cell A3, guarded by `core/CHECKSUMS.sha256` (checked at every start).
2. Don't install packages or download large files without asking. Don't fabricate outputs or metrics.
3. Work in controlled stages: test each stage, report, wait for approval before the next.
4. Never present scores as probability/confidence/accuracy. The "27 of 29 genuine phone clips called REAL"
   (93.1%) result is **not** an overall accuracy (false-alarm check on a small set, thresholds partly tuned on it).
5. Separate verified facts / estimates / recommendations in reports.

## How the detector works (from the notebook, unchanged)
PyAV decode → 25 fps frames (≤640 px) + 16 kHz mono audio (first 30 s) → MTCNN (facenet-pytorch, every 2nd frame)
→ aligned 96×96 grey mouth crops → frozen AV-HuBERT Large (`large_vox_iter5.pt`, 24 layers, dim 1024; layers 24/18/12
used) video-only and audio-only passes → cosine agreement matrix at offsets ±15 frames (±600 ms) →
**sync check** (StandardScaler + LogisticRegression C=0.3, 26 features) and **deepfake check** (LR C=0.1, 123 features
per 6 s segment) → `band_percent` (thresholds sync 0.583/0.77, deepfake 0.267/0.747 → 35/65) → fake score = max of the
two → REAL ≤35 / UNCERTAIN / FAKE ≥65; forced UNCERTAIN 50 if <0.6 s speech or face in <50% frames.
Bundle `lipsync_detector_v4.pkl` was saved with **scikit-learn 1.6.1** (pin it).

## What was built (this repo)
* `worker/` loads the model once (slim or original checkpoint), runs the **original** `process_video` with
  pass-through proxies that report real stage progress, then the original `scores` + `verdict_from_scores`
  (same steps as `LipSyncDetectorV3.analyze`), builds a JSON-safe `AnalysisResult`.
* `api/` FastAPI on 127.0.0.1 only: upload, SSE progress (`fastapi.sse`), job registry (one job at a time),
  preflight file check (PyAV), TrustedHost + required header `X-LipSync-Client: 1`, uploads auto-deleted,
  worker crash → auto restart.
* `frontend/` React 19 + TS 5.9 + Vite 8 + Tailwind 4 + Recharts 3 + lucide-react 1.69 (icon names changed in 1.x:
  e.g. `CircleCheck`, `TriangleExclamationPoint`, `ShieldQuestionMark`, `FileVideoCamera`). Built copy in `frontend/dist`.
* `common/model_docs.py` = single source for score definitions + notebook evaluation numbers shown in the UI.
* Tools: `extract_core.py`, `make_slim_ckpt.py`, `verify_reference.py`, `compare_devices.py`, `e2e_check.py`;
  `tests/test_units.py`.

## Verified on the 8 GB M1 (Python 3.13 + temporary test packages, CPU)
| check | result |
|---|---|
| core/ == notebook | byte-exact |
| slim checkpoint | 533 tensors identical; embeddings bit-identical; load ~5 s (original 42 s) |
| app engine vs original `analyze()` | identical on all 6 reference videos |
| verdicts vs notebook cell D2 | 6/6 identical |
| fake score vs notebook | 3/6 within ±1; **drift** 1.6, 1.9, **7.1** points (181.mp4: 75% real here vs 82% in Colab) |
| cause of drift | raw lip–speech agreement already differs before the classifiers (notebook B3 self-test: offsets identical, values differ ≤0.018) → hardware/library differences (Colab T4 GPU vs Mac CPU, OpenCV/PyAV versions), not the app |
| `tools/e2e_check.py` | 47/47 pass (errors, security, SSE, cancel, crash recovery, file deletion, memory soak) |
| memory | original loader peak footprint 5.4 GB (CPU) / 7.6 GB (MPS); slim 3.1 GB (CPU) / 4.5 GB (MPS); worker 1.5–2.7 GB over 15 runs, no leak |
| speed (CPU, 2–4 s clips) | ~2–9 s per video; AV-HuBERT on a 30 s clip: ~6–9 s CPU, ~2.2 s MPS |
| frontend | `tsc` 0 errors; production build OK; pages checked in headless Chrome |

## NOT yet verified
* The exact pinned environment (`requirements.txt`, Python 3.12) — `setup.sh` has not been run anywhere yet.
* MPS (Apple GPU) on real videos — run `tools/compare_devices.py`; default stays CPU until it passes.
* MTCNN on MPS. Manual upload through the browser UI (the upload path was tested via the API).
* Behaviour on the 16 GB Mac.

## Next steps (on the 16 GB Apple-Silicon Mac)
1. Clone this repo; copy `large_vox_iter5.pt` (3,905,065,977 bytes) and `lipsync_detector_v4.pkl` (7,001 bytes) into `models/`.
   (They were in Google Drive: `btp/lipsync_detector/ckpt/` and `btp/lipsync_detector/model/`.)
2. `brew install python@3.12` → `./setup.sh` → report any install problem (watch: scikit-learn 1.6.1 with numpy 2.5 / scipy 1.18).
3. With the LipSyncTIMIT dataset + manifest (`btp/ics_av_deepfake/manifest.csv` reproduces the notebook's demo picks):
   `python tools/verify_reference.py` → expect same verdicts; compare drift with the table above.
4. `python tools/compare_devices.py` (decides MPS), then `./run.sh` and `python tools/e2e_check.py`.
5. Try real phone videos (real and fake) in the browser.

## Pinned versions (checked against PyPI/npm metadata, Nov 2026)
torch 2.15.0, torchvision 0.29.2, av 19.0.2, opencv-python-headless 5.0.0.93, numpy 2.5.4, scipy 1.18.1,
scikit-learn 1.6.1, python_speech_features 0.6, facenet-pytorch 2.6.0 (`--no-deps`; needs requests + tqdm),
fastapi 0.162.0 (built-in SSE), uvicorn 0.54.0, python-multipart 0.0.33, pydantic 2.14.3 (needs pydantic-core 2.50.4),
psutil 7.2.2. Node ≥ 20.19 / 22.12 for Vite 8. PyTorch 2.15 has **no Intel-Mac build**.
