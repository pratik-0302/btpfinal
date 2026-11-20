"""Stage-1 / Stage-2 check: does this machine reproduce the notebook?

Runs the ORIGINAL LipSyncDetectorV3.analyze (and the app's engine) on the six videos of notebook cell D2
and compares with the notebook's printed results.
Per video:
  PASS  = same verdict, engine == original analyze(), fake score within ±1 point of the notebook (+0.5 rounding)
  DRIFT = same verdict and engine == analyze(), but the fake score differs by more (reported, see below)
  FAIL  = the verdict differs, or the app's engine differs from the original analyze()  -> exit code 1
Colab ran on a T4 GPU with other library versions; on the audit Mac (CPU) 3 of 6 videos drifted by 1.6–7.1 points
with identical verdicts, and the raw lip–speech agreement already differed slightly before any classifier.

usage:
  LIPSYNC_MANIFEST=~/Downloads/btp/ics_av_deepfake/manifest.csv \
  LIPSYNC_DATASET=~/Downloads/btp/LipSyncTimitDataset \
  python tools/verify_reference.py
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402

import config                                        # noqa: E402
from tools.reference import find_reference_videos, headline_percent   # noqa: E402
from worker.engine import run_analysis               # noqa: E402
from worker.loader import load_detector              # noqa: E402

TOL_POINTS = 1.0


def same(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and not isinstance(a, str):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=True)
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def main() -> None:
    refs = find_reference_videos()
    loaded = load_detector(progress=lambda m: print("  ", m, flush=True))
    print("loaded:", json.dumps({k: loaded.meta[k] for k in ("device", "cropper_device", "checkpoint", "load_seconds",
                                                               "torch", "sklearn")}))
    ok_all = True
    drift = 0
    rows = []
    for path, want_v, want_h in refs:
        t = time.time()
        orig = loaded.detector.analyze(str(path), max_seconds=config.MAX_SECONDS)       # original method
        t_orig = time.time() - t
        eng, _, _ = run_analysis(loaded, str(path), {}, lambda ev: None, None)          # app engine
        parity = same(orig, eng)
        want_pct = headline_percent(want_h)
        d = abs(orig["percent_fake"] - want_pct)
        hard_ok = orig["verdict"] == want_v and parity                 # must hold on every machine
        status = "FAIL" if not hard_ok else ("PASS" if d <= TOL_POINTS + 0.5 else "DRIFT")   # notebook rounds
        ok_all &= hard_ok
        drift += status == "DRIFT"
        rows.append({"file": path.name, "notebook": want_h, "this_machine": orig["headline"],
                     "fake_pct": round(orig["percent_fake"], 2), "diff_points": round(d, 2),
                     "engine_equals_analyze": parity, "seconds": round(t_orig, 1), "status": status})
        print(f"{status:5s} {path.name[:48]:48s} notebook: {want_h:24s} here: {orig['headline']:24s} "
              f"(fake {orig['percent_fake']:.2f}%, Δ {d:.2f}) engine==analyze: {parity}  {t_orig:.1f}s", flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3
    out = {"verdicts_and_engine_ok": ok_all, "score_drift_videos": drift, "device": loaded.meta["device"],
           "cropper_device": loaded.meta["cropper_device"], "checkpoint": loaded.meta["checkpoint"],
           "torch": loaded.meta["torch"], "sklearn": loaded.meta["sklearn"], "peak_rss_gb": round(peak, 2), "videos": rows}
    (config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "verify_reference.json").write_text(json.dumps(out, indent=2))
    print(f"\n{'verdicts identical to the notebook and engine == analyze()' if ok_all else 'FAILED: a verdict differs or the engine differs from analyze()'}"
          f" | {drift} of {len(rows)} videos with score drift > {TOL_POINTS} point vs Colab"
          f" | peak RSS {peak:.2f} GB | report: {config.DATA_DIR / 'verify_reference.json'}")
    if drift:
        print("DRIFT = same verdict, but the fake score differs from the notebook's Colab (T4 GPU) run. On the audit Mac "
              "the raw lip–speech agreement already differed slightly before the classifiers (different hardware and "
              "library versions), so this is expected; it is reported, not hidden.")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
