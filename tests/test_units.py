"""Fast checks that need no model: run with  python tests/test_units.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_core_checksums():
    from worker.core_import import verify_core
    verify_core()


def test_mtcnn_batch_count_matches_original_rule():
    from worker.stages import mtcnn_batches
    for n in (1, 2, 3, 31, 32, 33, 64, 65, 750, 751):
        det_idx = list(range(0, n, 2))                      # MouthCropper.landmarks
        if det_idx[-1] != n - 1:
            det_idx.append(n - 1)
        assert mtcnn_batches(n, 2, 32) == math.ceil(len(det_idx) / 32), n


def test_json_safe():
    import numpy as np
    from worker.result import json_safe
    out = json_safe({"a": np.float32("nan"), "b": np.array([1.5, np.inf]), "c": np.bool_(True), "d": (np.int64(3),)})
    assert out == {"a": None, "b": [1.5, None], "c": True, "d": [3]}


def test_error_classification():
    import subprocess
    from common.errors import Cancelled, classify_exception

    class NoFaceError(RuntimeError):
        pass

    cases = [
        (NoFaceError("no face was found in the video"), "NO_FACE"),
        (ValueError("the file has no audio track: lip sync cannot be checked"), "NO_AUDIO"),
        (ValueError("the file has no video stream"), "NO_VIDEO_STREAM"),
        (ValueError("video shorter than 1 second"), "TOO_SHORT"),
        (ValueError("no video frames could be decoded"), "FILE_UNREADABLE"),
        (MemoryError(), "OUT_OF_MEMORY"),
        (RuntimeError("MPS backend out of memory (MPS allocated: 3 GB)"), "OUT_OF_MEMORY"),
        (subprocess.CalledProcessError(1, ["ffmpeg"]), "FILE_UNREADABLE"),
        (FileNotFoundError(2, "No such file or directory: 'ffmpeg'"), "FILE_UNREADABLE"),
        (Cancelled(), "CANCELLED"),
        (KeyError("x"), "INTERNAL"),
    ]
    for exc, code in cases:
        assert classify_exception(exc).code == code, (exc, classify_exception(exc).code)


def test_reference_headline_parsing():
    from tools.reference import headline_percent
    assert headline_percent("LOOKS REAL — 82% real") == 18
    assert headline_percent("LOOKS FAKE — 98% fake") == 98


def test_docs_have_every_displayed_score():
    from common.model_docs import SCORE_DEFINITIONS
    for k in ("verdict", "fake_score", "sync_score", "deepfake_score", "raw_p", "sync_z", "matched_windows",
              "best_offset_ms", "agreement", "segment_p", "face_detected_frac", "mouth_width_px"):
        assert k in SCORE_DEFINITIONS and SCORE_DEFINITIONS[k]["meaning"], k


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
