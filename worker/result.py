"""AnalysisResult: the structured, JSON-safe object sent to the UI.

Every verdict / check / timeline / quality value is copied unchanged from the dictionary returned by the
original LipSyncDetectorV3.analyze. Values under "visuals" (except mouth crops) are new display calculations
and are labelled as such in common/model_docs.py.
"""
from __future__ import annotations

import math

import numpy as np

TAU = 15                 # lipsync_detector.TAU (offsets -15..+15 frames)
FPS = 25


def json_safe(x):
    """numpy -> Python; NaN/inf -> None (JSON has no NaN)."""
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def build_result(out: dict, extras: dict, file_info: dict, meta: dict, stage_seconds: dict, total_seconds: float) -> dict:
    fi = out.get("face_info", {}) or {}
    windows = out.get("windows", []) or []
    res = {
        "status": "ok",
        "file": file_info,
        "run": {
            **{k: meta.get(k) for k in ("device", "cropper_device", "device_reason", "checkpoint", "torch", "sklearn",
                                         "avhubert", "layers_used", "bundle", "max_seconds")},
            "analysed_seconds": out["seconds"],
            "processing_seconds": round(total_seconds, 2),
            "stage_seconds": stage_seconds,
        },
        "verdict": {
            "label": out["verdict"], "headline": out["headline"],
            "fake_score": out["percent_fake"], "real_score": out["percent_real"],
            "decided_by": out["why"], "reason": out["reason"], "notes": out["notes"], "hard_flags": out["hard_flags"],
        },
        "sync_check": {
            "score": out["pct_sync"], "raw_p": out["p_out_of_sync"], "z": out["sync_z"],
            "matched_windows": out["matched_windows"], "n_windows": len(out.get("window_z", [])),
            "best_offset_ms": out["best_offset_ms"], "speech_seconds": out["speech_seconds"],
        },
        "deepfake_check": {
            "score": out["pct_deepfake"], "raw_p": out["p_deepfake"],
            "segment_raw_p": out["segment_p_deepfake"], "segment_spans_s": extras.get("segment_spans_s"),
        },
        "timeline": {
            "fps": FPS,
            "agreement": out["per_frame_sync"],
            "speech_mask": out["speech_mask"],
            "energy_db": out["energy"],
            "windows_s": [[s / FPS, e / FPS] for s, e in windows],
            "window_z": out["window_z"],
        },
        "quality": {
            "face_detected_frac": fi.get("face_detected_frac"),
            "frames_with_several_faces": fi.get("frames_with_several_faces"),
            "mouth_width_px": fi.get("mouth_width_px"),
            "audio_gap_s": fi.get("audio_gap_s"), "audio_largest_gap_s": fi.get("audio_largest_gap_s"),
            "audio_gaps": fi.get("audio_gaps"), "audio_overlap_s": fi.get("audio_overlap_s"),
            "timing": fi.get("timing"), "stored_fps": fi.get("stored_fps"), "retimed_to_fps": fi.get("retimed_to_fps"),
            "timing_z_all": extras.get("timing_z_all"),
        },
        "visuals": {
            "mouth_crops": extras.get("crops"),
            "waveform": extras.get("waveform"),
            "audio_features": extras.get("audio_features"),
            "sync_matrix": {"offsets_ms": [d * 40 for d in range(-TAU, TAU + 1)], "values": extras.get("sync_matrix")},
            "offset_drift": extras.get("offset_drift"),
        },
        "warnings": list(file_info.get("warnings", [])),
        "error": None,
    }
    return json_safe(res)
