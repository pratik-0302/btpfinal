"""One analysis = exactly what LipSyncDetectorV3.analyze(path, max_seconds=30) does, in visible stages.

Original analyze():
    r = process_video(path, self.crop, self.enc, max_seconds=max_seconds)
    if r["T"] < 25: raise ValueError("video shorter than 1 second")
    g, c, ev, seg = self.scores(self.b, r["Cs"], r["E"], self.enc.device)
    out = self.verdict_from_scores(self.b, g, c, ev, r["info"], r["T"] / FPS)
    out.update({...})

Here the same original functions are called in the same order. Differences, none of which change a number:
  * read_av(..., timing="both") is called first (process_video would call it identically) so that decoding
    is a visible stage; its output is passed to process_video as `prefetched`, which the original supports;
  * the cropper/encoder are pass-through proxies that report progress;
  * a few extra display values are computed afterwards from data the pipeline already produced.
tests/test_engine_parity.py checks that the result equals the original analyze() field by field.
"""
from __future__ import annotations

import base64
import gc
import time

import numpy as np

import config
from worker.result import build_result
from worker.stages import CropperProxy, EncoderProxy, StageReporter

N_CROPS_SHOWN = 16


def _analyze_dict(LSD, b, r, g, c, ev, seg, device):
    """Copy of the dictionary assembled at the end of LipSyncDetectorV3.analyze (core/lipsync_detector.py)."""
    out = LSD.LipSyncDetectorV3.verdict_from_scores(b, g, c, ev, r["info"], r["T"] / LSD.FPS)
    out.update({"p_out_of_sync": g, "p_deepfake": c, "segment_p_deepfake": list(map(float, seg)),
                "seconds": r["T"] / LSD.FPS, "best_offset_ms": int(ev["offset"] * 1000 / LSD.FPS),
                "matched_windows": ev["hit_frac"], "sync_z": ev["z"], "speech_seconds": ev["speech_seconds"],
                "per_frame_sync": ev["per_frame"].tolist(), "speech_mask": ev["mask"].tolist(),
                "windows": ev["windows"], "window_z": ev["window_z"].tolist(),
                "energy": np.asarray(r["E"]).tolist(), "mouth_crops": r["crops"], "face_info": r["info"]})
    return out


# ------------------------------------------------------------------ extra display values (new calculations)
def _waveform_envelope(audio_int16, T, spf):
    """Min/max of the model's 16 kHz audio per 40 ms frame (display only)."""
    a = np.asarray(audio_int16[: T * spf], np.float32) / 32768.0
    if len(a) < T * spf:
        a = np.pad(a, (0, T * spf - len(a)))
    a = a.reshape(T, spf)
    return {"min": np.round(a.min(1), 4).tolist(), "max": np.round(a.max(1), 4).tolist()}


def _audio_features(audio_int16, T):
    """The model's own audio features: the same logfbank call as lipsync_detector.audio_input (26 log-mel
    filterbank energies at 100 Hz). For display they are averaged in groups of 4 -> 25 Hz, one row per
    video frame (the model instead stacks the 4 rows side by side)."""
    from python_speech_features import logfbank
    f = logfbank(np.asarray(audio_int16).astype(np.float64), samplerate=16000).astype(np.float32)
    r = (-len(f)) % 4
    if r:
        f = np.concatenate([f, np.repeat(f[-1:], r, 0)])
    f = f.reshape(-1, 4, f.shape[1]).mean(1)[:T]
    return {"bands": int(f.shape[1]), "values": np.round(f, 2).tolist(),
            "min": float(np.min(f)) if f.size else 0.0, "max": float(np.max(f)) if f.size else 0.0}


def _crops_png(crops, cv2):
    """Up to 16 evenly spaced 96x96 mouth crops as one PNG strip (base64) + their times."""
    if len(crops) == 0:
        return None
    idx = np.unique(np.linspace(0, len(crops) - 1, min(N_CROPS_SHOWN, len(crops))).astype(int))
    strip = np.concatenate([crops[i] for i in idx], 1)
    ok, buf = cv2.imencode(".png", strip)
    if not ok:
        return None
    return {"png_base64": base64.b64encode(buf.tobytes()).decode("ascii"), "count": int(len(idx)),
            "size": int(crops.shape[1]), "times_s": [round(float(i) / 25.0, 2) for i in idx]}


def _segment_spans(LSD, Cs, E, T):
    """Time spans of the 6 s segments the deepfake check scored (same trim + bounds as the original code)."""
    Ct, _ = LSD.trim_edges(Cs, E)
    off = LSD.EDGE if len(Ct[0]) != T else 0
    return [[round((s + off) / LSD.FPS, 2), round((e + off) / LSD.FPS, 2)] for s, e in LSD.segment_bounds(len(Ct[0]))]


# ------------------------------------------------------------------ the job
def run_analysis(loaded, path: str, file_info: dict, emit, cancel_event) -> dict:
    """Returns (original analyze() dict, extras, stage times). Memory is released by the caller
    (worker/main.py) once this function has returned and its locals are gone."""
    LSD = loaded.LSD
    rep = StageReporter(emit, cancel_event)

    # ---- decode (the call process_video would make itself)
    rep.start("decode")
    read_info: dict = {}
    versions = LSD.read_av(path, max_seconds=config.MAX_SECONDS, info=read_info, timing="both")
    n_versions = len(versions)
    rep.end("decode", versions=n_versions, frames=int(len(versions[0][0])))

    # ---- faces + embeddings: the original process_video with pass-through proxies
    cropper = CropperProxy(loaded.cropper, rep, n_versions)
    encoder = EncoderProxy(loaded.encoder, rep, n_versions)
    r = LSD.process_video(path, cropper, encoder, max_seconds=config.MAX_SECONDS,
                          prefetched=versions, read_info=read_info)
    del versions                                                   # decoded frames are no longer needed
    gc.collect()

    # ---- scoring: same calls as LipSyncDetectorV3.analyze
    rep.start("scoring")
    if r["T"] < 25:
        raise ValueError("video shorter than 1 second")
    g, c, ev, seg = LSD.LipSyncDetectorV3.scores(loaded.bundle, r["Cs"], r["E"], loaded.encoder.device)
    out = _analyze_dict(LSD, loaded.bundle, r, g, c, ev, seg, loaded.encoder.device)

    import cv2
    extras = {
        "segment_spans_s": _segment_spans(LSD, r["Cs"], r["E"], r["T"]),
        "waveform": _waveform_envelope(r["audio"], r["T"], LSD.SPF),
        "audio_features": _audio_features(r["audio"], r["T"]),
        "sync_matrix": np.round(np.asarray(r["Cs"][0], np.float32), 3),
        "offset_drift": [[round(t, 2), int(o) * 40] for t, o in LSD.window_offsets(r["Cs"][0])],
        "crops": _crops_png(out["mouth_crops"], cv2),
        "timing_z_all": r.get("timing_z_all"),
    }
    rep.end("scoring")
    return out, extras, rep.times


def analyze_to_result(loaded, path: str, file_info: dict, emit, cancel_event) -> dict:
    t0 = time.time()
    out, extras, times = run_analysis(loaded, path, file_info, emit, cancel_event)
    return build_result(out, extras, file_info, loaded.meta, times, time.time() - t0)
