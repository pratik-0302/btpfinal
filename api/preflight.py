"""Quick file check in the API process (PyAV only, no torch): fills FileInfo and rejects files that the
detector would certainly reject, before the worker spends time on them."""
from __future__ import annotations

import shutil
from pathlib import Path

import config
from common.errors import ErrorInfo, make_error


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def probe(path: Path, original_name: str) -> tuple[dict, ErrorInfo | None]:
    import av

    info: dict = {"name": original_name, "size_bytes": path.stat().st_size, "warnings": []}
    try:
        c = av.open(str(path))
    except Exception as e:                                     # noqa: BLE001
        if shutil.which("ffmpeg"):
            # the detector's ffmpeg fallback may still read it: let the worker try
            info["warnings"].append("PyAV could not open the file; the detector will try its ffmpeg fallback.")
            info["container"] = None
            return info, None
        return info, make_error("FILE_UNREADABLE", f"{type(e).__name__}: {e}", "preflight")
    try:
        info["container"] = c.format.name if c.format else None
        dur = _f(c.duration / 1_000_000) if c.duration else None
        vs = c.streams.video[0] if c.streams.video else None
        as_ = c.streams.audio[0] if c.streams.audio else None
        if vs is not None:
            cc = vs.codec_context
            info.update({"video_codec": cc.name if cc else None, "width": cc.width if cc else None,
                         "height": cc.height if cc else None,
                         "stored_fps": _f(vs.average_rate) if vs.average_rate else None,
                         "frames": int(vs.frames) if vs.frames else None})
            rot = None
            try:
                rot = vs.metadata.get("rotate")
            except Exception:
                pass
            info["rotate_tag"] = rot
            if vs.duration and vs.time_base:
                dur = dur or _f(vs.duration * vs.time_base)
        if as_ is not None:
            ac = as_.codec_context
            info.update({"audio_codec": ac.name if ac else None, "audio_rate": as_.rate,
                         "audio_channels": getattr(ac, "channels", None) if ac else None})
        info["duration_s"] = dur
        info["has_video"], info["has_audio"] = vs is not None, as_ is not None
    finally:
        c.close()

    if vs is None:
        return info, make_error("NO_VIDEO_STREAM", stage="preflight")
    if as_ is None:
        return info, make_error("NO_AUDIO", stage="preflight")
    if dur is not None and dur < config.MIN_PREFLIGHT_SECONDS:
        return info, make_error("TOO_SHORT", f"duration {dur:.2f} s", "preflight")
    if dur is not None and dur > config.MAX_SECONDS:
        info["warnings"].append(f"The video is {dur:.0f} s long; only the first {config.MAX_SECONDS} s are analysed.")
    return info, None
