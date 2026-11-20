"""Error codes shared by the API and the worker, and the mapping from exceptions to codes.

Only plain Python here: this module is imported by the API process, which must stay free of torch.
"""
from __future__ import annotations

import subprocess
import traceback
from dataclasses import asdict, dataclass

MESSAGES = {
    "UNSUPPORTED_EXTENSION": "This file type is not supported. Use a common video format such as MP4, MOV, MKV or WEBM.",
    "UPLOAD_TOO_LARGE": "The file is larger than the upload limit.",
    "EMPTY_FILE": "The uploaded file is empty.",
    "FILE_UNREADABLE": "This file cannot be read: the format is not supported or the file is damaged.",
    "NO_VIDEO_STREAM": "The file has no video track.",
    "NO_AUDIO": "The file has no audio track, so lip sync cannot be checked.",
    "TOO_SHORT": "The video is shorter than 1 second.",
    "NO_FACE": "No face was found in the video.",
    "OUT_OF_MEMORY": "The analysis ran out of memory. Close other applications and try again, or use a shorter video.",
    "WORKER_CRASHED": "The analysis process stopped unexpectedly.",
    "WORKER_NOT_READY": "The model is not ready yet.",
    "MODEL_FILES_MISSING": "The model files were not found.",
    "BUSY": "Another video is being analysed. Wait for it to finish.",
    "CANCELLED": "The analysis was cancelled.",
    "NOT_FOUND": "No such analysis.",
    "INTERNAL": "An unexpected error occurred.",
}


@dataclass
class ErrorInfo:
    code: str
    message: str
    detail: str = ""
    stage: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def make_error(code: str, detail: str = "", stage: str | None = None, message: str | None = None) -> ErrorInfo:
    return ErrorInfo(code=code, message=message or MESSAGES.get(code, MESSAGES["INTERNAL"]), detail=detail, stage=stage)


class Cancelled(Exception):
    """Raised inside the worker when the user cancels a job."""


def classify_exception(e: BaseException, stage: str | None = None) -> ErrorInfo:
    """Map an exception raised by the original detector code (or the worker) to an error code."""
    name = type(e).__name__
    text = str(e)
    low = text.lower()
    detail = f"{name}: {text}"[:2000]
    if isinstance(e, Cancelled):
        return make_error("CANCELLED", stage=stage)
    if name == "NoFaceError":                                  # lipsync_detector.NoFaceError
        return make_error("NO_FACE", detail, stage)
    if isinstance(e, MemoryError) or "out of memory" in low:   # incl. "MPS backend out of memory"
        return make_error("OUT_OF_MEMORY", detail, stage)
    if isinstance(e, ValueError):
        if "no audio track" in low:
            return make_error("NO_AUDIO", detail, stage)
        if "no video stream" in low:
            return make_error("NO_VIDEO_STREAM", detail, stage)
        if "shorter than 1 second" in low:
            return make_error("TOO_SHORT", detail, stage)
        if "no video frames" in low:
            return make_error("FILE_UNREADABLE", detail, stage)
    if isinstance(e, subprocess.CalledProcessError):           # ffmpeg fallback could not convert the file
        return make_error("FILE_UNREADABLE", detail + " (ffmpeg fallback failed)", stage)
    if isinstance(e, FileNotFoundError) and "ffmpeg" in low:   # ffmpeg fallback needed but not installed
        return make_error("FILE_UNREADABLE", detail + " (PyAV could not read the file and ffmpeg is not installed)", stage)
    if type(e).__module__.startswith("av"):                    # PyAV / FFmpeg errors
        return make_error("FILE_UNREADABLE", detail, stage)
    return make_error("INTERNAL", detail + "\n" + "".join(traceback.format_exception(e))[-3000:], stage)
