"""Real progress reporting without changing the original pipeline.

process_video(path, cropper, encoder, ...) only calls cropper.crops(frames) and encoder.embed(crops, audio)
and reads encoder.layers / encoder.device. The proxies below pass those calls straight through to the
original objects and emit "stage started / finished" events around them. Face detection also reports
the real number of MTCNN batches done (the total is known before it starts).
"""
from __future__ import annotations

import math
import time

from common.errors import Cancelled

STAGES = ["preflight", "decode", "faces", "embeddings", "scoring"]


class StageReporter:
    def __init__(self, emit, cancel_event):
        self.emit, self.cancel = emit, cancel_event
        self.times: dict[str, float] = {}
        self._t0: dict[str, float] = {}

    def check_cancel(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled()

    def start(self, stage: str, **extra) -> None:
        self.check_cancel()
        self._t0[stage] = time.time()
        self.emit({"type": "stage", "stage": stage, "state": "running", **extra})

    def progress(self, stage: str, done: int, total: int, **extra) -> None:
        self.emit({"type": "stage", "stage": stage, "state": "running", "done": done, "total": total, **extra})

    def end(self, stage: str, **extra) -> None:
        dt = time.time() - self._t0.pop(stage, time.time())
        self.times[stage] = round(self.times.get(stage, 0.0) + dt, 3)
        self.emit({"type": "stage", "stage": stage, "state": "done", "seconds": round(dt, 3), **extra})


def mtcnn_batches(n_frames: int, stride: int, batch: int) -> int:
    """Number of MTCNN calls MouthCropper.landmarks makes (same index rule as the original code)."""
    if n_frames <= 0:
        return 0
    det_idx = list(range(0, n_frames, stride))
    if det_idx[-1] != n_frames - 1:
        det_idx.append(n_frames - 1)
    return math.ceil(len(det_idx) / batch)


class _CountingDetector:
    """Wraps MTCNN: forwards detect() unchanged and counts the calls."""

    def __init__(self, mtcnn, on_call):
        self._m, self._on_call = mtcnn, on_call

    def detect(self, *args, **kwargs):
        out = self._m.detect(*args, **kwargs)
        self._on_call()
        return out

    def __getattr__(self, name):
        return getattr(self._m, name)


class CropperProxy:
    def __init__(self, cropper, rep: StageReporter, n_versions: int):
        self._c, self._rep, self._n = cropper, rep, n_versions
        self._calls = 0

    def crops(self, frames, pts=None, info=None):
        self._calls += 1
        ver = {"version": self._calls, "versions": self._n}
        total = mtcnn_batches(len(frames), self._c.det_stride, self._c.batch) if pts is None else 0
        done = [0]

        def on_call():
            done[0] += 1
            self._rep.progress("faces", done[0], total, **ver)
            self._rep.check_cancel()

        self._rep.start("faces", total=total, **ver)
        original = self._c.mtcnn
        self._c.mtcnn = _CountingDetector(original, on_call)
        try:
            out = self._c.crops(frames, pts, info)
        finally:
            self._c.mtcnn = original
        self._rep.end("faces", **ver)
        return out

    def __getattr__(self, name):
        return getattr(self._c, name)


class EncoderProxy:
    def __init__(self, encoder, rep: StageReporter, n_versions: int):
        self._e, self._rep, self._n = encoder, rep, n_versions
        self._calls = 0

    def embed(self, crops, audio):
        self._calls += 1
        ver = {"version": self._calls, "versions": self._n}
        self._rep.start("embeddings", **ver)
        out = self._e.embed(crops, audio)
        self._rep.end("embeddings", **ver)
        return out

    def __getattr__(self, name):
        return getattr(self._e, name)
