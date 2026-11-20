"""The inference worker process: loads the model once, then analyses one video at a time.

Started by the API with multiprocessing (spawn). Messages over a Pipe:
  worker -> API:  {"type": "loading", "message"}  {"type": "ready", "meta"}  {"type": "fatal", "error"}
                  {"type": "event", "job_id", "event": {...}}  {"type": "result", "job_id", "result"}
                  {"type": "error", "job_id", "error"}
  API -> worker:  {"type": "analyze", "job_id", "path", "file_info", "tmp_dir"}  {"type": "shutdown"}
Cancellation uses a shared Event that is checked between stages and between MTCNN batches.
"""
from __future__ import annotations

import gc
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _setup_path() -> None:
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)


def _free_memory_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        return None


def worker_entry(conn, cancel_event) -> None:
    _setup_path()
    import config
    from common.errors import classify_exception, make_error

    def send(msg: dict) -> None:
        try:
            conn.send(msg)
        except (BrokenPipeError, EOFError, OSError):
            os._exit(0)                                   # the API is gone: nothing left to do

    try:
        from worker.loader import load_detector
        loaded = load_detector(progress=lambda m: send({"type": "loading", "message": m}))
    except FileNotFoundError as e:
        send({"type": "fatal", "error": make_error("MODEL_FILES_MISSING", str(e), "startup").to_dict()})
        return
    except BaseException as e:                            # noqa: BLE001  (report everything at startup)
        send({"type": "fatal", "error": classify_exception(e, "startup").to_dict()})
        return
    send({"type": "ready", "meta": loaded.meta})

    from worker.engine import analyze_to_result
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg.get("type") == "shutdown":
            break
        if msg.get("type") != "analyze":
            continue
        job_id = msg["job_id"]
        stage = {"name": None}

        def emit(ev: dict, job_id=job_id) -> None:
            if ev.get("type") == "stage":
                stage["name"] = ev.get("stage")
            send({"type": "event", "job_id": job_id, "event": ev})

        file_info = dict(msg.get("file_info") or {})
        free = _free_memory_gb()
        if free is not None and free < config.LOW_MEMORY_WARNING_GB:
            w = f"Only {free:.1f} GB of memory is free; the analysis may be slow or fail."
            file_info.setdefault("warnings", []).append(w)
            emit({"type": "warning", "message": w})

        # the original ffmpeg fallback writes to tempfile.mkdtemp(): keep it inside this job's folder
        tmp_dir = msg.get("tmp_dir")
        if tmp_dir:
            os.makedirs(tmp_dir, exist_ok=True)
            tempfile.tempdir = tmp_dir
        try:
            result = analyze_to_result(loaded, msg["path"], file_info, emit, cancel_event)
            send({"type": "result", "job_id": job_id, "result": result})
        except BaseException as e:                        # noqa: BLE001  (MemoryError, KeyboardInterrupt, ...)
            send({"type": "error", "job_id": job_id, "error": classify_exception(e, stage["name"]).to_dict()})
        finally:
            tempfile.tempdir = None
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            result = None
            gc.collect()
            if "mps" in (loaded.device, loaded.cropper_device):
                try:
                    import torch
                    torch.mps.empty_cache()
                except Exception:
                    pass
