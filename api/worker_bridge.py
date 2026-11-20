"""Starts and supervises the single inference worker process and relays its messages."""
from __future__ import annotations

import multiprocessing as mp
import threading
import time
from typing import Callable


class WorkerBridge:
    """state: starting -> ready <-> busy; crashed / failed when the process dies or cannot load the model."""

    def __init__(self, on_message: Callable[[dict], None]):
        self._on_message = on_message
        self._ctx = mp.get_context("spawn")       # macOS default; never fork a process that may hold torch
        self._lock = threading.Lock()
        self.proc = None
        self.conn = None
        self.cancel_event = None
        self.state = "stopped"
        self.meta: dict = {}
        self.loading_message = ""
        self.fatal_error: dict | None = None
        self.started_at = 0.0
        self.restarts = 0

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        with self._lock:
            from worker.main import worker_entry
            parent, child = self._ctx.Pipe()
            self.cancel_event = self._ctx.Event()
            self.proc = self._ctx.Process(target=worker_entry, args=(child, self.cancel_event),
                                          name="lipsync-worker", daemon=True)
            self.state, self.meta, self.fatal_error = "starting", {}, None
            self.loading_message, self.started_at = "starting the worker process", time.time()
            self.proc.start()
            child.close()
            self.conn = parent
            threading.Thread(target=self._reader, args=(parent, self.proc), daemon=True, name="worker-reader").start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            proc, conn = self.proc, self.conn
            self.proc = self.conn = None
            self.state = "stopped"
        if conn is not None:
            try:
                conn.send({"type": "shutdown"})
            except Exception:
                pass
        if proc is not None:
            proc.join(timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join(2)

    def restart(self) -> None:
        self.stop(timeout=2.0)
        self.restarts += 1
        self.start()

    # ------------------------------------------------------------ jobs
    def submit(self, job_id: str, path: str, file_info: dict, tmp_dir: str) -> None:
        with self._lock:
            if self.state != "ready" or self.conn is None:
                raise RuntimeError(f"worker is {self.state}")
            self.cancel_event.clear()
            self.state = "busy"
            self.conn.send({"type": "analyze", "job_id": job_id, "path": path, "file_info": file_info,
                            "tmp_dir": tmp_dir})

    def cancel(self) -> None:
        if self.cancel_event is not None:
            self.cancel_event.set()

    # ------------------------------------------------------------ messages
    def _reader(self, conn, proc) -> None:
        while True:
            try:
                if conn.poll(0.5):
                    msg = conn.recv()
                elif not proc.is_alive():
                    raise EOFError
                else:
                    continue
            except (EOFError, OSError):
                if self.proc is proc and self.state != "failed":   # not an intentional stop / failed load
                    proc.join(1)
                    self.state = "crashed"
                    self._on_message({"type": "died", "exitcode": proc.exitcode})
                return
            t = msg.get("type")
            if t == "loading":
                self.loading_message = msg.get("message", "")
            elif t == "ready":
                self.state, self.meta = "ready", msg.get("meta", {})
            elif t == "fatal":
                self.state, self.fatal_error = "failed", msg.get("error")
            elif t in ("result", "error"):
                if self.state == "busy":
                    self.state = "ready"
            self._on_message(msg)
