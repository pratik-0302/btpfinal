"""Job registry (in memory, one job at a time) + local storage of uploads (deleted automatically)."""
from __future__ import annotations

import shutil
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import config
from common.errors import make_error

TERMINAL = {"done", "error", "cancelled"}


@dataclass
class Job:
    id: str
    name: str
    state: str = "queued"                     # queued -> running -> done | error | cancelled
    created: float = field(default_factory=time.time)
    file_info: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    result: dict | None = None
    error: dict | None = None

    @property
    def dir(self):
        return config.JOBS_DIR / self.id

    def public(self, with_result: bool = True) -> dict:
        d = {"id": self.id, "name": self.name, "state": self.state, "created": self.created,
             "file": self.file_info, "error": self.error}
        if with_result:
            d["result"] = self.result
        return d


class JobManager:
    def __init__(self):
        self.jobs: "OrderedDict[str, Job]" = OrderedDict()
        self.lock = threading.Lock()
        self.bridge = None                    # set by the app

    # ------------------------------------------------------------ storage
    @staticmethod
    def clean_storage() -> None:
        """Remove leftovers of earlier runs (uploads are never kept between runs)."""
        if config.JOBS_DIR.exists():
            shutil.rmtree(config.JOBS_DIR, ignore_errors=True)
        config.JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def new_job(self, name: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:16], name=name)
        job.dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.jobs[job.id] = job
            while len(self.jobs) > config.MAX_RESULTS_IN_MEMORY:
                old_id, old = self.jobs.popitem(last=False)
                shutil.rmtree(old.dir, ignore_errors=True)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def active(self) -> Job | None:
        for j in self.jobs.values():
            if j.state in ("queued", "running"):
                return j
        return None

    def _cleanup_files(self, job: Job) -> None:
        if not config.KEEP_UPLOADS:
            shutil.rmtree(job.dir, ignore_errors=True)

    def add_event(self, job: Job, ev: dict) -> None:
        ev = {**ev, "t": time.time()}
        with self.lock:
            job.events.append(ev)

    def finish(self, job: Job, state: str, result: dict | None = None, error: dict | None = None) -> None:
        if job.state in TERMINAL:
            return
        job.result, job.error = result, error
        job.state = state
        if state == "done":
            self.add_event(job, {"type": "result"})
        else:
            self.add_event(job, {"type": "error" if state == "error" else "cancelled", "error": error})
        self._cleanup_files(job)

    def remove(self, job: Job) -> None:
        shutil.rmtree(job.dir, ignore_errors=True)
        with self.lock:
            self.jobs.pop(job.id, None)

    # ------------------------------------------------------------ worker messages (reader thread)
    def on_worker_message(self, msg: dict) -> None:
        t = msg.get("type")
        job = self.jobs.get(msg.get("job_id", "")) if "job_id" in msg else self.active()
        if t == "event" and job:
            self.add_event(job, dict(msg.get("event") or {}))
        elif t == "result" and job:
            self.finish(job, "done", result=msg.get("result"))
        elif t == "error" and job:
            err = msg.get("error") or make_error("INTERNAL").to_dict()
            self.finish(job, "cancelled" if err.get("code") == "CANCELLED" else "error", error=err)
        elif t == "died":
            code = msg.get("exitcode")
            if job and job.state not in TERMINAL:
                if code is not None and code < 0:
                    detail = f"worker stopped by signal {-code}" + (" (the system may have run out of memory)" if code == -9 else "")
                    err = make_error("OUT_OF_MEMORY" if code == -9 else "WORKER_CRASHED", detail, None)
                else:
                    err = make_error("WORKER_CRASHED", f"exit code {code}")
                self.finish(job, "error", error=err.to_dict())
            if self.bridge is not None:                        # bring the model back for the next video
                threading.Thread(target=self.bridge.restart, daemon=True).start()
