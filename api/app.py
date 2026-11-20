"""FastAPI application: local-only REST API + SSE progress + the built frontend.

Run with:  python server.py        (binds 127.0.0.1 only)
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

import config
from api.jobs import TERMINAL, Job, JobManager
from api.preflight import probe
from api.worker_bridge import WorkerBridge
from common import model_docs
from common.errors import make_error

jobs = JobManager()
bridge = WorkerBridge(jobs.on_worker_message)
jobs.bridge = bridge
_upload_lock = asyncio.Lock()
CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    JobManager.clean_storage()
    bridge.start()
    yield
    bridge.stop()
    if not config.KEEP_UPLOADS:
        JobManager.clean_storage()


app = FastAPI(title="Lip-sync deepfake detector (local)", lifespan=lifespan, docs_url="/api/docs", redoc_url=None,
              openapi_url="/api/openapi.json")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.ALLOWED_HOSTS)   # blocks DNS rebinding


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    return resp


def error_response(status: int, code: str, detail: str = "", stage: str | None = None, job_id: str | None = None):
    body = {"error": make_error(code, detail, stage).to_dict()}
    if job_id:
        body["job_id"] = job_id
    return JSONResponse(body, status_code=status)


def require_client_header(request: Request) -> None:
    """State-changing requests must carry a custom header. Browsers do not let other websites add custom
    headers without a CORS preflight, which this server never approves, so a page on another site cannot
    upload or cancel anything here."""
    if request.headers.get(config.CLIENT_HEADER) != "1":
        raise HTTPException(status_code=403, detail="missing client header")


def get_job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    return job


# ------------------------------------------------------------------ status
@app.get("/api/health")
def health():
    active = jobs.active()
    return {"worker": bridge.state, "loading_message": bridge.loading_message, "meta": bridge.meta,
            "fatal_error": bridge.fatal_error, "restarts": bridge.restarts,
            "worker_pid": bridge.proc.pid if bridge.proc is not None else None,
            "active_job": active.id if active else None}


@app.get("/api/model")
def model_info():
    return {"meta": bridge.meta, "definitions": model_docs.SCORE_DEFINITIONS, "new_displays": model_docs.NEW_DISPLAYS,
            "evaluation": model_docs.EVALUATION, "limitations": model_docs.LIMITATIONS,
            "limits": {"max_upload_mb": config.MAX_UPLOAD_BYTES // (1024 * 1024), "max_seconds": config.MAX_SECONDS,
                       "extensions": sorted(config.VIDEO_EXT), "keep_uploads": config.KEEP_UPLOADS}}


# ------------------------------------------------------------------ jobs
@app.post("/api/jobs", status_code=202, dependencies=[Depends(require_client_header)])
async def create_job(request: Request, file: UploadFile = File(...)):
    name = Path(file.filename or "video").name
    ext = Path(name).suffix.lower()
    if ext not in config.VIDEO_EXT:
        return error_response(415, "UNSUPPORTED_EXTENSION", f"extension '{ext or '(none)'}'")
    if _upload_lock.locked() or jobs.active() or bridge.state == "busy":
        return error_response(409, "BUSY")
    if bridge.state != "ready":
        code = "MODEL_FILES_MISSING" if (bridge.fatal_error or {}).get("code") == "MODEL_FILES_MISSING" else "WORKER_NOT_READY"
        return error_response(503, code, f"worker state: {bridge.state}")
    async with _upload_lock:
        job = jobs.new_job(name)
        target = job.dir / f"upload{ext}"
        size = 0
        try:
            with open(target, "wb") as f:
                while chunk := await file.read(CHUNK):
                    size += len(chunk)
                    if size > config.MAX_UPLOAD_BYTES:
                        raise ValueError("too large")
                    f.write(chunk)
        except ValueError:
            jobs.remove(job)
            return error_response(413, "UPLOAD_TOO_LARGE", f"limit {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        finally:
            await file.close()
        if size == 0:
            jobs.remove(job)
            return error_response(400, "EMPTY_FILE")

        job.state = "running"
        jobs.add_event(job, {"type": "stage", "stage": "preflight", "state": "running"})
        info, err = await run_in_threadpool(probe, target, name)
        job.file_info = info
        if err is not None:
            jobs.finish(job, "error", error=err.to_dict())
            return error_response(422, err.code, err.detail, "preflight", job.id)
        jobs.add_event(job, {"type": "stage", "stage": "preflight", "state": "done"})
        for w in info.get("warnings", []):
            jobs.add_event(job, {"type": "warning", "message": w})
        try:
            bridge.submit(job.id, str(target), info, str(job.dir / "tmp"))
        except RuntimeError as e:
            jobs.finish(job, "error", error=make_error("WORKER_NOT_READY", str(e)).to_dict())
            return error_response(503, "WORKER_NOT_READY", str(e), job_id=job.id)
    return {"job": job.public(with_result=False)}


@app.get("/api/jobs/{job_id}")
def job_status(job: Job = Depends(get_job)):
    return {"job": job.public()}


@app.get("/api/jobs/{job_id}/events", response_class=EventSourceResponse)
async def job_events(request: Request, job: Job = Depends(get_job)) -> AsyncIterable[ServerSentEvent]:
    """Replays all events of the job, then streams new ones until the job ends."""
    try:
        i = int(request.headers.get("last-event-id", "0"))
    except ValueError:
        i = 0
    while True:
        with jobs.lock:
            new = job.events[i:]
        for ev in new:
            i += 1
            yield ServerSentEvent(data=ev, event=ev.get("type", "message"), id=str(i))
        if job.state in TERMINAL and i >= len(job.events):
            break
        if await request.is_disconnected():
            break
        await asyncio.sleep(0.25)


@app.delete("/api/jobs/{job_id}", dependencies=[Depends(require_client_header)])
def delete_job(job: Job = Depends(get_job)):
    if job.state not in TERMINAL:
        bridge.cancel()                      # the worker stops at the next stage / MTCNN batch
        return JSONResponse({"state": "cancelling"}, status_code=202)
    jobs.remove(job)
    return {"state": "deleted"}


@app.post("/api/worker/restart", dependencies=[Depends(require_client_header)])
def restart_worker():
    active = jobs.active()
    if active is not None:
        jobs.finish(active, "cancelled", error=make_error("CANCELLED", "worker restarted").to_dict())
    threading.Thread(target=bridge.restart, daemon=True).start()
    return {"worker": "restarting"}


# ------------------------------------------------------------------ frontend
if (config.FRONTEND_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/", response_class=HTMLResponse)
    def no_frontend():
        return ("<h3>Frontend not built</h3><p>Run <code>cd frontend && npm install && npm run build</code>, "
                "then restart the server. The API is available under /api.</p>")
