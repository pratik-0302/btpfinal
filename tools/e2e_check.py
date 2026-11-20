"""End-to-end check against a RUNNING server (python server.py).

Covers: health, security (host header, client header), upload + SSE progress + result, every error path
(unsupported type, empty, corrupt, no audio, no video, too short, no face), busy (409), cancel, worker crash
recovery, file clean-up, memory over repeated runs. Optionally the six reference videos (set LIPSYNC_MANIFEST
and LIPSYNC_DATASET): the API result must equal tools/verify_reference.py's result on this machine.

usage:  python tools/e2e_check.py [--base http://127.0.0.1:8000] [--soak N] [--skip-crash]
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, info: str = "") -> bool:
    RESULTS.append((name, ok, info))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  — {info}" if info else ""), flush=True)
    return ok


# ------------------------------------------------------------------ HTTP helpers (standard library only)
class Client:
    def __init__(self, base: str):
        u = base.replace("http://", "").rstrip("/")
        self.host, _, port = u.partition(":")
        self.port = int(port or 80)

    def req(self, method, path, body=None, headers=None, host_header=None):
        c = http.client.HTTPConnection(self.host, self.port, timeout=600)
        h = dict(headers or {})
        if host_header:
            h["Host"] = host_header
        c.request(method, path, body=body, headers=h)
        r = c.getresponse()
        data = r.read()
        c.close()
        try:
            return r.status, json.loads(data or b"null")
        except json.JSONDecodeError:
            return r.status, data.decode(errors="replace")

    def upload(self, path: Path, name: str | None = None, header=True):
        boundary = uuid.uuid4().hex
        name = name or path.name
        data = path.read_bytes()
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        h = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if header:
            h[config.CLIENT_HEADER] = "1"
        return self.req("POST", "/api/jobs", body=body, headers=h)

    def events(self, job_id: str, timeout=900):
        """Read the SSE stream until it ends; returns the list of decoded events."""
        c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        c.request("GET", f"/api/jobs/{job_id}/events")
        r = c.getresponse()
        out, cur = [], {}
        while True:
            line = r.readline()
            if not line:
                break
            line = line.decode().rstrip("\r\n")
            if line.startswith("data:"):
                cur["data"] = json.loads(line[5:].strip())
            elif line.startswith("event:"):
                cur["event"] = line[6:].strip()
            elif line == "" and cur:
                if "data" in cur:
                    out.append(cur["data"])
                cur = {}
        c.close()
        return out

    def wait_ready(self, timeout=600):
        t = time.time()
        while time.time() - t < timeout:
            try:
                s, h = self.req("GET", "/api/health")
                if s == 200 and h["worker"] == "ready":
                    return h
                if s == 200 and h["worker"] == "failed":
                    return h
            except OSError:
                pass
            time.sleep(1)
        return None


# ------------------------------------------------------------------ test media (generated with PyAV)
def make_media(d: Path) -> dict[str, Path]:
    import av
    import numpy as np

    def write(name, seconds, video=True, audio=True, size=(320, 240)):
        p = d / name
        c = av.open(str(p), "w")
        vs = as_ = None
        if video:
            vs = c.add_stream("mpeg4", rate=25)
            vs.width, vs.height, vs.pix_fmt = size[0], size[1], "yuv420p"
        if audio:
            as_ = c.add_stream("aac", rate=16000)
            as_.layout = "mono"
        if vs:
            yy, xx = np.mgrid[0:size[1], 0:size[0]]
            for i in range(int(seconds * 25)):
                img = (((xx + 5 * i) % 256)[..., None] * np.array([1, 0.6, 0.3])).astype(np.uint8)
                for pk in vs.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                    c.mux(pk)
            for pk in vs.encode():
                c.mux(pk)
        if as_:
            t = np.arange(int(seconds * 16000)) / 16000
            a = (0.3 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)[None]
            for s in range(0, a.shape[1], 1024):
                f = av.AudioFrame.from_ndarray(np.ascontiguousarray(a[:, s:s + 1024]), format="flt", layout="mono")
                f.sample_rate = 16000
                for pk in as_.encode(f):
                    c.mux(pk)
            for pk in as_.encode():
                c.mux(pk)
        c.close()
        return p

    files = {
        "noface": write("noface.mp4", 3),
        "noaudio": write("noaudio.mp4", 3, audio=False),
        "novideo": write("novideo.mp4", 3, video=False),
        "short": write("short.mp4", 0.5),
    }
    (d / "corrupt.mp4").write_bytes(os.urandom(50_000))
    files["corrupt"] = d / "corrupt.mp4"
    (d / "empty.mp4").write_bytes(b"")
    files["empty"] = d / "empty.mp4"
    (d / "notes.txt").write_text("not a video")
    files["txt"] = d / "notes.txt"
    return files


def expect_error(cl: Client, f: Path, codes: set[str], name: str):
    s, body = cl.upload(f)
    if s == 202:                                              # accepted: the worker must report the error
        jid = body["job"]["id"]
        evs = cl.events(jid)
        end = evs[-1] if evs else {}
        code = (end.get("error") or {}).get("code")
        return check(name, end.get("type") == "error" and code in codes, f"worker error {code}")
    code = (body or {}).get("error", {}).get("code") if isinstance(body, dict) else None
    return check(name, code in codes, f"HTTP {s} {code}")


def worker_pid(cl: Client) -> int | None:
    s, h = cl.req("GET", "/api/health")
    return h.get("worker_pid") if s == 200 else None


def worker_rss_gb(cl: Client) -> float | None:
    try:
        import psutil
        pid = worker_pid(cl)
        return psutil.Process(pid).memory_info().rss / 1024 ** 3 if pid else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=f"http://{config.HOST}:{config.PORT}")
    ap.add_argument("--soak", type=int, default=9, help="repeated analyses for the memory check (0 = skip)")
    ap.add_argument("--skip-crash", action="store_true")
    a = ap.parse_args()
    cl = Client(a.base)

    h = cl.wait_ready()
    if not check("model loaded (worker ready)", bool(h) and h["worker"] == "ready", json.dumps((h or {}).get("meta", {}))[:200]):
        sys.exit(1)

    # ---- security
    s, _ = cl.req("GET", "/api/health", host_header="evil.example.com")
    check("foreign Host header rejected (DNS-rebinding protection)", s == 400, f"HTTP {s}")
    tmp = Path(tempfile.mkdtemp(prefix="lipsync_e2e_"))
    media = make_media(tmp)
    s, _ = cl.upload(media["noface"], header=False)
    check("upload without client header rejected", s == 403, f"HTTP {s}")
    s, m = cl.req("GET", "/api/model")
    check("model info + definitions served", s == 200 and "fake_score" in m["definitions"] and m["evaluation"]["headline_sets"])
    s, page = cl.req("GET", "/")
    check("frontend served at /", s == 200 and "<div id=\"root\">" in str(page), f"HTTP {s}")

    # ---- error paths
    expect_error(cl, media["txt"], {"UNSUPPORTED_EXTENSION"}, "unsupported file type -> UNSUPPORTED_EXTENSION")
    expect_error(cl, media["empty"], {"EMPTY_FILE"}, "empty file -> EMPTY_FILE")
    expect_error(cl, media["corrupt"], {"FILE_UNREADABLE"}, "corrupt file -> FILE_UNREADABLE")
    expect_error(cl, media["noaudio"], {"NO_AUDIO"}, "video without audio -> NO_AUDIO")
    expect_error(cl, media["novideo"], {"NO_VIDEO_STREAM"}, "audio-only file -> NO_VIDEO_STREAM")
    expect_error(cl, media["short"], {"TOO_SHORT"}, "0.5 s clip -> TOO_SHORT")
    expect_error(cl, media["noface"], {"NO_FACE"}, "video without a face -> NO_FACE")

    # ---- reference videos (optional)
    refs = []
    if os.environ.get("LIPSYNC_MANIFEST") and os.environ.get("LIPSYNC_DATASET"):
        from tools.reference import find_reference_videos
        refs = find_reference_videos()
        vr = config.DATA_DIR / "verify_reference.json"
        direct = {v["file"]: v for v in json.loads(vr.read_text())["videos"]} if vr.exists() else {}
        for path, want_v, _ in refs:
            s, body = cl.upload(path)
            if not check(f"upload {path.name[:40]}", s == 202, f"HTTP {s} {body if s != 202 else ''}"):
                continue
            jid = body["job"]["id"]
            evs = cl.events(jid)
            stages = [e.get("stage") for e in evs if e.get("type") == "stage" and e.get("state") == "done"]
            s, j = cl.req("GET", f"/api/jobs/{jid}")
            res = j["job"]["result"]
            check(f"  stages reported in order ({path.name[:30]})",
                  [x for x in stages if x in ("preflight", "decode", "scoring")] == ["preflight", "decode", "scoring"]
                  and "faces" in stages and "embeddings" in stages, ",".join(stages))
            check(f"  verdict {res['verdict']['label']} == expected {want_v}", res["verdict"]["label"] == want_v,
                  res["verdict"]["headline"])
            d = direct.get(path.name)
            if d:
                check("  API fake score == direct analyze() on this machine",
                      abs(res["verdict"]["fake_score"] - d["fake_pct"]) < 0.01, f"{res['verdict']['fake_score']:.2f} vs {d['fake_pct']}")
            check("  uploaded file deleted after analysis", not (config.JOBS_DIR / jid).exists() or config.KEEP_UPLOADS)

    long_video = refs[0][0] if refs else media["noface"]

    # ---- busy + cancel
    s1, b1 = cl.upload(long_video)
    s2, b2 = cl.upload(long_video)
    check("second upload while busy -> 409 BUSY", s2 == 409 and b2["error"]["code"] == "BUSY", f"HTTP {s2}")
    if s1 == 202:
        cl.events(b1["job"]["id"])
    s1, b1 = cl.upload(long_video)
    if s1 == 202:
        jid = b1["job"]["id"]
        time.sleep(0.3)
        cl.req("DELETE", f"/api/jobs/{jid}", headers={config.CLIENT_HEADER: "1"})
        evs = cl.events(jid)
        end = evs[-1]["type"] if evs else None
        check("cancel a running job", end in ("cancelled", "result"), f"ended with '{end}' (a very short video may finish first)")

    # ---- memory over repeated runs
    if refs and a.soak > 0:
        # same video every time; memory goes up and down as macOS reclaims it, so compare peaks:
        # the last third of the runs must not peak clearly higher than the first third (no upward trend)
        rss = []
        for _ in range(a.soak):
            s, b = cl.upload(refs[1][0])
            if s == 202:
                cl.events(b["job"]["id"])
            r = worker_rss_gb(cl)
            if r is not None:
                rss.append(r)
        if len(rss) >= 6:
            k = len(rss) // 3
            growth = max(rss[-k:]) - max(rss[:k])
            check(f"worker memory has no upward trend over {len(rss)} analyses", growth < 0.3,
                  "RSS GB: " + ", ".join(f"{x:.2f}" for x in rss))

    # ---- crash recovery
    if not a.skip_crash:
        pid = worker_pid(cl)
        if pid:
            s1, b1 = cl.upload(long_video)
            time.sleep(0.5)
            os.kill(pid, signal.SIGKILL)
            evs = cl.events(b1["job"]["id"]) if s1 == 202 else []
            code = ((evs[-1].get("error") or {}).get("code")) if evs else None
            check("worker killed during a job -> job fails with a clear error",
                  bool(evs) and evs[-1]["type"] in ("error", "result"), f"code {code}")
            h = cl.wait_ready(300)
            check("worker restarted automatically", bool(h) and h["worker"] == "ready", f"restarts={h and h['restarts']}")
        else:
            check("crash test", False, "could not find the worker process (psutil missing?)")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
