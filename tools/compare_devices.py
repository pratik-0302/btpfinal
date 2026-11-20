"""Stage-3 check: may this Mac use the Apple GPU (MPS)?

Runs the original analyze() on the six reference videos three times, each in its own process:
  cpu      : AV-HuBERT and MTCNN on the CPU (reference)
  mps      : AV-HuBERT on MPS, MTCNN on the CPU
  mps_all  : AV-HuBERT and MTCNN on MPS
and compares with the CPU run. A configuration passes when, on every video, the verdict is identical,
the fake score differs by at most 0.5 points and both raw classifier outputs differ by at most 1e-3.
Writes models/device_check.json, which LIPSYNC_DEVICE=auto reads (MPS is used only if it passed).

usage:
  LIPSYNC_MANIFEST=... LIPSYNC_DATASET=... python tools/compare_devices.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAX_PCT, MAX_P = 0.5, 1e-3
CONFIGS = {"cpu": ("cpu", "cpu"), "mps": ("mps", "cpu"), "mps_all": ("mps", "mps")}


def child(out_path: str) -> None:
    import resource
    import config
    from tools.reference import find_reference_videos
    from worker.loader import load_detector
    t = time.time()
    loaded = load_detector()
    load_s = time.time() - t
    rows = []
    for path, _, _ in find_reference_videos():
        t = time.time()
        r = loaded.detector.analyze(str(path), max_seconds=config.MAX_SECONDS)
        rows.append({"file": path.name, "verdict": r["verdict"], "pct": r["percent_fake"], "p_sync": r["p_out_of_sync"],
                     "p_deepfake": r["p_deepfake"], "seconds": time.time() - t})
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3
    Path(out_path).write_text(json.dumps({"rows": rows, "load_s": load_s, "peak_rss_gb": peak,
                                          "torch": loaded.meta["torch"]}))


def main() -> None:
    import config
    import torch
    if not (torch.backends.mps.is_available()):
        raise SystemExit("MPS is not available on this machine: the app will use the CPU.")
    res = {}
    for name, (dev, crop) in CONFIGS.items():
        print(f"running {name} (AV-HuBERT on {dev}, MTCNN on {crop}) ...", flush=True)
        out = config.DATA_DIR / f"device_{name}.json"
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "LIPSYNC_DEVICE": dev, "LIPSYNC_CROPPER_DEVICE": crop}
        p = subprocess.run([sys.executable, __file__, "--child", str(out)], env=env, cwd=ROOT)
        res[name] = json.loads(out.read_text()) if p.returncode == 0 and out.exists() else None
        if res[name] is None:
            print(f"  {name} FAILED to run (exit code {p.returncode})")
    ref = res["cpu"]
    if ref is None:
        raise SystemExit("the CPU reference run failed")

    def passes(name):
        r = res.get(name)
        if r is None:
            return False, ["did not run"]
        bad = []
        for a, b in zip(ref["rows"], r["rows"]):
            if a["verdict"] != b["verdict"]:
                bad.append(f"{a['file']}: verdict {a['verdict']} vs {b['verdict']}")
            if abs(a["pct"] - b["pct"]) > MAX_PCT:
                bad.append(f"{a['file']}: fake score Δ {abs(a['pct'] - b['pct']):.3f}")
            for k in ("p_sync", "p_deepfake"):
                if abs(a[k] - b[k]) > MAX_P:
                    bad.append(f"{a['file']}: {k} Δ {abs(a[k] - b[k]):.2e}")
        return not bad, bad

    report = {"torch": ref["torch"], "date": time.strftime("%Y-%m-%d %H:%M")}
    for name in ("mps", "mps_all"):
        ok, bad = passes(name)
        report[name] = {"pass": ok, "problems": bad, **({k: res[name][k] for k in ("load_s", "peak_rss_gb")} if res[name] else {}),
                        "seconds_per_video": [round(x["seconds"], 1) for x in res[name]["rows"]] if res[name] else None}
        print(f"{name}: {'PASS' if ok else 'FAIL'} {bad[:3]}")
    report["cpu"] = {"load_s": ref["load_s"], "peak_rss_gb": ref["peak_rss_gb"],
                     "seconds_per_video": [round(x["seconds"], 1) for x in ref["rows"]]}
    report["avhubert_mps_ok"] = report["mps"]["pass"]
    report["mtcnn_mps_ok"] = report["mps"]["pass"] and report["mps_all"]["pass"]
    config.DEVICE_CHECK.write_text(json.dumps(report, indent=2))
    print(f"\nwritten {config.DEVICE_CHECK}: AV-HuBERT on MPS {'allowed' if report['avhubert_mps_ok'] else 'NOT allowed'}, "
          f"MTCNN on MPS {'allowed' if report['mtcnn_mps_ok'] else 'NOT allowed'}")
    print("CPU s/video:", report["cpu"]["seconds_per_video"], "| MPS s/video:", report["mps"].get("seconds_per_video"))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
