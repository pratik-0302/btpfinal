"""Load the detector once: AV-HuBERT (slim or original checkpoint), the MTCNN mouth cropper and the bundle.

Slim checkpoint: tools/make_slim_ckpt.py runs the ORIGINAL load_avhubert once and saves only the weights the
model uses, plus the constructor arguments. Loading it memory-mapped needs ~1.3 GB instead of the ~5 GB peak of
reading the full 3.9 GB checkpoint. make_slim_ckpt.py verifies that every tensor is identical.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import config
from worker.core_import import load_core

SLIM_FORMAT = "avhubert-slim-v1"


@dataclass
class LoadedDetector:
    detector: object                 # lipsync_detector.LipSyncDetectorV3
    encoder: object                  # lipsync_detector.SyncEncoder
    cropper: object                  # lipsync_detector.MouthCropper
    bundle: dict
    LSD: object                      # the lipsync_detector module
    device: str
    cropper_device: str
    meta: dict = field(default_factory=dict)


def _device_check() -> dict:
    try:
        return json.loads(config.DEVICE_CHECK.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_devices(torch) -> tuple[str, str, str]:
    """-> (AV-HuBERT device, MTCNN device, reason)."""
    want = config.DEVICE
    chk = _device_check()
    mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if want == "cpu":
        return "cpu", "cpu", "set by LIPSYNC_DEVICE=cpu"
    if want == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("LIPSYNC_DEVICE=cuda but CUDA is not available")
        return "cuda", "cuda", "set by LIPSYNC_DEVICE=cuda"
    if want == "mps":
        if not mps_ok:
            raise RuntimeError("LIPSYNC_DEVICE=mps but MPS is not available")
        crop = "mps" if chk.get("mtcnn_mps_ok") else "cpu"
        return "mps", crop, "set by LIPSYNC_DEVICE=mps"
    # auto: use MPS only where compare_devices.py verified it on this machine with this torch version
    if mps_ok and chk.get("avhubert_mps_ok") and chk.get("torch") == torch.__version__:
        crop = "mps" if chk.get("mtcnn_mps_ok") else "cpu"
        return "mps", crop, "auto: MPS verified by tools/compare_devices.py"
    reason = "auto: CPU (run tools/compare_devices.py to verify MPS)" if mps_ok else "auto: CPU (no GPU available)"
    return "cpu", "cpu", reason


def _load_slim(torch, avh, device: str):
    ck = torch.load(config.SLIM_CKPT, map_location="cpu", mmap=True, weights_only=True)
    if ck.get("format") != SLIM_FORMAT:
        raise RuntimeError(f"{config.SLIM_CKPT.name} is not a {SLIM_FORMAT} file")
    model = avh.AVHubertLite(**ck["kwargs"])
    model.load_state_dict(ck["state_dict"], strict=True, assign=True)
    info = dict(ck["info"])
    del ck
    return model.to(device).eval(), info


def _make_encoder(LSD, model, info, device):
    """A SyncEncoder around an already-built model.

    SyncEncoder.__init__ only does: load_avhubert(path) -> self.model, self.info; layers = [L, 3L//4, L//2];
    self.device, self.chunk = device, 500. The same attributes are set here, with the notebook's defaults."""
    enc = LSD.SyncEncoder.__new__(LSD.SyncEncoder)
    enc.model, enc.info = model, info
    L = info["layers"]
    enc.layers = [L, (3 * L) // 4, L // 2]
    enc.device, enc.chunk = device, 500
    return enc


def load_detector(progress=lambda msg: None) -> LoadedDetector:
    for p, what in ((config.BUNDLE_PATH, "classifier bundle"),):
        if not p.exists():
            raise FileNotFoundError(f"{what} not found: {p}")
    if not config.SLIM_CKPT.exists() and not config.ORIGINAL_CKPT.exists():
        raise FileNotFoundError(f"AV-HuBERT checkpoint not found: {config.SLIM_CKPT} or {config.ORIGINAL_CKPT}")

    t0 = time.time()
    progress("importing PyTorch and the detector code")
    import torch
    LSD, avh = load_core()
    device, crop_device, reason = resolve_devices(torch)
    if config.CROPPER_DEVICE:
        crop_device = config.CROPPER_DEVICE

    progress(f"loading AV-HuBERT on {device}")
    if config.SLIM_CKPT.exists():
        model, info = _load_slim(torch, avh, device)
        enc = _make_encoder(LSD, model, info, device)
        ckpt_kind = "slim (derived, verified identical)"
    else:
        enc = LSD.SyncEncoder(str(config.ORIGINAL_CKPT), device=device)      # original path (high memory peak)
        ckpt_kind = "original"

    progress("loading the face detector (MTCNN)")
    cropper = LSD.MouthCropper(device=crop_device, det_stride=2)              # same settings as notebook cell A4

    progress("loading the trained classifiers")
    bundle = LSD.LipSyncDetectorV3.load_bundle(config.BUNDLE_PATH, device)
    if bundle.get("v") != 4:
        raise RuntimeError(f"expected a v4 bundle, found v={bundle.get('v')}")
    if bundle["layers"] != enc.layers:                                       # same assert as notebook cell D1
        raise RuntimeError("model was trained with a different AV-HuBERT layer set")
    detector = LSD.LipSyncDetectorV3(bundle, enc, cropper)

    import sklearn
    meta = {
        "device": device, "cropper_device": crop_device, "device_reason": reason,
        "checkpoint": ckpt_kind, "load_seconds": round(time.time() - t0, 1),
        "torch": torch.__version__, "sklearn": sklearn.__version__,
        "avhubert": {k: enc.info[k] for k in ("layers", "dim", "heads", "ffn", "ln_first") if k in enc.info},
        "layers_used": list(enc.layers),
        "bundle": {"version": bundle.get("v"), "trained": bundle.get("trained"), "chosen": bundle.get("chosen"),
                   "thresholds": bundle.get("thresholds"), "n_your_videos": bundle.get("n_your_videos"),
                   "ckpt": bundle.get("ckpt")},
        "max_seconds": config.MAX_SECONDS,
    }
    return LoadedDetector(detector, enc, cropper, bundle, LSD, device, crop_device, meta)
