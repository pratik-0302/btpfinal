"""Create models/avhubert_slim.pt from the original checkpoint (one-time; the original is not modified).

1. loads the ORIGINAL checkpoint with the ORIGINAL loader (core/avhubert_lite.load_avhubert)
   -> peak memory ~5.4 GB (measured on an 8 GB M1); done once;
2. saves only the model's weights (fp32, unchanged) + its constructor arguments;
3. re-loads the slim file the way the app does (memory-mapped) and checks every tensor is identical.

usage:  python tools/make_slim_ckpt.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                    # noqa: E402
from worker.core_import import load_core         # noqa: E402
from worker.loader import SLIM_FORMAT            # noqa: E402


def constructor_kwargs(sd: dict, info: dict) -> dict:
    """Same derivation as load_avhubert, from the (already renamed) model state dict."""
    w = sd["encoder.pos_conv.conv.weight"]
    dim = info["dim"]
    return {"dim": dim, "ffn": info["ffn"], "heads": info["heads"], "layers": info["layers"],
            "ln_first": bool(info["ln_first"]),
            "audio_feat_dim": int(sd["feature_extractor_audio.proj.weight"].shape[1]),
            "prelu": "feature_extractor_video.resnet.frontend3D.2.weight" in sd,
            "conv_pos": int(w.shape[-1]), "conv_pos_groups": int(dim // w.shape[1])}


def main() -> None:
    import torch
    _, avh = load_core()
    src, dst = config.ORIGINAL_CKPT, config.SLIM_CKPT
    if not src.exists():
        raise SystemExit(f"original checkpoint not found: {src}")
    if dst.exists():
        raise SystemExit(f"{dst} already exists; delete it first if you want to rebuild it")
    t = time.time()
    print(f"loading {src.name} with the original loader (needs ~5.5 GB of memory) ...", flush=True)
    model, info = avh.load_avhubert(str(src), "cpu")
    sd = model.state_dict()
    kwargs = constructor_kwargs(sd, info)
    tmp = dst.with_suffix(".part")
    torch.save({"format": SLIM_FORMAT, "kwargs": kwargs, "info": info, "state_dict": sd,
                "source": {"file": src.name, "size_bytes": src.stat().st_size}}, tmp)
    print(f"saved in {time.time() - t:.0f}s; verifying ...", flush=True)

    ck = torch.load(tmp, map_location="cpu", mmap=True, weights_only=True)
    m2 = avh.AVHubertLite(**ck["kwargs"])
    m2.load_state_dict(ck["state_dict"], strict=True, assign=True)
    s2 = m2.state_dict()
    same = sd.keys() == s2.keys() and all(torch.equal(sd[k], s2[k]) for k in sd)
    if not same:
        tmp.unlink()
        raise SystemExit("verification FAILED: slim weights differ from the original; nothing was written")
    os.replace(tmp, dst)
    print(f"OK: {dst} ({dst.stat().st_size / 1e9:.2f} GB), {len(sd)} tensors identical to the original loader's model")


if __name__ == "__main__":
    main()
