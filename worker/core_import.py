"""Import the original detector modules from core/ without modifying them.

The notebook modules use a top-level import (`from avhubert_lite import load_avhubert`), so the
core/ folder itself is put on sys.path instead of turning it into a package.
"""
from __future__ import annotations

import hashlib
import sys

import config


def verify_core() -> None:
    """Fail loudly if core/ no longer matches the hashes recorded when it was extracted from the notebook."""
    sums = (config.CORE_DIR / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines()
    for line in sums:
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(None, 1)
        data = (config.CORE_DIR / name.strip()).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"core/{name.strip()} differs from the notebook version (checksum mismatch)")


def load_core():
    """-> (lipsync_detector module, avhubert_lite module)."""
    verify_core()
    p = str(config.CORE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    import avhubert_lite                      # noqa: E402  (original module)
    import lipsync_detector                   # noqa: E402  (original module)
    return lipsync_detector, avhubert_lite
