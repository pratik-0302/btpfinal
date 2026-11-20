"""Copy the two detector modules out of the v4 notebook, byte for byte.

The notebook stores them as raw strings in cell A3:
    (CODE / "avhubert_lite.py").write_text(r'''...''')
    (CODE / "lipsync_detector.py").write_text(r'''...''')

This tool opens the notebook read-only, writes the exact string contents to core/,
and records SHA-256 hashes in core/CHECKSUMS.sha256 together with the notebook's hash.
It never modifies the notebook.

usage:  python tools/extract_core.py "/path/to/LipSync_Deepfake_Detector_v4 (2).ipynb"
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
MODULES = ("avhubert_lite.py", "lipsync_detector.py")
CELL_MARKER = "# Cell A3"


def extract(notebook: Path) -> dict[str, str]:
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"
             and "".join(c["source"]).lstrip().startswith(CELL_MARKER)]
    if len(cells) != 1:
        raise SystemExit(f"expected exactly one '{CELL_MARKER}' cell, found {len(cells)}")
    src = "".join(cells[0]["source"])
    out = {}
    for name in MODULES:
        m = re.search(r'\(CODE / "%s"\)\.write_text\(r\'\'\'(.*?)\'\'\'\)' % re.escape(name), src, re.S)
        if not m:
            raise SystemExit(f"{name} not found in cell A3")
        out[name] = m.group(1)
    return out


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    notebook = Path(sys.argv[1]).expanduser()
    mods = extract(notebook)
    CORE.mkdir(exist_ok=True)
    lines = [f"# notebook {notebook.name} sha256={sha256(notebook.read_bytes())}"]
    for name, text in mods.items():
        data = text.encode("utf-8")
        target = CORE / name
        if target.exists() and target.read_bytes() != data:
            raise SystemExit(f"{target} exists and differs from the notebook; refusing to overwrite")
        target.write_bytes(data)
        lines.append(f"{sha256(data)}  {name}")
        print(f"wrote core/{name} ({len(data)} bytes)")
    (CORE / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote core/CHECKSUMS.sha256")


if __name__ == "__main__":
    main()
