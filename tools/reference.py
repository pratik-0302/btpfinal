"""Reference videos with the exact results printed by the v4 notebook (cell D2), and how to find them.

The notebook picked them with pandas .sample(random_state=1) from the LipSyncTIMIT manifest; the same
call on the local manifest returns the same six files (verified during the audit).
"""
from __future__ import annotations

import os
from pathlib import Path

# filename -> (verdict, headline printed by notebook cell D2)
NOTEBOOK_D2 = {
    "181.mp4": ("REAL", "LOOKS REAL — 82% real"),
    "96.mp4": ("REAL", "LOOKS REAL — 87% real"),
    "27.mp4": ("REAL", "LOOKS REAL — 93% real"),
    "W2LGAN_Vid178_Aud-ORG.mp4": ("FAKE", "LOOKS FAKE — 98% fake"),
    "W2LGAN_Vid178_Aud-LSR2-120.mp4": ("FAKE", "LOOKS FAKE — 99% fake"),
    "W2LGAN_Vid178_Aud-Fake-wavegrad_26_495_000035_000003_gen.mp4": ("FAKE", "LOOKS FAKE — 97% fake"),
}


def headline_percent(headline: str) -> float:
    """'LOOKS REAL — 82% real' -> fake % (18); 'LOOKS FAKE — 98% fake' -> 98."""
    num = float(headline.split("—")[1].strip().split("%")[0])
    return 100.0 - num if "real" in headline.split("—")[1] and "fake" not in headline.split("—")[1] else num


def find_reference_videos(manifest: str | None = None, dataset: str | None = None) -> list[tuple[Path, str, str]]:
    """-> [(path, expected verdict, expected headline)] using the manifest's demo selection."""
    import pandas as pd
    manifest = manifest or os.environ.get("LIPSYNC_MANIFEST")
    dataset = dataset or os.environ.get("LIPSYNC_DATASET")
    if not manifest or not dataset:
        raise SystemExit("set LIPSYNC_MANIFEST=/path/to/manifest.csv and LIPSYNC_DATASET=/path/to/LipSyncTimitDataset")
    man = pd.read_csv(manifest)
    t = man[man.split == "test"]
    demo = pd.concat([t[t.label == 1].sample(3, random_state=1),
                      t[t.category == "FakeVideo-OriginalAudio"].sample(1, random_state=1),
                      t[t.category == "FakeVideo-LRS2Audio"].sample(1, random_state=1),
                      t[t.category == "FakeVideo-FakeAudio"].sample(1, random_state=1)])
    out = []
    for r in demo.itertuples():
        p = Path(dataset) / r.relpath
        if not p.exists():
            raise SystemExit(f"reference video missing: {p}")
        v, h = NOTEBOOK_D2[r.filename]
        out.append((p, v, h))
    return out
