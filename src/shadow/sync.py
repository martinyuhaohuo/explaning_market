"""Download the shadow Volume to a local folder and load the records into tables.

    from shadow import sync
    sync.download()                      # Volume -> data/surprise/live/
    sync.load_predictions()              # one row per event and ticker
    sync.load_features()                 # one row per event: the model's input
    sync.load_preview_runs()             # one row per own-preview run

Equivalent CLI: `uv run modal volume get em-shadow-data / data/surprise/live`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .store import FEATURES, LOCAL_ROOT, OWN_PREVIEW_RUN, PREDICTION, VOLUME_NAME


def download(local_dir: Path | str = LOCAL_ROOT, volume_name: str = VOLUME_NAME) -> int:
    """Copy every file on the Volume to `local_dir`, keeping paths; returns the number of files."""
    import modal

    volume = modal.Volume.from_name(volume_name)
    local_dir = Path(local_dir)
    count = 0
    for entry in volume.listdir("/", recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        target = local_dir / entry.path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)
        count += 1
    return count


def _records(local_dir: Path | str, name: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(local_dir).glob(f"events/*/{name}"))]


def load_predictions(local_dir: Path | str = LOCAL_ROOT) -> pd.DataFrame:
    rows = []
    for record in _records(local_dir, PREDICTION):
        for pred in record.get("predictions", []):
            rows.append({k: v for k, v in record.items() if k != "predictions"} | pred)
    return pd.DataFrame(rows)


def load_features(local_dir: Path | str = LOCAL_ROOT) -> pd.DataFrame:
    return pd.DataFrame([r["features"] | {"preview_source": r.get("preview_source")} for r in _records(local_dir, FEATURES)])


def load_preview_runs(local_dir: Path | str = LOCAL_ROOT) -> pd.DataFrame:
    return pd.DataFrame(_records(local_dir, OWN_PREVIEW_RUN))
