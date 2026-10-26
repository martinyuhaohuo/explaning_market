"""Per-event records: `<root>/events/<event_id>/<file>`.

Files per event (each written when its step runs):

    own_preview.md            our preview text (only when produced)
    own_preview_run.json      the preview run record (settings, timestamps, cost, `usable`)
    competition_preview.md    the competition's preview, when the event has one
    surprises.json            extracted and classified surprises
    features.json             the feature vector given to the model, with its preview source
    prediction.json           the submitted prediction(s), branch, and model details

On Modal the root is a mounted Volume: writes are committed so other containers
and `sync.download` see them, and reads reload first so they see other
containers' writes. Locally the root is `data/surprise/live/` (or `SHADOW_STORE_DIR`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = ROOT / "data" / "surprise" / "live"

VOLUME_NAME = "em-shadow-data"
MOUNT_PATH = "/shadow"
CLAIMS_DICT = "em-preview-claims"

OWN_PREVIEW = "own_preview.md"
OWN_PREVIEW_RUN = "own_preview_run.json"
COMPETITION_PREVIEW = "competition_preview.md"
SURPRISES = "surprises.json"
FEATURES = "features.json"
PREDICTION = "prediction.json"


class EventStore:
    def __init__(self, root: Path | str, *, commit: Callable[[], None] | None = None,
                 reload: Callable[[], None] | None = None):
        self.root = Path(root)
        self._commit = commit
        self._reload = reload

    def path(self, event_id: str, name: str) -> Path:
        return self.root / "events" / event_id / name

    def refresh(self) -> None:
        """See writes made by other containers (no-op locally)."""
        if self._reload:
            self._reload()

    def _write(self, event_id: str, name: str, text: str) -> Path:
        path = self.path(event_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if self._commit:
            self._commit()
        return path

    def write_text(self, event_id: str, name: str, text: str) -> Path:
        return self._write(event_id, name, text)

    def write_json(self, event_id: str, name: str, payload: dict) -> Path:
        return self._write(event_id, name, json.dumps(payload, indent=2, default=str))

    def read_text(self, event_id: str, name: str) -> str | None:
        path = self.path(event_id, name)
        return path.read_text() if path.exists() else None

    def read_json(self, event_id: str, name: str) -> dict | None:
        text = self.read_text(event_id, name)
        return json.loads(text) if text is not None else None

    def has_own_preview_run(self, event_id: str) -> bool:
        return self.path(event_id, OWN_PREVIEW_RUN).exists()

    def save_own_preview(self, event_id: str, record: dict) -> None:
        """Store a `preview_agent.generate_preview` record: the text (if any) and the run record."""
        record = dict(record)
        preview = record.pop("preview", None)
        record.pop("files", None)
        if preview:
            self.write_text(event_id, OWN_PREVIEW, preview)
        self.write_json(event_id, OWN_PREVIEW_RUN, record)

    def own_preview(self, event_id: str) -> str | None:
        """Our preview for the event, only if its run finished before the knowledge cutoff."""
        self.refresh()
        run = self.read_json(event_id, OWN_PREVIEW_RUN)
        if not run or not run.get("usable"):
            return None
        return self.read_text(event_id, OWN_PREVIEW)


def configured_store() -> EventStore | None:
    """The store when recording is switched on (on Modal, or with SHADOW_STORE_DIR set); None otherwise.

    Local runs and tests therefore never write into data/ unless asked to.
    """
    if os.environ.get("SHADOW_VOLUME") or os.environ.get("SHADOW_STORE_DIR"):
        return from_env()
    return None


def from_env() -> EventStore:
    """The store for wherever this code runs: the mounted Volume on Modal, else a local folder."""
    if os.environ.get("SHADOW_VOLUME") and Path(MOUNT_PATH).is_dir():
        import modal

        volume = modal.Volume.from_name(os.environ["SHADOW_VOLUME"])
        return EventStore(MOUNT_PATH, commit=volume.commit, reload=volume.reload)
    return EventStore(os.environ.get("SHADOW_STORE_DIR", LOCAL_ROOT))
