"""Generate our own preview for one event and store it."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from .store import EventStore

PREVIEW_SETTINGS = "cheap"


def run_preview(event: dict, store: EventStore, *, settings: str = PREVIEW_SETTINGS) -> dict:
    """Run `preview_agent.generate_preview` (cut off at the knowledge cutoff) and save its record.

    A run that is refused because the cutoff has passed still leaves a run record
    (`usable: false`), so the event is not retried.
    """
    from preview_agent.generator import CutoffPassedError, generate_preview

    with tempfile.TemporaryDirectory() as tmp:
        try:
            record = asyncio.run(generate_preview(event, settings=settings, out_dir=Path(tmp), verbose=False))
        except CutoffPassedError as exc:
            record = {"event_id": event["event_id"], "usable": False, "subtype": "cutoff_passed", "error": str(exc)}
    store.save_own_preview(event["event_id"], record)
    return record
