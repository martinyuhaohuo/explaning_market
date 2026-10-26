"""What the graph runs on: the event with its live materials bundle, and which preview it uses."""

import copy
from dataclasses import dataclass

from .materials import PREVIEW_ID


@dataclass
class SurpriseContext:
    event: dict  # the event, with the live bundle under "disclosure"
    preview_source: str | None  # "competition", "own" or None (no preview at all)
    knowledge_cutoff: str


def _has_preview(bundle: dict) -> bool:
    for item in bundle.get("items") or []:
        if isinstance(item, dict) and item.get("id") == PREVIEW_ID and str(item.get("content") or "").strip():
            return True
    return False


def build_context(event: dict, bundle: dict, own_preview: str | None = None) -> SurpriseContext:
    """The competition's preview if the bundle has one, else our own (put in its place), else none."""
    bundle = copy.deepcopy(bundle or {})
    items = [i for i in bundle.get("items") or [] if isinstance(i, dict)]
    if _has_preview(bundle):
        source = "competition"
    elif own_preview and own_preview.strip():
        items = [i for i in items if i.get("id") != PREVIEW_ID]
        items.append({"id": PREVIEW_ID, "kind": "text", "source": "own_preview_agent", "content": own_preview})
        source = "own"
    else:
        source = None
    bundle["items"] = items
    live_event = {**event, "disclosure": bundle}
    cutoff = event.get("knowledge_cutoff") or event.get("event_datetime") or "unknown"
    return SurpriseContext(event=live_event, preview_source=source, knowledge_cutoff=str(cutoff))
