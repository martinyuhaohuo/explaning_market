"""Which calendar events need our own preview now, and claiming them so only one worker runs each.

An event is due from `LEAD` (12 hours) before its knowledge cutoff until the
cutoff itself. Most cutoffs fall at the same moment (16:00 ET), so hundreds of
events become due together; at most `MAX_PARALLEL_PREVIEWS` runs execute at once
(the Modal worker's container cap) and the rest wait in Modal's queue, still
holding their claims. The long lead leaves time to work through that queue, and
a run that fails is retried by a later check as long as the cutoff has not
passed. Every event gets a preview: whether the competition's preview will be
missing is only known at release.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .store import EventStore

LEAD = timedelta(hours=12)
MAX_PARALLEL_PREVIEWS = 5
CHECK_EVERY = timedelta(minutes=15)


def _time(value: str) -> datetime:
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def due_events(events: list[dict], now: datetime | None = None, *, lead: timedelta = LEAD,
               store: EventStore | None = None) -> list[dict]:
    """Events inside their preview window (cutoff - lead <= now < cutoff) without a stored preview run."""
    now = now or datetime.now(timezone.utc)
    due = []
    for event in events:
        cutoff = _time(event["knowledge_cutoff"])
        if cutoff - lead <= now < cutoff and not (store and store.has_own_preview_run(event["event_id"])):
            due.append(event)
    return sorted(due, key=lambda e: (e["knowledge_cutoff"], e["event_id"]))


def claim(claims, event_id: str) -> bool:
    """Reserve an event for one worker. `claims` is a modal.Dict (atomic) or, locally, a plain dict."""
    if hasattr(claims, "put"):
        return bool(claims.put(event_id, "in_flight", skip_if_exists=True))
    if event_id in claims:
        return False
    claims[event_id] = "in_flight"
    return True


def release(claims, event_id: str, *, done: bool) -> None:
    """Mark the event done, or free it so a later check can retry."""
    if done:
        claims[event_id] = "done"
    else:
        claims.pop(event_id, None)


def calendar_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    """The calendar range a check needs: every event whose cutoff could be due now.

    A cutoff can come days before its release (a Monday release often has Friday's
    close as its cutoff), so the range reaches a week ahead.
    """
    now = now or datetime.now(timezone.utc)
    return now - timedelta(hours=1), now + timedelta(days=7)
