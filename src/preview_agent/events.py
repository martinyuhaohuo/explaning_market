"""The competition's events calendar: `GET {EM_API_BASE_URL}/events` with the `X-API-Key` header."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
from dotenv import load_dotenv

from .generator import ROOT

DEFAULT_API_BASE_URL = "https://api.explainingmarkets.ai/v1"

load_dotenv(ROOT / ".env")


def _iso_z(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_events(start: datetime | None = None, end: datetime | None = None, *, timeout: float = 30.0) -> list[dict]:
    """Scheduled events between `start` and `end` (default: now to 60 days ahead)."""
    start = start or datetime.now(timezone.utc)
    end = end or start + timedelta(days=60)
    api_key = os.environ.get("EM_API_KEY")
    if not api_key:
        raise RuntimeError("EM_API_KEY is not set; add it to .env")
    base = os.environ.get("EM_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
    response = httpx.get(
        f"{base}/events",
        params={"start_date": _iso_z(start), "end_date": _iso_z(end)},
        headers={"X-API-Key": api_key},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def events_table(events: list[dict], now: datetime | None = None) -> pd.DataFrame:
    """One row per event, sorted by release time, with whether its knowledge cutoff is still ahead."""
    now = now or datetime.now(timezone.utc)
    rows = [
        {
            "event_id": e["event_id"],
            "tickers": ", ".join(a["identifier_value"] for a in e.get("focal_assets") or []),
            "event_datetime": pd.Timestamp(e["event_datetime"]),
            "knowledge_cutoff": pd.Timestamp(e["knowledge_cutoff"]),
            "event_type": e.get("event_type"),
        }
        for e in events
    ]
    table = pd.DataFrame(rows, columns=["event_id", "tickers", "event_datetime", "knowledge_cutoff", "event_type"])
    if table.empty:
        return table.assign(cutoff_ahead=pd.Series(dtype=bool), hours_to_cutoff=pd.Series(dtype=float))
    table["hours_to_cutoff"] = ((table["knowledge_cutoff"] - pd.Timestamp(now)).dt.total_seconds() / 3600).round(1)
    table["cutoff_ahead"] = table["hours_to_cutoff"] > 0
    return table.sort_values(["event_datetime", "event_id"]).reset_index(drop=True)


def next_event(events: list[dict], now: datetime | None = None, *, ticker: str | None = None) -> dict:
    """The next scheduled event whose knowledge cutoff has not passed (optionally for one ticker)."""
    now = now or datetime.now(timezone.utc)
    open_events = [
        e for e in events
        if datetime.fromisoformat(e["knowledge_cutoff"].replace("Z", "+00:00")) > now
        and (ticker is None or any(a["identifier_value"] == ticker for a in e.get("focal_assets") or []))
    ]
    if not open_events:
        raise LookupError("no scheduled event with a knowledge cutoff still ahead" + (f" for {ticker}" if ticker else ""))
    return min(open_events, key=lambda e: (e["event_datetime"], e["event_id"]))
