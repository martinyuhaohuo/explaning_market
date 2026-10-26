"""Archive loading and the train/test/discovery split."""

from __future__ import annotations

import gzip
import json
import random
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT / "data" / "archive"
OUT_DIR = ROOT / "data" / "surprise"
SPLIT_PATH = OUT_DIR / "split.json"
NO_PREVIEW_SPLIT_PATH = OUT_DIR / "split_no_preview.json"
EXTRACTIONS_PATH = OUT_DIR / "discovery_extractions.jsonl"
TAXONOMY_DIR = OUT_DIR / "taxonomy"

FACTS_ID = "earnings-call-facts"
PREVIEW_ID = "earnings-preview"


def extractions_path(set_name: str) -> Path:
    """Extraction file of an event set; "discovery" gives `EXTRACTIONS_PATH`."""
    return OUT_DIR / f"{set_name}_extractions.jsonl"


def items_by_id(event: dict) -> dict:
    items = (event.get("disclosure") or {}).get("items") or []
    return {i.get("id"): i for i in items if isinstance(i, dict)}


def ticker(event: dict) -> str:
    return event["focal_assets"][0]["identifier_value"]


def car1(event: dict) -> float | None:
    """Next-day abnormal return of the event's single focal asset."""
    ret = (event.get("event_returns") or {}).get(ticker(event)) or {}
    if ret.get("return_status") != "ok":
        return None
    return ret.get("car1")


def earnings_surprise(event: dict) -> float | None:
    """The archive's EPS surprise metric (None unless its status is "ok").

    The competition scores predictions by how much they add to a regression of
    the target on this value. Archive only: not part of the live payload.
    """
    block = (event.get("metrics") or {}).get("earnings_surprise") or {}
    if block.get("surprise_status") != "ok":
        return None
    return block.get("surprise")


def baseline_predictions(event: dict) -> dict[str, float]:
    """The competition's baseline percentiles for the event's focal asset, keyed by provider."""
    preds = event.get("baseline_predictions") or {}
    return {name.split("/")[0]: values.get(ticker(event)) for name, values in preds.items() if isinstance(values, dict)}


def event_time(event: dict) -> datetime:
    return datetime.fromisoformat(event["event_datetime"].replace("Z", "+00:00"))


def load_eligible_events(
    archive_dir: Path = ARCHIVE_DIR, quarters: tuple[str, ...] | None = ("2026Q3",)
) -> list[dict]:
    """Events with facts, a non-empty preview, and a realized return.

    `quarters` picks archive files by suffix (e.g. "2026Q3"); None loads all.
    """
    events = []
    for path in sorted(archive_dir.glob("EARNINGS_RELEASE_*.jsonl.gz")):
        if quarters is not None and path.name.removesuffix(".jsonl.gz").rsplit("_", 1)[-1] not in quarters:
            continue
        with gzip.open(path, "rt") as f:
            for line in f:
                event = json.loads(line)
                items = items_by_id(event)
                facts = (items.get(FACTS_ID) or {}).get("content")
                preview = (items.get(PREVIEW_ID) or {}).get("content")
                if not (isinstance(facts, list) and facts):
                    continue
                if not (isinstance(preview, str) and preview.strip()):
                    continue
                if len(event.get("focal_assets") or []) != 1 or car1(event) is None:
                    continue
                events.append(event)
    return events


def quarter_car1(quarter: str = "2026Q3", archive_dir: Path = ARCHIVE_DIR) -> dict[str, float]:
    """car1 of every single-asset event in a quarter's archive, eligible or not.

    The competition ranks a prediction against all of the quarter's event
    outcomes, so percentile targets are computed over this full set.
    """
    out = {}
    with gzip.open(archive_dir / f"EARNINGS_RELEASE_{quarter}.jsonl.gz", "rt") as f:
        for line in f:
            event = json.loads(line)
            if len(event.get("focal_assets") or []) == 1 and (value := car1(event)) is not None:
                out[event["event_id"]] = value
    return out


def quarter_earnings_surprise(quarter: str = "2026Q3", archive_dir: Path = ARCHIVE_DIR) -> dict[str, float | None]:
    """`earnings_surprise` of every event in a quarter's archive, keyed by event id."""
    with gzip.open(archive_dir / f"EARNINGS_RELEASE_{quarter}.jsonl.gz", "rt") as f:
        return {(event := json.loads(line))["event_id"]: earnings_surprise(event) for line in f}


def quarter_earnings_surprise_pct(quarter: str = "2026Q3", archive_dir: Path = ARCHIVE_DIR) -> dict[str, float]:
    """earnings_surprise as a within-quarter percentile rank, rank / N over events with a valid value.

    This is the control the benchmark uses (Koijen and Levy, section 2.3.3):
    ranking curbs the outliers that dominate a regression on the raw value.
    """
    values = {eid: v for eid, v in quarter_earnings_surprise(quarter, archive_dir).items() if v is not None}
    ranks = pd.Series(values).rank(method="average")
    return (ranks / len(ranks)).to_dict()


def load_no_preview_events(quarter: str = "2026Q3", archive_dir: Path = ARCHIVE_DIR) -> list[dict]:
    """Events of a quarter with call facts, one focal asset and a realized return, but no preview."""
    events = []
    with gzip.open(archive_dir / f"EARNINGS_RELEASE_{quarter}.jsonl.gz", "rt") as f:
        for line in f:
            event = json.loads(line)
            items = items_by_id(event)
            facts = (items.get(FACTS_ID) or {}).get("content")
            preview = (items.get(PREVIEW_ID) or {}).get("content")
            if not (isinstance(facts, list) and facts):
                continue
            if isinstance(preview, str) and preview.strip():
                continue
            if len(event.get("focal_assets") or []) != 1 or car1(event) is None:
                continue
            events.append(event)
    return events


def make_no_preview_split(events: list[dict], train_frac: float = 0.8) -> dict:
    """Chronological train/test split of no-preview events (no discovery sample)."""
    train, test = chronological_split(events, train_frac)
    return {
        "train_frac": train_frac,
        "n_eligible": len(events),
        "test_start": test[0]["event_datetime"],
        "train": [e["event_id"] for e in train],
        "test": [e["event_id"] for e in test],
    }


def chronological_split(events: list[dict], train_frac: float = 0.8) -> tuple[list[dict], list[dict]]:
    """Earliest `train_frac` of events are train, the rest test.

    The cut falls on an event time, so events released at the same moment
    always land on the same side.
    """
    ordered = sorted(events, key=lambda e: (event_time(e), e["event_id"]))
    cutoff = event_time(ordered[int(len(ordered) * train_frac)])
    train = [e for e in ordered if event_time(e) < cutoff]
    test = [e for e in ordered if event_time(e) >= cutoff]
    return train, test


def stratified_sample(events: list[dict], n: int, rng: random.Random) -> list[str]:
    """Sample event ids evenly across car1 deciles so both tails are covered."""
    ranked = sorted(events, key=car1)
    n_bins = 10
    bins = [ranked[i * len(ranked) // n_bins:(i + 1) * len(ranked) // n_bins] for i in range(n_bins)]

    sample: list[str] = []
    for i, bucket in enumerate(bins):
        # Spread the remainder over the first bins so the total is exact.
        take = n // n_bins + (1 if i < n % n_bins else 0)
        sample += [e["event_id"] for e in rng.sample(bucket, min(take, len(bucket)))]
    rng.shuffle(sample)
    return sample


def make_split(
    events: list[dict], train_frac: float = 0.8, n_discovery: int = 500, seed: int = 42
) -> dict:
    """Chronological train/test split, then a discovery sample from train only.

    Test events are later than every train event, so dimensions found on the
    discovery sample can be validated out of time. The return decides the
    discovery sample only; it is never shown to the LLM.
    """
    train, test = chronological_split(events, train_frac)
    discovery = stratified_sample(train, n_discovery, random.Random(seed))
    return {
        "seed": seed,
        "train_frac": train_frac,
        "n_eligible": len(events),
        "test_start": test[0]["event_datetime"],
        "train": [e["event_id"] for e in train],
        "test": [e["event_id"] for e in test],
        "discovery": discovery,
    }


def save_split(split: dict, path: Path = SPLIT_PATH, overwrite: bool = False) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2))
    return path


def load_split(path: Path = SPLIT_PATH) -> dict:
    return json.loads(path.read_text())
