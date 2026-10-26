"""Event sets and the combined one-row-per-surprise table.

Every set goes through the same two steps as discovery: free-form extraction
(`extraction.run_extraction`) and assignment to a frozen taxonomy
(`grouping.assign_items`). This module names the files for each set and joins
them into one table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import OUT_DIR, extractions_path, quarter_earnings_surprise, quarter_earnings_surprise_pct
from .extraction import load_extractions
from .grouping import load_assignments, load_taxonomy, surprise_items

SETS = ("discovery", "train_rest", "test")

COLUMNS = [
    "split", "event_id", "ticker", "event_datetime", "car1", "earnings_surprise", "earnings_surprise_pct",
    "item_id", "position", "n_items",
    "dimension", "dimension_name", "direction", "magnitude", "expectation_source",
    "description", "fact_evidence", "expectation_evidence",
    "taxonomy_version", "extraction_model",
]


def event_sets(split: dict, events_by_id: dict[str, dict]) -> dict[str, list[dict]]:
    """discovery, train_rest (train events outside discovery) and test, as event lists."""
    discovery = set(split["discovery"])
    return {
        "discovery": [events_by_id[eid] for eid in split["discovery"]],
        "train_rest": [events_by_id[eid] for eid in split["train"] if eid not in discovery],
        "test": [events_by_id[eid] for eid in split["test"]],
    }


def assignment_version(taxonomy_version: str, set_name: str) -> str:
    """Assignment file name for a set; discovery keeps the plain version name it was built with."""
    return taxonomy_version if set_name == "discovery" else f"{taxonomy_version}_{set_name}"


def surprises_path(taxonomy_version: str) -> Path:
    return OUT_DIR / f"surprises_{taxonomy_version}.csv"


def _set_frame(records: list[dict], assignments: pd.DataFrame, label: str, taxonomy_version: str) -> pd.DataFrame:
    items = surprise_items(records)
    models = {r["event_id"]: r.get("model") for r in records}
    return items.merge(assignments[["item_id", "dimension"]], on="item_id", how="left").assign(
        split=label,
        taxonomy_version=taxonomy_version,
        extraction_model=lambda t: t["event_id"].map(models),
    )


def _finish(frames: list[pd.DataFrame], taxonomy_version: str, quarter: str) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    names = {d.key: d.name for d in load_taxonomy(taxonomy_version).dimensions} | {"other": "Other"}
    frame = pd.concat(frames, ignore_index=True)
    frame["n_items"] = frame.groupby("event_id")["item_id"].transform("size")
    frame["earnings_surprise"] = frame["event_id"].map(quarter_earnings_surprise(quarter))
    frame["earnings_surprise_pct"] = frame["event_id"].map(quarter_earnings_surprise_pct(quarter))
    frame["dimension_name"] = frame["dimension"].map(names)
    return frame[COLUMNS]


def build_surprise_frame(
    split: dict, taxonomy_version: str = "v2", sets: tuple[str, ...] = SETS, quarter: str = "2026Q3"
) -> pd.DataFrame:
    """One row per surprise across `sets`, with its event, labels, evidence and dimension.

    Only extractions of events in each set's split ids are used, so stale
    records from an earlier split are ignored. Items not yet assigned keep a
    missing `dimension`. Events whose extraction found no surprise have no row.
    """
    set_ids = {
        "discovery": set(split["discovery"]),
        "train_rest": set(split["train"]) - set(split["discovery"]),
        "test": set(split["test"]),
    }
    frames = []
    for name in sets:
        records = [r for r in load_extractions(extractions_path(name)) if r["event_id"] in set_ids[name]]
        if records:
            frames.append(_set_frame(records, load_assignments(assignment_version(taxonomy_version, name)), name, taxonomy_version))
    return _finish(frames, taxonomy_version, quarter)


def build_extraction_frame(
    extraction_name: str,
    assignment_name: str,
    labels: dict[str, str],
    taxonomy_version: str = "v2",
    quarter: str = "2026Q3",
) -> pd.DataFrame:
    """The same table for any extraction run: `data/surprise/<extraction_name>_extractions.jsonl`
    with assignments `taxonomy/assignments_<assignment_name>.jsonl`.

    `labels` maps each event id to keep onto its `split` label (e.g. "train" / "test").
    """
    records = [r for r in load_extractions(extractions_path(extraction_name)) if r["event_id"] in labels]
    frames = [
        _set_frame([r for r in records if labels[r["event_id"]] == label], load_assignments(assignment_name), label, taxonomy_version)
        for label in sorted({labels[r["event_id"]] for r in records})
    ]
    return _finish(frames, taxonomy_version, quarter)
