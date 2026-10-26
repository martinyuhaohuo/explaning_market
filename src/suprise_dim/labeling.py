"""Separate labelling step: direction, magnitude and sentiment for classified surprises.

The extraction call already labels direction and magnitude while it lists the
surprises. This step re-labels them in calls of their own, once each surprise
has its v2 category, and adds management sentiment. The labeller sees only
the surprises (category, description, evidence, expectation source), not the
call facts, preview or option statistics, so it batches surprises from many
events into one call, like taxonomy assignment.

Results are appended to `data/surprise/labels_<name>.jsonl`, one line per
surprise, batch by batch, so an interrupted run resumes. Surprises the model
skips or numbers wrongly stay unlabelled and are retried on the next call.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .data import OUT_DIR, ROOT
from .prompts import LABEL_SYSTEM, build_label_message
from .schemas import SurpriseLabels

load_dotenv(ROOT / ".env")

LABEL_MODEL = "gpt-6-luna"
LABEL_EFFORT = "medium"
LABEL_COLUMNS = ["item_id", "event_id", "direction", "magnitude", "sentiment", "reason"]

# New integer labels back onto the extraction's vocabulary, so feature code can treat both alike.
DIRECTION_NAME = {1: "positive", -1: "negative", 0: "mixed", None: "unclear"}
MAGNITUDE_NAME = {1: "small", 2: "moderate", 3: "large"}


def labels_path(name: str) -> Path:
    return OUT_DIR / f"labels_{name}.jsonl"


def make_labeler(model: str = LABEL_MODEL, effort: str = LABEL_EFFORT):
    return ChatOpenAI(model=model, reasoning={"effort": effort}).with_structured_output(SurpriseLabels)


def event_items(surprises: pd.DataFrame, event_id: str) -> pd.DataFrame:
    """An event's surprises in extraction order."""
    return surprises[surprises["event_id"] == event_id].sort_values("position")


def label_batch(items: pd.DataFrame, labeler=None) -> list[dict]:
    """Label a batch of surprises; returns one dict per surprise the model labelled.

    Labels with an out-of-range or repeated number are dropped (the first one
    wins); magnitude is forced to None when direction is None.
    """
    if items.empty:
        return []
    labeler = labeler or make_labeler()
    result = labeler.invoke(
        [SystemMessage(LABEL_SYSTEM), HumanMessage(build_label_message(items))],
        config={"tags": ["surprise_labeling"], "metadata": {"n_items": len(items)}},
    )
    rows = items[["item_id", "event_id"]].to_dict("records")
    out = {}
    for label in result.labels:
        if 0 <= label.index < len(rows) and label.index not in out:
            out[label.index] = {
                **rows[label.index],
                "direction": label.direction,
                "magnitude": label.magnitude if label.direction is not None else None,
                "sentiment": label.sentiment,
                "reason": label.reason,
            }
    return [out[i] for i in sorted(out)]


def done_item_ids(name: str) -> set[str]:
    path = labels_path(name)
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["item_id"] for line in f if line.strip()}


def run_labeling(
    items: pd.DataFrame,
    name: str,
    *,
    batch_size: int = 40,
    model: str = LABEL_MODEL,
    effort: str = LABEL_EFFORT,
    workers: int = 8,
) -> dict:
    """Label every surprise in `items` not yet labelled under `name`, appending batch by batch.

    Batches follow the order of `items`. Calling again retries skipped
    surprises and failed batches.
    """
    done = done_item_ids(name)
    todo = items[~items["item_id"].isin(done)]
    summary = {"requested": len(items), "skipped": len(items) - len(todo), "written": 0, "missing": 0, "failed_batches": 0}
    if todo.empty:
        print(f"All {len(items)} surprises already labelled in {labels_path(name)}")
        return summary

    labeler = make_labeler(model, effort)
    batches = [todo.iloc[i:i + batch_size] for i in range(0, len(todo), batch_size)]

    print(f"Labelling {len(todo)} surprises in {len(batches)} batches ({summary['skipped']} already done) with {model}...")
    path = labels_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(label_batch, b, labeler): len(b) for b in batches}
        for n, future in enumerate(as_completed(futures), start=1):
            size = futures[future]
            try:
                rows = future.result()
            except Exception as exc:
                summary["failed_batches"] += 1
                print(f"[{n}/{len(batches)}] batch FAILED: {exc}")
                continue
            for row in rows:
                out.write(json.dumps({**row, "model": model, "effort": effort}) + "\n")
            out.flush()
            summary["written"] += len(rows)
            summary["missing"] += size - len(rows)
            print(f"[{n}/{len(batches)}] {len(rows)}/{size} labelled")

    print(f"Done. {summary['written']} written, {summary['missing']} missing, "
          f"{summary['failed_batches']} batches failed; call again to retry.")
    return summary


def load_labels(name: str) -> pd.DataFrame:
    """One row per labelled surprise (the last label wins if one was written twice)."""
    path = labels_path(name)
    if not path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows, columns=LABEL_COLUMNS).drop_duplicates("item_id", keep="last")


def complete_events(surprises: pd.DataFrame, labels: pd.DataFrame) -> set[str]:
    """Events whose every surprise has a label; only these are compared, so partial events never mix label sets."""
    labelled = surprises["item_id"].isin(labels["item_id"])
    per_event = labelled.groupby(surprises["event_id"]).all()
    return set(per_event.index[per_event])


def apply_labels(surprises: pd.DataFrame, labels: pd.DataFrame, event_ids: set[str]) -> pd.DataFrame:
    """The surprise rows of `event_ids` with direction and magnitude replaced by the new labels.

    The extraction's labels are kept as `direction_extraction` and
    `magnitude_extraction`; `sentiment` is added (NaN when null). A surprise
    without a new label gets direction "unclear" and no magnitude, so it counts 0.
    """
    rows = surprises[surprises["event_id"].isin(event_ids)].rename(
        columns={"direction": "direction_extraction", "magnitude": "magnitude_extraction"}
    )
    merged = rows.merge(labels[["item_id", "direction", "magnitude", "sentiment"]], on="item_id", how="left")
    merged["direction"] = merged["direction"].map(lambda v: DIRECTION_NAME[None if pd.isna(v) else int(v)])
    merged["magnitude"] = merged["magnitude"].map(lambda v: None if pd.isna(v) else MAGNITUDE_NAME[int(v)])
    merged["sentiment"] = pd.to_numeric(merged["sentiment"])
    return merged


def label_agreement(labelled: pd.DataFrame) -> dict:
    """How the new labels compare with the extraction's, on the same surprises."""
    direction = pd.crosstab(labelled["direction_extraction"], labelled["direction"], margins=True)
    magnitude = pd.crosstab(labelled["magnitude_extraction"], labelled["magnitude"].fillna("none"), margins=True)
    return {
        "n": len(labelled),
        "direction_same": float((labelled["direction"] == labelled["direction_extraction"]).mean()),
        "magnitude_same": float((labelled["magnitude"] == labelled["magnitude_extraction"]).mean()),
        "direction_table": direction,
        "magnitude_table": magnitude,
        "sentiment_by_direction": pd.crosstab(labelled["direction"], labelled["sentiment"].fillna(-9).map(
            {1: "upbeat", 0: "neutral", -1: "cautious", -9: "null"}), margins=True),
    }
