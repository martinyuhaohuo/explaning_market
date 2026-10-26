"""Grouping free-form surprise items into a codebook of dimensions.

The loop is: propose a taxonomy from a sample of events, assign every item
to it (with an "other" escape), review the counts and the "other" bucket,
refine, and assign again. Each taxonomy is saved as a numbered version, and
each version's assignments go to their own JSONL file, appended batch by
batch so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import Field, create_model

from .data import ROOT, TAXONOMY_DIR
from .prompts import (
    ASSIGN_SYSTEM,
    REFINE_SYSTEM,
    SEED_DIMENSIONS,
    TAXONOMY_SYSTEM,
    format_items,
    format_taxonomy,
)
from .schemas import Taxonomy

load_dotenv(ROOT / ".env")

# Designing the codebook is one hard call; assignment is many easy ones.
PROPOSE_MODEL = "gpt-6.1-sol"
PROPOSE_EFFORT = "medium"
ASSIGN_MODEL = "gpt-6-luna"
ASSIGN_EFFORT = "low"

OTHER = "other"
DIRECTION_SIGN = {"positive": 1, "negative": -1, "mixed": 0, "unclear": 0}
MAGNITUDE_WEIGHT = {"small": 1, "moderate": 2, "large": 3}


def _structured(schema, model: str, effort: str):
    return ChatOpenAI(model=model, reasoning={"effort": effort}).with_structured_output(schema)


# ----------------------------------------------------------------------
# Items
# ----------------------------------------------------------------------


def surprise_items(records: list[dict]) -> pd.DataFrame:
    """One row per surprise item, with a stable `item_id` of `<event_id>#<position>`."""
    rows = [
        {
            "item_id": f"{r['event_id']}#{i}",
            "event_id": r["event_id"],
            "ticker": r.get("ticker"),
            "event_datetime": r.get("event_datetime"),
            "position": i,
            "car1": r["car1"],
            "description": s["description"],
            "direction": s["direction"],
            "magnitude": s["magnitude"],
            "expectation_source": s["expectation_source"],
            "fact_evidence": s.get("fact_evidence"),
            "expectation_evidence": s.get("expectation_evidence"),
        }
        for r in records
        for i, s in enumerate(r["surprises"])
    ]
    return pd.DataFrame(rows)


def sample_event_items(items: pd.DataFrame, n_events: int, seed: int) -> pd.DataFrame:
    """All items of `n_events` randomly chosen events, so the sample keeps each release whole."""
    event_ids = sorted(items["event_id"].unique())
    chosen = random.Random(seed).sample(event_ids, min(n_events, len(event_ids)))
    return items[items["event_id"].isin(chosen)]


# ----------------------------------------------------------------------
# Taxonomy versions
# ----------------------------------------------------------------------


def taxonomy_path(version: str) -> Path:
    return TAXONOMY_DIR / f"taxonomy_{version}.json"


def assignments_path(version: str) -> Path:
    return TAXONOMY_DIR / f"assignments_{version}.jsonl"


def save_taxonomy(taxonomy: Taxonomy, version: str, overwrite: bool = False) -> Path:
    path = taxonomy_path(version)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True or use a new version.")
    keys = [d.key for d in taxonomy.dimensions]
    if len(set(keys)) != len(keys) or OTHER in keys:
        raise ValueError(f"Dimension keys must be unique and not '{OTHER}': {keys}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(taxonomy.model_dump_json(indent=2))
    return path


def load_taxonomy(version: str) -> Taxonomy:
    return Taxonomy.model_validate_json(taxonomy_path(version).read_text())


def propose_taxonomy(
    items: pd.DataFrame,
    *,
    n_events: int = 80,
    seed: int = 0,
    seed_dimensions: list[str] = SEED_DIMENSIONS,
    model: str = PROPOSE_MODEL,
    effort: str = PROPOSE_EFFORT,
) -> Taxonomy:
    """Draft a taxonomy from the items of a random sample of events."""
    sample = sample_event_items(items, n_events, seed)
    user = (
        "Starting list of candidate dimensions:\n"
        + "\n".join(f"- {d}" for d in seed_dimensions)
        + f"\n\nSurprise items ({len(sample)} items from {sample['event_id'].nunique()} releases):\n"
        + format_items(sample["description"].tolist())
    )
    return _structured(Taxonomy, model, effort).invoke(
        [SystemMessage(TAXONOMY_SYSTEM), HumanMessage(user)],
        config={"tags": ["propose_taxonomy"], "metadata": {"n_items": len(sample), "seed": seed}},
    )


def refine_taxonomy(
    taxonomy: Taxonomy,
    items: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    n_per_dimension: int = 8,
    n_other: int = 150,
    seed: int = 0,
    model: str = PROPOSE_MODEL,
    effort: str = PROPOSE_EFFORT,
) -> Taxonomy:
    """Revise a taxonomy from its assignment counts, per-dimension samples and the 'other' bucket."""
    merged = items.merge(assignments, on="item_id")
    counts = merged["dimension"].value_counts()

    sections = []
    for d in taxonomy.dimensions:
        rows = merged[merged["dimension"] == d.key]
        sample = rows.sample(min(n_per_dimension, len(rows)), random_state=seed)
        sections.append(
            f"### {d.key} ({counts.get(d.key, 0)} items)\n" + format_items(sample["description"].tolist())
        )
    other = merged[merged["dimension"] == OTHER]
    other_sample = other.sample(min(n_other, len(other)), random_state=seed)

    user = (
        f"Current codebook:\n{format_taxonomy(taxonomy)}\n\n"
        f"Assigned items: {len(merged)} total, {len(other)} in '{OTHER}'.\n\n"
        "Sample of items per dimension:\n\n" + "\n\n".join(sections)
        + f"\n\nItems assigned to '{OTHER}' (sample of {len(other_sample)}):\n"
        + format_items(other_sample["description"].tolist())
    )
    return _structured(Taxonomy, model, effort).invoke(
        [SystemMessage(REFINE_SYSTEM), HumanMessage(user)],
        config={"tags": ["refine_taxonomy"]},
    )


# ----------------------------------------------------------------------
# Assignment
# ----------------------------------------------------------------------


def _assignment_schema(keys: list[str]):
    """A batch schema whose `dimension` can only be one of the taxonomy's keys or 'other'."""
    key_type = Literal[tuple(keys + [OTHER])]
    item = create_model(
        "ItemAssignment",
        index=(int, Field(description="The item's number.")),
        dimension=(key_type, Field(description="Key of the dimension this item belongs to.")),
    )
    return create_model(
        "AssignmentBatch",
        assignments=(list[item], Field(description="One assignment per item.")),
    )


def done_item_ids(version: str) -> set[str]:
    path = assignments_path(version)
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["item_id"] for line in f if line.strip()}


def make_assigner(taxonomy: Taxonomy, model: str = ASSIGN_MODEL, effort: str = ASSIGN_EFFORT):
    return _structured(_assignment_schema([d.key for d in taxonomy.dimensions]), model, effort)


def assign_batch(batch: pd.DataFrame, taxonomy: Taxonomy, assigner=None, *, tag: str = "assign_dimensions") -> list[dict]:
    """Assign one batch of items in memory; returns `{item_id, dimension}` for each item the model numbered correctly."""
    if batch.empty:
        return []
    assigner = assigner or make_assigner(taxonomy)
    user = f"Codebook:\n{format_taxonomy(taxonomy)}\n\nItems:\n{format_items(batch['description'].tolist())}"
    result = assigner.invoke([SystemMessage(ASSIGN_SYSTEM), HumanMessage(user)], config={"tags": [tag]})
    ids = batch["item_id"].tolist()
    by_index = {a.index: a.dimension for a in result.assignments if 0 <= a.index < len(ids)}
    return [{"item_id": ids[i], "dimension": dim} for i, dim in sorted(by_index.items())]


def assign_items(
    items: pd.DataFrame,
    taxonomy: Taxonomy,
    version: str,
    *,
    batch_size: int = 40,
    model: str = ASSIGN_MODEL,
    effort: str = ASSIGN_EFFORT,
    workers: int = 8,
) -> dict:
    """Assign every item not yet assigned under `version`, appending batch by batch.

    Items the model skips or numbers wrongly are left unassigned; calling
    again retries them.
    """
    done = done_item_ids(version)
    todo = items[~items["item_id"].isin(done)]
    summary = {"requested": len(items), "skipped": len(items) - len(todo), "written": 0, "missing": 0, "failed_batches": 0}
    if todo.empty:
        print(f"All {len(items)} items already assigned under {version}")
        return summary

    assigner = make_assigner(taxonomy, model, effort)
    batches = [todo.iloc[i:i + batch_size] for i in range(0, len(todo), batch_size)]

    def run(batch: pd.DataFrame) -> list[dict]:
        return assign_batch(batch, taxonomy, assigner)

    print(f"Assigning {len(todo)} items in {len(batches)} batches ({summary['skipped']} already done) with {model}...")
    path = assignments_path(version)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, b): len(b) for b in batches}
        for n, future in enumerate(as_completed(futures), start=1):
            size = futures[future]
            try:
                rows = future.result()
            except Exception as exc:
                summary["failed_batches"] += 1
                print(f"[{n}/{len(batches)}] batch FAILED: {exc}")
                continue
            for row in rows:
                out.write(json.dumps({**row, "version": version}) + "\n")
            out.flush()
            summary["written"] += len(rows)
            summary["missing"] += size - len(rows)
            print(f"[{n}/{len(batches)}] {len(rows)}/{size} assigned")

    print(
        f"Done. {summary['written']} written, {summary['missing']} missing, "
        f"{summary['failed_batches']} batches failed; call again to retry."
    )
    return summary


def load_assignments(version: str) -> pd.DataFrame:
    path = assignments_path(version)
    if not path.exists():
        return pd.DataFrame(columns=["item_id", "dimension", "version"])
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # If an item was ever written twice, the last assignment wins.
    return pd.DataFrame(rows).drop_duplicates("item_id", keep="last")


# ----------------------------------------------------------------------
# Review
# ----------------------------------------------------------------------


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Normalized mutual information between two assignments of the same items, plus their crosstab.

    1 means the two taxonomies partition the items identically (labels may
    differ); near 0 means they are unrelated.
    """
    both = a[["item_id", "dimension"]].merge(b[["item_id", "dimension"]], on="item_id", suffixes=("_a", "_b"))
    table = pd.crosstab(both["dimension_a"], both["dimension_b"])
    p = table.to_numpy() / table.to_numpy().sum()
    pa, pb = p.sum(axis=1), p.sum(axis=0)
    mi = sum(
        p[i, j] * math.log(p[i, j] / (pa[i] * pb[j]))
        for i in range(p.shape[0])
        for j in range(p.shape[1])
        if p[i, j] > 0
    )
    ha = -sum(x * math.log(x) for x in pa if x > 0)
    hb = -sum(x * math.log(x) for x in pb if x > 0)
    nmi = mi / math.sqrt(ha * hb) if ha > 0 and hb > 0 else 0.0
    return nmi, table


def compare_assignments(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Label-level agreement between two assignments of the same items under the same taxonomy.

    Returns the share of items with the same dimension, Cohen's kappa (agreement
    beyond what matching label frequencies would give by chance), agreement per
    dimension of `a`, and the most common disagreements.
    """
    both = a[["item_id", "dimension"]].merge(b[["item_id", "dimension"]], on="item_id", suffixes=("_a", "_b"))
    same = both["dimension_a"] == both["dimension_b"]
    observed = same.mean()
    freq_a = both["dimension_a"].value_counts(normalize=True)
    freq_b = both["dimension_b"].value_counts(normalize=True)
    expected = (freq_a * freq_b.reindex(freq_a.index, fill_value=0)).sum()
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0

    per_dimension = (
        both.assign(same=same)
        .groupby("dimension_a")
        .agg(items=("same", "size"), agreement=("same", "mean"))
        .sort_values("items", ascending=False)
    )
    confusions = (
        both[~same]
        .groupby(["dimension_a", "dimension_b"])
        .size()
        .sort_values(ascending=False)
        .rename("items")
        .reset_index()
    )
    return {
        "n": len(both),
        "agreement": observed,
        "kappa": kappa,
        "per_dimension": per_dimension,
        "confusions": confusions,
    }


def reliability_check(
    items: pd.DataFrame,
    taxonomy: Taxonomy,
    version: str,
    *,
    n_items: int = 400,
    seed: int = 0,
    **assign_kwargs,
) -> dict:
    """Reassign a random sample of items under the same taxonomy and compare with `version`.

    The sample is in random order, so its batches mix items from different
    events, unlike the original pass, which batches items event by event. The
    rerun is saved as its own version (`<version>_rerun_s<seed>`), so it
    resumes like any assignment. Extra keyword arguments go to `assign_items`,
    e.g. a different `model` to compare assigners.
    """
    rerun = f"{version}_rerun_s{seed}"
    sample = items.sample(min(n_items, len(items)), random_state=seed)
    assign_items(sample, taxonomy, rerun, **assign_kwargs)
    return compare_assignments(load_assignments(version), load_assignments(rerun))


def dimension_net_scores(items: pd.DataFrame, assignments: pd.DataFrame, weighted: bool = False) -> pd.DataFrame:
    """Per event and dimension: positive minus negative items (optionally magnitude-weighted).

    Events with no item in a dimension get 0. `car1` is attached as the last column.
    """
    merged = items.merge(assignments[["item_id", "dimension"]], on="item_id")
    score = merged["direction"].map(DIRECTION_SIGN)
    if weighted:
        score = score * merged["magnitude"].map(MAGNITUDE_WEIGHT)
    table = merged.assign(score=score).pivot_table(
        index="event_id", columns="dimension", values="score", aggfunc="sum", fill_value=0
    )
    car1 = items.drop_duplicates("event_id").set_index("event_id")["car1"]
    return table.reindex(car1.index, fill_value=0).assign(car1=car1)
