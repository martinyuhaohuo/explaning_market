"""Event-level feature vectors from the one-row-per-surprise table.

One row per event in the split, including events whose extraction found no
surprise (all counts zero, `no_surprises` = 1). Discovery and train_rest are
both labelled "train"; test stays "test".

Features:
- per dimension: magnitude-weighted net (`net_w__<dim>`), and the positive and
  negative parts separately (`pos_w__<dim>`, `neg_w__<dim>`; net = pos - neg).
  Direction counts +1 / -1 for positive / negative and 0 for mixed or unclear;
  magnitude weighs 1 / 2 / 3 for small / moderate / large. Items in "other"
  are left out of the per-dimension features but count in the totals.
- totals: item counts, net and weighted net, the same over the first three
  items (the extraction lists the most important first), and net per item.
- option-implied statistics from the event's materials, NaN when absent.
- only when the surprise table has a `sentiment` column (from the separate
  labelling step): management sentiment summed per dimension
  (`sent__<dim>`, +1 upbeat / -1 cautious / 0 neutral or null), plus
  `net_sentiment` and `net_sentiment_first3`.

Not features: `earnings_surprise` (the archive's EPS surprise, NaN when its
status is not "ok"), `earnings_surprise_pct` (its within-quarter percentile
rank, the control the competition's ΔR² regresses on) and `baseline__<provider>` (the competition's baseline predictions, kept
for comparison).

Target: `target_pct`, car1's rank among all of the quarter's events (not just
the eligible ones), scaled so 0 is the most negative and 1 the most positive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    OUT_DIR, baseline_predictions, car1, earnings_surprise, items_by_id, quarter_car1,
    quarter_earnings_surprise_pct, ticker,
)
from .grouping import DIRECTION_SIGN, MAGNITUDE_WEIGHT, OTHER
from .schemas import Taxonomy

OPTION_STATS_ID = "option-implied-stats"
OPTION_FIELDS = ("implied_earnings_volatility", "implied_absolute_earnings_move", "skew_25_delta")
META = ["event_id", "ticker", "event_datetime", "split", "car1", "target_pct", "earnings_surprise", "earnings_surprise_pct"]


def features_path(taxonomy_version: str) -> Path:
    return OUT_DIR / f"features_{taxonomy_version}.csv"


def quarter_percentile(quarter: str = "2026Q3") -> pd.Series:
    """car1 percentile within the quarter: (rank - 1) / (n - 1), ties averaged."""
    returns = pd.Series(quarter_car1(quarter))
    return (returns.rank(method="average") - 1) / (len(returns) - 1)


def option_features(event: dict) -> dict:
    content = (items_by_id(event).get(OPTION_STATS_ID) or {}).get("content")
    row = {f"opt_{name}": np.nan for name in OPTION_FIELDS}
    if isinstance(content, dict):
        for name in OPTION_FIELDS:
            block = content.get(name)
            if isinstance(block, dict) and block.get("status") == "ok":
                row[f"opt_{name}"] = block.get("value")
    row["has_option_stats"] = int(not np.isnan(row["opt_implied_earnings_volatility"]))
    return row


def surprise_feature_table(items: pd.DataFrame, dims: list[str], event_ids) -> pd.DataFrame:
    """Surprise features for `event_ids` (index) from their surprise rows; events without rows get zeros.

    `items` needs `event_id`, `position`, `direction`, `magnitude` and `dimension`
    (and optionally `sentiment`). Shared by training (`build_event_features`) and
    live prediction (`live_feature_row`), so both compute features identically.
    """
    items = items[items["event_id"].isin(event_ids)].copy()
    items["sign"] = items["direction"].map(DIRECTION_SIGN).fillna(0)
    items["weight"] = items["magnitude"].map(MAGNITUDE_WEIGHT).fillna(0)
    items["signed_w"] = items["sign"] * items["weight"]
    items["pos_w"] = items["signed_w"].clip(lower=0)
    items["neg_w"] = (-items["signed_w"]).clip(lower=0)
    has_sentiment = "sentiment" in items.columns
    if has_sentiment:
        items["sentiment"] = pd.to_numeric(items["sentiment"]).fillna(0)
    first3 = items[items["position"] < 3]

    def per_dimension(column: str, prefix: str) -> pd.DataFrame:
        rows = items[items["dimension"] != OTHER]
        if rows.empty:
            table = pd.DataFrame(index=pd.Index([], name="event_id"))
        else:
            table = rows.pivot_table(index="event_id", columns="dimension", values=column, aggfunc="sum", fill_value=0)
        return table.reindex(columns=dims, fill_value=0).add_prefix(f"{prefix}__")

    by_event = items.groupby("event_id")
    totals = pd.DataFrame({
        "n_items": by_event.size(),
        "n_pos": by_event["sign"].apply(lambda s: (s > 0).sum()),
        "n_neg": by_event["sign"].apply(lambda s: (s < 0).sum()),
        "n_mixed_unclear": by_event["sign"].apply(lambda s: (s == 0).sum()),
        "net": by_event["sign"].sum(),
        "net_w": by_event["signed_w"].sum(),
        "net_first3": first3.groupby("event_id")["sign"].sum(),
        "net_w_first3": first3.groupby("event_id")["signed_w"].sum(),
    })
    totals["net_per_item"] = totals["net"] / totals["n_items"]

    blocks = [per_dimension("signed_w", "net_w"), per_dimension("pos_w", "pos_w"), per_dimension("neg_w", "neg_w"), totals]
    if has_sentiment:
        blocks.append(per_dimension("sentiment", "sent"))
        blocks.append(pd.DataFrame({
            "net_sentiment": by_event["sentiment"].sum(),
            "net_sentiment_first3": first3.groupby("event_id")["sentiment"].sum(),
        }))
    table = pd.concat(blocks, axis=1).reindex(list(event_ids))
    columns = list(table.columns)
    table[columns] = table[columns].fillna(0)
    count_columns = [c for c in columns if c != "net_per_item"]
    table[count_columns] = table[count_columns].astype(int)
    table["no_surprises"] = (table["n_items"] == 0).astype(int)
    table.index.name = "event_id"
    return table


def build_event_features(
    surprises: pd.DataFrame,
    split: dict,
    events_by_id: dict[str, dict],
    taxonomy: Taxonomy,
    quarter: str = "2026Q3",
) -> pd.DataFrame:
    """One feature row per train and test event; see the module docstring."""
    dims = [d.key for d in taxonomy.dimensions]
    labels = {eid: "train" for eid in split["train"]} | {eid: "test" for eid in split["test"]}
    surprise_features = surprise_feature_table(surprises, dims, list(labels))
    feature_columns = [c for c in surprise_features.columns if c != "no_surprises"]

    pct = quarter_percentile(quarter)
    es_pct = quarter_earnings_surprise_pct(quarter)
    rows = []
    for eid, label in labels.items():
        event = events_by_id[eid]
        rows.append({
            "event_id": eid,
            "ticker": ticker(event),
            "event_datetime": event.get("event_datetime"),
            "split": label,
            "car1": car1(event),
            "target_pct": pct.get(eid, np.nan),
            "earnings_surprise": earnings_surprise(event),
            "earnings_surprise_pct": es_pct.get(eid, np.nan),
            **option_features(event),
            **{f"baseline__{name}": value for name, value in baseline_predictions(event).items()},
        })
    frame = pd.DataFrame(rows).set_index("event_id").join(surprise_features, how="left")

    option_columns = [f"opt_{name}" for name in OPTION_FIELDS] + ["has_option_stats"]
    baseline_columns = sorted(c for c in frame.columns if c.startswith("baseline__"))
    ordered = [c for c in META if c != "event_id"] + feature_columns + ["no_surprises"] + option_columns + baseline_columns
    return frame[ordered].sort_values("event_datetime").reset_index()


def live_feature_row(event: dict, items: pd.DataFrame, taxonomy: Taxonomy) -> pd.DataFrame:
    """The one-row feature table for a live event (no target, returns or baselines).

    `event` carries the materials under `disclosure` (for the option statistics);
    `items` are its classified surprises, as `grouping.surprise_items` returns them
    plus a `dimension` column.
    """
    dims = [d.key for d in taxonomy.dimensions]
    eid = event["event_id"]
    row = surprise_feature_table(items, dims, [eid])
    for key, value in option_features(event).items():
        row[key] = value
    row.insert(0, "ticker", ticker(event))
    row.insert(1, "event_datetime", event.get("event_datetime"))
    return row.reset_index()
