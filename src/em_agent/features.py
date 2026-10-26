"""Surprises -> the feature vector the pooled ridge model expects.

A frozen copy of the training feature code (suprise_dim.features and
ml.preprocess.add_source_features), written for one live event. The tests check
it reproduces the training rows exactly.

Per dimension: `net_w__<dim>` = sum of direction x magnitude (+1/-1 for
positive/negative, 0 for mixed/unclear; 1/2/3 for small/moderate/large), and its
positive and negative parts `pos_w__<dim>` / `neg_w__<dim>`. Surprises classified
as "other" (or not classified) count only in the totals.
"""

from .materials import option_features
from .schemas import OTHER, Taxonomy

DIRECTION_SIGN = {"positive": 1, "negative": -1, "mixed": 0, "unclear": 0}
MAGNITUDE_WEIGHT = {"small": 1, "moderate": 2, "large": 3}
SOURCES = ("competition", "own")
SOURCE_INTERACTIONS = ["net_w", "net_w_first3", "n_items", "no_surprises"]


def surprise_features(items: list[dict], dims: list[str]) -> dict:
    """Features from one event's classified surprises (each with position, direction, magnitude, dimension)."""
    row = {f"{prefix}__{d}": 0 for prefix in ("net_w", "pos_w", "neg_w") for d in dims}
    totals = dict.fromkeys(["n_items", "n_pos", "n_neg", "n_mixed_unclear", "net", "net_w", "net_first3", "net_w_first3"], 0)
    for item in items:
        sign = DIRECTION_SIGN.get(item.get("direction"), 0)
        signed = sign * MAGNITUDE_WEIGHT.get(item.get("magnitude"), 0)
        totals["n_items"] += 1
        totals["n_pos"] += sign > 0
        totals["n_neg"] += sign < 0
        totals["n_mixed_unclear"] += sign == 0
        totals["net"] += sign
        totals["net_w"] += signed
        if item.get("position", 0) < 3:
            totals["net_first3"] += sign
            totals["net_w_first3"] += signed
        dim = item.get("dimension")
        if dim in dims and dim != OTHER:
            row[f"net_w__{dim}"] += signed
            row[f"pos_w__{dim}"] += max(signed, 0)
            row[f"neg_w__{dim}"] += max(-signed, 0)
    row.update({k: int(v) for k, v in totals.items()})
    row["net_per_item"] = totals["net"] / totals["n_items"] if totals["n_items"] else 0.0
    row["no_surprises"] = int(totals["n_items"] == 0)
    return row


def add_source_features(row: dict, source: str, *, reference: str = "competition") -> dict:
    """The preview-source flag `src_own` and its interactions with the main scores."""
    if source not in SOURCES:
        raise ValueError(f"unknown preview source {source!r}; expected one of {SOURCES}")
    row = {**row, "preview_source": source}
    for s in SOURCES:
        if s == reference:
            continue
        flag = f"src_{s}"
        row[flag] = int(source == s)
        for col in SOURCE_INTERACTIONS:
            row[f"{col}__x__{flag}"] = row[col] * row[flag]
    return row


def live_features(event: dict, items: list[dict], taxonomy: Taxonomy, source: str) -> dict:
    """The full feature vector for a live event: surprise features, option statistics and the source flag."""
    row = surprise_features(items, [d.key for d in taxonomy.dimensions])
    row.update(option_features(event))
    return add_source_features({"event_id": event["event_id"], **row}, source)
