"""Reading an event and its materials bundle (the event carries the bundle under "disclosure")."""

import math

FACTS_ID = "earnings-call-facts"
PREVIEW_ID = "earnings-preview"
OPTION_STATS_ID = "option-implied-stats"
OPTION_FIELDS = ("implied_earnings_volatility", "implied_absolute_earnings_move", "skew_25_delta")


def items_by_id(event: dict) -> dict:
    items = (event.get("disclosure") or {}).get("items") or []
    return {i.get("id"): i for i in items if isinstance(i, dict)}


def ticker(event: dict) -> str:
    return event["focal_assets"][0]["identifier_value"]


def call_facts(event: dict) -> list[str]:
    facts = (items_by_id(event).get(FACTS_ID) or {}).get("content")
    return facts if isinstance(facts, list) else []


def option_features(event: dict) -> dict:
    """`opt_<field>` values (NaN when absent or not "ok") and `has_option_stats`."""
    content = (items_by_id(event).get(OPTION_STATS_ID) or {}).get("content")
    row = {f"opt_{name}": math.nan for name in OPTION_FIELDS}
    if isinstance(content, dict):
        for name in OPTION_FIELDS:
            block = content.get(name)
            if isinstance(block, dict) and block.get("status") == "ok":
                row[f"opt_{name}"] = block.get("value")
    volatility = row["opt_implied_earnings_volatility"]
    row["has_option_stats"] = int(volatility is not None and not (isinstance(volatility, float) and math.isnan(volatility)))
    return row
