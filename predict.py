"""★ THIS IS THE ONLY FILE YOU NEED TO EDIT. ★

`predict(event)` is called once per competition event, after the webhook has
already been verified for you. Return one prediction per focal asset. Everything
else in this repo (webhook verification, dedupe, submission) is plumbing.

The default implementation runs the multi-model ``em_agent`` graph. If either
required provider key is missing, or the graph fails, it returns a 0.5 baseline
so the receive → predict → submit round-trip still completes.
"""

from __future__ import annotations

import os

import httpx

_agent_warned = False

# Keep the disclosure fetch bounded within the five-minute prediction window.
SUMMARY_TIMEOUT_SECONDS = 15.0


def predict(event: dict) -> list[dict]:
    """Return predictions for one Explaining Markets event.

    `event` is the verified webhook payload. Useful fields:
      event["event_type"]          e.g. "EARNINGS_RELEASE"
      event["focal_assets"]        list of {"identifier_type", "identifier_value"}
      event["information_url"]     short-lived signed URL to the event's materials
                                   (a JSON bundle of items; see `format_materials`)
      event["prediction_deadline"] ISO timestamp; submit before this fires

    Required return: a list of dicts, one per focal asset:
      [{"identifier_value": "AAPL", "predicted_percentile": 0.71}, ...]

    `predicted_percentile` is a float in [0, 1] — where you predict the asset's
    next-day abnormal (market-adjusted) return will rank across all of the
    quarter's event outcomes: 0 = the quarter's most negative reaction,
    0.50 = median, 1 = its most positive. It's a cross-sectional rank across the
    quarter's events, not a percentile within the asset's own history.
    """
    summary = httpx.get(event["information_url"], timeout=SUMMARY_TIMEOUT_SECONDS)
    summary.raise_for_status()
    summary_json = summary.json()

    return [
        {
            "identifier_value": asset["identifier_value"],
            "predicted_percentile": _ask_agent(
                event=event,
                disclosure=summary_json,
                ticker=asset["identifier_value"],
            ),
        }
        for asset in event["focal_assets"]
    ]


# ----------------------------------------------------------------------
# Reading the event's materials.
#
# The document behind `information_url` is a bundle: a list of `items`, each
# with an `id`, a `kind`, and a `content` whose JSON type follows the kind.
# Today an earnings event can carry three:
#
#   earnings-call-facts   kind "facts"  a list of strings     always present
#   earnings-preview      kind "text"   one markdown string   may be absent
#   option-implied-stats  kind "stats"  an OBJECT of numbers  may be absent
#
# Select items by `id`, never by position, and expect kinds you have not seen:
# new ones can be added at any time.
# ----------------------------------------------------------------------

FACTS_ID = "earnings-call-facts"
PREVIEW_ID = "earnings-preview"
OPTION_STATS_ID = "option-implied-stats"

# The preview is by far the largest item (often ~8,000 characters). It is the
# only one that gets cut, and it goes LAST in the prompt so a cut never costs
# you the facts or the option statistics.
PREVIEW_MAX_CHARS = 8000
OTHER_ITEM_MAX_CHARS = 2000


def _percent(block: object) -> str | None:
    """A `{value, status}` statistic as a percentage, or None if unavailable."""
    if not isinstance(block, dict) or block.get("status") != "ok":
        return None
    value = block.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return f"{value * 100:+.1f}" if value < 0 else f"{value * 100:.1f}"


def _format_option_stats(stats: dict) -> str | None:
    volatility = _percent(stats.get("implied_earnings_volatility"))
    if volatility is None:
        return None
    lines = [
        f"- Implied earnings volatility: {volatility}% (the standard deviation of the "
        "stock's move on this release that option prices imply)",
    ]
    move = _percent(stats.get("implied_absolute_earnings_move"))
    if move is not None:
        lines.append(
            f"- Implied absolute move: {move}% (the size of move options price in; "
            "it says nothing about direction)"
        )
    skew = _percent(stats.get("skew_25_delta"))
    if skew is not None:
        lines.append(
            f"- 25-delta skew: {skew} volatility points (call minus put implied "
            "volatility; negative means downside protection is priced richer)"
        )
    else:
        lines.append("- 25-delta skew: unavailable (options too thinly traded)")
    as_of = stats.get("as_of")
    header = "Option-market expectations, measured before the release"
    if isinstance(as_of, str) and as_of:
        header += f" (as of {as_of})"
    return header + ":\n" + "\n".join(lines)


def format_materials(bundle: object) -> str:
    """Turn the event's bundle into the text the model reads.

    Each item gets its own labelled section. Items are found by `id`, so the
    order they arrive in does not matter, and an item this code has never heard
    of is included if it is text and skipped otherwise, never an error.
    """
    if not isinstance(bundle, dict):
        return ""
    raw_items = bundle.get("items")
    items = [i for i in raw_items if isinstance(i, dict)] if isinstance(raw_items, list) else []
    by_id = {i.get("id"): i for i in items}
    sections: list[str] = []

    facts = (by_id.get(FACTS_ID) or {}).get("content")
    if isinstance(facts, list) and facts:
        sections.append(
            "Facts from the earnings call:\n"
            + "\n".join(f"{n}. {fact}" for n, fact in enumerate(facts, start=1))
        )

    stats = (by_id.get(OPTION_STATS_ID) or {}).get("content")
    if isinstance(stats, dict):
        formatted = _format_option_stats(stats)
        if formatted:
            sections.append(formatted)

    for item in items:
        if item.get("id") in (FACTS_ID, PREVIEW_ID, OPTION_STATS_ID):
            continue
        content = item.get("content")
        if isinstance(content, list) and all(isinstance(c, str) for c in content):
            content = "\n".join(content)
        if isinstance(content, str) and content.strip():
            label = item.get("id") or item.get("kind") or "item"
            sections.append(f"Additional material ({label}):\n{content[:OTHER_ITEM_MAX_CHARS]}")

    preview = (by_id.get(PREVIEW_ID) or {}).get("content")
    if isinstance(preview, str) and preview.strip():
        text = preview[:PREVIEW_MAX_CHARS]
        if len(preview) > PREVIEW_MAX_CHARS:
            text += "\n[preview truncated]"
        sections.append(
            "Research note written BEFORE the release (expectations, not results):\n" + text
        )

    return "\n\n".join(sections)


def _bundle_content(bundle: object, item_id: str) -> object | None:
    """Return an event material's content by ID, independent of item order."""
    if not isinstance(bundle, dict) or not isinstance(bundle.get("items"), list):
        return None
    for item in bundle["items"]:
        if isinstance(item, dict) and item.get("id") == item_id:
            return item.get("content")
    return None


def _ask_agent(*, event: dict, disclosure: object, ticker: str) -> float:
    """Run one bounded agent round and return its final prediction.

    Missing provider credentials and runtime failures fall back to the neutral
    percentile so an event still receives a valid competition submission.
    """
    global _agent_warned
    required_keys = ("OPENAI_API_KEY", "GOOGLE_API_KEY")
    missing = [name for name in required_keys if not os.environ.get(name)]
    if missing:
        if not _agent_warned:
            print(
                f"[WARN] Missing {', '.join(missing)} — submitting 0.5 placeholder."
            )
            _agent_warned = True
        return 0.5

    try:
        # Lazy imports keep importing predict.py cheap and allow the credential
        # fallback above to work without initializing provider clients.
        from em_agent.context import AnalysisContext
        from em_agent.graph import graph
        from em_agent.tracing import langfuse_handler

        context = AnalysisContext(
            stock_id=ticker,
            earnings_call_date=event.get("event_datetime") or "unknown",
            earnings_call_facts=_bundle_content(disclosure, FACTS_ID) or [],
            earnings_preview=_bundle_content(disclosure, PREVIEW_ID),
            option_stats=_bundle_content(disclosure, OPTION_STATS_ID),
            knowledge_cutoff=(
                event.get("knowledge_cutoff")
                or event.get("event_datetime")
                or "unknown"
            ),
            # Four sequential LLM calls already occur in one round. Start with
            # one round to stay safely inside the five-minute submission window.
            num_rounds=2,
        )
        result = graph.invoke(
            {},
            context=context,
            config={
                "callbacks": [langfuse_handler],
                "metadata": {
                    "event_id": event.get("event_id"),
                    "ticker": ticker,
                    "knowledge_cutoff": context.knowledge_cutoff,
                },
            },
        )
        return result["current_analysis"].prediction
    except Exception as exc:
        print(
            f"[ERROR] agent prediction failed for event {event.get('event_id')} "
            f"ticker {ticker}: {exc}"
        )
        return 0.5
