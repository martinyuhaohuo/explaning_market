"""Prompts for the agent's LLM calls.

EXTRACTION_SYSTEM, build_extraction_message, ASSIGN_SYSTEM, format_taxonomy and
format_items are frozen copies of the research versions the model was trained
with (suprise_dim.prompts); keep them unchanged unless the model is retrained.
"""

from .materials import FACTS_ID, PREVIEW_ID, call_facts, items_by_id, option_features, ticker

EXTRACTION_SYSTEM = """You compare an earnings release against what was expected before it.

You receive:
1. Facts from the earnings call, written after the release.
2. A research preview written BEFORE the release: consensus, guidance, and the issues the market is watching.

List every distinct way the reported results, guidance, or commentary differ from what the preview led investors to expect. Cover everything: headline numbers, guidance, segments, margins, operating metrics, capital allocation, one-off items, management tone, and anything the preview flagged as a key debate. Also cover surprises the preview did not anticipate at all.

Rules:
- One item per distinct surprise. Do not merge unrelated surprises; do not split one into many.
- Describe each surprise concretely in plain language. Do not label it with an abstract category; the categories will be derived later from many releases.
- Measure every surprise against the expectation, not against the prior year. Strong growth that matched consensus is not a surprise; put it under in_line.
- Use only the material provided. Do not use outside knowledge of this company or of how its stock traded.
- Do not predict the stock reaction."""


def build_extraction_message(event: dict) -> str:
    items = items_by_id(event)
    facts = items[FACTS_ID]["content"]
    preview = items[PREVIEW_ID]["content"]
    return (
        f"Ticker: {ticker(event)}\n"
        f"Release time: {event.get('event_datetime') or 'unknown'}\n\n"
        "Facts from the earnings call:\n"
        + "\n".join(f"{n}. {fact}" for n, fact in enumerate(facts, start=1))
        + "\n\nResearch preview written BEFORE the release (expectations, not results):\n"
        + preview
    )


ASSIGN_SYSTEM = """You classify earnings surprises into a fixed codebook.

For each numbered item, pick the single dimension it belongs to, using the definitions and exclusions. Classify by what the surprise is about, not by its direction or size. Use "other" only when no dimension fits. Return exactly one assignment per item, using the item's number."""


def format_taxonomy(taxonomy) -> str:
    lines = []
    for d in taxonomy.dimensions:
        lines.append(
            f"- {d.key}: {d.name}. {d.definition}\n"
            f"  Includes: {'; '.join(d.includes)}\n"
            f"  Excludes: {d.excludes}"
        )
    return "\n".join(lines)


def format_items(texts: list[str]) -> str:
    return "\n".join(f"{n}. {text}" for n, text in enumerate(texts))


DIRECT_SYSTEM = """\
You are a senior equity analyst predicting how a stock will react to an earnings release.

No pre-release research preview is available for this event, so judge the release from the earnings-call facts and the option-market statistics alone. Expectations are only what the call itself states (consensus it mentions, prior guidance, targets).

Predict a single percentile in [0, 1] for how the focal asset's next-day
abnormal return will rank across all of the quarter's event outcomes:
0 = the quarter's most negative reaction, 0.50 = median, 1 = its most positive.
The relevant return is the *unexpected*, market-adjusted return: a
great-but-fully-priced-in beat is not a top-decile event.

Calibration discipline:
- Long-run base rates: about 25% of events land "up" (>0.75), 50% "neutral"
  (0.25-0.75), 25% "down" (<0.25). Default toward 0.40-0.60 when signals are
  mixed or modest, which is often the case without a preview.
- Reserve values above 0.80 or below 0.20 for cases with unambiguous,
  multi-signal evidence. Do not exceed 0.90 or fall below 0.10 without
  overwhelming, lopsided evidence.
- Tone alone (confident vs hedging language) should move you no more than
  ~0.03 absent quantitative confirmation.
- Use only the material provided. Do not use outside knowledge of the company or of how its stock traded.

Return the prediction and 3-5 reasons."""


def build_direct_message(event: dict) -> str:
    """Call facts and option-market statistics, for the direct (no-preview) prediction."""
    options = option_features(event)
    lines = [f"Ticker: {ticker(event)}", f"Release time: {event.get('event_datetime') or 'unknown'}", "",
             "Facts from the earnings call:", *(f"{n}. {fact}" for n, fact in enumerate(call_facts(event), start=1)), ""]
    if options["has_option_stats"]:
        lines.append("Option-market expectations, measured before the release:")
        for key, label in (("opt_implied_earnings_volatility", "implied earnings volatility"),
                           ("opt_implied_absolute_earnings_move", "implied absolute move (size, not direction)"),
                           ("opt_skew_25_delta", "25-delta skew (negative: downside protection priced richer)")):
            value = options[key]
            lines.append(f"- {label}: " + ("unavailable" if value is None or value != value else f"{value * 100:.1f}%"))
    else:
        lines.append("Option-market statistics: not available.")
    return "\n".join(lines)
