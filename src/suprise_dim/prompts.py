from __future__ import annotations

from .data import FACTS_ID, PREVIEW_ID, items_by_id, ticker

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


def build_user_message(event: dict) -> str:
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


# Variant for events without a preview. The preview prompt above stays unchanged,
# so extractions made with it remain comparable.
EXTRACTION_SYSTEM_NO_PREVIEW = """You compare an earnings release against what was expected before it.

You receive facts from the earnings call, written after the release. No research preview is available, so the only expectations you can use are the ones the call itself states: analyst consensus or estimates the call mentions, the company's prior guidance or targets, or what management says it or the market expected.

List every distinct way the reported results, guidance, or commentary differ from those stated expectations. Cover everything the facts allow: headline numbers, guidance, segments, margins, operating metrics, capital allocation, one-off items, and management tone.

Rules:
- One item per distinct surprise. Do not merge unrelated surprises; do not split one into many.
- Describe each surprise concretely in plain language. Do not label it with an abstract category; the categories will be derived later from many releases.
- A surprise needs a stated expectation. Growth or a change against the prior year or prior quarter is not a surprise by itself; list it only if the call compares it with guidance, consensus or a stated target.
- When the call stresses something as better or worse than anticipated but names no benchmark, you may list it with expectation_source "none" and leave expectation_evidence empty; judge its magnitude conservatively.
- Management presents its own results. Judge direction from the substance of each fact, not from upbeat or cautious wording.
- Results the call reports as in line with guidance or consensus go under in_line.
- Use only the material provided. Do not use outside knowledge of this company or of how its stock traded.
- Do not predict the stock reaction."""


def build_user_message_no_preview(event: dict) -> str:
    """Ticker, release time and call facts only; any preview the event carries is left out."""
    facts = items_by_id(event)[FACTS_ID]["content"]
    return (
        f"Ticker: {ticker(event)}\n"
        f"Release time: {event.get('event_datetime') or 'unknown'}\n\n"
        "Facts from the earnings call:\n"
        + "\n".join(f"{n}. {fact}" for n, fact in enumerate(facts, start=1))
        + "\n\nResearch preview: not available for this release."
    )


# Textbook starting point; the proposal step may keep, split, merge or drop
# these, and should add whatever the data shows that is missing.
SEED_DIMENSIONS = [
    "revenue / top line vs. consensus or guidance",
    "earnings per share / net income vs. consensus",
    "margins and profitability (gross, operating, EBITDA)",
    "costs and operating expenses",
    "forward guidance (raised, cut, initiated, withdrawn)",
    "segment or product-line performance",
    "industry-specific operating KPIs (same-store sales, subscribers, net interest margin, bookings, utilization)",
    "cash flow, balance sheet and liquidity",
    "capital allocation (buybacks, dividends, capex, debt repayment)",
    "one-off or non-recurring items (impairments, litigation, tax, accounting effects)",
    "strategic and corporate events (M&A, partnerships, leadership changes, restructuring)",
    "management tone and credibility (confidence, hedging, missed promises)",
]

TAXONOMY_SYSTEM = """You design a codebook for classifying earnings surprises.

Each item you see is one surprise from one earnings release: a concrete way the reported results, guidance, or commentary differed from what was expected. You also get a starting list of candidate dimensions.

Produce a set of dimensions such that:
- Every item fits exactly one dimension. Dimensions are mutually exclusive, with clear boundaries spelled out in `excludes`.
- Dimensions are general across industries. An industry-specific metric belongs in a general dimension (e.g. "key operating KPI"), not a dimension of its own.
- A dimension describes WHAT the surprise is about, never its direction or size. "Revenue beat" and "revenue miss" are the same dimension.
- Keep, split, merge, or drop starting dimensions as the items warrant, and add dimensions the items show are missing. Split only when the two parts are common and clearly different in kind; merge dimensions that would rarely be told apart.
- Aim for roughly 8 to 15 dimensions. Do not create a catch-all; items that fit nothing will be reviewed separately."""

REFINE_SYSTEM = """You revise a codebook for classifying earnings surprises.

You get the current dimensions, how many items were assigned to each, a sample of items per dimension, and items that fit no dimension ("other").

Revise the codebook:
- Add a dimension when "other" contains a common, coherent group; otherwise extend an existing definition to cover those items.
- Merge dimensions that are small or whose samples look indistinguishable. Split a dimension whose samples clearly contain two different kinds of surprise.
- Tighten `excludes` wherever samples show items landing in the wrong dimension.
- Keep keys unchanged for dimensions whose meaning is unchanged.
- A dimension describes WHAT the surprise is about, never its direction or size.
List every change in `changes`; if nothing needs to change, return the codebook as is with an empty `changes`."""

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


LABEL_SYSTEM = """You label surprises from earnings releases.

You receive a numbered list of surprises, possibly from different companies. Each surprise is one way a company's reported results, guidance or commentary differed from what was expected before the release. Each comes with its category, a description, the earnings-call evidence, the pre-release expectation evidence, and where the expectation came from (the research preview, the call itself, or implicit). Judge each surprise on its own, from this information only.

For each surprise, give:
- direction: 1 if it is good news for shareholders relative to what was expected, -1 if bad news, 0 if it combines good and bad parts that roughly offset. null only if the information does not let you tell.
- magnitude: the size of the gap relative to the expectation, not how important the topic is. 1 small (a slight beat or miss, within normal noise), 2 moderate (clearly outside the expected range), 3 large (far outside it, or a reset of the outlook). null when direction is null.
- sentiment: management's tone about this topic in the call evidence, separate from the numbers themselves. 1 confident or upbeat, -1 cautious, defensive or negative, 0 neutral or purely factual. null if the call evidence carries no management framing.
- reason: one short sentence.

Rules:
- Judge each surprise against the expectation, not against the prior year.
- Use the category only to understand what the surprise is about.
- Do not use outside knowledge of the companies or of how their stocks traded.
- Do not predict the stock reaction.
- Return exactly one label per surprise, using its number."""


def build_label_message(items) -> str:
    """The numbered surprises with their category, description, evidence and expectation source.

    `items` needs `dimension_name`, `description`, `fact_evidence`,
    `expectation_evidence` and `expectation_source` columns.
    """
    def text(value) -> str:
        # Empty evidence comes back from CSV as NaN.
        return value.strip() if isinstance(value, str) and value.strip() else "-"

    parts = ["Surprises to label:"]
    for n, row in enumerate(items.itertuples(index=False)):
        parts += [
            f"{n}. [{row.dimension_name}] {text(row.description)}",
            f"   Call evidence: {text(row.fact_evidence)}",
            f"   Expectation evidence: {text(row.expectation_evidence)}",
            f"   Expectation source: {text(row.expectation_source)}",
        ]
    return "\n".join(parts)
