"""Synthetic thin previews: the competition's preview cut down to what our cheap agent produces.

Our own previews (`preview_agent`, cheap preset) are thinner than the
competition's: a few searches, no page reads, no price or implied move. A model
trained only on competition previews may misjudge them. Generating our own
previews for past events would leak their results (the web now contains them),
so instead each competition preview is condensed by a cheap model into the
shape and depth of a cheap preview. Every fact comes from the competition's
preview, written before the knowledge cutoff, so nothing leaks.

Results are appended to `data/surprise/thin_previews_<name>.jsonl`, one record
per event, so an interrupted run resumes.
"""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from .data import OUT_DIR, PREVIEW_ID, ROOT, items_by_id, ticker

load_dotenv(ROOT / ".env")

THIN_MODEL = "claude-haiku-5-5"
THIN_EFFORT = "low"
EXAMPLE_PATH = Path(__file__).with_name("thin_preview_example.md")  # a real cheap-preset preview (AEYE, Q3 FY2026)

THIN_SYSTEM = """You rewrite an equity research earnings preview into a much thinner one.

The thin preview imitates what a quick, low-budget research agent produces after three or four web searches and no page reads. You receive a real example of such a thin preview (for a different company) and the full preview to condense.

Write the thin preview for the full preview's company, in the example's structure and tone:
- Keep: the consensus and company-guidance figures for the quarter and the year, the prior quarter's headline results and guidance, and the three to five issues the full preview treats as most important.
- Keep each figure's source and date when the full preview gives one; where sources disagree, keep both.
- Drop: the share price, market capitalization, options-implied move, whisper numbers, detailed segment analysis, and history of past reactions. Where the example says something could not be found, say the same.
- Use only information in the full preview. Never add facts, numbers, or sources it does not contain, and never mention the full preview itself.
- Aim for roughly the example's length (about 5,000 to 6,000 characters).

Return only the thin preview in Markdown."""


def make_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def build_thin_message(event: dict, example: str | None = None) -> str:
    preview = items_by_id(event)[PREVIEW_ID]["content"]
    example = example if example is not None else EXAMPLE_PATH.read_text()
    return (
        "Example of a thin preview (different company):\n<example>\n" + example + "\n</example>\n\n"
        f"Full preview to condense ({ticker(event)}):\n<full_preview>\n" + preview + "\n</full_preview>"
    )


def thin_preview(event: dict, client: anthropic.Anthropic | None = None, *, model: str = THIN_MODEL,
                 effort: str = THIN_EFFORT) -> dict:
    """Condense one event's competition preview; returns the text plus usage and stop reason."""
    client = client or make_client()
    response = client.messages.create(
        model=model,
        max_tokens=8000,
        system=THIN_SYSTEM,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": build_thin_message(event)}],
    )
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise RuntimeError(f"refused ({category})")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("thin preview cut off at max_tokens")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise RuntimeError("empty thin preview")
    return {
        "preview": text,
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def thin_path(name: str) -> Path:
    return OUT_DIR / f"thin_previews_{name}.jsonl"


def done_event_ids(name: str) -> set[str]:
    path = thin_path(name)
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["event_id"] for line in f if line.strip()}


def run_thin_previews(events: list[dict], name: str, *, model: str = THIN_MODEL, effort: str = THIN_EFFORT,
                      workers: int = 8) -> dict:
    """Condense every event not yet in run `name`, appending as each finishes; calling again retries failures."""
    done = done_event_ids(name)
    todo = [e for e in events if e["event_id"] not in done]
    summary = {"requested": len(events), "skipped": len(events) - len(todo), "written": 0, "failed": []}
    if not todo:
        print(f"All {len(events)} events already condensed in {thin_path(name)}")
        return summary

    client = make_client()
    print(f"Condensing {len(todo)} previews ({summary['skipped']} already done) with {model}...")
    path = thin_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(thin_preview, e, client, model=model, effort=effort): e["event_id"] for e in todo}
        for n, future in enumerate(as_completed(futures), start=1):
            event_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                summary["failed"].append(event_id)
                print(f"[{n}/{len(todo)}] {event_id} FAILED: {exc}")
                continue
            out.write(json.dumps({"event_id": event_id, "model": model, "effort": effort, **result}) + "\n")
            out.flush()
            summary["written"] += 1
            print(f"[{n}/{len(todo)}] {event_id}: {len(result['preview'])} chars")

    print(f"Done. {summary['written']} written, {len(summary['failed'])} failed; call again to retry failures.")
    return summary


def load_thin_previews(name: str) -> dict[str, str]:
    """event id -> thin preview text (the last record wins if an event was written twice)."""
    path = thin_path(name)
    if not path.exists():
        return {}
    with path.open() as f:
        return {r["event_id"]: r["preview"] for r in map(json.loads, filter(str.strip, f))}


def with_preview(event: dict, preview: str) -> dict:
    """A copy of the event whose preview is replaced, so the normal extraction runs on the thin version."""
    out = copy.deepcopy(event)
    for item in out["disclosure"]["items"]:
        if item.get("id") == PREVIEW_ID:
            item["content"] = preview
            return out
    raise ValueError(f"{event['event_id']} has no preview to replace")
