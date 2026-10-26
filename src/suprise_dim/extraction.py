"""Free-form surprise extraction over archived events.

Results are appended to a JSONL file one event at a time, so an interrupted
run resumes where it stopped.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .data import EXTRACTIONS_PATH, ROOT, car1, ticker
from .prompts import (
    EXTRACTION_SYSTEM, EXTRACTION_SYSTEM_NO_PREVIEW, build_user_message, build_user_message_no_preview,
)
from .schemas import SurpriseExtraction, SurpriseExtractionNoPreview

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_EFFORT = "medium"


def make_extractor(model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, *, preview: bool = True):
    llm = ChatOpenAI(model=model, reasoning={"effort": effort})
    return llm.with_structured_output(SurpriseExtraction if preview else SurpriseExtractionNoPreview)


def extract_event(event: dict, extractor=None, *, preview: bool = True) -> SurpriseExtraction:
    """Run the extraction on a single event; handy for inspecting the prompt's behaviour.

    `preview=False` uses the no-preview prompt and leaves out any preview the
    event carries, so preview events can also be extracted as if they had none.
    """
    extractor = extractor or make_extractor(preview=preview)
    system, user = (
        (EXTRACTION_SYSTEM, build_user_message(event)) if preview
        else (EXTRACTION_SYSTEM_NO_PREVIEW, build_user_message_no_preview(event))
    )
    return extractor.invoke(
        [SystemMessage(system), HumanMessage(user)],
        config={"tags": ["surprise_extraction"], "metadata": {"event_id": event["event_id"], "preview": preview}},
    )


def done_event_ids(path: Path = EXTRACTIONS_PATH) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["event_id"] for line in f if line.strip()}


def run_extraction(
    events: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    workers: int = 8,
    out_path: Path = EXTRACTIONS_PATH,
    preview: bool = True,
) -> dict:
    """Extract every event not already in `out_path`, appending as each finishes.

    `preview=False` uses the no-preview prompt (see `extract_event`). Returns
    counts plus the ids that failed; calling again retries them.
    """
    done = done_event_ids(out_path)
    todo = [e for e in events if e["event_id"] not in done]
    summary = {"requested": len(events), "skipped": len(events) - len(todo), "written": 0, "failed": []}
    if not todo:
        print(f"All {len(events)} events already extracted in {out_path}")
        return summary

    extractor = make_extractor(model, effort, preview=preview)

    def run(event: dict) -> dict:
        result = extract_event(event, extractor, preview=preview)
        return {
            "event_id": event["event_id"],
            "ticker": ticker(event),
            "event_datetime": event.get("event_datetime"),
            # Kept for later validation against outcomes; never part of the prompt.
            "car1": car1(event),
            "model": model,
            "effort": effort,
            "preview": preview,
            **result.model_dump(),
        }

    prompt = "preview" if preview else "no-preview"
    print(f"Extracting {len(todo)} events ({summary['skipped']} already done) with {model}, {prompt} prompt...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, e): e["event_id"] for e in todo}
        for n, future in enumerate(as_completed(futures), start=1):
            event_id = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                summary["failed"].append(event_id)
                print(f"[{n}/{len(todo)}] {event_id} FAILED: {exc}")
                continue
            out.write(json.dumps(record) + "\n")
            out.flush()
            summary["written"] += 1
            print(f"[{n}/{len(todo)}] {event_id}: {len(record['surprises'])} surprises")

    print(f"Done. {summary['written']} written, {len(summary['failed'])} failed; call again to retry failures.")
    return summary


def load_extractions(path: Path = EXTRACTIONS_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
