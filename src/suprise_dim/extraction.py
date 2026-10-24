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
from .prompts import EXTRACTION_SYSTEM, build_user_message
from .schemas import SurpriseExtraction

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_EFFORT = "medium"


def make_extractor(model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT):
    llm = ChatOpenAI(model=model, reasoning={"effort": effort})
    return llm.with_structured_output(SurpriseExtraction)


def extract_event(event: dict, extractor=None) -> SurpriseExtraction:
    """Run the extraction on a single event; handy for inspecting the prompt's behaviour."""
    extractor = extractor or make_extractor()
    return extractor.invoke(
        [SystemMessage(EXTRACTION_SYSTEM), HumanMessage(build_user_message(event))],
        config={"tags": ["surprise_extraction"], "metadata": {"event_id": event["event_id"]}},
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
) -> dict:
    """Extract every event not already in `out_path`, appending as each finishes.

    Returns counts plus the ids that failed; calling again retries them.
    """
    done = done_event_ids(out_path)
    todo = [e for e in events if e["event_id"] not in done]
    summary = {"requested": len(events), "skipped": len(events) - len(todo), "written": 0, "failed": []}
    if not todo:
        print(f"All {len(events)} events already extracted in {out_path}")
        return summary

    extractor = make_extractor(model, effort)

    def run(event: dict) -> dict:
        result = extract_event(event, extractor)
        return {
            "event_id": event["event_id"],
            "ticker": ticker(event),
            "event_datetime": event.get("event_datetime"),
            # Kept for later validation against outcomes; never part of the prompt.
            "car1": car1(event),
            "model": model,
            "effort": effort,
            **result.model_dump(),
        }

    print(f"Extracting {len(todo)} events ({summary['skipped']} already done) with {model}...")
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
