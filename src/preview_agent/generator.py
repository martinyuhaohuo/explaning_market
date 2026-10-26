"""Run the earnings-preview skill for one calendar event with the Claude Agent SDK.

The request mirrors the competition's: one line naming the company, ticker,
fiscal period and scheduled event time; the agent researches the rest on the
public web. The skill is dispatched by name, so it always runs.

Knowledge cutoff: a preview may only use information public before the event's
`knowledge_cutoff`. A run refuses to start after the cutoff and is stopped when
the cutoff arrives; only a preview finished before the cutoff is marked
`usable`. A preview generated after an event has happened will contain its
results and must not be used for training or evaluation.

Each run writes `<event_id>_<UTC start>.md` (the preview) and a `.json` record
(request, model, skill hash, timestamps, cost, turns, tool calls) to
`data/surprise/own_previews/`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)
from dotenv import load_dotenv

from .names import company_name

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKSPACE = HERE / "workspace"  # the agent's working directory; its .claude/skills/ holds the skill
SKILL_NAME = "earnings-preview"
SKILL_PATH = WORKSPACE / ".claude" / "skills" / SKILL_NAME / "SKILL.md"
OUT_DIR = ROOT / "data" / "surprise" / "own_previews"



@dataclass(frozen=True)
class PreviewSettings:
    """How much one preview run may spend.

    `max_budget_usd` stops the run when the SDK's cost estimate reaches it (the
    estimate can overshoot, and a stopped run returns no preview), so the real
    levers are the model, effort, turns and the web caps. `max_searches` and
    `max_fetches` are hard caps: calls beyond them are denied, and the agent is
    told the limit up front. None means no cap / the model's default effort.
    """

    model: str = "claude-opus-5-5"
    effort: str | None = None  # low | medium | high | xhigh | max
    max_turns: int = 10
    max_budget_usd: float = 0.05
    max_searches: int | None = None
    max_fetches: int | None = None


PRESETS = {
    "default": PreviewSettings(),
    # About $0.10 a preview: web searches cost $0.01 each, and Haiku 5.5 tokens $0.10 / $0.50 per 1M
    # (prompts up to 100K tokens; 5x beyond, which the fetch cap helps avoid).
    "cheap": PreviewSettings(model="claude-haiku-5-5", effort="low", max_turns=12, max_budget_usd=0.10,
                             max_searches=4, max_fetches=3),
}


def resolve_settings(settings: PreviewSettings | str = "default", **overrides) -> PreviewSettings:
    """A preset name or settings object, with any field overridden (e.g. `max_searches=6`)."""
    base = PRESETS[settings] if isinstance(settings, str) else settings
    return replace(base, **overrides) if overrides else base

load_dotenv(ROOT / ".env")  # ANTHROPIC_API_KEY; the SDK passes the environment to Claude Code


class CutoffPassedError(RuntimeError):
    """Raised when a preview is requested after the event's knowledge cutoff."""


def parse_time(value: str) -> datetime:
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def fiscal_period(event_id: str) -> str | None:
    """'ea_TWST_Q3_2026' -> 'Q3 FY2026' (the id carries the company's fiscal quarter and year)."""
    match = re.search(r"_Q([1-4])_(\d{4})$", event_id)
    return f"Q{match.group(1)} FY{match.group(2)}" if match else None


def build_request(event: dict, company: str | None = None, *, lookup: bool = True) -> str:
    """The one-line request: company, ticker, fiscal period, scheduled event time.

    Without `company`, the name is looked up from the ticker (`names.company_name`,
    cached); if that fails, or `lookup=False`, the request names the ticker only.
    """
    ticker = event["focal_assets"][0]["identifier_value"]
    if company is None and lookup:
        company = company_name(ticker)
    name = f"{company} ({ticker})" if company else f"Ticker {ticker}"
    period = fiscal_period(event["event_id"])
    when = parse_time(event["event_datetime"]).strftime("%Y-%m-%d %H:%M UTC")
    request = name + (f", fiscal period {period}" if period else "") + f", earnings release scheduled {when}."
    return request + f" Knowledge cutoff: {event['knowledge_cutoff']}."


def web_cap_hook(settings: PreviewSettings, web_calls: dict):
    """A PreToolUse hook that counts WebSearch / WebFetch calls in `web_calls` and denies those beyond the caps."""
    caps = {"WebSearch": settings.max_searches, "WebFetch": settings.max_fetches}

    async def hook(input_data, tool_use_id, context):
        tool = input_data.get("tool_name")
        if tool not in caps:
            return {}
        cap = caps[tool]
        if cap is not None and web_calls.get(tool, 0) >= cap:
            web_calls["denied"] = web_calls.get("denied", 0) + 1
            return {"hookSpecificOutput": {
                "hookEventName": input_data["hook_event_name"],
                "permissionDecision": "deny",
                "permissionDecisionReason": f"{tool} limit of {cap} reached. Write the preview from what you have.",
            }}
        web_calls[tool] = web_calls.get(tool, 0) + 1
        return {}

    return hook


def cost_note(settings: PreviewSettings) -> str | None:
    """The limits, told to the agent up front so it plans within them."""
    limits = [f"at most {n} {what}" for n, what in ((settings.max_searches, "web searches"), (settings.max_fetches, "page fetches")) if n is not None]
    if not limits:
        return None
    return f"Cost limit for this task: use {' and '.join(limits)}, then write the preview from what you have. Calls beyond the limit are refused."


def build_options(settings: PreviewSettings | str = "default", web_calls: dict | None = None) -> ClaudeAgentOptions:
    """Agent options for one preview run; `web_calls` collects the search / fetch counts."""
    settings = resolve_settings(settings)
    web_calls = {} if web_calls is None else web_calls
    note = cost_note(settings)
    return ClaudeAgentOptions(
        model=settings.model,
        effort=settings.effort,
        cwd=str(WORKSPACE),              # .claude/skills/ is discovered from here
        setting_sources=["project"],     # this workspace's skills only; personal ~/.claude skills stay out
        skills=[SKILL_NAME],
        allowed_tools=["WebSearch", "WebFetch", "Read"],  # pre-approved: web research and reading the skill's files
        # Nothing touches this machine, and no sub-agents, which would multiply cost.
        disallowed_tools=["Bash", "Write", "Edit", "Agent", "Task"],
        system_prompt={"type": "preset", "preset": "claude_code", "append": note} if note else None,
        hooks={"PreToolUse": [HookMatcher(matcher="WebSearch|WebFetch", hooks=[web_cap_hook(settings, web_calls)])]},
        max_turns=settings.max_turns,
        max_budget_usd=settings.max_budget_usd,
    )


def skill_sha256() -> str:
    return hashlib.sha256(SKILL_PATH.read_bytes()).hexdigest()


async def _stream(prompt: str, options: ClaudeAgentOptions, record: dict, verbose: bool) -> str | None:
    """Run the session, recording which skills loaded and which tools ran; return the final text."""
    final = None
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            final = _handle(message, record, verbose) or final
    return final


def _handle(message, record: dict, verbose: bool) -> str | None:
    """Record one streamed message; returns the final text when the result arrives, else None."""
    if isinstance(message, SystemMessage) and message.subtype == "init":
        record["skills_loaded"] = message.data.get("skills", [])
        record["session_id"] = message.data.get("session_id")
        if verbose:
            print(f"session started; skills loaded: {record['skills_loaded']}")
    elif isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                detail = block.input.get("query") or block.input.get("url") or block.input.get("skill") or ""
                record["tool_calls"].append({"tool": block.name, "detail": str(detail)[:200]})
                if verbose:
                    print(f"  {block.name}: {str(detail)[:120]}")
            elif isinstance(block, TextBlock) and block.text.strip() and verbose:
                print(f"  (text, {len(block.text)} chars)")
    elif isinstance(message, ResultMessage):
        record.update({
            "subtype": message.subtype,
            "is_error": message.is_error,
            "num_turns": message.num_turns,
            "duration_ms": message.duration_ms,
            "total_cost_usd": message.total_cost_usd,
            "usage": message.usage,
        })
        return message.result
    return None


async def generate_preview(
    event: dict,
    *,
    company: str | None = None,
    settings: PreviewSettings | str = "default",
    out_dir: Path = OUT_DIR,
    verbose: bool = True,
    **overrides,
) -> dict:
    """Generate one preview for a calendar event; returns the run record (with `preview` text when produced).

    `settings` is a preset name ("default", "cheap") or a `PreviewSettings`;
    keyword overrides change single fields (`max_searches=6`). Await it from a
    notebook (`record = await generate_preview(event, settings="cheap")`); from
    a plain script use `generate_preview_sync`.
    """
    settings = resolve_settings(settings, **overrides)
    cutoff = parse_time(event["knowledge_cutoff"])
    started = datetime.now(timezone.utc)
    if started >= cutoff:
        raise CutoffPassedError(f"{event['event_id']}: knowledge cutoff {cutoff.isoformat()} has already passed")

    ticker = event["focal_assets"][0]["identifier_value"]
    company_source = "given" if company else None
    if company is None:
        company = company_name(ticker)
        company_source = "yahoo" if company else "ticker only"
    request = build_request(event, company, lookup=False)
    prompt = f"/{SKILL_NAME} {request}"  # dispatch the skill by name; the request becomes its arguments
    record = {
        "event_id": event["event_id"],
        "ticker": ticker,
        "company": company,
        "company_source": company_source,
        "event_datetime": event["event_datetime"],
        "knowledge_cutoff": event["knowledge_cutoff"],
        "request": request,
        "prompt": prompt,
        "model": settings.model,
        "settings": asdict(settings),
        "skill_path": str(SKILL_PATH.relative_to(ROOT)),
        "skill_sha256": skill_sha256(),
        "started_at": started.isoformat(),
        "tool_calls": [],
        "web_calls": {},
    }
    if verbose:
        print(f"prompt: {prompt}")
    try:
        preview = await asyncio.wait_for(
            _stream(prompt, build_options(settings, record["web_calls"]), record, verbose),
            timeout=(cutoff - started).total_seconds(),
        )
    except asyncio.TimeoutError:
        preview, record["subtype"] = None, "stopped_at_cutoff"
    finished = datetime.now(timezone.utc)
    record["finished_at"] = finished.isoformat()
    record["finished_before_cutoff"] = finished < cutoff
    record["usable"] = bool(preview) and not record.get("is_error") and finished < cutoff

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{event['event_id']}_{started.strftime('%Y%m%dT%H%M%SZ')}"
    if preview:
        (out_dir / f"{stem}.md").write_text(preview)
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2, default=str))
    record["preview"] = preview
    record["files"] = [str((out_dir / f"{stem}.{ext}")) for ext in (("md", "json") if preview else ("json",))]
    if verbose:
        print(f"{record.get('subtype')} | turns {record.get('num_turns')} | cost ${record.get('total_cost_usd') or 0:.3f} | "
              f"web {record['web_calls']} | usable {record['usable']}")
    return record


def generate_preview_sync(event: dict, **kwargs) -> dict:
    """`generate_preview` for plain scripts (not inside a running event loop such as Jupyter)."""
    return asyncio.run(generate_preview(event, **kwargs))
