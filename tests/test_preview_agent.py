"""preview_agent: calendar helpers, request building and the cutoff-guarded run, fully offline."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from preview_agent import events, generator, names

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


def make_event(event_id="ea_VZ_Q4_2026", ticker="VZ", when="2026-10-27T11:00:00Z", cutoff="2026-10-26T20:00:00Z"):
    return {
        "event_id": event_id, "event_type": "EARNINGS_RELEASE", "timing_category": "SCHEDULED",
        "event_datetime": when, "knowledge_cutoff": cutoff,
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": ticker}],
    }


def iso(moment):
    return moment.isoformat().replace("+00:00", "Z")


@pytest.fixture(autouse=True)
def offline_names(tmp_path, monkeypatch):
    """No real Yahoo calls in tests: lookups fail unless a test installs its own fake; cache in tmp."""
    monkeypatch.setattr(names, "CACHE_PATH", tmp_path / "company_names.json")
    monkeypatch.setattr(names, "_memory", {})

    def no_network(symbol):
        raise ConnectionError("network disabled in tests")

    monkeypatch.setattr(names, "_fetch_name", no_network)
    # company_name's default cache_path was bound at import; route lookups through the tmp cache.
    real = names.company_name
    monkeypatch.setattr(generator, "company_name", lambda t: real(t, cache_path=tmp_path / "company_names.json"))
    return tmp_path / "company_names.json"


def test_fiscal_period_and_request():
    assert generator.fiscal_period("ea_MMYT_Q2_2027") == "Q2 FY2027"
    assert generator.fiscal_period("ea_ODD_ID") is None
    assert generator.build_request(make_event(), "Verizon Communications Inc.") == (
        "Verizon Communications Inc. (VZ), fiscal period Q4 FY2026, earnings release scheduled "
        "2026-10-27 11:00 UTC. Knowledge cutoff: 2026-10-26T20:00:00Z."
    )
    assert generator.build_request(make_event(), lookup=False).startswith("Ticker VZ, fiscal period Q4 FY2026")
    # A failed lookup (network disabled here) also falls back to the ticker.
    assert generator.build_request(make_event()).startswith("Ticker VZ, fiscal period Q4 FY2026")


def test_options_load_only_this_skill_and_keep_the_machine_read_only():
    options = generator.build_options()
    assert options.cwd == str(generator.WORKSPACE) and options.setting_sources == ["project"]
    assert options.skills == [generator.SKILL_NAME] and generator.SKILL_PATH.exists()
    assert {"Bash", "Write", "Edit"} <= set(options.disallowed_tools)
    assert {"WebSearch", "WebFetch"} <= set(options.allowed_tools)


def test_next_event_skips_passed_cutoffs_and_filters_by_ticker():
    cal = [
        make_event("ea_OLD_Q3_2026", "OLD", "2026-10-24T13:00:00Z", "2026-10-23T20:00:00Z"),  # cutoff passed
        make_event("ea_B_Q3_2026", "B", "2026-10-28T11:00:00Z", "2026-10-27T20:00:00Z"),
        make_event("ea_A_Q3_2026", "A", "2026-10-27T11:00:00Z", "2026-10-26T20:00:00Z"),
    ]
    assert events.next_event(cal, NOW)["event_id"] == "ea_A_Q3_2026"
    assert events.next_event(cal, NOW, ticker="B")["event_id"] == "ea_B_Q3_2026"
    with pytest.raises(LookupError):
        events.next_event(cal, NOW, ticker="OLD")
    table = events.events_table(cal, NOW)
    assert table["event_id"].tolist() == ["ea_OLD_Q3_2026", "ea_A_Q3_2026", "ea_B_Q3_2026"]
    assert table["cutoff_ahead"].tolist() == [False, True, True]


def test_fetch_events_sends_the_key_and_date_window(monkeypatch):
    seen = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [make_event()]

    def fake_get(url, params, headers, timeout):
        seen.update(url=url, params=params, headers=headers)
        return Response()

    monkeypatch.setenv("EM_API_KEY", "test-key")
    monkeypatch.delenv("EM_API_BASE_URL", raising=False)
    monkeypatch.setattr(events.httpx, "get", fake_get)
    out = events.fetch_events(NOW, NOW + timedelta(days=2))
    assert out[0]["event_id"] == "ea_VZ_Q4_2026"
    assert seen["url"] == "https://api.explainingmarkets.ai/v1/events"
    assert seen["params"] == {"start_date": "2026-10-24T12:00:00Z", "end_date": "2026-10-26T12:00:00Z"}
    assert seen["headers"] == {"X-API-Key": "test-key"}


def test_generate_preview_refuses_after_the_cutoff(tmp_path):
    past = make_event(cutoff=iso(datetime.now(timezone.utc) - timedelta(minutes=1)))
    with pytest.raises(generator.CutoffPassedError):
        asyncio.run(generator.generate_preview(past, out_dir=tmp_path, verbose=False))
    assert not list(tmp_path.iterdir())


def test_generate_preview_saves_the_preview_and_record(tmp_path, monkeypatch):
    async def fake_stream(prompt, options, record, verbose):
        record["skills_loaded"] = [generator.SKILL_NAME]
        record["tool_calls"].append({"tool": "WebSearch", "detail": "VZ consensus"})
        record.update(subtype="success", is_error=False, num_turns=7, total_cost_usd=0.42)
        return "# Verizon (NYSE: VZ) — Q4 FY2026 Earnings Preview"

    monkeypatch.setattr(generator, "_stream", fake_stream)
    event = make_event(cutoff=iso(datetime.now(timezone.utc) + timedelta(hours=1)))
    record = asyncio.run(generator.generate_preview(event, company="Verizon", out_dir=tmp_path, verbose=False))
    assert record["usable"] and record["finished_before_cutoff"] and record["preview"].startswith("# Verizon")
    assert record["prompt"].startswith("/earnings-preview Verizon (VZ)")
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["skill_sha256"] == generator.skill_sha256() and "preview" not in saved
    assert next(tmp_path.glob("*.md")).read_text().startswith("# Verizon")


def test_generate_preview_stops_at_the_cutoff(tmp_path, monkeypatch):
    async def slow_stream(prompt, options, record, verbose):
        await asyncio.sleep(5)
        return "too late"

    monkeypatch.setattr(generator, "_stream", slow_stream)
    event = make_event(cutoff=iso(datetime.now(timezone.utc) + timedelta(seconds=0.5)))
    record = asyncio.run(generator.generate_preview(event, out_dir=tmp_path, verbose=False))
    assert record["subtype"] == "stopped_at_cutoff" and not record["usable"] and record["preview"] is None
    assert not list(tmp_path.glob("*.md")) and len(list(tmp_path.glob("*.json"))) == 1


def test_company_name_uses_yahoo_symbols_and_caches_on_disk(offline_names, monkeypatch):
    calls = []

    def fake_fetch(symbol):
        calls.append(symbol)
        return {"BRK-B": "Berkshire Hathaway Inc.", "VZ": "Verizon Communications Inc."}.get(symbol)

    monkeypatch.setattr(names, "_fetch_name", fake_fetch)
    assert names.company_name("BRK.B", cache_path=offline_names) == "Berkshire Hathaway Inc."
    assert names.company_name("VZ", cache_path=offline_names) == "Verizon Communications Inc."
    assert names.company_name("VZ", cache_path=offline_names) == "Verizon Communications Inc."
    assert calls == ["BRK-B", "VZ"]  # dot -> dash, second VZ served from memory
    assert json.loads(offline_names.read_text()) == {"BRK.B": "Berkshire Hathaway Inc.", "VZ": "Verizon Communications Inc."}

    monkeypatch.setattr(names, "_memory", {})  # a new process: the disk cache answers without a fetch
    assert names.company_name("BRK.B", cache_path=offline_names) == "Berkshire Hathaway Inc." and len(calls) == 2


def test_failed_or_empty_lookups_return_none_and_are_retried(offline_names, monkeypatch):
    calls = []
    monkeypatch.setattr(names, "_fetch_name", lambda symbol: calls.append(symbol))  # returns None
    assert names.company_name("ZZZZ", cache_path=offline_names) is None
    assert names.company_name("ZZZZ", cache_path=offline_names) is None
    assert calls == ["ZZZZ", "ZZZZ"] and not offline_names.exists()  # nothing cached, so retried

    def boom(symbol):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(names, "_fetch_name", boom)
    assert names.company_name("VZ", cache_path=offline_names) is None


def test_build_request_looks_up_the_company_unless_given(monkeypatch):
    monkeypatch.setattr(names, "_fetch_name", lambda symbol: "Verizon Communications Inc.")
    assert generator.build_request(make_event()).startswith("Verizon Communications Inc. (VZ), fiscal period Q4 FY2026")
    assert generator.build_request(make_event(), "Verizon").startswith("Verizon (VZ)")


def test_generate_preview_records_where_the_name_came_from(tmp_path, monkeypatch):
    async def fake_stream(prompt, options, record, verbose):
        record.update(subtype="success", is_error=False)
        return "preview"

    monkeypatch.setattr(generator, "_stream", fake_stream)
    event = make_event(cutoff=iso(datetime.now(timezone.utc) + timedelta(hours=1)))
    out = tmp_path / "out"

    looked_up = make_event(cutoff=event["knowledge_cutoff"])
    monkeypatch.setattr(names, "_fetch_name", lambda symbol: "Verizon Communications Inc.")
    record = asyncio.run(generator.generate_preview(looked_up, out_dir=out, verbose=False))
    assert record["company"] == "Verizon Communications Inc." and record["company_source"] == "yahoo"
    assert record["prompt"].startswith("/earnings-preview Verizon Communications Inc. (VZ)")

    given = asyncio.run(generator.generate_preview(event, company="Verizon", out_dir=out, verbose=False))
    assert given["company_source"] == "given" and given["prompt"].startswith("/earnings-preview Verizon (VZ)")

    monkeypatch.setattr(names, "_memory", {})
    monkeypatch.setattr(names, "_fetch_name", lambda symbol: None)
    nameless = asyncio.run(generator.generate_preview(make_event("ea_NEW_Q1_2027", "NEW", cutoff=event["knowledge_cutoff"]), out_dir=out, verbose=False))
    assert nameless["company"] is None and nameless["company_source"] == "ticker only"
    assert nameless["prompt"].startswith("/earnings-preview Ticker NEW")


def test_presets_and_overrides():
    cheap = generator.PRESETS["cheap"]
    assert (cheap.model, cheap.effort, cheap.max_turns, cheap.max_budget_usd, cheap.max_searches, cheap.max_fetches) == (
        "claude-haiku-5-5", "low", 12, 0.10, 4, 3)
    assert generator.resolve_settings("cheap", max_searches=6).max_searches == 6
    assert generator.resolve_settings("cheap") is cheap and generator.PRESETS["cheap"].max_searches == 4  # presets untouched
    custom = generator.PreviewSettings(model="claude-sonnet-5-5")
    assert generator.resolve_settings(custom) is custom
    with pytest.raises(KeyError):
        generator.resolve_settings("unknown")


def test_options_follow_the_settings():
    cheap = generator.build_options("cheap")
    assert (cheap.model, cheap.effort, cheap.max_turns, cheap.max_budget_usd) == ("claude-haiku-5-5", "low", 12, 0.10)
    assert "at most 4 web searches and at most 3 page fetches" in cheap.system_prompt["append"]
    assert cheap.system_prompt["preset"] == "claude_code"
    assert {"Agent", "Task"} <= set(cheap.disallowed_tools)  # no sub-agents
    (matcher,) = cheap.hooks["PreToolUse"]
    assert matcher.matcher == "WebSearch|WebFetch"
    uncapped = generator.build_options(generator.PreviewSettings())
    assert uncapped.system_prompt is None and generator.cost_note(generator.PreviewSettings()) is None


def test_web_cap_hook_counts_and_denies_beyond_the_caps():
    web_calls = {}
    hook = generator.web_cap_hook(generator.PreviewSettings(max_searches=2, max_fetches=None), web_calls)
    call = lambda tool: asyncio.run(hook({"tool_name": tool, "hook_event_name": "PreToolUse", "tool_input": {}}, None, None))
    assert call("WebSearch") == {} and call("WebSearch") == {}
    denied = call("WebSearch")["hookSpecificOutput"]
    assert denied["permissionDecision"] == "deny" and "limit of 2" in denied["permissionDecisionReason"]
    assert all(call("WebFetch") == {} for _ in range(5))  # no fetch cap
    assert call("Read") == {}  # other tools are not counted
    assert web_calls == {"WebSearch": 2, "denied": 1, "WebFetch": 5}


def test_generate_preview_records_settings_and_web_calls(tmp_path, monkeypatch):
    async def fake_stream(prompt, options, record, verbose):
        hook = options.hooks["PreToolUse"][0].hooks[0]
        for _ in range(5):  # the agent tries 5 searches against a cap of 4
            await hook({"tool_name": "WebSearch", "hook_event_name": "PreToolUse", "tool_input": {}}, None, None)
        record.update(subtype="success", is_error=False, total_cost_usd=0.07)
        return "preview"

    monkeypatch.setattr(generator, "_stream", fake_stream)
    event = make_event(cutoff=iso(datetime.now(timezone.utc) + timedelta(hours=1)))
    record = asyncio.run(generator.generate_preview(event, company="Verizon", settings="cheap", max_fetches=1,
                                                    out_dir=tmp_path, verbose=False))
    assert record["model"] == "claude-haiku-5-5" and record["settings"]["max_fetches"] == 1
    assert record["web_calls"] == {"WebSearch": 4, "denied": 1}
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["settings"]["max_searches"] == 4 and saved["web_calls"]["denied"] == 1
