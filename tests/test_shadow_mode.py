"""Shadow mode and the surprise graph, fully offline: no Modal connection, no LLM calls.

Parity with the research code (prompts, features, model) is in test_em_agent.py.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import modal_app
import predict as predict_module
from em_agent import nodes
from em_agent.context import build_context
from em_agent.graph import graph
from em_agent.schemas import SurpriseExtraction
from shadow import runner, schedule, store as store_mod, sync

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
PREVIEW = "# Acme preview\nConsensus revenue $10.0M."
FACTS = ["Revenue was $11.0M, above the $10.0M consensus.", "Full-year guidance was raised."]


def iso(moment):
    return moment.isoformat().replace("+00:00", "Z")


def make_event(event_id="ea_ACME_Q3_2026", cutoff=NOW + timedelta(hours=1), ticker="ACME"):
    return {"id": f"evt_{event_id}", "event_id": event_id, "event_type": "EARNINGS_RELEASE",
            "event_datetime": iso(cutoff + timedelta(minutes=30)), "knowledge_cutoff": iso(cutoff),
            "focal_assets": [{"identifier_type": "TICKER", "identifier_value": ticker}],
            "information_url": "https://example.test/bundle"}


def make_bundle(preview=PREVIEW, stats=True):
    items = [{"id": "earnings-call-facts", "kind": "facts", "content": FACTS}]
    if preview is not None:
        items.append({"id": "earnings-preview", "kind": "text", "content": preview})
    if stats:
        items.append({"id": "option-implied-stats", "kind": "stats", "content": {
            "implied_earnings_volatility": {"value": 0.08, "status": "ok"},
            "implied_absolute_earnings_move": {"value": 0.06, "status": "ok"},
            "skew_25_delta": {"value": -0.02, "status": "ok"}}})
    return {"schema_version": "1.0", "items": items}


def surprise(desc, direction="positive", magnitude="moderate"):
    return {"description": desc, "direction": direction, "magnitude": magnitude, "fact_evidence": "f",
            "expectation_evidence": "e", "expectation_source": "preview"}


class FakeLLM:
    """Stands in for a structured-output client: records each call and returns `reply(messages)`."""

    def __init__(self, calls, name, reply):
        self.calls, self.name, self.reply = calls, name, reply

    def invoke(self, messages, config=None):
        self.calls[self.name] += 1
        self.calls[f"{self.name}_message"] = messages[1].content
        return self.reply(messages)


@pytest.fixture
def fake_llms(monkeypatch):
    """Extraction, classification and the direct prediction, without any LLM call."""
    calls = {"extract": 0, "classify": 0, "direct": 0}
    extracted = SurpriseExtraction(surprises=[surprise("revenue beat"), surprise("guidance raised"),
                                              surprise("margin miss", "negative", "small")], in_line=[])
    dims = ["revenue", "forward_guidance", "operating_profitability"]
    assigned = SimpleNamespace(assignments=[SimpleNamespace(index=i, dimension=d) for i, d in enumerate(dims)])
    monkeypatch.setattr(nodes, "extractor", lambda: FakeLLM(calls, "extract", lambda m: extracted))
    monkeypatch.setattr(nodes, "assigner", lambda: FakeLLM(calls, "classify", lambda m: assigned))
    monkeypatch.setattr(nodes, "direct_predictor", lambda: FakeLLM(
        calls, "direct", lambda m: SimpleNamespace(prediction=0.62, reasons="guidance raised")))
    return calls


# --------------------------------------------------------------------- preview choice

def test_preview_choice_prefers_the_competition_then_ours():
    event = make_event()
    bundle = make_bundle()
    assert build_context(event, bundle, own_preview="OURS").preview_source == "competition"

    ours = build_context(event, make_bundle(preview=None), own_preview="OURS")
    assert ours.preview_source == "own"
    assert [i["content"] for i in ours.event["disclosure"]["items"] if i["id"] == "earnings-preview"] == ["OURS"]

    empty = build_context(event, make_bundle(preview="   "), own_preview="OURS")
    assert [i["content"] for i in empty.event["disclosure"]["items"] if i["id"] == "earnings-preview"] == ["OURS"]

    assert build_context(event, make_bundle(preview=None), own_preview=None).preview_source is None
    assert bundle["items"][1]["content"] == PREVIEW  # the caller's bundle is never changed


# --------------------------------------------------------------------- graph

@pytest.mark.parametrize("preview, own, source, flag", [(PREVIEW, None, "competition", 0), (None, "OURS", "own", 1)])
def test_graph_with_a_preview_runs_the_surprise_pipeline(fake_llms, preview, own, source, flag):
    context = build_context(make_event(), make_bundle(preview=preview), own_preview=own)
    state = graph.invoke({}, context=context)
    assert state["branch"] == "surprise_ridge" and 0.0 <= state["prediction"] <= 1.0
    assert len(state["surprises"]) == 3 and {i["dimension"] for i in state["items"]} == {"revenue", "forward_guidance", "operating_profitability"}
    feats = state["features"]
    assert feats["src_own"] == flag and feats["preview_source"] == source
    assert feats["net_w__revenue"] == 2 and feats["net_w__operating_profitability"] == -1 and feats["n_items"] == 3
    assert feats["opt_implied_absolute_earnings_move"] == 0.06
    assert fake_llms["direct"] == 0


def test_graph_without_any_preview_predicts_directly(fake_llms):
    context = build_context(make_event(), make_bundle(preview=None), own_preview=None)
    state = graph.invoke({}, context=context)
    assert state == {"prediction": 0.62, "reasons": "guidance raised", "branch": "direct"}
    assert fake_llms["extract"] == 0 and fake_llms["direct"] == 1
    assert "1. Revenue was $11.0M" in fake_llms["direct_message"] and "implied absolute move" in fake_llms["direct_message"]


# --------------------------------------------------------------------- predict() and the records

def test_predict_records_everything_and_uses_our_preview(tmp_path, monkeypatch, fake_llms):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("SHADOW_STORE_DIR", str(tmp_path))
    monkeypatch.delenv("SHADOW_VOLUME", raising=False)
    event = make_event()
    store = store_mod.EventStore(tmp_path)
    store.save_own_preview(event["event_id"], {"preview": "OUR PREVIEW", "usable": True})
    monkeypatch.setattr(predict_module.httpx, "get", lambda *a, **k: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: make_bundle(preview=None)))

    preds = predict_module.predict(event)
    assert len(preds) == 1 and 0.0 <= preds[0]["predicted_percentile"] <= 1.0
    record = store.read_json(event["event_id"], store_mod.PREDICTION)
    assert record["preview_source"] == "own" and record["branch"] == "surprise_ridge"
    assert record["predictions"] == preds
    assert store.read_json(event["event_id"], store_mod.FEATURES)["features"]["src_own"] == 1
    assert len(store.read_json(event["event_id"], store_mod.SURPRISES)["items"]) == 3
    assert store.read_text(event["event_id"], store_mod.COMPETITION_PREVIEW) is None  # the event had none


def test_predict_saves_the_competition_preview_and_falls_back_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("SHADOW_STORE_DIR", str(tmp_path))
    monkeypatch.delenv("SHADOW_VOLUME", raising=False)

    class Broken:
        def invoke(self, messages, config=None):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(nodes, "extractor", lambda: Broken())
    monkeypatch.setattr(predict_module.httpx, "get", lambda *a, **k: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: make_bundle()))
    event = make_event()
    preds = predict_module.predict(event)
    store = store_mod.EventStore(tmp_path)
    record = store.read_json(event["event_id"], store_mod.PREDICTION)
    assert preds[0]["predicted_percentile"] == 0.5 and record["branch"] == "fallback" and "model unavailable" in record["error"]
    assert store.read_text(event["event_id"], store_mod.COMPETITION_PREVIEW) == PREVIEW


def test_predict_writes_nothing_locally_unless_asked(monkeypatch, tmp_path, fake_llms):
    monkeypatch.delenv("SHADOW_STORE_DIR", raising=False)
    monkeypatch.delenv("SHADOW_VOLUME", raising=False)
    assert store_mod.configured_store() is None


# --------------------------------------------------------------------- store, schedule, runner, sync

def test_store_only_serves_previews_finished_before_the_cutoff(tmp_path):
    store = store_mod.EventStore(tmp_path)
    store.save_own_preview("ea_A", {"preview": "A", "usable": True, "files": ["x"]})
    store.save_own_preview("ea_B", {"preview": "B", "usable": False})
    store.save_own_preview("ea_C", {"usable": False, "subtype": "stopped_at_cutoff"})
    assert store.own_preview("ea_A") == "A" and store.own_preview("ea_B") is None and store.own_preview("ea_C") is None
    assert "preview" not in store.read_json("ea_A", store_mod.OWN_PREVIEW_RUN) and store.has_own_preview_run("ea_C")


def test_due_events_window_and_skips(tmp_path):
    store = store_mod.EventStore(tmp_path)
    events = [
        make_event("ea_TOO_EARLY", cutoff=NOW + schedule.LEAD + timedelta(minutes=1)),
        make_event("ea_START", cutoff=NOW + schedule.LEAD),
        make_event("ea_LATE_CATCHUP", cutoff=NOW + timedelta(minutes=10)),
        make_event("ea_PASSED", cutoff=NOW),
        make_event("ea_DONE", cutoff=NOW + timedelta(hours=1)),
    ]
    store.save_own_preview("ea_DONE", {"preview": "x", "usable": True})
    assert [e["event_id"] for e in schedule.due_events(events, NOW, store=store)] == ["ea_LATE_CATCHUP", "ea_START"]
    start, end = schedule.calendar_window(NOW)
    assert end - NOW == timedelta(days=7) and NOW - start == timedelta(hours=1)


def test_previews_start_12_hours_ahead_with_at_most_5_at_once():
    assert schedule.LEAD == timedelta(hours=12) and schedule.MAX_PARALLEL_PREVIEWS == 5
    # Modal does not expose the cap on the function object, so check the decorator.
    assert "max_containers=MAX_PARALLEL_PREVIEWS" in inspect.getsource(modal_app)
    assert modal_app.MAX_PARALLEL_PREVIEWS is schedule.MAX_PARALLEL_PREVIEWS


def test_claims_are_exclusive_and_released_on_failure():
    claims = {}
    assert schedule.claim(claims, "ea_A") and not schedule.claim(claims, "ea_A")
    schedule.release(claims, "ea_A", done=False)
    assert schedule.claim(claims, "ea_A")
    schedule.release(claims, "ea_A", done=True)
    assert claims == {"ea_A": "done"} and not schedule.claim(claims, "ea_A")


def test_runner_saves_the_run_and_records_a_passed_cutoff(tmp_path, monkeypatch):
    from preview_agent import generator

    async def fake_generate(event, *, settings, out_dir, verbose):
        assert settings == "cheap"
        return {"event_id": event["event_id"], "preview": "OURS", "usable": True, "total_cost_usd": 0.04}

    monkeypatch.setattr(generator, "generate_preview", fake_generate)
    store = store_mod.EventStore(tmp_path)
    runner.run_preview(make_event("ea_A"), store)
    assert store.own_preview("ea_A") == "OURS"

    async def too_late(event, **kwargs):
        raise generator.CutoffPassedError("passed")

    monkeypatch.setattr(generator, "generate_preview", too_late)
    runner.run_preview(make_event("ea_B"), store)
    assert store.read_json("ea_B", store_mod.OWN_PREVIEW_RUN)["subtype"] == "cutoff_passed" and store.own_preview("ea_B") is None


def test_sync_loads_the_records_into_tables(tmp_path):
    store = store_mod.EventStore(tmp_path)
    store.save_own_preview("ea_A", {"event_id": "ea_A", "preview": "x", "usable": True, "total_cost_usd": 0.05})
    store.write_json("ea_A", store_mod.FEATURES, {"preview_source": "own", "features": {"event_id": "ea_A", "net_w": 3}})
    store.write_json("ea_A", store_mod.PREDICTION, {"event_id": "ea_A", "branch": "surprise_ridge",
                                                    "predictions": [{"identifier_value": "ACME", "predicted_percentile": 0.7}]})
    assert sync.load_predictions(tmp_path).loc[0, "predicted_percentile"] == 0.7
    assert sync.load_features(tmp_path).loc[0, "preview_source"] == "own"
    assert sync.load_preview_runs(tmp_path).loc[0, "total_cost_usd"] == 0.05


# --------------------------------------------------------------------- Modal functions (unwrapped)

def test_schedule_previews_spawns_each_due_event_once(tmp_path, monkeypatch):
    from preview_agent import events as events_mod

    due = make_event("ea_DUE", cutoff=datetime.now(timezone.utc) + timedelta(minutes=30))
    later = make_event("ea_LATER", cutoff=datetime.now(timezone.utc) + schedule.LEAD + timedelta(hours=1))
    monkeypatch.setattr(events_mod, "fetch_events", lambda start, end: [due, later])
    monkeypatch.setattr(store_mod, "from_env", lambda: store_mod.EventStore(tmp_path))
    spawned = []
    monkeypatch.setattr(modal_app, "generate_own_preview", SimpleNamespace(spawn=lambda e: spawned.append(e["event_id"])))
    monkeypatch.setattr(modal_app, "preview_claims", {})

    modal_app.schedule_previews.get_raw_f()()
    modal_app.schedule_previews.get_raw_f()()  # a second check must not start it again
    assert spawned == ["ea_DUE"]


def test_generate_own_preview_releases_the_claim_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "from_env", lambda: store_mod.EventStore(tmp_path))
    claims = {"ea_A": "in_flight"}
    monkeypatch.setattr(modal_app, "preview_claims", claims)

    def boom(event, store, **kwargs):
        raise RuntimeError("sdk crashed")

    monkeypatch.setattr(runner, "run_preview", boom)
    modal_app.generate_own_preview.get_raw_f()(make_event("ea_A"))
    assert "ea_A" not in claims  # freed, so the next check retries

    monkeypatch.setattr(runner, "run_preview", lambda event, store, **kw: {"subtype": "success", "usable": True})
    claims["ea_A"] = "in_flight"
    modal_app.generate_own_preview.get_raw_f()(make_event("ea_A"))
    assert claims["ea_A"] == "done"
