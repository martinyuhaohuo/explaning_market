"""The no-preview extraction variant and the helpers around it, fully offline."""

from __future__ import annotations

import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ml import model
from suprise_dim import data, dataset, extraction, grouping, prompts
from suprise_dim.schemas import Dimension, SurpriseExtraction, SurpriseExtractionNoPreview, Taxonomy

# Fingerprint of the preview prompt used for every existing extraction; it must never change.
PREVIEW_PROMPT_SHA256 = "dcd4cab0d79dde9eeb5594ef29a3c96510dea42c31cc9a664d987631d260910e"


def make_event(event_id="ea_T_Q3_2026", preview="Consensus EPS $1.00.", facts=("EPS was $1.20.", "Guidance raised."),
               when="2026-08-05T20:30:00+00:00", assets=1, ret=0.01):
    items = []
    if facts is not None:
        items.append({"id": "earnings-call-facts", "content": list(facts)})
    if preview is not None:
        items.append({"id": "earnings-preview", "content": preview})
    tickers = [f"T{i}" for i in range(assets)]
    return {
        "event_id": event_id,
        "event_datetime": when,
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": t} for t in tickers],
        "event_returns": {t: {"return_status": "ok", "car1": ret} for t in tickers},
        "disclosure": {"items": items},
    }


def surprise(source="call", **overrides):
    return {"description": "d", "direction": "positive", "magnitude": "small", "fact_evidence": "f",
            "expectation_evidence": "e", "expectation_source": source, **overrides}


class FakeExtractor:
    def __init__(self, result):
        self.result = result
        self.messages = []

    def invoke(self, messages, config=None):
        self.messages.append(messages)
        return self.result


def test_preview_prompt_is_unchanged():
    assert hashlib.sha256(prompts.EXTRACTION_SYSTEM.encode()).hexdigest() == PREVIEW_PROMPT_SHA256
    msg = prompts.build_user_message(make_event())
    assert msg == (
        "Ticker: T0\nRelease time: 2026-08-05T20:30:00+00:00\n\n"
        "Facts from the earnings call:\n1. EPS was $1.20.\n2. Guidance raised.\n\n"
        "Research preview written BEFORE the release (expectations, not results):\nConsensus EPS $1.00."
    )


def test_no_preview_message_leaves_out_any_preview():
    msg = prompts.build_user_message_no_preview(make_event())
    assert "1. EPS was $1.20." in msg and "2. Guidance raised." in msg
    assert "Consensus EPS" not in msg and "not available" in msg


def test_extract_event_uses_the_matching_prompt():
    fake = FakeExtractor(SurpriseExtractionNoPreview(surprises=[surprise()], in_line=[]))
    extraction.extract_event(make_event(), fake, preview=False)
    system, user = fake.messages[-1]
    assert system.content == prompts.EXTRACTION_SYSTEM_NO_PREVIEW and "Consensus EPS" not in user.content

    fake = FakeExtractor(SurpriseExtraction(surprises=[surprise("preview")], in_line=[]))
    extraction.extract_event(make_event(), fake, preview=True)
    system, user = fake.messages[-1]
    assert system.content == prompts.EXTRACTION_SYSTEM and "Consensus EPS $1.00." in user.content


def test_no_preview_schema_rejects_a_preview_source_and_keeps_the_same_fields():
    with pytest.raises(ValidationError):
        SurpriseExtractionNoPreview(surprises=[surprise("preview")], in_line=[])
    ok = SurpriseExtractionNoPreview(surprises=[surprise("none")], in_line=["x"])
    assert set(ok.model_dump()["surprises"][0]) == set(SurpriseExtraction(surprises=[surprise("preview")], in_line=[]).model_dump()["surprises"][0])


def test_run_extraction_records_the_prompt_and_resumes(tmp_path, monkeypatch):
    fake = FakeExtractor(SurpriseExtractionNoPreview(surprises=[surprise(), surprise("none")], in_line=[]))
    seen = []
    monkeypatch.setattr(extraction, "make_extractor", lambda model, effort, preview=True: seen.append(preview) or fake)
    out = tmp_path / "np_extractions.jsonl"
    events = [make_event("ea_A"), make_event("ea_B")]

    first = extraction.run_extraction(events, out_path=out, preview=False, workers=1)
    assert first["written"] == 2 and seen == [False]
    records = extraction.load_extractions(out)
    assert {r["preview"] for r in records} == {False} and len(records[0]["surprises"]) == 2
    assert extraction.run_extraction(events, out_path=out, preview=False)["skipped"] == 2 and len(fake.messages) == 2


def test_load_no_preview_events_keeps_only_valid_events_without_a_preview(tmp_path):
    events = [
        make_event("ea_with_preview"),
        make_event("ea_no_preview", preview=None),
        make_event("ea_blank_preview", preview="   "),
        make_event("ea_no_facts", preview=None, facts=None),
        make_event("ea_two_assets", preview=None, assets=2),
        make_event("ea_no_return", preview=None, ret=None) | {"event_returns": {"T0": {"return_status": "missing"}}},
    ]
    with gzip.open(tmp_path / "EARNINGS_RELEASE_2026Q3.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(e) + "\n" for e in events)
    kept = [e["event_id"] for e in data.load_no_preview_events("2026Q3", tmp_path)]
    assert kept == ["ea_no_preview", "ea_blank_preview"]


def test_no_preview_split_is_chronological():
    events = [make_event(f"ea_{i}", preview=None, when=f"2026-07-{i + 1:02d}T12:00:00+00:00") for i in range(10)]
    split = data.make_no_preview_split(events, 0.8)
    assert split["train"] == [f"ea_{i}" for i in range(8)] and split["test"] == ["ea_8", "ea_9"]


def test_out_of_fold_predict_on_uses_the_other_features():
    rng = np.random.default_rng(0)
    n = 120
    train = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(n)],
        "event_datetime": pd.date_range("2026-08-01", periods=n, freq="h").astype(str),
        "net_w": rng.normal(size=n),
        "earnings_surprise_pct": rng.uniform(size=n),
    })
    train["target_pct"] = (train["net_w"].rank() / n + rng.uniform(0, 0.5, n)).clip(0, 1)
    same, _ = model.out_of_fold(train, ["net_w"], 1.0, log_counts=False, predict_on=train.sample(frac=1, random_state=1))
    plain, _ = model.out_of_fold(train, ["net_w"], 1.0, log_counts=False)
    assert np.allclose(same["oof"], plain["oof"], equal_nan=True)
    flipped, _ = model.out_of_fold(train, ["net_w"], 1.0, log_counts=False, predict_on=train.assign(net_w=-train["net_w"]))
    scored = plain["oof"].notna()
    assert np.corrcoef(plain["oof"][scored], flipped["oof"][scored])[0, 1] < 0


def test_build_extraction_frame_labels_events_from_any_run(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "OUT_DIR", tmp_path)
    monkeypatch.setattr(grouping, "TAXONOMY_DIR", tmp_path / "taxonomy")
    monkeypatch.setattr(dataset, "quarter_earnings_surprise", lambda q: {"ea_A": 0.01})
    monkeypatch.setattr(dataset, "quarter_earnings_surprise_pct", lambda q: {"ea_A": 0.7})
    grouping.save_taxonomy(Taxonomy(dimensions=[Dimension(key="revenue", name="Revenue", definition="d", includes=["i"], excludes="e")], changes=[]), "v2")
    records = [{"event_id": eid, "ticker": "T", "event_datetime": "2026-07-01", "car1": 0.0, "model": "m",
                "surprises": [surprise()], "in_line": []} for eid in ("ea_A", "ea_B", "ea_C")]
    (tmp_path / "np_extractions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (tmp_path / "taxonomy" / "assignments_v2_np.jsonl").write_text(json.dumps({"item_id": "ea_A#0", "dimension": "revenue"}) + "\n")

    frame = dataset.build_extraction_frame("np", "v2_np", {"ea_A": "train", "ea_B": "test"})
    assert frame.set_index("event_id")["split"].to_dict() == {"ea_A": "train", "ea_B": "test"}  # ea_C not requested
    assert frame.set_index("event_id").loc["ea_A", "dimension_name"] == "Revenue"
    assert pd.isna(frame.set_index("event_id").loc["ea_B", "dimension"])  # not assigned yet
    assert frame.set_index("event_id").loc["ea_A", "earnings_surprise_pct"] == 0.7
