"""The separate labelling step and the sentiment features, fully offline (fake labeller)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml import model, preprocess
from suprise_dim import features, labeling
from suprise_dim.prompts import build_label_message
from suprise_dim.schemas import Dimension, SurpriseLabel, SurpriseLabels, Taxonomy


def make_event(event_id="ea_TEST_Q3_2026", preview=True, move=0.06):
    items = [{"id": "earnings-call-facts", "content": ["Revenue was $10M.", "Guidance was raised."]}]
    if preview:
        items.append({"id": "earnings-preview", "content": "Consensus revenue $9M."})
    if move is not None:
        items.append({"id": "option-implied-stats", "content": {
            "implied_earnings_volatility": {"value": 0.08, "status": "ok"},
            "implied_absolute_earnings_move": {"value": move, "status": "ok"},
            "skew_25_delta": {"value": -0.01, "status": "ok"},
        }})
    return {
        "event_id": event_id,
        "event_datetime": "2026-08-05T20:30:00+00:00",
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "TEST"}],
        "disclosure": {"items": items},
    }


def make_items(event_id="ea_TEST_Q3_2026", n=3):
    return pd.DataFrame({
        "item_id": [f"{event_id}#{i}" for i in range(n)],
        "event_id": event_id,
        "position": range(n),
        "dimension": ["revenue", "forward_guidance", "revenue"][:n],
        "dimension_name": ["Revenue", "Guidance", "Revenue"][:n],
        "description": [f"surprise {i}" for i in range(n)],
        "fact_evidence": ["call says so"] * n,
        "expectation_evidence": [""] * n,
        "expectation_source": ["preview"] * n,
        "direction": ["positive", "negative", "mixed"][:n],
        "magnitude": ["large", "small", "moderate"][:n],
    })


class FakeLabeler:
    """Returns `labels` if given, else one label per surprise in the message."""

    def __init__(self, labels=None):
        self.labels = labels
        self.calls = 0
        self.batch_sizes = []

    def invoke(self, messages, config=None):
        self.calls += 1
        n = sum(1 for line in messages[1].content.splitlines() if line[:1].isdigit())
        self.batch_sizes.append(n)
        return SurpriseLabels(labels=self.labels if self.labels is not None else [label(i) for i in range(n)])


def label(i, d=1, m=2, s=0):
    return SurpriseLabel(index=i, reason="r", direction=d, magnitude=m, sentiment=s)


def test_label_message_holds_only_the_surprises():
    msg = build_label_message(make_items())
    assert "0. [Revenue] surprise 0" in msg and "2. [Revenue] surprise 2" in msg
    assert "Expectation source: preview" in msg and "Call evidence: call says so" in msg
    assert "Ticker" not in msg and "preview written" not in msg and "expected move" not in msg


def test_label_message_shows_missing_evidence_as_dash():
    items = make_items().assign(expectation_evidence=[np.nan, "", "  "], fact_evidence=["said so", np.nan, "x"])
    msg = build_label_message(items)
    assert "nan" not in msg
    assert msg.count("Expectation evidence: -") == 3 and "Call evidence: -" in msg


def test_label_batch_keeps_first_valid_label_per_number():
    fake = FakeLabeler([label(0, d=1, m=3), label(0, d=-1), label(1, d=None, m=2, s=None), label(7)])
    out = labeling.label_batch(make_items(), fake)
    assert [o["item_id"] for o in out] == ["ea_TEST_Q3_2026#0", "ea_TEST_Q3_2026#1"]
    assert out[0]["direction"] == 1 and out[0]["magnitude"] == 3 and out[0]["event_id"] == "ea_TEST_Q3_2026"
    assert out[1]["direction"] is None and out[1]["magnitude"] is None  # magnitude forced to None


def test_label_batch_spans_events_and_skips_empty_batches():
    fake = FakeLabeler()
    items = pd.concat([make_items("ea_A", 2), make_items("ea_B", 2)])
    out = labeling.label_batch(items, fake)
    assert [o["event_id"] for o in out] == ["ea_A", "ea_A", "ea_B", "ea_B"]
    assert labeling.label_batch(items.iloc[:0], fake) == [] and fake.calls == 1


def test_run_labeling_batches_resumes_and_retries_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(labeling, "OUT_DIR", tmp_path)
    items = pd.concat([make_items("ea_A", 3), make_items("ea_B", 2)])
    skipping = FakeLabeler([label(0)])  # labels only the first surprise of each batch
    monkeypatch.setattr(labeling, "make_labeler", lambda model, effort: skipping)
    first = labeling.run_labeling(items, "t", batch_size=2, workers=1)
    assert skipping.batch_sizes == [2, 2, 1] and first["written"] == 3 and first["missing"] == 2

    complete = FakeLabeler()
    monkeypatch.setattr(labeling, "make_labeler", lambda model, effort: complete)
    second = labeling.run_labeling(items, "t", batch_size=2, workers=1)
    assert second["skipped"] == 3 and complete.batch_sizes == [2]
    assert labeling.run_labeling(items, "t", batch_size=2, workers=1)["skipped"] == 5

    rows = [json.loads(line) for line in (tmp_path / "labels_t.jsonl").read_text().splitlines()]
    assert len(rows) == 5 and {r["model"] for r in rows} == {labeling.LABEL_MODEL}
    assert len(labeling.load_labels("t")) == 5


def test_complete_events_only_counts_fully_labelled_events():
    surprises = pd.concat([make_items("ea_A", 2), make_items("ea_B", 2)])
    labels = pd.DataFrame({"item_id": ["ea_A#0", "ea_A#1", "ea_B#0"]})
    assert labeling.complete_events(surprises, labels) == {"ea_A"}


def test_apply_labels_replaces_labels_and_keeps_the_extraction_ones():
    surprises = make_items()
    labels = pd.DataFrame([
        {"item_id": "ea_TEST_Q3_2026#0", "direction": -1, "magnitude": 1, "sentiment": 1, "reason": ""},
        {"item_id": "ea_TEST_Q3_2026#1", "direction": None, "magnitude": None, "sentiment": None, "reason": ""},
    ])  # item #2 left unlabelled
    out = labeling.apply_labels(surprises, labels, {"ea_TEST_Q3_2026"})
    assert out["direction"].tolist() == ["negative", "unclear", "unclear"]
    assert out["magnitude"].tolist()[0] == "small" and out["magnitude"].isna().tolist()[1:] == [True, True]
    assert out["direction_extraction"].tolist() == ["positive", "negative", "mixed"]
    assert out["sentiment"].tolist()[0] == 1 and np.isnan(out["sentiment"].tolist()[1])


def test_event_features_add_sentiment_only_when_present(monkeypatch):
    monkeypatch.setattr(features, "quarter_percentile", lambda q: pd.Series({"ea_A": 0.2, "ea_B": 0.8}))
    monkeypatch.setattr(features, "quarter_earnings_surprise_pct", lambda q: {"ea_A": 0.5})
    tax = Taxonomy(dimensions=[
        Dimension(key=k, name=k, definition="d", includes=["i"], excludes="e") for k in ("revenue", "forward_guidance")
    ], changes=[])
    events = {eid: make_event(eid) for eid in ("ea_A", "ea_B")}
    split = {"train": ["ea_A"], "test": ["ea_B"]}
    surprises = make_items("ea_A").assign(sentiment=[1, -1, np.nan])

    with_sent = features.build_event_features(surprises, split, events, tax).set_index("event_id")
    assert with_sent.loc["ea_A", "sent__revenue"] == 1 and with_sent.loc["ea_A", "sent__forward_guidance"] == -1
    assert with_sent.loc["ea_A", "net_sentiment"] == 0 and with_sent.loc["ea_B", "net_sentiment"] == 0
    assert with_sent.loc["ea_B", "no_surprises"] == 1

    without = features.build_event_features(surprises.drop(columns="sentiment"), split, events, tax)
    assert not any(c.startswith("sent__") or c.startswith("net_sentiment") for c in without.columns)
    assert "net_sent" in preprocess.feature_sets(with_sent.columns)
    assert "net_sent" not in preprocess.feature_sets(without.columns)


def test_out_of_fold_leaves_first_block_unscored_and_lines_up():
    rng = np.random.default_rng(0)
    n = 120
    train = pd.DataFrame({
        "event_datetime": pd.date_range("2026-08-01", periods=n, freq="h").astype(str),
        "net_w": rng.normal(size=n),
        "earnings_surprise_pct": rng.uniform(size=n),
    })
    train["target_pct"] = (train["net_w"].rank() / n + rng.uniform(0, 0.5, n)).clip(0, 1)
    out, folds = model.out_of_fold(train.sample(frac=1, random_state=0), ["net_w"], 1.0, log_counts=False)
    assert len(folds) == 5 and out["event_datetime"].is_monotonic_increasing
    assert out["oof"].isna().sum() == pytest.approx(n / 6, abs=1)
    assert out["oof"].between(0, 1).sum() == n - out["oof"].isna().sum()
