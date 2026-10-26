"""em_agent never imports suprise_dim, and its frozen copies still match the research code it was trained with.

The parity tests compare against `suprise_dim` and `ml`; those using local
research data (data/surprise/) are skipped when it is not present.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from em_agent import features as agent_features, nodes, prompts as agent_prompts
from ml import preprocess
from suprise_dim import data, dataset, features, grouping, prompts as research_prompts

EVENT = {
    "event_id": "ea_ACME_Q3_2026", "event_datetime": "2026-10-26T13:30:00Z",
    "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "ACME"}],
    "disclosure": {"items": [
        {"id": "earnings-call-facts", "kind": "facts", "content": ["Revenue was $11.0M.", "Guidance was raised."]},
        {"id": "earnings-preview", "kind": "text", "content": "# Acme preview\nConsensus revenue $10.0M."},
        {"id": "option-implied-stats", "kind": "stats", "content": {
            "implied_earnings_volatility": {"value": 0.08, "status": "ok"},
            "implied_absolute_earnings_move": {"value": None, "status": "unavailable"},
            "skew_25_delta": {"value": -0.02, "status": "ok"}}},
    ]},
}


def test_the_agent_never_imports_suprise_dim():
    code = (
        "import sys, predict, em_agent.graph, em_agent.nodes; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'suprise_dim'); "
        "print(','.join(bad))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == ""


def test_prompts_match_the_research_prompts():
    assert agent_prompts.EXTRACTION_SYSTEM == research_prompts.EXTRACTION_SYSTEM
    assert agent_prompts.ASSIGN_SYSTEM == research_prompts.ASSIGN_SYSTEM
    assert agent_prompts.build_extraction_message(EVENT) == research_prompts.build_user_message(EVENT)
    texts = ["revenue beat", "guidance raised"]
    assert agent_prompts.format_items(texts) == research_prompts.format_items(texts)
    tax = nodes.taxonomy()
    assert agent_prompts.format_taxonomy(tax) == research_prompts.format_taxonomy(tax)


@pytest.mark.skipif(not grouping.taxonomy_path("v2").exists(), reason="local research data not present")
def test_frozen_taxonomy_matches_v2():
    assert nodes.taxonomy().model_dump() == grouping.load_taxonomy("v2").model_dump()


def test_model_artifact_matches_its_description():
    meta = json.loads((nodes.ARTIFACTS / "pooled_ridge_v2.joblib.json").read_text())
    assert nodes.pooled_model().features == meta["features"] and "src_own" in meta["features"]


def test_option_statistics_that_are_not_ok_count_as_missing():
    row = agent_features.live_features(EVENT, [], nodes.taxonomy(), "own")
    assert row["opt_implied_earnings_volatility"] == 0.08 and np.isnan(row["opt_implied_absolute_earnings_move"])
    assert row["no_surprises"] == 1 and row["src_own"] == 1 and row["no_surprises__x__src_own"] == 1
    assert 0.0 <= nodes.pooled_model().predict(pd.DataFrame([row]))[0] <= 1.0


@pytest.fixture(scope="module")
def training_rows():
    if not features.features_path("v2").exists():
        pytest.skip("local research data not present")
    events = {e["event_id"]: e for e in data.load_eligible_events()}
    table = pd.read_csv(features.features_path("v2"))
    items = pd.read_csv(dataset.surprises_path("v2"))
    return events, table, {e: g.dropna(subset=["item_id"]) for e, g in items.groupby("event_id")}


@pytest.mark.parametrize("source", ["competition", "own"])
def test_live_features_reproduce_the_training_rows(training_rows, source):
    events, table, items = training_rows
    expected = preprocess.add_source_features(table, source)
    columns = nodes.pooled_model().features
    for row in expected.to_dict("records"):
        group = items.get(row["event_id"])
        live = agent_features.live_features(events[row["event_id"]], [] if group is None else group.to_dict("records"),
                                            nodes.taxonomy(), source)
        assert np.allclose([live[c] for c in columns], [row[c] for c in columns], equal_nan=True), row["event_id"]


def test_ridge_node_scores_the_feature_row_like_the_pipeline(training_rows):
    _, table, _ = training_rows
    rows = preprocess.add_source_features(table.head(20), "own")
    expected = nodes.pooled_model().predict(rows)
    for row, want in zip(rows.to_dict("records"), expected):
        assert nodes.ridge_predict({"features": row}, None)["prediction"] == pytest.approx(want)
