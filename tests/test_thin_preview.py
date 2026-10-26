"""Synthetic thin previews and preview-source features, fully offline (fake Anthropic client)."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml import model, preprocess
from suprise_dim import thin_preview as tp

FULL = "# Acme (ACME) Q3 Preview\nConsensus revenue $10.0M (FactSet, Oct 1, 2026). Last price $42.10. Implied move 8%."


def make_event(event_id="ea_ACME_Q3_2026", preview=FULL):
    items = [{"id": "earnings-call-facts", "content": ["Revenue was $11M."]}]
    if preview is not None:
        items.append({"id": "earnings-preview", "content": preview})
    return {"event_id": event_id, "event_datetime": "2026-08-05T20:30:00+00:00",
            "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "ACME"}], "disclosure": {"items": items}}


def response(text="# Acme thin preview", stop_reason="end_turn", category=None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if stop_reason == "refusal" else None,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=900),
    )


class FakeClient:
    def __init__(self, reply=None):
        self.calls = []
        self.reply = reply or response()
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def test_thin_message_holds_the_example_and_the_full_preview():
    msg = tp.build_thin_message(make_event(), example="EXAMPLE TEXT")
    assert "<example>\nEXAMPLE TEXT\n</example>" in msg and FULL in msg and "(ACME)" in msg
    assert "Revenue was $11M." not in msg  # call facts (results) are never shown
    assert tp.EXAMPLE_PATH.exists() and len(tp.EXAMPLE_PATH.read_text()) > 1000


def test_thin_preview_sends_the_cheap_setup_and_returns_text():
    client = FakeClient()
    out = tp.thin_preview(make_event(), client)
    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5" and call["output_config"] == {"effort": "low"}
    assert call["system"] == tp.THIN_SYSTEM and call["messages"][0]["role"] == "user"
    assert out == {"preview": "# Acme thin preview", "stop_reason": "end_turn", "input_tokens": 1200, "output_tokens": 900}


@pytest.mark.parametrize("reply, message", [
    (response(stop_reason="refusal", category="cyber"), "refused (cyber)"),
    (response(stop_reason="max_tokens"), "max_tokens"),
    (response(text="   "), "empty"),
])
def test_thin_preview_rejects_unusable_responses(reply, message):
    with pytest.raises(RuntimeError, match=re.escape(message)):
        tp.thin_preview(make_event(), FakeClient(reply))


def test_run_thin_previews_resumes_and_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "OUT_DIR", tmp_path)
    client = FakeClient()
    monkeypatch.setattr(tp, "make_client", lambda: client)
    events = [make_event("ea_A"), make_event("ea_B")]
    assert tp.run_thin_previews(events, "t", workers=1)["written"] == 2
    assert tp.run_thin_previews(events, "t", workers=1)["skipped"] == 2 and len(client.calls) == 2
    records = [json.loads(line) for line in (tmp_path / "thin_previews_t.jsonl").read_text().splitlines()]
    assert {r["model"] for r in records} == {"claude-haiku-5-5"}
    assert tp.load_thin_previews("t") == {"ea_A": "# Acme thin preview", "ea_B": "# Acme thin preview"}


def test_with_preview_replaces_a_copy():
    event = make_event()
    thin = tp.with_preview(event, "THIN")
    assert [i["content"] for i in thin["disclosure"]["items"] if i["id"] == "earnings-preview"] == ["THIN"]
    assert [i["content"] for i in event["disclosure"]["items"] if i["id"] == "earnings-preview"] == [FULL]
    with pytest.raises(ValueError):
        tp.with_preview(make_event(preview=None), "THIN")


def test_source_features_flag_and_interact():
    frame = pd.DataFrame({"net_w": [3, -1], "net_w_first3": [2, 0], "n_items": [5, 2], "no_surprises": [0, 1]})
    own = preprocess.add_source_features(frame, "own")
    comp = preprocess.add_source_features(frame, "competition")
    assert own["src_own"].tolist() == [1, 1] and comp["src_own"].tolist() == [0, 0]
    assert own["net_w__x__src_own"].tolist() == [3, -1] and comp["net_w__x__src_own"].tolist() == [0, 0]
    assert own["preview_source"].unique().tolist() == ["own"]
    assert preprocess.with_source(["net_w"], own.columns) == ["net_w", "src_own"] + [f"{c}__x__src_own" for c in preprocess.SOURCE_INTERACTIONS]
    with pytest.raises(ValueError):
        preprocess.add_source_features(frame, "unknown")


def test_pooled_rows_of_one_event_share_a_fold():
    rng = np.random.default_rng(0)
    n = 60
    base = pd.DataFrame({"event_id": [f"e{i}" for i in range(n)],
                         "event_datetime": pd.date_range("2026-08-01", periods=n, freq="h").astype(str),
                         "net_w": rng.normal(size=n)})
    pooled = pd.concat([base.assign(src="competition"), base.assign(src="own")]).sort_values("event_datetime").reset_index(drop=True)
    for fit_idx, val_idx in model.time_folds(pooled, 5):
        fit_ids, val_ids = set(pooled.loc[fit_idx, "event_id"]), set(pooled.loc[val_idx, "event_id"])
        assert not fit_ids & val_ids  # no event has one version in training and the other in validation
