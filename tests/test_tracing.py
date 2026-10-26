"""Tracing turns itself off when Langfuse is not configured or its host is down."""

from __future__ import annotations

import importlib
import socket

from langchain_core.callbacks import BaseCallbackHandler

from em_agent import tracing


def _reload(monkeypatch, host):
    monkeypatch.setenv("LANGFUSE_HOST", host)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    return importlib.reload(tracing)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_an_unreachable_host_turns_tracing_off(monkeypatch, capsys):
    module = _reload(monkeypatch, f"http://127.0.0.1:{_free_port()}")
    assert module.langfuse is None and type(module.langfuse_handler) is BaseCallbackHandler
    assert "tracing is off" in capsys.readouterr().out


def test_a_listening_host_counts_as_reachable():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        assert tracing.host_reachable(f"http://127.0.0.1:{server.getsockname()[1]}")
    assert not tracing.host_reachable("")


def test_missing_settings_turn_tracing_off(monkeypatch):
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    module = importlib.reload(tracing)
    assert module.langfuse is None and type(module.langfuse_handler) is BaseCallbackHandler
