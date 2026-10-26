"""Company names for calendar tickers, looked up on Yahoo Finance (yfinance) and cached.

The calendar gives tickers only, while the competition's preview request names
the company. Names are cached in memory and in `data/surprise/company_names.json`,
so each ticker is looked up once across runs. A failed lookup returns None and is
not cached, so it is retried next time; callers then fall back to the ticker.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "surprise" / "company_names.json"

_memory: dict[str, str] = {}


def yahoo_symbol(ticker: str) -> str:
    """Yahoo writes class shares with a dash: BRK.B -> BRK-B."""
    return ticker.strip().upper().replace(".", "-")


def _fetch_name(symbol: str) -> str | None:
    import yfinance as yf  # imported here: slow to import, and only needed on a cache miss

    info = yf.Ticker(symbol).info or {}
    # longName is the full legal name; shortName is sometimes truncated, so it is only a fallback.
    return info.get("longName") or info.get("shortName")


def _load(path: Path) -> dict[str, str]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def company_name(ticker: str, *, cache_path: Path = CACHE_PATH) -> str | None:
    """The company's name for a calendar ticker, or None if it cannot be found."""
    if ticker in _memory:
        return _memory[ticker]
    cache = _load(cache_path)
    if ticker in cache:
        _memory[ticker] = cache[ticker]
        return cache[ticker]
    try:
        name = _fetch_name(yahoo_symbol(ticker))
    except Exception:  # network errors, rate limits, format changes: fall back to the ticker
        return None
    if not name:
        return None
    _memory[ticker] = name
    cache[ticker] = name
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(dict(sorted(cache.items())), indent=1))
    return name
