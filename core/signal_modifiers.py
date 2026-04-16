"""Signal modifiers — blend sentiment, LLM conviction, and RL selection into trading signals.

Each modifier reads from a JSON data file written by its corresponding agent.
If the data file is missing or the feature is disabled, the signal passes through unchanged.

Sprint 6D: every modifier emits an A/B record via `analytics.modifier_ab.log_delta`
so the weekly performance job can measure whether the modifier is actually
pulling its weight. Logging is best-effort and wrapped in a try/except so a
bad disk or permissions error never breaks the signal path.
"""

import json
import os
from datetime import datetime

from config import Config


def _ab_log(symbol: str, modifier_name: str, before: dict, after: dict) -> None:
    """Best-effort call into the A/B logger. Swallows any error.

    Kept as a private helper so modifier functions stay readable and the
    modifier_ab import is lazy (avoids a circular import during module load
    since analytics imports Config, which imports parts of core).
    """
    try:
        from analytics.modifier_ab import log_delta
        log_delta(symbol, modifier_name, before, after)
    except Exception:
        # A/B instrumentation must never influence live trading.
        pass

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_SENTIMENT_FILE = os.path.join(_DATA_DIR, "sentiment", "scores.json")
_LLM_FILE = os.path.join(_DATA_DIR, "llm_analyst", "convictions.json")
_EARNINGS_FILE = os.path.join(_DATA_DIR, "earnings_calendar", "output.json")


def _clamp(value: float, lo: float = 0.3, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _load_json(path: str) -> dict:
    """Load a JSON file, returning empty dict on any error."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def validate_data_freshness(data: dict, max_age_hours: float = 24.0) -> bool:
    """Check if data has a recent timestamp. Returns True if fresh, False if stale/missing."""
    timestamp_str = data.get("timestamp") or data.get("updated_at")
    if not timestamp_str:
        return False
    try:
        ts = datetime.fromisoformat(str(timestamp_str).replace("Z", "+00:00"))
        # Make naive for comparison if needed
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        age_hours = (datetime.now() - ts).total_seconds() / 3600
        return age_hours <= max_age_hours
    except (ValueError, TypeError):
        return False


def apply_sentiment(signal: dict, symbol: str, weight: float = 0.15) -> dict:
    """Adjust signal strength based on FinBERT sentiment score.

    Reads from data/sentiment/scores.json written by the sentiment agent.
    score range: [-1, 1] where positive = bullish, negative = bearish.

    - Buy signals get boosted by positive sentiment, dampened by negative
    - Sell signals get boosted by negative sentiment, dampened by positive
    - If sentiment strongly opposes the signal direction, strength is reduced by 30%
    """
    # Sprint 6D: snapshot before-state so the A/B logger can record the
    # delta. Shallow copy is sufficient (we only read action + strength).
    before = {"action": signal.get("action", "hold"),
              "strength": signal.get("strength", 0.0)}

    data = _load_json(_SENTIMENT_FILE)

    # v3: Skip stale data when freshness check is enabled
    if Config.SENTIMENT_FRESHNESS_CHECK and not validate_data_freshness(data, Config.SENTIMENT_MAX_AGE_HOURS):
        _ab_log(symbol, "sentiment", before, signal)
        return signal

    sym_data = data.get("scores", {}).get(symbol)
    if sym_data is None:
        _ab_log(symbol, "sentiment", before, signal)
        return signal

    sent_score = sym_data.get("sentiment_score", 0)
    if not isinstance(sent_score, (int, float)):
        _ab_log(symbol, "sentiment", before, signal)
        return signal

    modifier = sent_score * weight

    if signal["action"] == "hold":
        signal["sentiment_score"] = round(sent_score, 3)
        _ab_log(symbol, "sentiment", before, signal)
        return signal

    if signal["action"] == "buy":
        signal["strength"] = signal.get("strength", 0.5) + modifier
        if sent_score < -0.6:
            signal["strength"] *= 0.7
    elif signal["action"] == "sell":
        signal["strength"] = signal.get("strength", 0.5) - modifier
        if sent_score > 0.6:
            signal["strength"] *= 0.7

    signal["strength"] = _clamp(signal["strength"])
    signal["sentiment_score"] = round(sent_score, 3)
    signal["reason"] = signal.get("reason", "") + f" | sentiment={sent_score:+.2f}"
    _ab_log(symbol, "sentiment", before, signal)
    return signal


def apply_llm_conviction(signal: dict, symbol: str, weight: float = 0.2) -> dict:
    """Adjust signal strength based on LLM analyst conviction score.

    Reads from data/llm_analyst/convictions.json written by the LLM analyst agent.
    conviction range: [-1, 1] where positive = bullish, negative = bearish.

    - Buy signals get boosted by bullish conviction, dampened by bearish
    - Sell signals get boosted by bearish conviction, dampened by bullish
    - If conviction strongly opposes the signal direction, strength is reduced by 30%
    """
    before = {"action": signal.get("action", "hold"),
              "strength": signal.get("strength", 0.0)}

    data = _load_json(_LLM_FILE)

    # v3: Skip stale data when freshness check is enabled
    if Config.LLM_FRESHNESS_CHECK and not validate_data_freshness(data, Config.LLM_MAX_AGE_HOURS):
        _ab_log(symbol, "llm", before, signal)
        return signal

    sym_data = data.get("convictions", {}).get(symbol)
    if sym_data is None:
        _ab_log(symbol, "llm", before, signal)
        return signal

    conviction = sym_data.get("score", 0)
    if not isinstance(conviction, (int, float)):
        _ab_log(symbol, "llm", before, signal)
        return signal

    modifier = conviction * weight

    if signal["action"] == "hold":
        signal["llm_conviction"] = round(conviction, 3)
        _ab_log(symbol, "llm", before, signal)
        return signal

    if signal["action"] == "buy":
        signal["strength"] = signal.get("strength", 0.5) + modifier
        if conviction < -0.6:
            signal["strength"] *= 0.7
    elif signal["action"] == "sell":
        signal["strength"] = signal.get("strength", 0.5) - modifier
        if conviction > 0.6:
            signal["strength"] *= 0.7

    signal["strength"] = _clamp(signal["strength"])
    signal["llm_conviction"] = round(conviction, 3)
    signal["reason"] = signal.get("reason", "") + f" | llm={conviction:+.2f}"
    _ab_log(symbol, "llm", before, signal)
    return signal


_earnings_cache: dict = {}
_earnings_cache_mtime: float = 0.0


def _get_earnings_data() -> dict:
    """Load earnings data with file-mtime caching (avoids re-reading unchanged files)."""
    global _earnings_cache, _earnings_cache_mtime
    try:
        mtime = os.path.getmtime(_EARNINGS_FILE)
    except OSError:
        return {}
    if mtime != _earnings_cache_mtime:
        _earnings_cache = _load_json(_EARNINGS_FILE)
        _earnings_cache_mtime = mtime
    return _earnings_cache


def is_in_earnings_blackout(symbol: str) -> bool:
    """Quick check used by engine for pre-earnings position closure."""
    data = _get_earnings_data()
    sym_data = data.get("earnings", {}).get(symbol)
    if not sym_data:
        return False
    return sym_data.get("in_blackout", False)


def apply_earnings_blackout(signal: dict, symbol: str) -> dict:
    """Reduce or block buy signals during earnings blackout window.

    Reads from data/earnings_calendar/output.json written by the earnings agent.
    Crypto symbols are never in blackout (no earnings).

    - Buy signals: strength multiplied by EARNINGS_SIZE_REDUCTION (0.0 = block entirely)
    - Sell signals: pass through unchanged
    - Hold signals: annotated with earnings_blackout flag
    """
    before = {"action": signal.get("action", "hold"),
              "strength": signal.get("strength", 0.0)}

    if Config.is_crypto(symbol):
        # No log here — crypto path is a structural no-op, not a decision.
        return signal

    data = _get_earnings_data()
    sym_data = data.get("earnings", {}).get(symbol)
    if not sym_data or not sym_data.get("in_blackout", False):
        _ab_log(symbol, "earnings_blackout", before, signal)
        return signal

    days_until = sym_data.get("days_until")
    signal["earnings_blackout"] = True
    signal["days_to_earnings"] = days_until

    if signal["action"] == "buy":
        reduction = Config.EARNINGS_SIZE_REDUCTION
        if reduction <= 0:
            signal["action"] = "hold"
            signal["strength"] = 0.0
            signal["reason"] = signal.get("reason", "") + f" | earnings_blackout({days_until}d) -> blocked"
        else:
            signal["strength"] = _clamp(signal.get("strength", 0.5) * reduction)
            signal["reason"] = signal.get("reason", "") + f" | earnings_blackout({days_until}d) size*{reduction}"
    elif signal["action"] == "hold":
        signal["reason"] = signal.get("reason", "") + f" | earnings_blackout({days_until}d)"

    # Sell signals pass through unchanged — selling before earnings is fine
    _ab_log(symbol, "earnings_blackout", before, signal)
    return signal
