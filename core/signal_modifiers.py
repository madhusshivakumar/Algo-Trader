"""Signal modifiers — blend sentiment, LLM conviction, and RL selection into trading signals.

Each modifier reads from a JSON data file written by its corresponding agent.
If the data file is missing or the feature is disabled, the signal passes through unchanged.
"""

import json
import os

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_SENTIMENT_FILE = os.path.join(_DATA_DIR, "sentiment", "scores.json")
_LLM_FILE = os.path.join(_DATA_DIR, "llm_analyst", "convictions.json")


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


def apply_sentiment(signal: dict, symbol: str, weight: float = 0.15) -> dict:
    """Adjust signal strength based on FinBERT sentiment score.

    Reads from data/sentiment/scores.json written by the sentiment agent.
    score range: [-1, 1] where positive = bullish, negative = bearish.

    - Buy signals get boosted by positive sentiment, dampened by negative
    - Sell signals get boosted by negative sentiment, dampened by positive
    - If sentiment strongly opposes the signal direction, strength is reduced by 30%
    """
    data = _load_json(_SENTIMENT_FILE)
    sym_data = data.get("scores", {}).get(symbol)
    if sym_data is None:
        return signal

    sent_score = sym_data.get("sentiment_score", 0)
    if not isinstance(sent_score, (int, float)):
        return signal

    modifier = sent_score * weight

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
    return signal


def apply_llm_conviction(signal: dict, symbol: str, weight: float = 0.2) -> dict:
    """Adjust signal strength based on LLM analyst conviction score.

    Reads from data/llm_analyst/convictions.json written by the LLM analyst agent.
    conviction range: [-1, 1] where positive = bullish, negative = bearish.

    - Buy signals get boosted by bullish conviction, dampened by bearish
    - Sell signals get boosted by bearish conviction, dampened by bullish
    - If conviction strongly opposes the signal direction, strength is reduced by 30%
    """
    data = _load_json(_LLM_FILE)
    sym_data = data.get("convictions", {}).get(symbol)
    if sym_data is None:
        return signal

    conviction = sym_data.get("score", 0)
    if not isinstance(conviction, (int, float)):
        return signal

    modifier = conviction * weight

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
    return signal
