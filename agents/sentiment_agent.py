"""Sentiment Agent — pre-market agent that scores news sentiment per symbol.

Runs daily at 5:45 AM ET (before market open).
Fetches headlines via Alpaca News API, scores with FinBERT,
writes results to data/sentiment/scores.json for signal_modifiers to read.
"""

import json
import os
import sys
from datetime import datetime

# Ensure project root is on sys.path when run as a script (python agents/sentiment_agent.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.news_client import fetch_headlines
from core.sentiment_analyzer import score_headlines
from utils.logger import log

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sentiment")
_SCORES_FILE = os.path.join(_DATA_DIR, "scores.json")


def _ensure_data_dir():
    os.makedirs(_DATA_DIR, exist_ok=True)


def analyze_symbol(symbol: str, lookback_hours: int = 24) -> dict | None:
    """Fetch and score news for a single symbol.

    Returns sentiment dict or None if no headlines found.
    """
    headlines = fetch_headlines(symbol, lookback_hours=lookback_hours)
    if not headlines:
        log.info(f"No news headlines for {symbol}, skipping sentiment")
        return None

    result = score_headlines(headlines)
    log.info(
        f"{symbol}: sentiment={result['sentiment_score']:+.3f} "
        f"({result['num_articles']} articles, "
        f"{result['positive_pct']:.0%} pos, {result['negative_pct']:.0%} neg)"
    )
    return result


def run_analysis(symbols: list[str] | None = None,
                 lookback_hours: int = 24) -> dict:
    """Run sentiment analysis for all configured symbols.

    Returns the full scores dict that gets written to disk.
    """
    if symbols is None:
        symbols = Config.SYMBOLS

    scores = {}
    for symbol in symbols:
        result = analyze_symbol(symbol, lookback_hours)
        if result is not None:
            scores[symbol] = result

    return {
        "timestamp": datetime.now().isoformat(),
        "lookback_hours": lookback_hours,
        "scores": scores,
    }


def write_scores(data: dict):
    """Write scores to the JSON file for signal_modifiers to read."""
    _ensure_data_dir()
    with open(_SCORES_FILE, "w") as f:
        json.dump(data, f, indent=2)
    log.success(f"Wrote sentiment scores for {len(data.get('scores', {}))} symbols")


def main():
    """Main entry point — run full sentiment analysis pipeline."""
    log.info("=" * 50)
    log.info("Sentiment Agent starting")
    log.info("=" * 50)

    if not Config.SENTIMENT_ENABLED:
        log.warning("SENTIMENT_ENABLED=false, running anyway (agent invoked directly)")

    data = run_analysis()
    write_scores(data)

    scored = len(data.get("scores", {}))
    total = len(Config.SYMBOLS)
    log.success(f"Sentiment analysis complete: {scored}/{total} symbols scored")
    return data


if __name__ == "__main__":
    main()
