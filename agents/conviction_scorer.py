"""Conviction Scorer — grades LLM analyst predictions against actual price moves.

Runs daily after market close (5 PM, alongside trade_analyzer).
No LLM calls — purely computes accuracy from price data.

Workflow:
  1. Load yesterday's archived predictions from prediction_history/
  2. Fetch actual price changes via broker
  3. Score direction accuracy, magnitude error
  4. Write scored feedback for next day's prompt injection
  5. Update rolling feedback summary
"""

import json
import os
import sys
from datetime import datetime, timedelta, date
from math import copysign

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.logger import log

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "llm_analyst")
_PREDICTION_DIR = os.path.join(_DATA_DIR, "prediction_history")
_FEEDBACK_DIR = os.path.join(_DATA_DIR, "feedback")
_SUMMARY_FILE = os.path.join(_DATA_DIR, "feedback_summary.json")

# Max expected daily move (in %) — used to normalize conviction scores to price moves
# A conviction of 1.0 roughly maps to a +3% move expectation
MAX_EXPECTED_MOVE_PCT = 3.0

# Rolling window for feedback summary
SUMMARY_WINDOW_DAYS = 30


def _ensure_dirs():
    os.makedirs(_PREDICTION_DIR, exist_ok=True)
    os.makedirs(_FEEDBACK_DIR, exist_ok=True)


def archive_predictions(convictions_data: dict):
    """Archive today's predictions for later scoring.

    Called from llm_analyst_v2.write_outputs() after writing convictions.json.

    Args:
        convictions_data: The full convictions dict (same as written to convictions.json).
    """
    _ensure_dirs()
    today = date.today().isoformat()
    archive_path = os.path.join(_PREDICTION_DIR, f"{today}.json")

    archive = {
        "date": today,
        "timestamp": convictions_data.get("timestamp", datetime.now().isoformat()),
        "convictions": convictions_data.get("convictions", {}),
        "portfolio_risk": convictions_data.get("portfolio_risk", "unknown"),
        "overall_bias": convictions_data.get("overall_bias", "neutral"),
    }

    with open(archive_path, "w") as f:
        json.dump(archive, f, indent=2)
    log.info(f"Archived predictions for {today} ({len(archive['convictions'])} symbols)")


def load_predictions(target_date: str) -> dict | None:
    """Load archived predictions for a specific date.

    Args:
        target_date: ISO format date string (YYYY-MM-DD).

    Returns:
        Predictions dict or None if not found.
    """
    path = os.path.join(_PREDICTION_DIR, f"{target_date}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def fetch_price_changes(symbols: list[str], broker=None) -> dict[str, float]:
    """Fetch actual daily price changes for symbols.

    Args:
        symbols: List of trading symbols.
        broker: Optional broker instance (for testing). Creates one if None.

    Returns:
        Dict mapping symbol to percent change (e.g., {"AAPL": 1.2, "XOM": -0.5}).
    """
    if broker is None:
        from core.broker import Broker
        broker = Broker()

    changes = {}
    for symbol in symbols:
        try:
            df = broker.get_historical_bars(symbol, days=3)
            if df is not None and len(df) >= 2:
                prev_close = float(df["close"].iloc[-2])
                curr_close = float(df["close"].iloc[-1])
                if prev_close > 0:
                    pct_change = ((curr_close - prev_close) / prev_close) * 100
                    changes[symbol] = round(pct_change, 4)
        except Exception as e:
            log.warning(f"Could not fetch price change for {symbol}: {e}")

    return changes


def score_predictions(predictions: dict, price_changes: dict) -> dict:
    """Score predictions against actual price changes.

    Args:
        predictions: Archived predictions dict with "convictions" key.
        price_changes: Dict mapping symbol to actual percent change.

    Returns:
        Scored feedback dict.
    """
    convictions = predictions.get("convictions", {})
    symbols_scored = {}
    sector_stats = {}  # sector -> {correct: int, total: int, pred_sum: float, actual_sum: float}

    for symbol, conv in convictions.items():
        if symbol not in price_changes:
            continue

        predicted_score = conv.get("score", conv.get("raw_score", 0))
        if not isinstance(predicted_score, (int, float)):
            predicted_score = 0

        actual_pct = price_changes[symbol]

        # Normalize predicted score to expected % move
        predicted_pct = predicted_score * MAX_EXPECTED_MOVE_PCT

        # Direction accuracy
        direction_correct = (
            (predicted_score > 0.05 and actual_pct > 0) or
            (predicted_score < -0.05 and actual_pct < 0) or
            (abs(predicted_score) <= 0.05 and abs(actual_pct) < 0.5)  # neutral = small move
        )

        # Magnitude error (how far off was the prediction)
        magnitude_error = abs(predicted_pct - actual_pct)

        # Determine sector
        from agents.llm_analyst_v2 import SYMBOL_SECTOR_MAP
        sector = SYMBOL_SECTOR_MAP.get(symbol, "Unknown")

        symbols_scored[symbol] = {
            "predicted_score": predicted_score,
            "predicted_bias": conv.get("bias", "neutral"),
            "actual_pct_change": actual_pct,
            "direction_correct": direction_correct,
            "magnitude_error": round(magnitude_error, 4),
            "sector": sector,
        }

        # Aggregate sector stats
        if sector not in sector_stats:
            sector_stats[sector] = {"correct": 0, "total": 0, "pred_sum": 0, "actual_sum": 0}
        sector_stats[sector]["total"] += 1
        if direction_correct:
            sector_stats[sector]["correct"] += 1
        sector_stats[sector]["pred_sum"] += predicted_score
        sector_stats[sector]["actual_sum"] += actual_pct

    # Compute sector summaries
    sector_summary = {}
    for sector, stats in sector_stats.items():
        total = stats["total"]
        sector_summary[sector] = {
            "avg_predicted": round(stats["pred_sum"] / total, 4) if total else 0,
            "avg_actual_pct": round(stats["actual_sum"] / total, 4) if total else 0,
            "direction_accuracy": round(stats["correct"] / total, 4) if total else 0,
            "num_symbols": total,
        }

    # Overall accuracy
    total_scored = len(symbols_scored)
    total_correct = sum(1 for s in symbols_scored.values() if s["direction_correct"])
    overall_accuracy = round(total_correct / total_scored, 4) if total_scored else 0

    return {
        "date": predictions.get("date", "unknown"),
        "scored_at": datetime.now().isoformat(),
        "symbols": symbols_scored,
        "sector_summary": sector_summary,
        "overall_direction_accuracy": overall_accuracy,
        "total_scored": total_scored,
        "total_correct": total_correct,
    }


def write_feedback(feedback: dict):
    """Write scored feedback to the feedback directory."""
    _ensure_dirs()
    feedback_date = feedback.get("date", date.today().isoformat())
    path = os.path.join(_FEEDBACK_DIR, f"{feedback_date}.json")
    with open(path, "w") as f:
        json.dump(feedback, f, indent=2)
    log.info(f"Wrote feedback for {feedback_date}: "
             f"{feedback['total_correct']}/{feedback['total_scored']} correct "
             f"({feedback['overall_direction_accuracy']:.0%})")


def update_summary():
    """Update the rolling feedback summary from recent feedback files.

    Aggregates the last N days of feedback into a single summary for quick loading.
    """
    _ensure_dirs()
    cutoff = (date.today() - timedelta(days=SUMMARY_WINDOW_DAYS)).isoformat()

    all_feedback = []
    if os.path.exists(_FEEDBACK_DIR):
        for fname in sorted(os.listdir(_FEEDBACK_DIR)):
            if not fname.endswith(".json"):
                continue
            fdate = fname.replace(".json", "")
            if fdate < cutoff:
                continue
            try:
                with open(os.path.join(_FEEDBACK_DIR, fname)) as f:
                    all_feedback.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue

    if not all_feedback:
        summary = {
            "updated": datetime.now().isoformat(),
            "window_days": SUMMARY_WINDOW_DAYS,
            "total_days": 0,
            "overall_accuracy": 0,
            "sector_accuracy": {},
            "symbol_accuracy": {},
            "recent_days": [],
        }
    else:
        # Aggregate
        total_correct = sum(f.get("total_correct", 0) for f in all_feedback)
        total_scored = sum(f.get("total_scored", 0) for f in all_feedback)

        # Per-sector accuracy
        sector_agg = {}
        for f in all_feedback:
            for sector, stats in f.get("sector_summary", {}).items():
                if sector not in sector_agg:
                    sector_agg[sector] = {"correct": 0, "total": 0}
                sector_agg[sector]["correct"] += int(stats.get("direction_accuracy", 0) * stats.get("num_symbols", 0))
                sector_agg[sector]["total"] += stats.get("num_symbols", 0)

        sector_accuracy = {
            s: round(v["correct"] / v["total"], 4) if v["total"] else 0
            for s, v in sector_agg.items()
        }

        # Per-symbol accuracy (last 7 days only for prompt injection)
        symbol_history = {}  # symbol -> list of recent outcomes
        for f in all_feedback[-7:]:
            fdate = f.get("date", "")
            for sym, data in f.get("symbols", {}).items():
                if sym not in symbol_history:
                    symbol_history[sym] = []
                symbol_history[sym].append({
                    "date": fdate,
                    "predicted": data.get("predicted_score", 0),
                    "actual_pct": data.get("actual_pct_change", 0),
                    "correct": data.get("direction_correct", False),
                })

        # Recent day summaries
        recent_days = []
        for f in all_feedback[-5:]:
            recent_days.append({
                "date": f.get("date"),
                "accuracy": f.get("overall_direction_accuracy", 0),
                "scored": f.get("total_scored", 0),
            })

        summary = {
            "updated": datetime.now().isoformat(),
            "window_days": SUMMARY_WINDOW_DAYS,
            "total_days": len(all_feedback),
            "overall_accuracy": round(total_correct / total_scored, 4) if total_scored else 0,
            "sector_accuracy": sector_accuracy,
            "symbol_accuracy": symbol_history,
            "recent_days": recent_days,
        }

    with open(_SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Updated feedback summary: {summary['total_days']} days, "
             f"accuracy={summary['overall_accuracy']:.0%}")


def load_summary() -> dict:
    """Load the feedback summary for prompt injection."""
    if not os.path.exists(_SUMMARY_FILE):
        return {}
    try:
        with open(_SUMMARY_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_sector_feedback(sector: str) -> str:
    """Get formatted feedback text for a sector prompt.

    Returns a concise accuracy summary for the last few days.
    """
    summary = load_summary()
    if not summary or summary.get("total_days", 0) == 0:
        return ""

    sector_acc = summary.get("sector_accuracy", {}).get(sector)
    if sector_acc is None:
        return ""

    overall_acc = summary.get("overall_accuracy", 0)
    recent = summary.get("recent_days", [])

    lines = ["## Your Recent Accuracy (calibrate accordingly)"]
    lines.append(f"- Overall direction accuracy: {overall_acc:.0%} ({summary['total_days']} days)")
    lines.append(f"- {sector} sector accuracy: {sector_acc:.0%}")

    if recent:
        last = recent[-1]
        lines.append(f"- Yesterday: {last['accuracy']:.0%} across {last['scored']} symbols")

    return "\n".join(lines)


def get_symbol_feedback(symbol: str) -> str:
    """Get formatted feedback text for a symbol prompt.

    Returns the last 3 days of prediction vs actual for this symbol.
    """
    summary = load_summary()
    if not summary:
        return ""

    history = summary.get("symbol_accuracy", {}).get(symbol, [])
    if not history:
        return ""

    lines = ["## Your Recent Predictions vs Reality"]
    for entry in history[-3:]:
        pred = entry.get("predicted", 0)
        actual = entry.get("actual_pct", 0)
        correct = "correct" if entry.get("correct") else "wrong direction"
        bias = "bullish" if pred > 0 else "bearish" if pred < 0 else "neutral"
        lines.append(
            f"- {entry['date']}: predicted {pred:+.2f} ({bias}), "
            f"actual {actual:+.2f}%. {correct.capitalize()}."
        )

    # Add accuracy rate
    if len(history) >= 2:
        acc = sum(1 for h in history if h.get("correct")) / len(history)
        lines.append(f"- {len(history)}-day direction accuracy for {symbol}: {acc:.0%}")

    return "\n".join(lines)


def run_scoring(target_date: str | None = None, broker=None) -> dict | None:
    """Run the full scoring pipeline for a given date.

    Args:
        target_date: Date to score (defaults to yesterday).
        broker: Optional broker instance for testing.

    Returns:
        Feedback dict or None if no predictions found.
    """
    if target_date is None:
        target_date = (date.today() - timedelta(days=1)).isoformat()

    log.info(f"Scoring predictions for {target_date}")

    predictions = load_predictions(target_date)
    if predictions is None:
        log.warning(f"No predictions found for {target_date}")
        return None

    symbols = list(predictions.get("convictions", {}).keys())
    if not symbols:
        log.warning(f"No symbols in predictions for {target_date}")
        return None

    price_changes = fetch_price_changes(symbols, broker)
    if not price_changes:
        log.warning(f"Could not fetch price changes for {target_date}")
        return None

    feedback = score_predictions(predictions, price_changes)
    write_feedback(feedback)
    update_summary()

    return feedback


def main():
    """Entry point — score yesterday's predictions."""
    log.info("=" * 50)
    log.info("Conviction Scorer — Grading Predictions")
    log.info("=" * 50)

    if not Config.LLM_FEEDBACK_LOOP_ENABLED:
        log.warning("LLM_FEEDBACK_LOOP_ENABLED=false, running anyway")

    result = run_scoring()
    if result:
        log.success(
            f"Scoring complete: {result['total_correct']}/{result['total_scored']} correct "
            f"({result['overall_direction_accuracy']:.0%})"
        )
    else:
        log.warning("No predictions to score")
    return result


if __name__ == "__main__":
    main()
