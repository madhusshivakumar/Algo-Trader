"""Signal-modifier A/B logging.

Sprint 6D: each signal modifier (sentiment, LLM, MTF, earnings blackout)
blindly adds weight to the base strategy signal. We had no way to tell
whether those modifiers were actually helping — or whether we were paying
API costs (FinBERT, Anthropic) to make worse trading decisions.

This module is the lightweight logging substrate. Every time a modifier
runs, it records a record with:
  - timestamp
  - symbol
  - modifier_name
  - base signal (action + strength) BEFORE the modifier touched it
  - final signal (action + strength) AFTER the modifier touched it

The companion module `analytics/modifier_performance.py` reads this log,
joins to the trades table, and estimates each modifier's Sharpe contribution.
If the 30-day contribution is non-positive, it flips the modifier's feature
flag to false and alerts — closing the loop.

Storage is JSONL (append-only, no read-modify-write) at
`data/modifier_ab/log.jsonl`. Rotated by the DB rotation sweep when the file
exceeds MODIFIER_AB_MAX_MB (~50MB).

Design notes:
  - Logging is best-effort — never raises into the calling signal path.
    A disk-full error must not stop trading.
  - The file is opened/closed per write (each record is flushed before we
    return); this is slow if we're logging millions of records, but at ~12
    signals/min × 4 modifiers × 390 min = ~18K records/day, that's fine.
  - Records include a `pre_strength`/`post_strength` delta AND the action
    pair so the performance module can distinguish three regimes:
      * strength-only change (modifier nudged intensity, same action)
      * action flip (modifier pushed us to hold / inverted direction)
      * no-op (modifier read data but didn't change anything — zero delta)
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta
from typing import Any, Iterator

from config import Config

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                         "modifier_ab")
_LOG_PATH = os.path.join(_DATA_DIR, "log.jsonl")

# Thread lock: engine is single-threaded today but TWAP slicing and position
# reconciliation can cross threads; cheap insurance against torn writes.
_WRITE_LOCK = threading.Lock()


def _ensure_dir() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)


def _serialize_signal(signal: dict) -> dict:
    """Extract only the fields we care about; avoid logging the full signal
    (which can include long `reason` strings that bloat the JSONL)."""
    return {
        "action": signal.get("action", "hold"),
        "strength": round(float(signal.get("strength", 0.0) or 0.0), 4),
    }


def log_delta(
    symbol: str,
    modifier_name: str,
    before: dict,
    after: dict,
    timestamp: datetime | None = None,
    log_path: str | None = None,
) -> bool:
    """Append one modifier-delta record to the JSONL log.

    Args:
        symbol: Symbol the signal applies to (e.g., "AAPL", "BTC/USD").
        modifier_name: Short identifier — "sentiment", "llm", "mtf",
            "earnings_blackout".
        before: Signal dict BEFORE the modifier was applied.
        after: Signal dict AFTER the modifier was applied (same ref, mutated,
            is fine — we snapshot `action` and `strength` only).
        timestamp: Optional override for testing. Defaults to now.
        log_path: Optional override for testing.

    Returns:
        True if the record was written, False if logging was disabled or a
        non-fatal error occurred.
    """
    if not getattr(Config, "MODIFIER_AB_ENABLED", True):
        return False

    path = log_path or _LOG_PATH
    try:
        _ensure_dir() if log_path is None else os.makedirs(os.path.dirname(path),
                                                          exist_ok=True)
        record = {
            "ts": (timestamp or datetime.now()).isoformat(),
            "symbol": symbol,
            "modifier": modifier_name,
            "before": _serialize_signal(before),
            "after": _serialize_signal(after),
            "delta": round(
                _serialize_signal(after)["strength"]
                - _serialize_signal(before)["strength"],
                4,
            ),
            "action_changed": before.get("action") != after.get("action"),
        }
        with _WRITE_LOCK:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        return True
    except Exception:
        # Logging must never break signal computation. Swallow.
        return False


def read_deltas(
    since: datetime | None = None,
    log_path: str | None = None,
) -> Iterator[dict]:
    """Yield modifier-delta records from the log, optionally filtered by time.

    Args:
        since: If provided, only records with `ts >= since` are yielded.
        log_path: Optional override for testing.

    Yields:
        Parsed record dicts. Malformed lines are skipped silently.
    """
    path = log_path or _LOG_PATH
    if not os.path.exists(path):
        return

    cutoff_iso = since.isoformat() if since else None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cutoff_iso and record.get("ts", "") < cutoff_iso:
                continue
            yield record


def read_deltas_for_modifier(
    modifier_name: str,
    days: int = 30,
    log_path: str | None = None,
) -> list[dict]:
    """Return records for one modifier over the last `days`.

    Convenience wrapper around `read_deltas` — the most common query for the
    performance report.
    """
    since = datetime.now() - timedelta(days=days)
    return [r for r in read_deltas(since=since, log_path=log_path)
            if r.get("modifier") == modifier_name]


def summarize_deltas(records: list[dict]) -> dict[str, Any]:
    """Aggregate a list of modifier deltas into summary stats.

    Returns a dict with:
        - count: total records
        - noop_count: records where the modifier made no change
        - action_flip_count: records where the modifier changed the action
        - mean_delta: average strength change (signed)
        - abs_mean_delta: average absolute strength change
        - unique_symbols: count of distinct symbols touched
    """
    if not records:
        return {
            "count": 0, "noop_count": 0, "action_flip_count": 0,
            "mean_delta": 0.0, "abs_mean_delta": 0.0, "unique_symbols": 0,
        }

    def _coerce(val: Any) -> float:
        """Be liberal: the log comes from a best-effort writer and any
        downstream corruption should sum to zero, not crash the report."""
        try:
            return float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    deltas = [_coerce(r.get("delta", 0.0)) for r in records]
    flips = sum(1 for r in records if r.get("action_changed"))
    noops = sum(1 for d in deltas if abs(d) < 1e-9)
    symbols = {r.get("symbol") for r in records}

    n = len(deltas)
    mean_d = sum(deltas) / n
    abs_mean_d = sum(abs(d) for d in deltas) / n

    return {
        "count": n,
        "noop_count": noops,
        "action_flip_count": flips,
        "mean_delta": round(mean_d, 5),
        "abs_mean_delta": round(abs_mean_d, 5),
        "unique_symbols": len(symbols),
    }
