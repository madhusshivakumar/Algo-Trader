"""Engine heartbeat — Sprint 8 (post-incident April 21, 2026).

Writes a liveness + activity snapshot to ``data/heartbeat.json`` on every
engine cycle. The watchdog reads this file during market hours and alerts
if:

    * timestamp is stale (engine is hung or dead in an otherwise-healthy
      container — the Monday Apr 20 failure mode), OR
    * signals_produced stays at zero for too long during market hours
      (engine is alive but not doing work).

Failure mode this fixes: container-level ``docker ps`` checks report
"healthy" for a zombie Python process that has stopped iterating. On
Monday Apr 20 2026, the engine was in this state for ~7 hours of market
time without any alert firing. This module closes that gap.

Design:
    * Writes to an atomic temp file then renames, so partial writes from a
      crashing engine can't corrupt the sentinel.
    * ``write_heartbeat`` is called from the engine's ``run_cycle()``
      tail, just before the next sleep.
    * Callers never block on IO errors — the heartbeat is observability,
      not control flow. If disk is full, log a warning and continue.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

from utils.logger import log


# Single source of truth for the sentinel path. Docker maps /app/data
# to the named db-data volume via docker-compose.
_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "heartbeat.json",
)


def heartbeat_path() -> str:
    """Resolved heartbeat path — uses $HEARTBEAT_PATH env if set.

    The watchdog reads the same env var so host + container agree on
    location even when the repo root differs.
    """
    return os.environ.get("HEARTBEAT_PATH", _DEFAULT_PATH)


def write_heartbeat(
    cycle_count: int,
    positions_evaluated: int = 0,
    signals_produced: int = 0,
    last_trade_ts: str | None = None,
    equity: float | None = None,
    halted: bool = False,
    extra: dict[str, Any] | None = None,
) -> bool:
    """Write a heartbeat record. Best-effort — never raises.

    Args:
        cycle_count: Engine cycle count (monotonic).
        positions_evaluated: How many symbols the current cycle processed.
        signals_produced: How many non-hold signals the cycle generated.
            Stays 0 when cycle is idle — the watchdog alarms on extended
            zeros during market hours.
        last_trade_ts: ISO timestamp of the most recent trade, if any.
        equity: Current account equity ($).
        halted: True if the risk manager has halted trading.
        extra: Free-form fields for future expansion.

    Returns:
        True on successful write, False on any IO error.
    """
    path = heartbeat_path()
    now_utc = datetime.now(timezone.utc)
    payload = {
        # Timezone-aware UTC ISO string. Container TZ (America/New_York)
        # differs from host (PT), so a naive timestamp here would cause
        # the watchdog's check to report negative elapsed times.
        "ts": now_utc.isoformat(timespec="seconds"),
        # Unix epoch seconds — timezone-agnostic fallback for callers
        # that want numeric delta without parsing ISO strings.
        "ts_epoch": int(now_utc.timestamp()),
        "cycle_count": int(cycle_count),
        "positions_evaluated": int(positions_evaluated),
        "signals_produced": int(signals_produced),
        "last_trade_ts": last_trade_ts,
        "equity": float(equity) if equity is not None else None,
        "halted": bool(halted),
    }
    if extra:
        payload["extra"] = extra

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Atomic write: tmp + rename. Prevents the watchdog from reading
        # a half-written file if the engine crashes mid-fsync.
        fd, tmp = tempfile.mkstemp(
            prefix=".heartbeat.", dir=os.path.dirname(path),
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            raise
        return True
    except OSError as e:
        log.warning(f"Heartbeat write failed: {e}")
        return False


def read_heartbeat() -> dict | None:
    """Read the last heartbeat. Returns None if missing or malformed.

    Callers (primarily the watchdog + its check script) use this to
    decide if the engine is alive.
    """
    path = heartbeat_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def seconds_since_heartbeat(now: float | None = None) -> float | None:
    """Elapsed seconds since the last heartbeat write. None if no file.

    Prefers the ``ts_epoch`` numeric field (timezone-agnostic) and falls
    back to parsing ``ts`` as a timezone-aware ISO string. Both are written
    by current ``write_heartbeat``; only old files (or malformed writes)
    may be missing one or the other.

    ``now`` override is for testing.
    """
    hb = read_heartbeat()
    if not hb:
        return None
    current = now if now is not None else time.time()

    # Preferred: numeric epoch field
    epoch = hb.get("ts_epoch")
    if isinstance(epoch, (int, float)):
        return current - float(epoch)

    # Fallback: parse ISO string. Accept both naive (legacy) and
    # tz-aware forms; naive is interpreted as UTC for safety, since a
    # container may produce either depending on TZ settings.
    ts_str = hb.get("ts")
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return current - dt.timestamp()
