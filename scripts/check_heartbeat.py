#!/usr/bin/env python3
"""Heartbeat check — Sprint 8 (post Monday Apr 20 incident).

Reads ``data/heartbeat.json`` written by the engine each cycle and fires
an AlertManager alert if the engine appears dead.

Exit codes:
    0 — heartbeat is fresh, engine is live
    1 — heartbeat is stale / missing (alert fired if channels configured)
    2 — heartbeat fresh but engine is halted (alert fired, info severity)

Called by the watchdog during market hours. Runs locally (host) — the
heartbeat file lives in the Docker-mounted data/ directory.

Usage:
    python scripts/check_heartbeat.py
    python scripts/check_heartbeat.py --stale-seconds 300 --silent
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from core.heartbeat import read_heartbeat, seconds_since_heartbeat


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check engine heartbeat")
    ap.add_argument("--stale-seconds", type=int, default=300,
                    help="Alert if heartbeat is older than this (default 300s).")
    ap.add_argument("--silent", action="store_true",
                    help="Suppress stdout (exit code is the signal).")
    args = ap.parse_args(argv)

    def say(msg: str):
        if not args.silent:
            print(msg)

    hb = read_heartbeat()
    if hb is None:
        say("HEARTBEAT MISSING — no heartbeat file at data/heartbeat.json")
        _fire_alert(
            event_type="heartbeat_missing",
            message="Engine heartbeat file is missing. Container may be up "
                    "but the engine Python process is not running.",
            level="critical",
            data={"file_exists": False},
        )
        return 1

    elapsed = seconds_since_heartbeat()
    if elapsed is None:
        say("HEARTBEAT UNREADABLE — file exists but timestamp is invalid")
        _fire_alert(
            event_type="heartbeat_corrupt",
            message="Heartbeat file is present but malformed.",
            level="warning",
            data={"hb": hb},
        )
        return 1

    if elapsed > args.stale_seconds:
        say(f"HEARTBEAT STALE — last write {int(elapsed)}s ago "
            f"(threshold {args.stale_seconds}s)")
        _fire_alert(
            event_type="heartbeat_stale",
            message=(f"Engine heartbeat is {int(elapsed)}s stale — the "
                     f"container may report healthy but the trading loop "
                     f"has stopped progressing. Last cycle: "
                     f"{hb.get('cycle_count')}."),
            level="critical",
            data={
                "seconds_since_heartbeat": int(elapsed),
                "cycle_count": hb.get("cycle_count"),
                "last_trade_ts": hb.get("last_trade_ts"),
            },
        )
        return 1

    if hb.get("halted"):
        say(f"HEARTBEAT FRESH ({int(elapsed)}s ago) — but engine is HALTED")
        _fire_alert(
            event_type="engine_halted",
            message="Engine is alive but trading is halted (risk manager).",
            level="warning",
            data={"cycle_count": hb.get("cycle_count"),
                  "equity": hb.get("equity")},
        )
        return 2

    say(f"HEARTBEAT FRESH — {int(elapsed)}s ago, cycle "
        f"{hb.get('cycle_count')}, equity ${hb.get('equity')}")
    return 0


def _fire_alert(event_type: str, message: str, level: str,
                data: dict | None = None) -> None:
    """Best-effort AlertManager dispatch. Silent on any failure.

    Bug #1 fix (Apr 28 incident): MUST call ``mgr.flush()`` before this
    function returns. The dispatch happens on a daemon thread; without
    flush, the script returns to caller, the script exits ~10ms later,
    and the daemon thread is killed mid-POST. AlertManager logs
    "dispatched" but no message reaches the channel. This silently lost
    every heartbeat-stale alert during the Apr 28 incident.
    """
    try:
        from core.alerting import AlertLevel, AlertManager
        mgr = AlertManager()
        lvl_map = {
            "info": AlertLevel.INFO,
            "warning": AlertLevel.WARNING,
            "critical": AlertLevel.CRITICAL,
        }
        mgr.alert(
            event_type=event_type,
            message=message,
            level=lvl_map.get(level, AlertLevel.WARNING),
            data=data,
        )
        # Wait for the daemon thread to actually deliver. CallMeBot's
        # WhatsApp endpoint usually responds in 1-3s; budget 10s to cover
        # transient slowness without blocking the watchdog forever.
        unfinished = mgr.flush(timeout=10.0)
        if unfinished > 0:
            # Re-raise into the swallow below so the caller sees it.
            raise RuntimeError(
                f"{unfinished} alert thread(s) timed out — channel likely down"
            )
    except Exception:
        # We're a monitoring script — never let alerting errors mask the
        # underlying state we're trying to report.
        pass


if __name__ == "__main__":
    sys.exit(main())
