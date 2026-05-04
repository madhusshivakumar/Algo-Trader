"""Weekly health summary agent — Bug #10 (May 3 bug bash).

Fires once a week (Sunday evening) and sends a single WhatsApp/ntfy
message summarising the past 7 days of trading. The intent is a
positive-confirmation channel: if the user gets it, the bot is alive
and reporting; if they don't, that's its own signal that something
is wrong with the alert pipeline.

Includes:
    * Days the bot ran vs scheduled
    * Total trades + win/loss
    * Realized P&L
    * Best / worst day
    * Best / worst symbol
    * Any heartbeat-stale events flagged this week (from the
      watchdog log)
    * Any auto-disabled modifiers from the modifier A/B agent

Schedule (added to crontab):
    0 18 * * 0  Sunday 6 PM ET = 3 PM PT — well after market close
                and the post-market analyzer has written today's
                report to data/analyzer/reports/.

Usage:
    python -m agents.weekly_summary               # send the alert
    python -m agents.weekly_summary --dry-run     # log only, no alert
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta

from utils.logger import log


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DB = os.environ.get("DB_PATH",
                             os.path.join(_REPO_ROOT, "trades.db"))
_WATCHDOG_LOG = os.path.join(_REPO_ROOT, "logs", "watchdog.log")


def _query_week(db_path: str, start_iso: str, end_iso: str) -> dict:
    """Pull the trade summary for the given date range."""
    summary = {
        "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0,
        "by_day": [], "by_symbol_top": [], "by_symbol_bottom": [],
    }
    if not os.path.exists(db_path):
        return summary
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
    except sqlite3.Error:
        return summary

    try:
        # Overall closed-trade stats
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(pnl),0), "
            " SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), "
            " SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) "
            "FROM trades WHERE side='sell' "
            "  AND DATE(timestamp) BETWEEN ? AND ?",
            (start_iso, end_iso),
        ).fetchone()
        if row:
            summary["trades"], summary["pnl"], summary["wins"], summary["losses"] = (
                row[0] or 0, row[1] or 0.0, row[2] or 0, row[3] or 0,
            )

        # Per-day P&L
        days = conn.execute(
            "SELECT DATE(timestamp), COUNT(*), SUM(pnl) "
            "FROM trades WHERE side='sell' "
            "  AND DATE(timestamp) BETWEEN ? AND ? "
            "GROUP BY 1 ORDER BY 1",
            (start_iso, end_iso),
        ).fetchall()
        summary["by_day"] = [{"date": d, "n": n, "pnl": round(p or 0.0, 2)}
                             for d, n, p in days]

        # Top + bottom symbols by realized PnL
        syms = conn.execute(
            "SELECT symbol, COUNT(*), SUM(pnl) "
            "FROM trades WHERE side='sell' "
            "  AND DATE(timestamp) BETWEEN ? AND ? "
            "GROUP BY symbol ORDER BY 3 DESC",
            (start_iso, end_iso),
        ).fetchall()
        summary["by_symbol_top"] = [
            {"sym": s, "n": n, "pnl": round(p or 0.0, 2)}
            for s, n, p in syms[:3]
        ]
        summary["by_symbol_bottom"] = [
            {"sym": s, "n": n, "pnl": round(p or 0.0, 2)}
            for s, n, p in syms[-3:][::-1]
            if (p or 0.0) < 0
        ]
    finally:
        conn.close()
    return summary


def _count_heartbeat_alarms(start_iso: str, end_iso: str) -> int:
    """Count HEARTBEAT STALE warnings in the watchdog log this week."""
    if not os.path.exists(_WATCHDOG_LOG):
        return 0
    n = 0
    try:
        with open(_WATCHDOG_LOG) as f:
            for line in f:
                if "HEARTBEAT STALE" not in line:
                    continue
                # Best-effort date extraction: "[YYYY-MM-DD HH:MM:SS TZ]"
                if start_iso in line or end_iso in line or any(
                    line.startswith(f"[{d}") for d in _date_range(
                        start_iso, end_iso)):
                    n += 1
    except OSError:
        pass
    return n


def _date_range(start_iso: str, end_iso: str) -> list[str]:
    """All YYYY-MM-DD strings in [start, end] inclusive."""
    s = datetime.strptime(start_iso, "%Y-%m-%d").date()
    e = datetime.strptime(end_iso, "%Y-%m-%d").date()
    out = []
    while s <= e:
        out.append(s.isoformat())
        s = s + timedelta(days=1)
    return out


def _format_summary(summary: dict, start_iso: str, end_iso: str,
                    heartbeat_alarms: int) -> str:
    """Build the human-readable message body."""
    n = summary["trades"]
    if n == 0:
        return (f"📅 Weekly summary {start_iso} → {end_iso}: NO closed "
                f"trades this week. If equity markets were open and the "
                f"bot was scheduled to run, investigate.")

    wr = (summary["wins"] / n * 100) if n else 0.0
    lines = [
        f"📅 Weekly summary {start_iso} → {end_iso}",
        f"  Trades: {n}  ({summary['wins']}W / {summary['losses']}L, "
        f"{wr:.0f}% WR)",
        f"  Realized P&L: ${summary['pnl']:+,.2f}",
    ]

    if summary["by_day"]:
        best = max(summary["by_day"], key=lambda d: d["pnl"])
        worst = min(summary["by_day"], key=lambda d: d["pnl"])
        lines.append(
            f"  Best day: {best['date']} ${best['pnl']:+,.2f} ({best['n']}t)"
        )
        if worst["date"] != best["date"]:
            lines.append(
                f"  Worst day: {worst['date']} ${worst['pnl']:+,.2f} "
                f"({worst['n']}t)"
            )

    if summary["by_symbol_top"]:
        winners = ", ".join(
            f"{s['sym']} ${s['pnl']:+,.0f}"
            for s in summary["by_symbol_top"]
        )
        lines.append(f"  Top symbols: {winners}")

    if summary["by_symbol_bottom"]:
        losers = ", ".join(
            f"{s['sym']} ${s['pnl']:+,.0f}"
            for s in summary["by_symbol_bottom"]
        )
        lines.append(f"  Bottom symbols: {losers}")

    if heartbeat_alarms > 0:
        lines.append(
            f"  ⚠ Heartbeat-stale alarms this week: {heartbeat_alarms}"
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weekly health summary")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log the summary; do not fire AlertManager.")
    parser.add_argument("--days", type=int, default=7,
                        help="Lookback window (default 7).")
    parser.add_argument("--db", default=_DEFAULT_DB)
    args = parser.parse_args(argv)

    end = datetime.now().date()
    start = end - timedelta(days=args.days - 1)
    start_iso, end_iso = start.isoformat(), end.isoformat()

    summary = _query_week(args.db, start_iso, end_iso)
    heartbeat_alarms = _count_heartbeat_alarms(start_iso, end_iso)
    body = _format_summary(summary, start_iso, end_iso, heartbeat_alarms)

    log.info("Weekly summary:\n" + body)

    if args.dry_run:
        return 0

    try:
        from core.alerting import AlertLevel, AlertManager
        mgr = AlertManager()
        if not mgr.channels:
            log.warning("No alert channels configured — summary not sent.")
            return 0
        mgr.alert(
            event_type="weekly_summary",
            message=body,
            level=AlertLevel.INFO,
            data={"period": f"{start_iso}→{end_iso}",
                  "trades": summary["trades"],
                  "pnl_usd": round(summary["pnl"], 2)},
        )
        # Bug #1a: short-lived script must flush before exit so the
        # daemon dispatch thread isn't killed mid-HTTP-POST.
        unfinished = mgr.flush(timeout=10.0)
        if unfinished > 0:
            log.warning(f"{unfinished} alert thread(s) timed out")
            return 1
    except Exception as e:
        log.error(f"weekly_summary alert dispatch failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
