"""Modifier performance scorer and auto-disable runner.

Sprint 6D: reads the A/B log from `analytics.modifier_ab` and the trades
table, computes a counterfactual Sharpe contribution per modifier over the
last N days, and flips the modifier's feature flag to false (with a loud
alert) if the contribution is non-positive.

Contribution model — keep it simple and honest:
  - For each trade in the window, find the most recent modifier record for
    the same symbol BEFORE the trade timestamp (within a short grace window,
    default 10 minutes).
  - The modifier's `delta` × trade's realized P&L is the per-trade
    contribution proxy. Intuition: if the modifier pushed strength UP and
    the trade made money, that's positive. If it pushed UP and lost, that's
    negative. If it pushed DOWN and we still traded anyway, the magnitude
    is smaller but the sign is consistent.
  - Sum over the window → 30-day contribution. If ≤ 0, the modifier is not
    paying for its API costs and gets auto-disabled.

Caveats (documented because they matter for interpretation):
  1. A modifier that simply refuses bad trades (flips action to hold) gets
     zero contribution from this model because we only see trades that
     actually executed. We flag `action_flip_count` separately so the user
     can see that signal.
  2. We use REALIZED P&L, which takes days-to-close to land. For symbols
     with long-held positions, the modifier record and the trade's exit P&L
     may be weeks apart — the 10-minute match window is for the ENTRY bar.
  3. Threshold of 0 is deliberately strict: a modifier adding real value
     should comfortably beat zero over 30 days and tens of trades. If it
     can't, the API + compute cost is better spent elsewhere.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

from config import Config
from utils.logger import log

from .modifier_ab import read_deltas_for_modifier, summarize_deltas

# Modifier registry — each entry is (modifier_name, config_flag_name).
# The config flag name is the attribute on `Config` we flip to disable.
MODIFIER_REGISTRY: list[tuple[str, str]] = [
    ("sentiment", "SENTIMENT_ENABLED"),
    ("llm", "LLM_ANALYST_ENABLED"),
    ("mtf", "MTF_ENABLED"),
    ("earnings_blackout", "EARNINGS_CALENDAR_ENABLED"),
]

# How close (in minutes) a modifier record must be to a trade timestamp
# to count as "attached to" that trade.
DEFAULT_MATCH_WINDOW_MIN = 10


@dataclass
class ModifierReport:
    """One row of the weekly modifier-performance report."""
    modifier: str
    config_flag: str
    days: int
    contribution: float
    n_trades_matched: int
    summary: dict = field(default_factory=dict)
    recommendation: str = "keep"  # "keep" | "disable" | "no_data"


def _get_trades_db_path() -> str:
    """Resolve the trades SQLite path in a way compatible with both the
    live runtime (reads Config) and tests (which may override)."""
    # Config doesn't expose a dedicated trades DB path; reuse the logger's
    # canonical location without importing the logger at module-import time
    # (avoids a circular init during analytics imports).
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "db", "trades.db")


def _load_trades(db_path: str, since: datetime) -> list[dict]:
    """Read trades executed since `since` as list of dicts."""
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT timestamp, symbol, side, pnl FROM trades WHERE timestamp >= ?",
            (since.isoformat(),),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except sqlite3.Error as e:
        log.warning(f"Could not read trades for A/B analysis: {e}")
        return []


def _attach_deltas_to_trades(
    trades: list[dict],
    deltas: list[dict],
    match_window_min: int = DEFAULT_MATCH_WINDOW_MIN,
) -> list[tuple[dict, dict]]:
    """Pair each trade with the most recent same-symbol delta within window.

    Returns a list of (trade, delta) tuples for trades that matched. Trades
    without a matching delta are dropped — this is what we want, because
    those trades weren't affected by the modifier being measured.
    """
    window = timedelta(minutes=match_window_min)
    # Bucket deltas by symbol for O(1) lookup, keep them sorted by ts.
    by_sym: dict[str, list[dict]] = {}
    for d in deltas:
        sym = d.get("symbol")
        if sym:
            by_sym.setdefault(sym, []).append(d)
    for lst in by_sym.values():
        lst.sort(key=lambda r: r.get("ts", ""))

    paired: list[tuple[dict, dict]] = []
    for t in trades:
        sym = t.get("symbol")
        ts_str = t.get("timestamp")
        if not sym or not ts_str:
            continue
        try:
            trade_ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            if trade_ts.tzinfo is not None:
                trade_ts = trade_ts.replace(tzinfo=None)
        except ValueError:
            continue

        candidates = by_sym.get(sym, [])
        best: dict | None = None
        for d in candidates:
            try:
                d_ts = datetime.fromisoformat(
                    str(d["ts"]).replace("Z", "+00:00")
                )
                if d_ts.tzinfo is not None:
                    d_ts = d_ts.replace(tzinfo=None)
            except (KeyError, ValueError):
                continue
            # Want deltas that occurred AT OR BEFORE the trade timestamp,
            # within the match window. Anything after the trade can't have
            # influenced it.
            if d_ts <= trade_ts and (trade_ts - d_ts) <= window:
                if best is None or d_ts > datetime.fromisoformat(
                    str(best["ts"]).replace("Z", "+00:00")
                ):
                    best = d
        if best is not None:
            paired.append((t, best))

    return paired


def _contribution(pairs: list[tuple[dict, dict]]) -> tuple[float, int]:
    """Sum delta × P&L across all matched (trade, delta) pairs.

    Returns (total_contribution, n_matched).
    """
    total = 0.0
    n = 0
    for trade, delta in pairs:
        try:
            d = float(delta.get("delta", 0.0) or 0.0)
            pnl = float(trade.get("pnl", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        total += d * pnl
        n += 1
    return total, n


def compute_report(
    modifier_name: str,
    config_flag: str,
    days: int = 30,
    db_path: str | None = None,
    log_path: str | None = None,
) -> ModifierReport:
    """Compute the report for one modifier.

    Args:
        modifier_name: e.g. "sentiment".
        config_flag: The Config attribute to disable if contribution ≤ 0.
        days: Analysis window.
        db_path: Optional trades-DB override (tests use this).
        log_path: Optional modifier_ab log override (tests use this).
    """
    deltas = read_deltas_for_modifier(modifier_name, days=days, log_path=log_path)
    summary = summarize_deltas(deltas)

    if summary["count"] == 0:
        return ModifierReport(
            modifier=modifier_name,
            config_flag=config_flag,
            days=days,
            contribution=0.0,
            n_trades_matched=0,
            summary=summary,
            recommendation="no_data",
        )

    since = datetime.now() - timedelta(days=days)
    trades = _load_trades(db_path or _get_trades_db_path(), since)
    pairs = _attach_deltas_to_trades(trades, deltas)
    contribution, n_matched = _contribution(pairs)

    # Recommendation: disable only if we have reasonable evidence. Require
    # at least 5 matched trades to avoid auto-disabling on noise.
    if n_matched < 5:
        recommendation = "no_data"
    elif contribution <= 0:
        recommendation = "disable"
    else:
        recommendation = "keep"

    return ModifierReport(
        modifier=modifier_name,
        config_flag=config_flag,
        days=days,
        contribution=round(contribution, 4),
        n_trades_matched=n_matched,
        summary=summary,
        recommendation=recommendation,
    )


def compute_all_reports(
    days: int = 30,
    db_path: str | None = None,
    log_path: str | None = None,
) -> list[ModifierReport]:
    """One report per known modifier. Entry point for the weekly job."""
    return [
        compute_report(name, flag, days=days,
                       db_path=db_path, log_path=log_path)
        for name, flag in MODIFIER_REGISTRY
    ]


def auto_disable_negative_modifiers(
    reports: Iterable[ModifierReport],
    alert_manager=None,
    dry_run: bool = False,
) -> list[str]:
    """Flip config flags for any modifier with `recommendation == 'disable'`.

    Alerts are sent for every disable action so the user sees them in the
    dashboard + Slack/Discord channel (via AlertManager dedup).

    Args:
        reports: Sequence of ModifierReport from compute_all_reports.
        alert_manager: Optional AlertManager instance. If None, alerts are
            logged but not dispatched.
        dry_run: If True, log what would be done but don't mutate Config.

    Returns:
        List of modifier names that were disabled.
    """
    disabled: list[str] = []
    for rep in reports:
        if rep.recommendation != "disable":
            continue
        disabled.append(rep.modifier)
        msg = (
            f"Modifier '{rep.modifier}' auto-disabled: "
            f"{rep.days}-day contribution = {rep.contribution:.4f} over "
            f"{rep.n_trades_matched} matched trades"
        )
        if dry_run:
            log.warning(f"[dry_run] Would disable {rep.modifier} | {msg}")
            continue
        try:
            setattr(Config, rep.config_flag, False)
            log.warning(msg)
        except Exception as e:
            log.error(f"Failed to disable {rep.modifier}: {e}")
            continue

        if alert_manager is not None:
            try:
                alert_manager.alert(
                    "warning",
                    msg,
                    {"modifier": rep.modifier,
                     "contribution": rep.contribution,
                     "n_trades": rep.n_trades_matched},
                )
            except Exception:
                # Alerting failure must not block the disable action.
                pass

    return disabled


def format_report_text(reports: Iterable[ModifierReport]) -> str:
    """Pretty-print reports as a human-readable block (for logs / stdout)."""
    lines = ["Modifier A/B Report", "=" * 60]
    for r in reports:
        lines.append(
            f"{r.modifier:20s} "
            f"contrib={r.contribution:+.4f} "
            f"trades={r.n_trades_matched:3d} "
            f"records={r.summary.get('count', 0):4d} "
            f"action_flips={r.summary.get('action_flip_count', 0):3d} "
            f"→ {r.recommendation}"
        )
    return "\n".join(lines)
