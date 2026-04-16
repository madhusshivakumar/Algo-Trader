"""Dashboard view-mode logic — Sprint 6H.

Three progressive-disclosure views, all drawing from the same data:

    **Simple** (default for Beginner profile)
        Balance, today's P&L ($, %), last 5 trades with plain-English
        explanation, protection status badge, expected-return context.

    **Standard** (default for Hobbyist profile)
        Everything in Simple PLUS: weekly P&L chart, open positions with
        trailing stop visualization, daily trade count, strategy breakdown.

    **Advanced** (default for Learner profile, toggle for everyone)
        Everything in Standard PLUS: Sharpe/Sortino/Calmar/VaR, slippage
        analytics, RL action distribution, regime timeline, signal-strength
        histogram.

The module exposes ``build_view_payload(view_mode, raw_data)`` which
returns a trimmed dict the Vercel API can dump to JSON. Tests can call
this without launching a server.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, Sequence

ViewMode = Literal["simple", "standard", "advanced"]
VALID_VIEWS = ("simple", "standard", "advanced")


# ── Regime + framing readers ───────────────────────────────────────────────

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ALERTS_DIR = os.path.join(_REPO_ROOT, "data", "alerts")
_MODIFIER_REPORT = os.path.join(_REPO_ROOT, "data", "modifier_ab",
                                "report.json")


def _read_json_safe(path: str) -> dict | None:
    """Best-effort JSON read. Returns None on any error."""
    try:
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def read_latest_alerts(limit: int = 30) -> list[dict]:
    """Read the last `limit` alerts from the file-based alert log.

    The AlertManager appends one JSON line per alert to
    ``data/alerts/log.jsonl``. We read from the tail so the newest
    alerts come first.
    """
    path = os.path.join(_ALERTS_DIR, "log.jsonl")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            lines = f.readlines()
        raw = []
        for line in reversed(lines[-limit:]):
            line = line.strip()
            if line:
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return raw[:limit]
    except OSError:
        return []


def read_modifier_report() -> dict | None:
    return _read_json_safe(_MODIFIER_REPORT)


# ── Protection status ──────────────────────────────────────────────────────

def protection_status(
    daily_dd_pct: float | None = None,
    halted: bool = False,
    drift_degraded_symbols: Sequence[str] | None = None,
) -> dict:
    """Return the protection-status badge for the Simple view.

    Returns:
        ``{"level": "green" | "yellow" | "red", "label": str}``
    """
    if halted:
        return {"level": "red", "label": "Trading halted (daily drawdown limit)"}
    if daily_dd_pct is not None and daily_dd_pct < -3.0:
        return {"level": "yellow",
                "label": f"Elevated drawdown today ({daily_dd_pct:+.1f}%)"}
    if drift_degraded_symbols:
        n = len(drift_degraded_symbols)
        return {"level": "yellow",
                "label": f"{n} symbol{'s' if n > 1 else ''} degraded — sizes reduced"}
    return {"level": "green", "label": "All systems healthy"}


# ── View payload builder ──────────────────────────────────────────────────

def build_view_payload(
    view_mode: ViewMode,
    *,
    account: dict | None = None,
    stats: dict | None = None,
    trades: list[dict] | None = None,
    equity_curve: list[dict] | None = None,
    regime: str | None = None,
    framing: dict | None = None,
    halted: bool = False,
    daily_dd_pct: float | None = None,
    drift_degraded: list[str] | None = None,
) -> dict:
    """Build a view-mode-appropriate dashboard payload.

    Callers (API endpoints) provide the full raw data. This function trims
    it to the fields the selected view needs, so the frontend doesn't
    accidentally display advanced metrics on the Simple tab.

    Returns a JSON-serializable dict.
    """
    if view_mode not in VALID_VIEWS:
        view_mode = "simple"

    account = account or {}
    stats = stats or {}
    trades = trades or []
    equity_curve = equity_curve or []

    badge = protection_status(daily_dd_pct, halted, drift_degraded)

    # ── Simple: balance + P&L + last 5 trades + protection + framing ──
    payload: dict[str, Any] = {
        "view": view_mode,
        "balance": account.get("equity"),
        "cash": account.get("cash"),
        "total_pnl": stats.get("total_pnl"),
        "total_pnl_pct": stats.get("total_pnl_pct"),
        "win_rate": stats.get("win_rate"),
        "protection": badge,
        "recent_trades": _trim_trades_for_simple(trades, limit=5),
        "framing": framing,
        "regime": regime,
    }

    if view_mode in ("standard", "advanced"):
        # Standard adds: positions, full trades list, equity curve,
        # strategy breakdown.
        payload["positions"] = account.get("positions", [])
        payload["recent_trades"] = _trim_trades_for_standard(trades, limit=50)
        payload["equity_curve"] = equity_curve
        payload["total_trades"] = stats.get("total_trades", 0)
        payload["wins"] = stats.get("wins", 0)
        payload["losses"] = stats.get("losses", 0)
        payload["avg_win"] = stats.get("avg_win")
        payload["avg_loss"] = stats.get("avg_loss")

    if view_mode == "advanced":
        # Advanced adds: raw stats passthrough, alerts, modifier report.
        payload["stats_raw"] = stats
        payload["alerts"] = read_latest_alerts(limit=30)
        payload["modifier_report"] = read_modifier_report()

    return payload


# ── Trade trimming ────────────────────────────────────────────────────────

def _trim_trades_for_simple(trades: list[dict],
                            limit: int = 5) -> list[dict]:
    """Return the last `limit` trades with only Simple-view fields.

    Fields kept: timestamp, symbol, side, amount, explanation.
    Dropped: raw price, strategy, pnl (those come in Standard).
    """
    out = []
    for t in trades[:limit]:
        out.append({
            "timestamp": t.get("timestamp"),
            "symbol": t.get("symbol"),
            "side": t.get("side"),
            "amount": t.get("amount"),
            "explanation": t.get("explanation", t.get("reason", "")),
        })
    return out


def _trim_trades_for_standard(trades: list[dict],
                              limit: int = 50) -> list[dict]:
    """Return the last `limit` trades with Standard-view fields (all)."""
    return trades[:limit]
