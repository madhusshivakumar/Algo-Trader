"""Sprint 5G: Paper→live promotion safety gate.

Refuses to honor `TRADING_MODE=live` until the bot has accumulated 30+ days of
paper trading history with a worst drawdown >= -3% of equity. The override flag
exists but is intentionally awkward (`I_UNDERSTAND_THE_RISK=yes-really`) so a
new user can't accidentally flip the switch.

This is not financial advice — it's a software safety belt for users who are
new to algo trading. Even with the gate, real-money trading carries real risk.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta


# Constants — keep in this module so they're importable for tests
MIN_PAPER_DAYS = 30
MAX_DRAWDOWN_THRESHOLD = -0.03  # -3% — must be at least this good to promote
OVERRIDE_ENV = "I_UNDERSTAND_THE_RISK"
OVERRIDE_VALUE = "yes-really"


@dataclass
class GateResult:
    """Result of a paper→live promotion check."""

    can_promote: bool
    days_observed: int
    worst_drawdown: float  # most negative day-on-day return seen
    reason: str

    def __bool__(self) -> bool:
        return self.can_promote


def _query_equity_history(db_path: str) -> list[tuple[str, float]]:
    """Return [(date_str, equity), ...] from equity_snapshots, daily resolution."""
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            """
            SELECT DATE(timestamp) AS day, equity
            FROM equity_snapshots
            ORDER BY timestamp
            """
        )
        rows = cursor.fetchall()
        conn.close()
    except sqlite3.Error:
        return []
    # Take last value per day (close-of-day equity)
    by_day: dict[str, float] = {}
    for day, equity in rows:
        if equity is not None:
            by_day[day] = float(equity)
    return sorted(by_day.items())


def evaluate_gate(db_path: str = "trades.db",
                  min_days: int = MIN_PAPER_DAYS,
                  max_drawdown: float = MAX_DRAWDOWN_THRESHOLD) -> GateResult:
    """Return a GateResult describing whether paper→live promotion is allowed."""
    history = _query_equity_history(db_path)
    if not history:
        return GateResult(
            can_promote=False, days_observed=0, worst_drawdown=0.0,
            reason=(f"No paper trading history — need {min_days} days "
                    f"with worst day-loss >= {max_drawdown:.0%}."),
        )

    # Compute day-on-day returns
    returns = []
    for i in range(1, len(history)):
        prev = history[i - 1][1]
        curr = history[i][1]
        if prev > 0:
            returns.append((curr - prev) / prev)

    days = len(history)
    worst = min(returns) if returns else 0.0

    if days < min_days:
        return GateResult(
            can_promote=False, days_observed=days, worst_drawdown=worst,
            reason=(f"Only {days} days of paper history — need at least {min_days}."),
        )

    if worst < max_drawdown:
        return GateResult(
            can_promote=False, days_observed=days, worst_drawdown=worst,
            reason=(f"Worst day-loss {worst:.2%} breaches the "
                    f"{max_drawdown:.0%} acceptable threshold — paper validation failed."),
        )

    return GateResult(
        can_promote=True, days_observed=days, worst_drawdown=worst,
        reason=f"Validated: {days} days of paper, worst day-loss {worst:.2%}.",
    )


def is_override_active() -> bool:
    """True if the user has typed the awkward override env var."""
    return os.getenv(OVERRIDE_ENV, "").strip().lower() == OVERRIDE_VALUE


def check_live_mode_allowed(trading_mode: str,
                           db_path: str = "trades.db",
                           min_days: int = MIN_PAPER_DAYS,
                           max_drawdown: float = MAX_DRAWDOWN_THRESHOLD,
                           ) -> tuple[str, GateResult | None, bool]:
    """Top-level helper used by Config.validate.

    Args:
        trading_mode: The requested mode ("paper" or "live").
        db_path: Path to trades SQLite DB.
        min_days: Minimum paper trading days required.
        max_drawdown: Worst acceptable single-day return (negative number).

    Returns:
        (effective_mode, gate_result, override_used)
        - effective_mode: "paper" or "live" — what the bot will actually trade as
        - gate_result: GateResult if mode was "live" and we evaluated, else None
        - override_used: True if effective_mode is "live" via override
    """
    if trading_mode != "live":
        return trading_mode, None, False

    if is_override_active():
        return "live", None, True

    result = evaluate_gate(db_path, min_days=min_days, max_drawdown=max_drawdown)
    if result.can_promote:
        return "live", result, False
    # Force back to paper
    return "paper", result, False
