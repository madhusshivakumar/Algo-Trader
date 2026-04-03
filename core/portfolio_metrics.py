"""Advanced portfolio metrics — Sortino, Calmar, expectancy, drawdown analysis.

Computes risk-adjusted performance metrics from equity curves and trade history.
Centralizes metric calculations used by dashboard, backtester, and agents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


# Annualization factor for 1-minute bars (252 trading days × 390 min/day)
_MINUTES_PER_YEAR = 252 * 390


@dataclass
class DrawdownInfo:
    """Details about a single drawdown period."""
    start_idx: int
    trough_idx: int
    end_idx: int | None      # None if not yet recovered
    depth: float              # Peak-to-trough as fraction (e.g., 0.05 = 5%)
    duration_bars: int        # Bars from peak to recovery (or current if open)
    recovery_bars: int | None # Bars from trough to recovery (None if open)


@dataclass
class PortfolioMetrics:
    """Container for all computed portfolio metrics."""
    # Performance
    total_return: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    wins: int = 0
    losses: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0           # avg profit per trade
    profit_factor: float = 0.0        # gross profit / gross loss

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0      # total profit / max drawdown

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0    # bars
    avg_drawdown: float = 0.0
    drawdown_count: int = 0

    # Trade stats
    total_trades: int = 0
    avg_hold_bars: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Monthly breakdown (month string -> return%)
    monthly_returns: dict[str, float] = field(default_factory=dict)


def compute_returns(equity_curve: list[float] | np.ndarray) -> np.ndarray:
    """Compute percentage returns from an equity curve."""
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2:
        return np.array([])
    returns = np.diff(eq) / eq[:-1]
    # Replace inf/nan with 0
    returns = np.where(np.isfinite(returns), returns, 0.0)
    return returns


def sharpe_ratio(returns: np.ndarray, annualize: int = _MINUTES_PER_YEAR,
                 risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 3:
        return 0.0
    excess = returns - risk_free / annualize
    std = np.std(excess, ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(annualize))


def sortino_ratio(returns: np.ndarray, annualize: int = _MINUTES_PER_YEAR,
                  risk_free: float = 0.0) -> float:
    """Annualized Sortino ratio — uses downside deviation instead of std."""
    if len(returns) < 3:
        return 0.0
    excess = returns - risk_free / annualize
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0  # Not enough downside data for meaningful ratio
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(annualize))


def calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """Calmar ratio — annualized return / max drawdown."""
    if max_drawdown <= 0:
        return 0.0
    return total_return / max_drawdown


def profit_factor(trades_pnl: list[float]) -> float:
    """Profit factor — gross profit / gross loss.

    Returns 999.0 (capped) when there are profits but no losses,
    to avoid inf which breaks JSON serialization.
    """
    gross_profit = sum(p for p in trades_pnl if p > 0)
    gross_loss = abs(sum(p for p in trades_pnl if p < 0))
    if gross_loss == 0:
        return 999.0 if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def expectancy(trades_pnl: list[float]) -> float:
    """Average profit per trade."""
    if not trades_pnl:
        return 0.0
    return sum(trades_pnl) / len(trades_pnl)


def max_drawdown(equity_curve: list[float] | np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a fraction."""
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd))


def analyze_drawdowns(equity_curve: list[float] | np.ndarray,
                      min_depth: float = 0.01) -> list[DrawdownInfo]:
    """Identify all drawdown periods exceeding min_depth.

    Args:
        equity_curve: Equity values over time.
        min_depth: Minimum drawdown depth to report (0.01 = 1%).

    Returns:
        List of DrawdownInfo sorted by depth (deepest first).
    """
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2:
        return []

    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1.0)

    drawdowns = []
    in_dd = False
    start = 0
    trough_idx = 0
    trough_depth = 0.0

    for i in range(len(dd)):
        if dd[i] > 0 and not in_dd:
            # Entering drawdown
            in_dd = True
            start = i
            trough_idx = i
            trough_depth = dd[i]
        elif in_dd:
            if dd[i] > trough_depth:
                trough_idx = i
                trough_depth = dd[i]
            if dd[i] == 0:
                # Recovered
                if trough_depth >= min_depth:
                    drawdowns.append(DrawdownInfo(
                        start_idx=start,
                        trough_idx=trough_idx,
                        end_idx=i,
                        depth=trough_depth,
                        duration_bars=i - start,
                        recovery_bars=i - trough_idx,
                    ))
                in_dd = False

    # Handle open drawdown (not yet recovered)
    if in_dd and trough_depth >= min_depth:
        drawdowns.append(DrawdownInfo(
            start_idx=start,
            trough_idx=trough_idx,
            end_idx=None,
            depth=trough_depth,
            duration_bars=len(eq) - 1 - start,
            recovery_bars=None,
        ))

    drawdowns.sort(key=lambda d: d.depth, reverse=True)
    return drawdowns


def consecutive_streaks(trades_pnl: list[float]) -> tuple[int, int]:
    """Compute max consecutive wins and max consecutive losses.

    Returns:
        (max_consecutive_wins, max_consecutive_losses)
    """
    if not trades_pnl:
        return 0, 0

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for pnl in trades_pnl:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            # Breakeven — reset both
            current_wins = 0
            current_losses = 0

    return max_wins, max_losses


def monthly_returns(equity_curve: list[float] | np.ndarray,
                    timestamps: list[datetime] | pd.DatetimeIndex | None = None
                    ) -> dict[str, float]:
    """Compute monthly returns from equity curve with timestamps.

    Args:
        equity_curve: Equity values over time.
        timestamps: Corresponding timestamps. If None, returns empty.

    Returns:
        Dict of "YYYY-MM" -> return percentage.
    """
    if timestamps is None or len(timestamps) < 2:
        return {}

    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) != len(timestamps):
        return {}

    df = pd.DataFrame({"equity": eq, "time": pd.to_datetime(timestamps)})
    df["month"] = df["time"].dt.to_period("M")

    result = {}
    for month, group in df.groupby("month"):
        start_eq = group["equity"].iloc[0]
        end_eq = group["equity"].iloc[-1]
        if start_eq > 0:
            result[str(month)] = (end_eq - start_eq) / start_eq
    return result


def compute_all(equity_curve: list[float] | np.ndarray,
                trades_pnl: list[float] | None = None,
                timestamps: list[datetime] | None = None,
                annualize: int = _MINUTES_PER_YEAR) -> PortfolioMetrics:
    """Compute all portfolio metrics from equity curve and trade history.

    Args:
        equity_curve: Equity values over time.
        trades_pnl: List of realized P&L per trade (sell trades only).
        timestamps: Timestamps corresponding to equity curve points.
        annualize: Annualization factor for Sharpe/Sortino.

    Returns:
        PortfolioMetrics with all fields populated.
    """
    metrics = PortfolioMetrics()
    eq = np.asarray(equity_curve, dtype=float)

    if len(eq) < 2:
        return metrics

    # Returns
    returns = compute_returns(eq)

    # Performance
    metrics.total_return = (eq[-1] - eq[0]) / eq[0] if eq[0] > 0 else 0.0
    metrics.total_pnl = float(eq[-1] - eq[0])

    # Risk-adjusted
    metrics.sharpe_ratio = sharpe_ratio(returns, annualize)
    metrics.sortino_ratio = sortino_ratio(returns, annualize)
    metrics.max_drawdown = max_drawdown(eq)
    metrics.calmar_ratio = calmar_ratio(metrics.total_return, metrics.max_drawdown)

    # Drawdown analysis
    drawdowns = analyze_drawdowns(eq)
    metrics.drawdown_count = len(drawdowns)
    if drawdowns:
        metrics.max_drawdown_duration = max(d.duration_bars for d in drawdowns)
        metrics.avg_drawdown = sum(d.depth for d in drawdowns) / len(drawdowns)

    # Trade-level metrics
    if trades_pnl:
        pnl = list(trades_pnl)
        metrics.total_trades = len(pnl)
        metrics.wins = sum(1 for p in pnl if p > 0)
        metrics.losses = sum(1 for p in pnl if p < 0)
        metrics.win_rate = metrics.wins / metrics.total_trades if metrics.total_trades > 0 else 0.0

        win_pnl = [p for p in pnl if p > 0]
        loss_pnl = [p for p in pnl if p < 0]
        metrics.avg_win = sum(win_pnl) / len(win_pnl) if win_pnl else 0.0
        metrics.avg_loss = sum(loss_pnl) / len(loss_pnl) if loss_pnl else 0.0

        metrics.expectancy = expectancy(pnl)
        metrics.profit_factor = profit_factor(pnl)
        # Recovery factor: total profit / max dollar drawdown
        # max_drawdown is fractional, so convert to dollar amount using starting equity
        dollar_dd = metrics.max_drawdown * eq[0] if eq[0] > 0 else 0.0
        metrics.recovery_factor = (sum(pnl) / dollar_dd
                                   if dollar_dd > 0 else 0.0)

        max_w, max_l = consecutive_streaks(pnl)
        metrics.max_consecutive_wins = max_w
        metrics.max_consecutive_losses = max_l

    # Monthly breakdown
    if timestamps is not None:
        metrics.monthly_returns = monthly_returns(eq, timestamps)

    return metrics


def compute_from_db(db_path: str = "trades.db",
                    annualize: int = _MINUTES_PER_YEAR) -> PortfolioMetrics:
    """Compute metrics from the trades database.

    Reads equity_snapshots for the equity curve and trades for P&L.
    """
    import sqlite3

    try:
        conn = sqlite3.connect(db_path)
        try:
            # Load equity curve
            eq_df = pd.read_sql_query(
                "SELECT timestamp, equity FROM equity_snapshots ORDER BY timestamp",
                conn,
            )
            # Load trade PnL (sells only — they have realized PnL)
            trades_df = pd.read_sql_query(
                "SELECT timestamp, pnl FROM trades WHERE side='sell' AND pnl IS NOT NULL ORDER BY timestamp",
                conn,
            )
        finally:
            conn.close()

        equity_curve = eq_df["equity"].tolist() if not eq_df.empty else []
        timestamps = pd.to_datetime(eq_df["timestamp"]).tolist() if not eq_df.empty else None
        trades_pnl = trades_df["pnl"].tolist() if not trades_df.empty else None

        return compute_all(equity_curve, trades_pnl, timestamps, annualize)

    except Exception as e:
        try:
            from utils.logger import log
            log.warning(f"compute_from_db failed: {e}")
        except Exception:
            pass
        return PortfolioMetrics()
