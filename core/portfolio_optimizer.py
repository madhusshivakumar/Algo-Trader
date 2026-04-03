"""Portfolio optimization — Kelly criterion sizing and mean-variance allocation.

Provides two complementary portfolio optimization techniques:

- **Kelly Criterion**: Optimal position sizing based on historical win rate and
  payoff ratio. Uses fractional Kelly (default 25%) to reduce variance while
  retaining most of the growth advantage.

- **Mean-Variance Optimization**: Markowitz-style allocation across multiple
  assets. Maximizes Sharpe ratio subject to constraints (no shorting, max
  single-position cap). Uses historical return covariance.

Both are feature-flagged and integrate into the engine's existing position
sizing pipeline as an additional layer.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import Config
from utils.logger import log


@dataclass
class KellyResult:
    """Result from Kelly criterion calculation."""
    full_kelly_fraction: float = 0.0   # raw Kelly %
    adjusted_fraction: float = 0.0     # after fractional Kelly scaling
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    num_trades: int = 0
    reliable: bool = False  # True if enough trades for confident estimate


@dataclass
class AllocationResult:
    """Result from mean-variance optimization."""
    weights: dict[str, float] = field(default_factory=dict)
    # e.g. {"AAPL": 0.25, "TSLA": 0.15, "BTC/USD": 0.30, ...}
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    num_assets: int = 0


def compute_kelly_fraction(trades_pnl: list[float],
                           kelly_fraction: float | None = None,
                           min_trades: int | None = None) -> KellyResult:
    """Compute the Kelly criterion position size from trade history.

    The Kelly criterion gives the optimal fraction of capital to risk:
        f* = (W/L) * p - (1 - p)
             ─────────────────────
                    W/L

    Where:
        p  = probability of winning (win rate)
        W  = average win amount
        L  = average loss amount (positive)
        W/L = payoff ratio

    The raw Kelly fraction is then scaled by kelly_fraction (default 0.25)
    to reduce variance ("fractional Kelly").

    Args:
        trades_pnl: List of realized P&L values per trade.
        kelly_fraction: Fraction of full Kelly to use (0.0 to 1.0).
        min_trades: Minimum trades needed for reliable estimate.

    Returns:
        KellyResult with computed fractions.
    """
    if kelly_fraction is None:
        kelly_fraction = Config.KELLY_FRACTION
    if min_trades is None:
        min_trades = Config.KELLY_MIN_TRADES

    result = KellyResult(num_trades=len(trades_pnl))

    if not trades_pnl:
        return result

    wins = [p for p in trades_pnl if p > 0]
    losses = [p for p in trades_pnl if p < 0]

    if not wins or not losses:
        # Can't compute Kelly without both wins and losses
        return result

    result.win_rate = len(wins) / len(trades_pnl)
    result.avg_win = sum(wins) / len(wins)
    result.avg_loss = abs(sum(losses) / len(losses))
    result.reliable = len(trades_pnl) >= min_trades

    # Payoff ratio (W/L)
    if result.avg_loss == 0:
        return result

    payoff_ratio = result.avg_win / result.avg_loss

    # Kelly formula: f* = p - (1-p)/payoff_ratio
    # Equivalent to: f* = (payoff_ratio * p - (1-p)) / payoff_ratio
    full_kelly = result.win_rate - (1 - result.win_rate) / payoff_ratio

    result.full_kelly_fraction = max(0.0, full_kelly)
    result.adjusted_fraction = result.full_kelly_fraction * kelly_fraction

    return result


def compute_kelly_position_size(equity: float, trades_pnl: list[float],
                                base_pct: float,
                                kelly_fraction: float | None = None,
                                max_pct: float | None = None) -> float:
    """Compute Kelly-adjusted position size in dollars.

    Returns a dollar amount that respects both the Kelly sizing and the
    maximum position cap. Falls back to base_pct if Kelly is unavailable
    or unreliable.

    Args:
        equity: Total portfolio equity.
        trades_pnl: Historical trade P&L list.
        base_pct: Base position size as fraction of equity (fallback).
        kelly_fraction: Fractional Kelly multiplier.
        max_pct: Maximum position as fraction of equity.

    Returns:
        Dollar amount for position sizing.
    """
    if max_pct is None:
        max_pct = Config.MAX_SINGLE_POSITION_PCT

    kelly = compute_kelly_fraction(trades_pnl, kelly_fraction)

    if not kelly.reliable or kelly.adjusted_fraction <= 0:
        # Fall back to base sizing
        return equity * min(base_pct, max_pct)

    # Use Kelly but cap at max_pct
    kelly_pct = min(kelly.adjusted_fraction, max_pct)
    return equity * kelly_pct


def compute_mean_variance_weights(
    returns_df: pd.DataFrame,
    risk_free_rate: float | None = None,
    max_weight: float | None = None,
) -> AllocationResult:
    """Compute mean-variance optimal portfolio weights (max Sharpe).

    Uses analytical solution for the tangency portfolio with no-short-selling
    constraint enforced via iterative clipping. For the typical 2-8 asset
    portfolios in this system, this is sufficient without scipy.

    Args:
        returns_df: DataFrame with columns as asset symbols and rows as
                    daily returns. Must have >= 5 rows and >= 2 columns.
        risk_free_rate: Annual risk-free rate (default from config).
        max_weight: Maximum weight per asset (default from config).

    Returns:
        AllocationResult with optimal weights and portfolio metrics.
    """
    if risk_free_rate is None:
        risk_free_rate = Config.MEAN_VARIANCE_RISK_FREE_RATE
    if max_weight is None:
        max_weight = Config.MAX_SINGLE_POSITION_PCT

    result = AllocationResult()

    if returns_df is None or returns_df.empty:
        return result

    # Need enough data and assets
    if len(returns_df) < 5 or len(returns_df.columns) < 2:
        return result

    # Drop columns with all NaN or zero variance
    valid_cols = []
    for col in returns_df.columns:
        series = returns_df[col].dropna()
        if len(series) >= 5 and series.std() > 0:
            valid_cols.append(col)

    if len(valid_cols) < 2:
        return result

    df = returns_df[valid_cols].dropna()
    if len(df) < 5:
        return result

    symbols = list(df.columns)
    n = len(symbols)
    result.num_assets = n

    # Expected returns (annualized from daily)
    mean_returns = df.mean().values * 252
    # Covariance matrix (annualized)
    cov_matrix = df.cov().values * 252

    # Daily risk-free rate for excess returns
    rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = df.mean().values - rf_daily

    # Inverse-variance weights as starting point (robust heuristic)
    variances = np.diag(cov_matrix)
    if np.any(variances <= 0):
        return result

    inv_var = 1.0 / variances
    weights = inv_var / inv_var.sum()

    # Iterative optimization: adjust towards max-Sharpe via gradient steps
    # For small portfolios this converges quickly
    best_sharpe = -np.inf
    best_weights = weights.copy()

    for _ in range(200):
        # Portfolio metrics
        port_return = weights @ mean_returns
        port_var = weights @ cov_matrix @ weights
        if port_var <= 0:
            break
        port_vol = np.sqrt(port_var)
        port_sharpe = (port_return - risk_free_rate) / port_vol

        if port_sharpe > best_sharpe:
            best_sharpe = port_sharpe
            best_weights = weights.copy()

        # Gradient of Sharpe ratio w.r.t. weights
        d_return = mean_returns
        d_var = 2 * cov_matrix @ weights
        grad = (d_return * port_vol - (port_return - risk_free_rate) * d_var / (2 * port_vol)) / port_var

        # Step
        step_size = 0.01
        new_weights = weights + step_size * grad

        # Project onto constraints: no short selling, max weight, sum to 1
        new_weights = np.maximum(new_weights, 0.0)
        new_weights = np.minimum(new_weights, max_weight)
        total = new_weights.sum()
        if total > 0:
            new_weights = new_weights / total
        else:
            break

        weights = new_weights

    # Final metrics with best weights
    port_return = best_weights @ mean_returns
    port_var = best_weights @ cov_matrix @ best_weights
    port_vol = np.sqrt(max(0, port_var))

    result.weights = {symbols[i]: float(best_weights[i]) for i in range(n)}
    result.expected_return = float(port_return)
    result.expected_volatility = float(port_vol)
    result.sharpe_ratio = float(best_sharpe) if best_sharpe > -np.inf else 0.0

    return result


def get_optimal_position_pct(symbol: str, allocation: AllocationResult,
                             base_pct: float) -> float:
    """Get the optimal position size for a symbol from allocation result.

    If the symbol is in the allocation, returns its weight (as fraction of equity).
    Otherwise falls back to base_pct.

    Args:
        symbol: Trading symbol.
        allocation: AllocationResult from mean-variance optimization.
        base_pct: Fallback position percentage.

    Returns:
        Position size as fraction of equity.
    """
    if not allocation.weights:
        return base_pct

    weight = allocation.weights.get(symbol, 0.0)
    if weight <= 0:
        return base_pct

    return weight


class PortfolioOptimizer:
    """Manages portfolio optimization state across engine cycles.

    Caches allocation results and Kelly estimates to avoid recomputing
    every cycle. Recomputes periodically or when positions change.
    """

    def __init__(self):
        self._allocation: AllocationResult | None = None
        self._kelly_cache: dict[str, KellyResult] = {}
        self._last_recompute_cycle: int = 0
        self._recompute_interval: int = 50  # cycles between recomputes

    def should_recompute(self, cycle_count: int) -> bool:
        """Check if allocation should be recomputed this cycle."""
        if self._allocation is None:
            return True
        return (cycle_count - self._last_recompute_cycle) >= self._recompute_interval

    def update_allocation(self, returns_df: pd.DataFrame,
                          cycle_count: int) -> AllocationResult:
        """Recompute mean-variance allocation from recent returns.

        Args:
            returns_df: DataFrame with columns as symbols, rows as daily returns.
            cycle_count: Current engine cycle number.

        Returns:
            Updated AllocationResult.
        """
        self._allocation = compute_mean_variance_weights(returns_df)
        self._last_recompute_cycle = cycle_count

        if self._allocation.weights:
            log.info(f"Portfolio allocation updated: {len(self._allocation.weights)} assets, "
                     f"Sharpe={self._allocation.sharpe_ratio:.2f}")
        return self._allocation

    def get_position_pct(self, symbol: str, base_pct: float) -> float:
        """Get optimal position percentage for a symbol.

        Uses cached allocation if available, otherwise returns base_pct.
        """
        if self._allocation is None:
            return base_pct
        return get_optimal_position_pct(symbol, self._allocation, base_pct)

    def update_kelly(self, symbol: str, trades_pnl: list[float]) -> KellyResult:
        """Update Kelly estimate for a symbol."""
        result = compute_kelly_fraction(trades_pnl)
        self._kelly_cache[symbol] = result
        return result

    def get_kelly_size(self, symbol: str, equity: float,
                       base_pct: float) -> float:
        """Get Kelly-adjusted position size for a symbol.

        Returns dollar amount. Falls back to base_pct if no Kelly data.
        """
        kelly = self._kelly_cache.get(symbol)
        if kelly is None or not kelly.reliable or kelly.adjusted_fraction <= 0:
            return equity * base_pct

        max_pct = Config.MAX_SINGLE_POSITION_PCT
        kelly_pct = min(kelly.adjusted_fraction, max_pct)
        return equity * kelly_pct

    @property
    def allocation(self) -> AllocationResult | None:
        """Current allocation result."""
        return self._allocation
