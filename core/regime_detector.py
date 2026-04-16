"""Market regime detector — Sprint 6C.

Classifies the current market regime from SPY's 20-day realized volatility
(a cheap VIX proxy, since Alpaca doesn't give us VIX intraday).

Used by:
  - strategies/router.py — mean-reversion strategies skip in high_vol/crisis,
    momentum strategies skip in low_vol (they need movement to ride).
  - trades table — persisted as `regime` column for post-hoc analysis.
  - RL feature extractor — one-hot regime input to state vector (Sprint 6E).

Regimes (annualized SPY realized vol):
  - low_vol:  < 15%   — calm, trending markets; momentum works, mean-rev starves
  - normal:   15-25%  — typical conditions; everything runs
  - high_vol: 25-40%  — choppy, news-driven; mean-rev gets run over by trends
  - crisis:   > 40%   — 2020-03 / 2008-style; most signals break down, size down

The classifier is deliberately conservative — we'd rather stay in `normal` one
day too long than flap between regimes every hour. Hysteresis comes for free
via the 20-day window (daily re-computation only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

RegimeName = Literal["low_vol", "normal", "high_vol", "crisis"]

# Annualized vol thresholds (fractions, not percentages)
THRESHOLD_LOW = 0.15
THRESHOLD_HIGH = 0.25
THRESHOLD_CRISIS = 0.40

# Trading days per year (standard Sharpe denominator)
TRADING_DAYS_PER_YEAR = 252

# Minimum bars needed before we'll produce a classification
MIN_BARS_REQUIRED = 20


@dataclass(frozen=True)
class RegimeSnapshot:
    """Point-in-time regime classification."""
    regime: RegimeName
    annualized_vol: float
    as_of: datetime
    bars_used: int

    def __str__(self) -> str:
        return f"{self.regime} (vol={self.annualized_vol:.1%}, n={self.bars_used})"


def annualize_vol(daily_returns: np.ndarray | pd.Series) -> float:
    """Annualize a sample std of daily returns: stddev × √252.

    Returns 0.0 on empty/all-NaN input (safe fallback for callers that
    pass in very short series during warmup).
    """
    if daily_returns is None:
        return 0.0
    arr = np.asarray(daily_returns, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    return float(np.std(arr, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def classify_vol(annualized_vol: float) -> RegimeName:
    """Map an annualized vol number to a regime label.

    Thresholds are right-exclusive at the low end (< 15% is low_vol, not normal).
    """
    if annualized_vol <= 0 or not math.isfinite(annualized_vol):
        # Degenerate case — treat as normal so we don't spuriously halt trading.
        return "normal"
    if annualized_vol < THRESHOLD_LOW:
        return "low_vol"
    if annualized_vol < THRESHOLD_HIGH:
        return "normal"
    if annualized_vol < THRESHOLD_CRISIS:
        return "high_vol"
    return "crisis"


def compute_realized_vol(
    df: pd.DataFrame | None,
    window: int = 20,
    price_col: str = "close",
) -> float:
    """Compute annualized realized volatility from a price DataFrame.

    Uses the last `window+1` bars so we get `window` returns. If fewer bars
    are available, uses whatever is there (min 2 returns) and returns 0.0
    otherwise.
    """
    if df is None or len(df) < 2 or price_col not in df.columns:
        return 0.0
    tail = df[price_col].tail(window + 1)
    returns = tail.pct_change().dropna()
    return annualize_vol(returns.values)


class RegimeDetector:
    """Stateful regime classifier with a small history buffer.

    Typical usage:
        >>> rd = RegimeDetector()
        >>> rd.update(spy_df)
        >>> rd.get_current_regime()
        'normal'
        >>> rd.get_regime_history()[-5:]
        [RegimeSnapshot(...), ...]

    The detector keeps only the last `max_history` snapshots to bound memory.
    It's not persisted across restarts — callers re-hydrate from SPY bars
    (which we fetch every cycle anyway).
    """

    def __init__(self, window: int = 20, max_history: int = 500):
        self.window = window
        self._history: list[RegimeSnapshot] = []
        self._max_history = max_history

    def update(self, df: pd.DataFrame | None, as_of: datetime | None = None) -> RegimeSnapshot | None:
        """Classify the current regime from the given price bars.

        Returns the new snapshot (also appended to history) or None if
        there's insufficient data.
        """
        if df is None or len(df) < MIN_BARS_REQUIRED:
            return None

        vol = compute_realized_vol(df, self.window)
        regime = classify_vol(vol)
        snap = RegimeSnapshot(
            regime=regime,
            annualized_vol=vol,
            as_of=as_of or datetime.now(),
            bars_used=min(len(df), self.window + 1),
        )
        self._history.append(snap)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        return snap

    def get_current_regime(self) -> RegimeName:
        """Most recent classification, or 'normal' if we haven't seen data yet."""
        if not self._history:
            return "normal"
        return self._history[-1].regime

    def get_current_vol(self) -> float:
        """Most recent annualized vol, or 0.0 if no history."""
        if not self._history:
            return 0.0
        return self._history[-1].annualized_vol

    def get_regime_history(self) -> list[RegimeSnapshot]:
        """Full history of snapshots (oldest first)."""
        return list(self._history)

    def reset(self) -> None:
        """Clear all history (useful in tests and for daily resets)."""
        self._history = []


# ─────────────────────────────────────────────────────────────────────────────
# Strategy-regime compatibility policy.
#
# These are HARD rules for Sprint 6C — if a strategy is explicitly incompatible
# with the current regime, the router skips it (returns hold). This is
# intentionally coarse; Sprint 6F will add data-driven refinement.
#
# Mean reversion strategies assume the price oscillates around a mean. When the
# market is trending hard (high_vol/crisis), those strategies just keep shorting
# breakouts and getting run over. Better to sit out.
#
# Momentum strategies need movement to ride. In low_vol regimes, breakouts don't
# persist — you buy the high and give it back. Better to sit out.
# ─────────────────────────────────────────────────────────────────────────────

_MEAN_REVERSION_STRATEGIES = {
    "mean_reversion",
    "mean_reversion_aggressive",
    "rsi_divergence",
    "volume_profile",  # treats volume-weighted mean as an anchor
}

_MOMENTUM_STRATEGIES = {
    "momentum",
    "macd_crossover",
    "triple_ema",
    "scalper",
}


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 6F: data-driven override set.
#
# ``analytics.strategy_regime_matrix`` writes a JSON file listing
# (strategy, regime) pairs with empirical evidence of negative edge. At module
# import we load that file (if present); ``is_strategy_allowed`` then checks it
# first and falls back to the Sprint 6C hardcoded rules.
# ─────────────────────────────────────────────────────────────────────────────

_HARD_SKIP_OVERRIDES: set[tuple[str, str]] = set()


def _load_overrides_from_disk() -> set[tuple[str, str]]:
    """Best-effort load of the data-driven overrides. Empty on any failure."""
    try:
        from analytics.strategy_regime_matrix import load_hard_skip
        return load_hard_skip()
    except Exception:
        return set()


def reload_overrides() -> set[tuple[str, str]]:
    """Re-read overrides from disk. Returns the freshly loaded set.

    Called at import time; tests can invoke it after monkey-patching the
    JSON path to swap overrides in/out without restarting the process.
    """
    global _HARD_SKIP_OVERRIDES
    _HARD_SKIP_OVERRIDES = _load_overrides_from_disk()
    return _HARD_SKIP_OVERRIDES


# Load at import. Cheap (single JSON read, empty on miss), keeps the runtime
# path branch-free.
reload_overrides()


def is_strategy_allowed(strategy_key: str, regime: RegimeName) -> bool:
    """Return True if the strategy is permitted in the current regime.

    Precedence:
      1. Data-driven overrides from the strategy-regime matrix (Sprint 6F).
         If the pair is listed there, block it.
      2. Hardcoded Sprint 6C policy (mean-rev skipped in high_vol/crisis,
         momentum skipped in low_vol).

    `ensemble` is always allowed — it combines momentum+mean-rev internally
    and already hedges across styles.
    """
    if strategy_key == "ensemble":
        return True

    # Sprint 6F: empirical overrides trump the hardcoded rules.
    if (strategy_key, regime) in _HARD_SKIP_OVERRIDES:
        return False

    if strategy_key in _MEAN_REVERSION_STRATEGIES and regime in ("high_vol", "crisis"):
        return False
    if strategy_key in _MOMENTUM_STRATEGIES and regime == "low_vol":
        return False
    return True


def regime_one_hot(regime: RegimeName) -> tuple[float, float, float, float]:
    """Encode a regime as a 4-dim one-hot vector for RL feature input.

    Order: (low_vol, normal, high_vol, crisis). Used by Sprint 6E to add a
    regime signal to the 16-dim state vector.
    """
    return (
        1.0 if regime == "low_vol" else 0.0,
        1.0 if regime == "normal" else 0.0,
        1.0 if regime == "high_vol" else 0.0,
        1.0 if regime == "crisis" else 0.0,
    )
