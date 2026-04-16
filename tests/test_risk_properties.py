"""Sprint 7F: property-based tests for risk math (hypothesis).

Tests three key invariants:
  1. Trailing stop price never *decreases* for a long position.
  2. ATR is always non-negative for any valid bar series.
  3. Volatility-adjusted sizing is always bounded within
     [MIN_PCT × equity, base_pct × equity].
  4. ATR stop pct is bounded [0.5%, 8%].
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from core.risk_manager import RiskManager, TrailingStop


# ── Trailing stop monotonicity ─────────────────────────────────────────────

@given(
    entry_price=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False),
    prices=st.lists(
        st.floats(min_value=0.01, max_value=50000.0, allow_nan=False,
                  allow_infinity=False),
        min_size=1, max_size=200,
    ),
    stop_pct=st.floats(min_value=0.001, max_value=0.30, allow_nan=False),
)
@settings(max_examples=300, deadline=None)
def test_trailing_stop_price_never_decreases(entry_price, prices, stop_pct):
    """For any price sequence, the trailing stop price must be monotonically
    non-decreasing until the stop is triggered."""
    ts = TrailingStop(symbol="TEST", entry_price=entry_price,
                      highest_price=entry_price, stop_pct=stop_pct)
    prev_stop = ts.stop_price
    for p in prices:
        triggered = ts.update(p)
        if triggered:
            break
        assert ts.stop_price >= prev_stop, (
            f"Stop price decreased: {prev_stop} -> {ts.stop_price} "
            f"at price {p}"
        )
        prev_stop = ts.stop_price


@given(
    entry_price=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False),
    stop_pct=st.floats(min_value=0.001, max_value=0.30, allow_nan=False),
)
@settings(max_examples=100, deadline=None)
def test_trailing_stop_nan_inf_does_not_trigger(entry_price, stop_pct):
    """NaN and inf prices must not trigger the stop or change highest_price."""
    ts = TrailingStop(symbol="X", entry_price=entry_price,
                      highest_price=entry_price, stop_pct=stop_pct)
    old_highest = ts.highest_price
    assert ts.update(float("nan")) is False
    assert ts.highest_price == old_highest
    assert ts.update(float("inf")) is False
    assert ts.highest_price == old_highest


# ── ATR always non-negative ────────────────────────────────────────────────

def _random_ohlcv(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, 1, n).cumsum()
    close = np.maximum(close, 0.01)  # floor at a penny
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    low = np.maximum(low, 0.001)
    return pd.DataFrame({
        "open": close, "high": high, "low": low, "close": close,
        "volume": np.full(n, 1000),
    })


@given(
    n=st.integers(min_value=2, max_value=200),
    period=st.integers(min_value=1, max_value=50),
    seed=st.integers(min_value=0, max_value=99999),
)
@settings(max_examples=200, deadline=None)
def test_atr_always_non_negative(n, period, seed):
    df = _random_ohlcv(n, seed)
    atr = RiskManager.calculate_atr(df, period=period)
    assert atr >= 0.0, f"ATR was negative: {atr}"
    assert math.isfinite(atr), f"ATR was not finite: {atr}"


@given(period=st.integers(min_value=1, max_value=50))
@settings(max_examples=30, deadline=None)
def test_atr_on_empty_or_tiny_df(period):
    """ATR with insufficient data must return 0, never crash."""
    assert RiskManager.calculate_atr(pd.DataFrame(), period=period) == 0.0
    assert RiskManager.calculate_atr(None, period=period) == 0.0
    one = _random_ohlcv(1)
    assert RiskManager.calculate_atr(one, period=period) >= 0.0


# ── ATR stop pct bounded ──────────────────────────────────────────────────

@given(
    n=st.integers(min_value=16, max_value=200),
    multiplier=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    seed=st.integers(min_value=0, max_value=99999),
)
@settings(max_examples=200, deadline=None)
def test_atr_stop_pct_bounded(n, multiplier, seed):
    df = _random_ohlcv(n, seed)
    pct = RiskManager.calculate_atr_stop_pct(df, multiplier=multiplier)
    assert 0.005 <= pct <= 0.08, f"ATR stop pct out of bounds: {pct}"


# ── Volatility sizing bounded ─────────────────────────────────────────────

@given(
    equity=st.floats(min_value=100.0, max_value=1e8, allow_nan=False),
    n=st.integers(min_value=25, max_value=300),
    base_pct=st.floats(min_value=0.01, max_value=0.50, allow_nan=False),
    seed=st.integers(min_value=0, max_value=99999),
)
@settings(max_examples=200, deadline=None)
def test_vol_sizing_always_bounded(equity, n, base_pct, seed):
    df = _random_ohlcv(n, seed)
    size = RiskManager.calculate_volatility_adjusted_size(
        equity, df, base_pct=base_pct,
    )
    floor = equity * 0.01
    ceiling = equity * base_pct
    assert size >= floor - 0.01, f"Size {size} < floor {floor}"
    assert size <= ceiling + 0.01, f"Size {size} > ceiling {ceiling}"


@given(equity=st.floats(min_value=100.0, max_value=1e8, allow_nan=False))
@settings(max_examples=30, deadline=None)
def test_vol_sizing_with_insufficient_data(equity):
    """Short data → fallback to equity × base_pct."""
    df = _random_ohlcv(5)
    size = RiskManager.calculate_volatility_adjusted_size(equity, df, base_pct=0.15)
    assert abs(size - equity * 0.15) < 0.01


# ── Flat-market edge case ────────────────────────────────────────────────

def test_flat_market_atr_near_zero():
    """Perfectly flat market should produce ATR ≈ 0."""
    df = pd.DataFrame({
        "open": [100.0] * 30, "high": [100.0] * 30,
        "low": [100.0] * 30, "close": [100.0] * 30,
        "volume": [1000] * 30,
    })
    atr = RiskManager.calculate_atr(df, period=14)
    assert atr == pytest.approx(0.0, abs=1e-10)
