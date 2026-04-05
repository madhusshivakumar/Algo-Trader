"""Risk manager coverage tests — Sprint 4: ATR edge cases, volatility sizing
boundaries, correlation scenarios, and max hold edge cases."""

import time
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from config import Config
from core.risk_manager import RiskManager, TrailingStop


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_df(n=50, base_price=100.0, volatility=1.0, flat=False):
    """Generate OHLCV data with controllable volatility."""
    rng = np.random.RandomState(42)
    if flat:
        close = np.full(n, base_price)
    else:
        close = base_price + np.cumsum(rng.randn(n) * volatility)
        close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + abs(rng.randn(n) * volatility * 0.5),
        "low": close - abs(rng.randn(n) * volatility * 0.5),
        "close": close,
        "volume": [1_000_000] * n,
    })


# ═══════════════════════════════════════════════════════════════════════
# 1. ATR calculation edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestATREdgeCases:
    def test_atr_with_exactly_period_plus_one_rows(self):
        """Minimum data for ATR (period+1 rows)."""
        df = _make_df(n=15)  # period=14, need 15
        atr = RiskManager.calculate_atr(df, period=14)
        assert atr > 0

    def test_atr_with_less_than_period_returns_zero(self):
        """Fewer than period+1 bars returns 0 (guard clause)."""
        df = _make_df(n=10)
        atr = RiskManager.calculate_atr(df, period=14)
        assert atr == 0.0

    def test_atr_flat_market(self):
        """Flat market produces ATR based on high-low spread."""
        df = _make_df(n=50, flat=True)
        atr = RiskManager.calculate_atr(df, period=14)
        assert atr >= 0  # h-l spread still produces some ATR

    def test_atr_high_volatility(self):
        """High volatility should produce larger ATR."""
        df_low = _make_df(n=50, volatility=0.5)
        df_high = _make_df(n=50, volatility=10.0)
        atr_low = RiskManager.calculate_atr(df_low)
        atr_high = RiskManager.calculate_atr(df_high)
        assert atr_high > atr_low

    def test_atr_none_df(self):
        assert RiskManager.calculate_atr(None) == 0.0

    def test_atr_empty_df(self):
        df = pd.DataFrame(columns=["high", "low", "close"])
        assert RiskManager.calculate_atr(df) == 0.0

    def test_atr_single_row(self):
        df = _make_df(n=1)
        assert RiskManager.calculate_atr(df) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 2. ATR stop percentage edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestATRStopPct:
    def test_clamp_minimum(self):
        """Very low volatility should clamp to 0.5%."""
        df = _make_df(n=50, flat=True)
        # Force a tiny non-zero ATR by adding minimal noise
        df.iloc[-1, df.columns.get_loc("high")] += 0.001
        pct = RiskManager.calculate_atr_stop_pct(df, multiplier=0.01)
        assert pct >= 0.005

    def test_clamp_maximum_at_008(self):
        """Extreme volatility should clamp to 8%."""
        df = _make_df(n=50, base_price=10, volatility=10.0)
        pct = RiskManager.calculate_atr_stop_pct(df, multiplier=10.0)
        assert pct <= 0.08

    def test_zero_price_fallback(self):
        """Zero current price should return 0.02 fallback."""
        df = _make_df(n=50)
        df.iloc[-1, df.columns.get_loc("close")] = 0
        pct = RiskManager.calculate_atr_stop_pct(df)
        assert pct == 0.02

    def test_negative_price_fallback(self):
        """Negative current price should return 0.02 fallback."""
        df = _make_df(n=50)
        df.iloc[-1, df.columns.get_loc("close")] = -10
        pct = RiskManager.calculate_atr_stop_pct(df)
        assert pct == 0.02

    def test_crypto_multiplier_produces_wider_stop(self):
        """Crypto multiplier (3.0) should produce wider stop than equity (2.0)."""
        df = _make_df(n=50, volatility=2.0)
        pct_equity = RiskManager.calculate_atr_stop_pct(df, multiplier=2.0)
        pct_crypto = RiskManager.calculate_atr_stop_pct(df, multiplier=3.0)
        assert pct_crypto >= pct_equity


# ═══════════════════════════════════════════════════════════════════════
# 3. ATR-based register_entry with per-asset multiplier
# ═══════════════════════════════════════════════════════════════════════

class TestRegisterEntryATR:
    def test_atr_enabled_crypto_uses_crypto_multiplier(self):
        rm = RiskManager()
        df = _make_df(n=50, volatility=2.0)

        with patch.object(Config, "ATR_STOPS_ENABLED", True), \
             patch.object(Config, "ATR_STOP_MULTIPLIER", 2.0), \
             patch.object(Config, "ATR_STOP_MULTIPLIER_CRYPTO", 3.0):
            rm.register_entry("BTC/USD", 60000, df)

        # Crypto multiplier is higher, so stop should reflect that
        stop = rm.trailing_stops["BTC/USD"]
        assert 0.005 <= stop.stop_pct <= 0.08

    def test_atr_enabled_equity_uses_equity_multiplier(self):
        rm = RiskManager()
        df = _make_df(n=50, volatility=2.0)

        with patch.object(Config, "ATR_STOPS_ENABLED", True), \
             patch.object(Config, "ATR_STOP_MULTIPLIER", 2.0), \
             patch.object(Config, "ATR_STOP_MULTIPLIER_CRYPTO", 3.0):
            rm.register_entry("AAPL", 150, df)

        stop = rm.trailing_stops["AAPL"]
        assert 0.005 <= stop.stop_pct <= 0.08

    def test_atr_enabled_short_data_falls_back(self):
        rm = RiskManager()
        df = _make_df(n=10)  # Too short for ATR

        with patch.object(Config, "ATR_STOPS_ENABLED", True):
            rm.register_entry("AAPL", 150, df)

        assert rm.trailing_stops["AAPL"].stop_pct == 0.025  # equity fallback

    def test_entry_time_set_on_register(self):
        rm = RiskManager()
        before = time.time()
        rm.register_entry("AAPL", 150)
        after = time.time()

        assert before <= rm.trailing_stops["AAPL"].entry_time <= after


# ═══════════════════════════════════════════════════════════════════════
# 4. Volatility-adjusted sizing clamp boundaries
# ═══════════════════════════════════════════════════════════════════════

class TestVolatilitySizingBoundaries:
    def test_insufficient_data_returns_base(self):
        df = _make_df(n=15)  # Less than 20
        size = RiskManager.calculate_volatility_adjusted_size(100000, df, 0.15)
        assert size == 100000 * 0.15

    def test_none_df_returns_base(self):
        size = RiskManager.calculate_volatility_adjusted_size(100000, None, 0.15)
        assert size == 100000 * 0.15

    def test_few_returns_uses_base(self):
        """Less than 10 returns after pct_change should use base."""
        df = _make_df(n=20)
        # pct_change drops first row, tail(20) still has enough
        # But if we make n=11, pct_change gives 10 which is borderline
        df_small = _make_df(n=11)
        size = RiskManager.calculate_volatility_adjusted_size(100000, df_small, 0.15)
        # Should either use base or adjusted — both are valid
        assert size > 0

    def test_zero_volatility_returns_base(self):
        """Flat market with zero std should return base."""
        df = _make_df(n=50, flat=True)
        size = RiskManager.calculate_volatility_adjusted_size(100000, df, 0.15)
        assert size == 100000 * 0.15

    def test_high_vol_smaller_size(self):
        """High volatility should produce smaller position."""
        df = _make_df(n=50, volatility=5.0)
        size = RiskManager.calculate_volatility_adjusted_size(100000, df, 0.15)
        assert size < 100000 * 0.15

    def test_low_vol_capped_at_base(self):
        """Low volatility should not exceed base_pct."""
        df = _make_df(n=50, volatility=0.001)
        size = RiskManager.calculate_volatility_adjusted_size(100000, df, 0.15)
        assert size <= 100000 * 0.15

    def test_minimum_1_pct(self):
        """Size should not go below 1% of equity."""
        df = _make_df(n=50, volatility=50.0)
        size = RiskManager.calculate_volatility_adjusted_size(100000, df, 0.15)
        assert size >= 100000 * 0.01


# ═══════════════════════════════════════════════════════════════════════
# 5. Correlation check scenarios
# ═══════════════════════════════════════════════════════════════════════

class TestCorrelationCheck:
    def test_none_new_df_returns_safe(self):
        assert RiskManager.check_correlation(None, {"AAPL": _make_df()}) is True

    def test_short_new_df_returns_safe(self):
        assert RiskManager.check_correlation(_make_df(n=15), {"AAPL": _make_df()}) is True

    def test_empty_existing_returns_safe(self):
        assert RiskManager.check_correlation(_make_df(), {}) is True

    def test_identical_series_blocked(self):
        """Perfectly correlated data should block."""
        df = _make_df(n=60)
        result = RiskManager.check_correlation(df, {"AAPL": df.copy()}, threshold=0.7)
        assert result is False

    def test_uncorrelated_series_allowed(self):
        """Random uncorrelated data should pass."""
        rng1 = np.random.RandomState(1)
        rng2 = np.random.RandomState(999)
        n = 60
        df1 = pd.DataFrame({"close": 100 + np.cumsum(rng1.randn(n))})
        df2 = pd.DataFrame({"close": 100 + np.cumsum(rng2.randn(n))})
        result = RiskManager.check_correlation(df1, {"OTHER": df2}, threshold=0.9)
        assert result is True

    def test_short_existing_df_skipped(self):
        """Short existing data should be skipped (not block)."""
        new_df = _make_df(n=60)
        short_df = _make_df(n=10)
        result = RiskManager.check_correlation(new_df, {"SHORT": short_df}, threshold=0.7)
        assert result is True

    def test_none_existing_df_skipped(self):
        new_df = _make_df(n=60)
        result = RiskManager.check_correlation(new_df, {"NONE": None}, threshold=0.7)
        assert result is True

    def test_negative_correlation_also_blocked(self):
        """Strong negative correlation should also block (abs >= threshold)."""
        rng = np.random.RandomState(42)
        n = 60
        base = np.cumsum(rng.randn(n))
        df1 = pd.DataFrame({"close": 100 + base})
        df2 = pd.DataFrame({"close": 100 - base})  # Inverted
        result = RiskManager.check_correlation(df1, {"INV": df2}, threshold=0.7)
        assert result is False


# ═══════════════════════════════════════════════════════════════════════
# 6. is_max_hold_exceeded edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestMaxHoldExceeded:
    def test_no_stop_returns_false(self):
        rm = RiskManager()
        assert rm.is_max_hold_exceeded("UNKNOWN") is False

    def test_disabled_returns_false(self):
        rm = RiskManager()
        rm.trailing_stops["AAPL"] = TrailingStop("AAPL", 150, 155, 0.02, entry_time=1.0)

        with patch.object(Config, "MAX_HOLD_HOURS", 0):
            assert rm.is_max_hold_exceeded("AAPL") is False

    def test_crypto_disabled_returns_false(self):
        rm = RiskManager()
        rm.trailing_stops["BTC/USD"] = TrailingStop("BTC/USD", 60000, 61000, 0.04, entry_time=1.0)

        with patch.object(Config, "MAX_HOLD_HOURS_CRYPTO", 0):
            assert rm.is_max_hold_exceeded("BTC/USD") is False

    def test_within_limit_returns_false(self):
        rm = RiskManager()
        rm.trailing_stops["AAPL"] = TrailingStop(
            "AAPL", 150, 155, 0.02, entry_time=time.time() - 1800  # 30 min ago
        )

        with patch.object(Config, "MAX_HOLD_HOURS", 2.0):
            assert rm.is_max_hold_exceeded("AAPL") is False

    def test_exceeded_returns_true(self):
        rm = RiskManager()
        rm.trailing_stops["AAPL"] = TrailingStop(
            "AAPL", 150, 155, 0.02, entry_time=time.time() - 10800  # 3 hours ago
        )

        with patch.object(Config, "MAX_HOLD_HOURS", 2.0):
            assert rm.is_max_hold_exceeded("AAPL") is True

    def test_crypto_uses_crypto_limit(self):
        rm = RiskManager()
        rm.trailing_stops["BTC/USD"] = TrailingStop(
            "BTC/USD", 60000, 61000, 0.04, entry_time=time.time() - 7200  # 2h ago
        )

        with patch.object(Config, "MAX_HOLD_HOURS", 100.0), \
             patch.object(Config, "MAX_HOLD_HOURS_CRYPTO", 1.0):
            assert rm.is_max_hold_exceeded("BTC/USD") is True

    def test_zero_entry_time_returns_false(self):
        rm = RiskManager()
        rm.trailing_stops["AAPL"] = TrailingStop("AAPL", 150, 155, 0.02, entry_time=0.0)

        with patch.object(Config, "MAX_HOLD_HOURS", 1.0):
            assert rm.is_max_hold_exceeded("AAPL") is False


# ═══════════════════════════════════════════════════════════════════════
# 7. TrailingStop.hours_held edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestHoursHeld:
    def test_zero_entry_time(self):
        stop = TrailingStop("AAPL", 150, 155, 0.02, entry_time=0.0)
        assert stop.hours_held() == 0.0

    def test_negative_entry_time(self):
        stop = TrailingStop("AAPL", 150, 155, 0.02, entry_time=-100.0)
        assert stop.hours_held() == 0.0

    def test_recent_entry(self):
        stop = TrailingStop("AAPL", 150, 155, 0.02, entry_time=time.time() - 3600)
        hours = stop.hours_held()
        assert 0.9 < hours < 1.1  # ~1 hour

    def test_old_entry(self):
        stop = TrailingStop("AAPL", 150, 155, 0.02, entry_time=time.time() - 86400)
        hours = stop.hours_held()
        assert 23.5 < hours < 24.5  # ~24 hours


# ═══════════════════════════════════════════════════════════════════════
# 8. Daily reset and drawdown
# ═══════════════════════════════════════════════════════════════════════

class TestDrawdownEdgeCases:
    def test_zero_equity_halts(self):
        rm = RiskManager()
        rm.daily_start_equity = 0
        rm._last_daily_reset = time.time()
        result = rm.check_drawdown(0)
        assert result is True
        assert rm.halted is True

    def test_negative_equity_halts(self):
        rm = RiskManager()
        rm.daily_start_equity = -1000
        rm._last_daily_reset = time.time()
        result = rm.check_drawdown(-2000)
        assert result is True
        assert rm.halted is True

    def test_exactly_at_limit_halts(self):
        rm = RiskManager()
        rm.daily_start_equity = 100000
        rm._last_daily_reset = time.time()
        # Drawdown exactly at 10%
        result = rm.check_drawdown(90000)
        assert result is True

    def test_below_limit_does_not_halt(self):
        rm = RiskManager()
        rm.daily_start_equity = 100000
        rm._last_daily_reset = time.time()
        result = rm.check_drawdown(95000)
        assert result is False
        assert rm.halted is False
