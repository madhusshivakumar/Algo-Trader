"""Sprint 4 — additional risk_manager.py tests for coverage gaps."""

import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.risk_manager import RiskManager, TrailingStop
from config import Config


def _make_df(n=50, seed=42, flat=False):
    np.random.seed(seed)
    if flat:
        close = np.full(n, 100.0)
    else:
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


# -- TrailingStop edge cases --

class TestTrailingStopEdgeCases:

    def test_update_with_nan(self):
        stop = TrailingStop("AAPL", 100.0, 105.0, 0.02)
        assert stop.update(float("nan")) is False
        assert stop.highest_price == 105.0

    def test_update_with_inf(self):
        stop = TrailingStop("AAPL", 100.0, 105.0, 0.02)
        assert stop.update(float("inf")) is False
        assert stop.highest_price == 105.0

    def test_update_with_neg_inf(self):
        stop = TrailingStop("AAPL", 100.0, 105.0, 0.02)
        assert stop.update(float("-inf")) is False

    def test_update_price_above_highest(self):
        stop = TrailingStop("AAPL", 100.0, 105.0, 0.02)
        triggered = stop.update(110.0)
        assert triggered is False
        assert stop.highest_price == 110.0

    def test_update_triggers_at_stop(self):
        stop = TrailingStop("AAPL", 100.0, 100.0, 0.02)
        triggered = stop.update(98.0)
        assert triggered is True

    def test_hours_held_zero_entry(self):
        stop = TrailingStop("AAPL", 100.0, 100.0, 0.02, entry_time=0.0)
        assert stop.hours_held() == 0.0

    def test_hours_held_positive(self):
        stop = TrailingStop("AAPL", 100.0, 100.0, 0.02, entry_time=time.time() - 7200)
        assert 1.9 < stop.hours_held() < 2.1

    def test_stop_price_property(self):
        stop = TrailingStop("AAPL", 100.0, 200.0, 0.05)
        assert stop.stop_price == 190.0


# -- ATR Calculation --

class TestATRCalculation:

    def test_atr_short_data_returns_zero(self):
        """With only 5 bars (need period+1=15), ATR returns 0."""
        df = _make_df(n=5)
        atr = RiskManager.calculate_atr(df, period=14)
        assert atr == 0.0

    def test_atr_enough_data(self):
        df = _make_df(n=30)
        atr = RiskManager.calculate_atr(df, period=14)
        assert atr > 0

    def test_atr_none_df(self):
        assert RiskManager.calculate_atr(None) == 0.0

    def test_atr_flat_market(self):
        df = _make_df(n=50, flat=True)
        atr = RiskManager.calculate_atr(df)
        assert atr >= 0

    def test_atr_stop_pct_short_data(self):
        df = _make_df(n=5)
        assert RiskManager.calculate_atr_stop_pct(df) == 0.02

    def test_atr_stop_pct_none(self):
        assert RiskManager.calculate_atr_stop_pct(None) == 0.02

    def test_atr_stop_pct_clamp_min(self):
        df = _make_df(n=50, flat=True)
        pct = RiskManager.calculate_atr_stop_pct(df, multiplier=0.001)
        assert pct >= 0.005

    def test_atr_stop_pct_clamp_max(self):
        df = _make_df(n=50)
        pct = RiskManager.calculate_atr_stop_pct(df, multiplier=100.0)
        assert pct <= 0.08

    def test_atr_stop_pct_zero_price(self):
        df = _make_df(n=50)
        df.iloc[-1, df.columns.get_loc("close")] = 0.0
        assert RiskManager.calculate_atr_stop_pct(df) == 0.02


# -- Volatility Sizing --

class TestVolatilitySizing:

    def test_none_df_returns_base(self):
        assert RiskManager.calculate_volatility_adjusted_size(100000, None, 0.15) == 15000.0

    def test_short_df_returns_base(self):
        df = _make_df(n=5)
        assert RiskManager.calculate_volatility_adjusted_size(100000, df, 0.15) == 15000.0

    def test_high_vol_reduces_size(self):
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 5.0)
        df = pd.DataFrame({
            "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": [1000] * 50
        })
        size = RiskManager.calculate_volatility_adjusted_size(100000, df, 0.15)
        assert size < 15000

    def test_low_vol_keeps_base(self):
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.01)
        df = pd.DataFrame({
            "open": close, "high": close + 0.01, "low": close - 0.01,
            "close": close, "volume": [1000] * 50
        })
        size = RiskManager.calculate_volatility_adjusted_size(100000, df, 0.15)
        assert size == 15000


# -- Correlation Check --

class TestCorrelationCheck:

    def test_empty_existing(self):
        assert RiskManager.check_correlation(_make_df(), {}) is True

    def test_none_df(self):
        assert RiskManager.check_correlation(None, {"A": _make_df()}) is True

    def test_perfect_correlation_blocks(self):
        df = _make_df(seed=42)
        assert RiskManager.check_correlation(df, {"CLONE": df.copy()}, threshold=0.7) is False

    def test_uncorrelated_allows(self):
        result = RiskManager.check_correlation(
            _make_df(seed=42), {"OTHER": _make_df(seed=999)}, threshold=0.95)
        assert result is True

    def test_short_existing_skipped(self):
        assert RiskManager.check_correlation(_make_df(), {"S": _make_df(n=5)}) is True


# -- Max Hold Exceeded --

class TestMaxHoldExceeded:

    def test_no_stop(self):
        assert RiskManager().is_max_hold_exceeded("AAPL") is False

    @patch.object(Config, "MAX_HOLD_HOURS", 0)
    def test_disabled(self):
        rm = RiskManager()
        rm.trailing_stops["AAPL"] = TrailingStop("AAPL", 100, 100, 0.02, time.time() - 999999)
        assert rm.is_max_hold_exceeded("AAPL") is False

    @patch.object(Config, "MAX_HOLD_HOURS", 24)
    def test_under_limit(self):
        rm = RiskManager()
        rm.trailing_stops["AAPL"] = TrailingStop("AAPL", 100, 100, 0.02, time.time() - 3600)
        assert rm.is_max_hold_exceeded("AAPL") is False

    @patch.object(Config, "MAX_HOLD_HOURS", 24)
    def test_over_limit(self):
        rm = RiskManager()
        rm.trailing_stops["AAPL"] = TrailingStop("AAPL", 100, 100, 0.02, time.time() - 100000)
        assert rm.is_max_hold_exceeded("AAPL") is True

    @patch.object(Config, "MAX_HOLD_HOURS_CRYPTO", 12)
    def test_crypto_uses_crypto_limit(self):
        rm = RiskManager()
        rm.trailing_stops["BTC/USD"] = TrailingStop("BTC/USD", 50000, 50000, 0.04, time.time() - 50000)
        assert rm.is_max_hold_exceeded("BTC/USD") is True

    @patch.object(Config, "MAX_HOLD_HOURS", 48)
    def test_entry_time_zero(self):
        rm = RiskManager()
        rm.trailing_stops["AAPL"] = TrailingStop("AAPL", 100, 100, 0.02, entry_time=0.0)
        assert rm.is_max_hold_exceeded("AAPL") is False


# -- Register Entry --

class TestRegisterEntry:

    @patch.object(Config, "ATR_STOPS_ENABLED", True)
    def test_atr_stop_with_data(self):
        rm = RiskManager()
        rm.register_entry("AAPL", 100.0, _make_df(n=50))
        stop = rm.trailing_stops["AAPL"]
        assert 0.005 <= stop.stop_pct <= 0.08
        assert stop.entry_time > 0

    @patch.object(Config, "ATR_STOPS_ENABLED", False)
    def test_fallback_crypto(self):
        rm = RiskManager()
        rm.register_entry("BTC/USD", 50000.0)
        assert rm.trailing_stops["BTC/USD"].stop_pct == 0.04

    @patch.object(Config, "ATR_STOPS_ENABLED", False)
    def test_fallback_equity(self):
        rm = RiskManager()
        rm.register_entry("AAPL", 150.0)
        assert rm.trailing_stops["AAPL"].stop_pct == 0.025
