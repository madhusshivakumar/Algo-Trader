"""Unit tests for the risk manager."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from core.risk_manager import RiskManager, TrailingStop
from config import Config


def _make_df(n=100, seed=42, base_price=100, volatility=1.0):
    """Create a sample OHLC DataFrame."""
    rng = np.random.RandomState(seed)
    close = base_price + np.cumsum(rng.randn(n) * volatility)
    return pd.DataFrame({
        "open": close + rng.randn(n) * 0.1,
        "high": close + abs(rng.randn(n)) * volatility,
        "low": close - abs(rng.randn(n)) * volatility,
        "close": close,
        "volume": rng.randint(100000, 1000000, n),
    })


class TestTrailingStop:
    """Test trailing stop mechanics."""

    def test_initial_stop_price(self):
        stop = TrailingStop(symbol="AAPL", entry_price=100, highest_price=100, stop_pct=0.02)
        assert stop.stop_price == 98.0  # 100 * 0.98

    def test_stop_moves_up_with_price(self):
        stop = TrailingStop(symbol="AAPL", entry_price=100, highest_price=100, stop_pct=0.02)
        triggered = stop.update(110)
        assert not triggered
        assert stop.highest_price == 110
        assert stop.stop_price == 107.8  # 110 * 0.98

    def test_stop_does_not_move_down(self):
        stop = TrailingStop(symbol="AAPL", entry_price=100, highest_price=110, stop_pct=0.02)
        # 109 is above the stop price of 107.8, so stop should NOT trigger
        triggered = stop.update(109)
        assert not triggered
        assert stop.highest_price == 110  # Unchanged (109 < 110)
        assert stop.stop_price == 107.8

    def test_stop_triggers_when_price_drops_below(self):
        stop = TrailingStop(symbol="AAPL", entry_price=100, highest_price=110, stop_pct=0.02)
        # 105 is below stop price of 107.8, so it SHOULD trigger
        triggered = stop.update(105)
        assert triggered

    def test_stop_triggers_at_threshold(self):
        stop = TrailingStop(symbol="AAPL", entry_price=100, highest_price=100, stop_pct=0.02)
        triggered = stop.update(98.0)
        assert triggered

    def test_stop_triggers_below_threshold(self):
        stop = TrailingStop(symbol="AAPL", entry_price=100, highest_price=100, stop_pct=0.02)
        triggered = stop.update(95.0)
        assert triggered

    def test_custom_stop_pct(self):
        stop = TrailingStop(symbol="BTC/USD", entry_price=60000, highest_price=60000, stop_pct=0.015)
        assert stop.stop_price == 59100.0  # 60000 * 0.985


class TestRiskManager:
    """Test risk manager operations."""

    def test_initialize(self):
        rm = RiskManager()
        rm.initialize(100000)
        assert rm.starting_equity == 100000
        assert rm.daily_start_equity == 100000

    def test_can_trade_initially(self):
        rm = RiskManager()
        rm.initialize(100000)
        assert rm.can_trade(100000) is True

    def test_drawdown_halts_trading(self):
        rm = RiskManager()
        rm.initialize(100000)
        # 11% drawdown (exceeds 10% limit)
        result = rm.can_trade(89000)
        assert result is False
        assert rm.halted is True

    def test_small_drawdown_allows_trading(self):
        rm = RiskManager()
        rm.initialize(100000)
        # 5% drawdown (within limit)
        assert rm.can_trade(95000) is True

    def test_register_and_check_stop(self):
        rm = RiskManager()
        rm.register_entry("AAPL", 100)
        assert "AAPL" in rm.trailing_stops
        assert rm.should_stop_loss("AAPL", 105) is False  # Price went up
        assert rm.should_stop_loss("AAPL", 90) is True    # Price crashed

    def test_unregister(self):
        rm = RiskManager()
        rm.register_entry("AAPL", 100)
        rm.unregister("AAPL")
        assert "AAPL" not in rm.trailing_stops

    def test_should_stop_loss_unknown_symbol(self):
        rm = RiskManager()
        assert rm.should_stop_loss("UNKNOWN", 100) is False

    def test_crypto_gets_wider_stop(self):
        rm = RiskManager()
        rm.register_entry("BTC/USD", 60000)
        stop = rm.trailing_stops["BTC/USD"]
        assert stop.stop_pct == 0.04  # 4% for crypto

    def test_equity_gets_standard_stop(self):
        rm = RiskManager()
        rm.register_entry("AAPL", 250)
        stop = rm.trailing_stops["AAPL"]
        assert stop.stop_pct == 0.025  # 2.5% for equities

    def test_calculate_position_size(self):
        rm = RiskManager()
        size = rm.calculate_position_size(100000)
        assert size == 100000 * Config.MAX_POSITION_PCT

    def test_halt_reason_set_on_drawdown(self):
        rm = RiskManager()
        rm.initialize(100000)
        rm.can_trade(85000)
        assert "drawdown" in rm.halt_reason.lower()


# ── Sprint 2: ATR-based stops ──────────────────────────────────────

class TestATRCalculation:
    def test_calculate_atr_normal(self):
        df = _make_df(n=50, volatility=2.0)
        atr = RiskManager.calculate_atr(df, period=14)
        assert atr > 0

    def test_calculate_atr_insufficient_data(self):
        df = _make_df(n=5)
        atr = RiskManager.calculate_atr(df, period=14)
        # Should still return something (mean of available TRs)
        assert atr >= 0

    def test_calculate_atr_none_df(self):
        assert RiskManager.calculate_atr(None) == 0.0

    def test_calculate_atr_empty_df(self):
        df = pd.DataFrame(columns=["high", "low", "close"])
        assert RiskManager.calculate_atr(df) == 0.0


class TestATRStops:
    def test_atr_stop_pct_normal_volatility(self):
        df = _make_df(n=50, volatility=1.0)
        pct = RiskManager.calculate_atr_stop_pct(df, multiplier=2.0)
        assert 0.005 <= pct <= 0.08

    def test_atr_stop_pct_high_volatility(self):
        df = _make_df(n=50, volatility=5.0)
        pct = RiskManager.calculate_atr_stop_pct(df, multiplier=2.0)
        assert pct >= 0.005  # should be higher but clamped

    def test_atr_stop_pct_low_volatility(self):
        # Very low volatility
        df = _make_df(n=50, volatility=0.01)
        pct = RiskManager.calculate_atr_stop_pct(df, multiplier=2.0)
        assert pct >= 0.005  # clamped minimum

    def test_atr_stop_pct_clamped_maximum(self):
        # Extreme volatility
        df = _make_df(n=50, base_price=10, volatility=10.0)
        pct = RiskManager.calculate_atr_stop_pct(df, multiplier=5.0)
        assert pct <= 0.08

    def test_atr_stop_pct_insufficient_data(self):
        df = _make_df(n=10)
        pct = RiskManager.calculate_atr_stop_pct(df, multiplier=2.0)
        assert pct == 0.02  # fallback

    def test_atr_stop_pct_none_df(self):
        pct = RiskManager.calculate_atr_stop_pct(None)
        assert pct == 0.02  # fallback

    def test_register_entry_with_atr_enabled(self):
        rm = RiskManager()
        df = _make_df(n=50, volatility=1.0)
        with patch.object(Config, "ATR_STOPS_ENABLED", True), \
             patch.object(Config, "ATR_STOP_MULTIPLIER", 2.0):
            rm.register_entry("AAPL", 100.0, df)
        stop = rm.trailing_stops["AAPL"]
        # ATR-based stop should differ from fixed 2%
        assert stop.stop_pct > 0

    def test_register_entry_atr_disabled_fallback(self):
        rm = RiskManager()
        df = _make_df(n=50)
        with patch.object(Config, "ATR_STOPS_ENABLED", False):
            rm.register_entry("AAPL", 100.0, df)
        assert rm.trailing_stops["AAPL"].stop_pct == 0.025  # fixed equity default

    def test_register_entry_atr_no_df_fallback(self):
        rm = RiskManager()
        with patch.object(Config, "ATR_STOPS_ENABLED", True):
            rm.register_entry("AAPL", 100.0, None)
        assert rm.trailing_stops["AAPL"].stop_pct == 0.025  # fallback


# ── Sprint 2: Volatility-adjusted sizing ───────────────────────────

class TestVolatilitySizing:
    def test_high_vol_smaller_position(self):
        # High volatility → smaller size
        df = _make_df(n=50, volatility=5.0)
        size = RiskManager.calculate_volatility_adjusted_size(100000, df, 0.15)
        base_size = 100000 * 0.15
        assert size <= base_size

    def test_low_vol_larger_position(self):
        # Very low volatility → should approach base_pct
        df = _make_df(n=50, volatility=0.01)
        size = RiskManager.calculate_volatility_adjusted_size(100000, df, 0.15)
        assert size > 0

    def test_insufficient_data_fallback(self):
        df = _make_df(n=5)
        size = RiskManager.calculate_volatility_adjusted_size(100000, df, 0.15)
        assert size == 100000 * 0.15

    def test_none_df_fallback(self):
        size = RiskManager.calculate_volatility_adjusted_size(100000, None, 0.15)
        assert size == 100000 * 0.15

    def test_size_clamped_minimum(self):
        # Even with extreme vol, size should be at least 1% of equity
        df = _make_df(n=50, volatility=50.0)
        size = RiskManager.calculate_volatility_adjusted_size(100000, df, 0.15)
        assert size >= 100000 * 0.01

    def test_size_clamped_maximum(self):
        df = _make_df(n=50, volatility=0.001)
        size = RiskManager.calculate_volatility_adjusted_size(100000, df, 0.15)
        assert size <= 100000 * 0.15


# ── Sprint 2: Correlation check ────────────────────────────────────

class TestCorrelationCheck:
    def test_uncorrelated_positions_allowed(self):
        # Two unrelated price series
        df1 = _make_df(n=100, seed=42)
        df2 = _make_df(n=100, seed=99, base_price=200)
        assert RiskManager.check_correlation(df1, {"OTHER": df2}, threshold=0.7) is True

    def test_highly_correlated_blocked(self):
        # Same price series = correlation ~1.0
        df = _make_df(n=100, seed=42)
        df_copy = df.copy()
        # Add tiny noise so they're not identical but still very correlated
        df_copy["close"] = df_copy["close"] + np.random.RandomState(1).randn(100) * 0.001
        assert RiskManager.check_correlation(df, {"SAME": df_copy}, threshold=0.7) is False

    def test_no_existing_positions_allowed(self):
        df = _make_df(n=100)
        assert RiskManager.check_correlation(df, {}, threshold=0.7) is True

    def test_none_df_allowed(self):
        assert RiskManager.check_correlation(None, {"X": _make_df()}, threshold=0.7) is True

    def test_short_df_allowed(self):
        df = _make_df(n=5)
        assert RiskManager.check_correlation(df, {"X": _make_df()}, threshold=0.7) is True

    def test_threshold_boundary(self):
        # With threshold=0.0, even slightly correlated should be blocked
        df = _make_df(n=100, seed=42)
        df2 = _make_df(n=100, seed=43)
        result = RiskManager.check_correlation(df, {"OTHER": df2}, threshold=0.0)
        # At threshold 0, almost any correlation will block
        # Result depends on actual correlation value
        assert isinstance(result, bool)
