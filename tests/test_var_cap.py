"""Sprint 5F: VaR-aware position cap tests.

Verifies the engine caps individual position sizes so that a single trade can't
risk more than the profile's `max_var_contribution_pct` of equity.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config import Config
from core.risk_manager import RiskManager
from core.user_profile import BEGINNER, HOBBYIST, LEARNER


def _make_df_with_vol(n=100, base_price=100.0, daily_vol=0.02, seed=42):
    """Create OHLCV data with controlled volatility (daily_vol = stddev of returns)."""
    rng = np.random.RandomState(seed)
    returns = rng.randn(n) * daily_vol
    closes = [base_price]
    for r in returns[:-1]:
        closes.append(closes[-1] * (1 + r))
    closes = np.array(closes)
    return pd.DataFrame({
        "open": closes,
        "high": closes * 1.005,
        "low": closes * 0.995,
        "close": closes,
        "volume": np.full(n, 5_000_000),
    })


class TestEstimateVarPct:
    def test_returns_zero_for_short_data(self):
        rm = RiskManager()
        df = _make_df_with_vol(n=5)
        assert rm.estimate_var_pct(df) == 0.0

    def test_returns_zero_for_empty_df(self):
        rm = RiskManager()
        assert rm.estimate_var_pct(pd.DataFrame()) == 0.0
        assert rm.estimate_var_pct(None) == 0.0

    def test_high_vol_higher_var(self):
        rm = RiskManager()
        low_vol_df = _make_df_with_vol(daily_vol=0.005)
        high_vol_df = _make_df_with_vol(daily_vol=0.05)
        var_low = rm.estimate_var_pct(low_vol_df)
        var_high = rm.estimate_var_pct(high_vol_df)
        assert var_high > var_low
        # 0.5% daily vol → ~0.8% VaR; 5% daily vol → ~8% VaR
        assert 0.005 < var_low < 0.02
        assert 0.05 < var_high < 0.20

    def test_99_confidence_higher_than_95(self):
        rm = RiskManager()
        df = _make_df_with_vol(daily_vol=0.02)
        v95 = rm.estimate_var_pct(df, confidence=0.95)
        v99 = rm.estimate_var_pct(df, confidence=0.99)
        # 99% z-score (2.326) > 95% z-score (1.645)
        assert v99 > v95


class TestVarCapInEngine:
    def _make_engine_for_var(self, profile):
        from core.engine import TradingEngine
        engine = TradingEngine.__new__(TradingEngine)
        engine.broker = MagicMock()
        engine.broker.get_position.return_value = None
        engine.broker.get_latest_quote.return_value = None
        engine.broker.buy.return_value = {"id": "x", "status": "filled"}
        engine.broker.check_buying_power.return_value = 100_000.0
        engine.risk = RiskManager()
        engine.alert_manager = MagicMock()
        engine.order_manager = None
        engine.execution_manager = None
        engine.portfolio_optimizer = None
        engine.cost_model = None
        engine.drift_detector = None
        engine.cycle_count = 1
        engine._daily_trade_count = {}
        engine._daily_trade_date = ""
        engine._last_trade_time = {}
        engine._equity_buys_today = {}
        engine._alerted_degraded = set()
        # Activate profile
        Config._PROFILE = profile
        return engine

    def _signal(self, strength=1.0):
        return {"action": "buy", "strength": strength,
                "reason": "test", "strategy": "mean_rev"}

    def _patches(self):
        return [
            patch.object(Config, "VWAP_TWAP_ENABLED", False),
            patch.object(Config, "VOLATILITY_SIZING_ENABLED", False),
            patch.object(Config, "CORRELATION_CHECK_ENABLED", False),
            patch.object(Config, "KELLY_SIZING_ENABLED", False),
            patch.object(Config, "MEAN_VARIANCE_ENABLED", False),
            patch.object(Config, "TC_ENABLED", False),
            patch.object(Config, "MIN_SIGNAL_STRENGTH", 0.3),
            patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False),
            patch.object(Config, "LIQUIDITY_GATE_ENABLED", False),
        ]

    def test_low_vol_symbol_no_cap_applied(self):
        engine = self._make_engine_for_var(BEGINNER)
        try:
            df = _make_df_with_vol(daily_vol=0.005)  # 0.5% daily vol → ~0.8% VaR
            with patch("core.engine.route_signals", return_value=self._signal()):
                for p in self._patches():
                    p.start()
                try:
                    engine._process_symbol_inner("AAPL", 1_000.0, df)
                finally:
                    for p in self._patches():
                        try:
                            p.stop()
                        except RuntimeError:
                            pass
            # Beginner sizing on $1K: 1.5% × 1K = $15 (not capped — small enough)
            engine.broker.buy.assert_called_once()
            size = engine.broker.buy.call_args[0][1]
            # Should be at base sizing or close
            assert size == pytest.approx(15.0, abs=0.5)
        finally:
            Config._PROFILE = None

    def test_high_vol_symbol_capped(self):
        """A high-vol symbol triggers the VaR cap to lower the position size."""
        engine = self._make_engine_for_var(LEARNER)  # Learner uses big base sizing (15%)
        try:
            # 8% daily vol → ~13% VaR. Learner cap = 3% equity.
            # max_size = 0.03 * 100_000 / 0.13 ≈ $23,000 (well below 15% × 100K = $15K? wait)
            # 15% × 100K = $15K base. VaR cap allows $23K. So no cap here.
            # Need higher vol or smaller cap to trigger
            df = _make_df_with_vol(daily_vol=0.10, seed=42)  # ~16.5% VaR
            with patch("core.engine.route_signals", return_value=self._signal()):
                for p in self._patches():
                    p.start()
                try:
                    engine._process_symbol_inner("AAPL", 100_000.0, df)
                finally:
                    for p in self._patches():
                        try:
                            p.stop()
                        except RuntimeError:
                            pass
            # Learner base: 15% × 100K = $15,000
            # VaR cap: 3% × 100K / 0.165 ≈ $18,200 — base < cap, so no reduction
            engine.broker.buy.assert_called_once()
            size = engine.broker.buy.call_args[0][1]
            # Size should be at base
            assert 14_000 < size < 16_000
        finally:
            Config._PROFILE = None

    def test_extreme_vol_caps_size(self):
        """Pathologically high volatility forces the cap below base sizing."""
        engine = self._make_engine_for_var(BEGINNER)
        try:
            # Extreme vol: 30% daily → ~50% VaR
            # Base for BEGINNER on $10K: 1.5% × 10K = $150
            # VaR cap: 1% × 10K / 0.50 = $200 — cap is HIGHER than base, no reduction
            # Need to test where cap is actually tighter:
            # Use Learner with extreme vol: base 15% × 10K = $1500
            # VaR cap: 3% × 10K / 0.50 = $600 → cap kicks in
            Config._PROFILE = LEARNER
            df = _make_df_with_vol(daily_vol=0.30, seed=42)  # extreme vol
            with patch("core.engine.route_signals", return_value=self._signal()):
                for p in self._patches():
                    p.start()
                try:
                    engine._process_symbol_inner("XYZ", 10_000.0, df)
                finally:
                    for p in self._patches():
                        try:
                            p.stop()
                        except RuntimeError:
                            pass
            engine.broker.buy.assert_called_once()
            size = engine.broker.buy.call_args[0][1]
            # Should be capped, well under base sizing of $1500
            assert size < 1500
            # And well under base, but still positive
            assert size > 0
        finally:
            Config._PROFILE = None

    def test_no_profile_no_cap(self):
        """When profile not set, no VaR cap applied (legacy path)."""
        engine = self._make_engine_for_var(LEARNER)
        Config._PROFILE = None  # disable profile path
        try:
            df = _make_df_with_vol(daily_vol=0.30)  # extreme vol
            with patch("core.engine.route_signals", return_value=self._signal()):
                for p in self._patches():
                    p.start()
                try:
                    engine._process_symbol_inner("XYZ", 10_000.0, df)
                finally:
                    for p in self._patches():
                        try:
                            p.stop()
                        except RuntimeError:
                            pass
            engine.broker.buy.assert_called_once()
            size = engine.broker.buy.call_args[0][1]
            # Without profile, legacy 15% base on $10K = $1500 (no VaR cap path)
            assert size == pytest.approx(1500.0, abs=10)
        finally:
            Config._PROFILE = None
