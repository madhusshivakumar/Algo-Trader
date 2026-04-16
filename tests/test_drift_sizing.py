"""Sprint 5E: drift detection wired to position sizing.

When a symbol's recent performance is degraded, the engine cuts the position
size to 30% of normal. The full RL/LLM/signal pipeline still runs.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config import Config
from core.drift_detector import DriftDetector, DriftMetrics


def _make_df(n=100, base_price=100.0):
    rng = np.random.RandomState(42)
    close = np.full(n, base_price) + rng.randn(n) * 0.5
    return pd.DataFrame({
        "open": close + rng.randn(n) * 0.1,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(n, 5_000_000),
    })


class TestIsDegradedHelper:
    def test_returns_false_when_cache_empty(self):
        detector = DriftDetector(db_path=":memory:")
        # No should_alert() call yet — cache is empty
        assert detector.is_degraded("AAPL") is False

    def test_returns_true_when_cached_degraded(self):
        detector = DriftDetector(db_path=":memory:")
        detector._degraded_cache = {
            "AAPL": DriftMetrics(symbol="AAPL", strategy="x",
                                 window_start="", window_end="",
                                 trade_count=10, win_rate=0.3, avg_pnl=-1.0,
                                 total_pnl=-10.0, is_degraded=True),
        }
        assert detector.is_degraded("AAPL") is True
        assert detector.is_degraded("MSFT") is False

    def test_get_recent_metrics_returns_cached(self):
        detector = DriftDetector(db_path=":memory:")
        m = DriftMetrics(symbol="AAPL", strategy="x",
                        window_start="", window_end="",
                        trade_count=10, win_rate=0.3, avg_pnl=-1.0,
                        total_pnl=-10.0, is_degraded=True)
        detector._degraded_cache = {"AAPL": m}
        assert detector.get_recent_metrics("AAPL") is m
        assert detector.get_recent_metrics("XYZ") is None


def _make_engine_with_drift():
    """Build a buy-path engine with a drift detector attached."""
    from core.engine import TradingEngine

    engine = TradingEngine.__new__(TradingEngine)
    engine.broker = MagicMock()
    engine.broker.get_position.return_value = None
    engine.broker.get_latest_quote.return_value = None
    engine.broker.buy.return_value = {"id": "x", "status": "filled"}
    engine.broker.check_buying_power.return_value = 100_000.0
    engine.risk = MagicMock()
    engine.risk.should_stop_loss.return_value = False
    engine.risk.is_max_hold_exceeded.return_value = False
    engine.risk.trailing_stops = {}
    engine.risk.check_correlation.return_value = True
    engine.risk.calculate_volatility_adjusted_size = lambda eq, df, pct: eq * pct
    engine.alert_manager = MagicMock()
    engine.order_manager = None
    engine.execution_manager = None
    engine.portfolio_optimizer = None
    engine.cost_model = None
    engine.cycle_count = 1
    engine._daily_trade_count = {}
    engine._daily_trade_date = ""
    engine._last_trade_time = {}
    engine._equity_buys_today = {}
    engine._alerted_degraded = set()
    engine.drift_detector = DriftDetector(db_path=":memory:")
    return engine


class TestDriftSizingInEngine:
    def test_normal_symbol_full_size(self):
        engine = _make_engine_with_drift()
        # No degraded entry → normal sizing
        signal = {"action": "buy", "strength": 1.0, "reason": "test", "strategy": "mean_rev"}
        with patch("core.engine.route_signals", return_value=signal), \
             patch.object(Config, "VWAP_TWAP_ENABLED", False), \
             patch.object(Config, "VOLATILITY_SIZING_ENABLED", False), \
             patch.object(Config, "CORRELATION_CHECK_ENABLED", False), \
             patch.object(Config, "KELLY_SIZING_ENABLED", False), \
             patch.object(Config, "MEAN_VARIANCE_ENABLED", False), \
             patch.object(Config, "TC_ENABLED", False), \
             patch.object(Config, "MIN_SIGNAL_STRENGTH", 0.3), \
             patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False), \
             patch.object(Config, "LIQUIDITY_GATE_ENABLED", False):
            engine._process_symbol_inner("AAPL", 10_000.0, _make_df())

        # Bought at full size — base_pct (legacy 0.15) × equity × strength
        engine.broker.buy.assert_called_once()
        bought_size = engine.broker.buy.call_args[0][1]
        assert bought_size == pytest.approx(10_000 * 0.15)
        engine.alert_manager.symbol_degraded.assert_not_called()

    def test_degraded_symbol_size_cut_70_percent(self):
        engine = _make_engine_with_drift()
        # Mark AAPL degraded
        engine.drift_detector._degraded_cache = {
            "AAPL": DriftMetrics(symbol="AAPL", strategy="mean_rev",
                                 window_start="", window_end="",
                                 trade_count=10, win_rate=0.2, avg_pnl=-1.5,
                                 total_pnl=-15.0, is_degraded=True),
        }
        signal = {"action": "buy", "strength": 1.0, "reason": "test", "strategy": "mean_rev"}
        with patch("core.engine.route_signals", return_value=signal), \
             patch.object(Config, "VWAP_TWAP_ENABLED", False), \
             patch.object(Config, "VOLATILITY_SIZING_ENABLED", False), \
             patch.object(Config, "CORRELATION_CHECK_ENABLED", False), \
             patch.object(Config, "KELLY_SIZING_ENABLED", False), \
             patch.object(Config, "MEAN_VARIANCE_ENABLED", False), \
             patch.object(Config, "TC_ENABLED", False), \
             patch.object(Config, "MIN_SIGNAL_STRENGTH", 0.3), \
             patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False), \
             patch.object(Config, "LIQUIDITY_GATE_ENABLED", False):
            engine._process_symbol_inner("AAPL", 10_000.0, _make_df())

        # Bought at 30% of normal size: 10_000 × 0.15 × 0.3 × 1.0 strength = 450
        engine.broker.buy.assert_called_once()
        bought_size = engine.broker.buy.call_args[0][1]
        assert bought_size == pytest.approx(10_000 * 0.15 * 0.3)
        # Alert fired exactly once
        engine.alert_manager.symbol_degraded.assert_called_once()

    def test_degraded_alert_fires_only_once(self):
        engine = _make_engine_with_drift()
        engine.drift_detector._degraded_cache = {
            "AAPL": DriftMetrics(symbol="AAPL", strategy="mean_rev",
                                 window_start="", window_end="",
                                 trade_count=10, win_rate=0.2, avg_pnl=-1.5,
                                 total_pnl=-15.0, is_degraded=True),
        }
        signal = {"action": "buy", "strength": 1.0, "reason": "test", "strategy": "mean_rev"}
        with patch("core.engine.route_signals", return_value=signal), \
             patch.object(Config, "VWAP_TWAP_ENABLED", False), \
             patch.object(Config, "VOLATILITY_SIZING_ENABLED", False), \
             patch.object(Config, "CORRELATION_CHECK_ENABLED", False), \
             patch.object(Config, "KELLY_SIZING_ENABLED", False), \
             patch.object(Config, "MEAN_VARIANCE_ENABLED", False), \
             patch.object(Config, "TC_ENABLED", False), \
             patch.object(Config, "MIN_SIGNAL_STRENGTH", 0.3), \
             patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False), \
             patch.object(Config, "LIQUIDITY_GATE_ENABLED", False):
            for _ in range(3):
                engine._process_symbol_inner("AAPL", 10_000.0, _make_df())
        # Alert manager called only on first cycle
        assert engine.alert_manager.symbol_degraded.call_count == 1
