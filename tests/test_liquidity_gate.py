"""Sprint 5C: pre-trade liquidity gate tests.

The engine should skip a buy when the bid-ask spread exceeds the configured
limit. Used to protect low-capital accounts from paying a fat spread on thinly
traded symbols.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config import Config
from core.engine import TradingEngine


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


def _make_engine_for_buy():
    """Build a minimal engine whose buy-path reaches the liquidity gate."""
    engine = TradingEngine.__new__(TradingEngine)
    engine.broker = MagicMock()
    engine.broker.get_position.return_value = None  # no existing position
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
    engine.portfolio_optimizer = None
    engine.cost_model = None
    engine.execution_manager = None
    engine.drift_detector = None
    engine.cycle_count = 1
    engine._daily_trade_count = {}
    engine._daily_trade_date = ""
    engine._last_trade_time = {}
    engine._equity_buys_today = {}
    return engine


class TestLiquidityGate:
    def test_tight_spread_allows_buy(self):
        engine = _make_engine_for_buy()
        engine.broker.get_latest_quote.return_value = {
            "bid": 99.98, "ask": 100.02, "mid": 100.0, "spread_bps": 4.0, "symbol": "AAPL",
        }
        signal = {"action": "buy", "strength": 0.8, "reason": "test", "strategy": "mean_rev"}
        with patch("core.engine.route_signals", return_value=signal), \
             patch.object(Config, "LIQUIDITY_GATE_ENABLED", True), \
             patch.object(Config, "MAX_SPREAD_BPS_EQUITY", 50.0), \
             patch.object(Config, "VWAP_TWAP_ENABLED", False), \
             patch.object(Config, "VOLATILITY_SIZING_ENABLED", False), \
             patch.object(Config, "CORRELATION_CHECK_ENABLED", False), \
             patch.object(Config, "KELLY_SIZING_ENABLED", False), \
             patch.object(Config, "MEAN_VARIANCE_ENABLED", False), \
             patch.object(Config, "TC_ENABLED", False), \
             patch.object(Config, "MIN_SIGNAL_STRENGTH", 0.3), \
             patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False):
            engine._process_symbol_inner("AAPL", 10_000.0, _make_df())
        engine.broker.buy.assert_called_once()
        # No liquidity-skip alert
        engine.alert_manager.liquidity_skip.assert_not_called()

    def test_wide_equity_spread_blocks_buy(self):
        engine = _make_engine_for_buy()
        # 80 bps spread — above 50 bps equity limit
        engine.broker.get_latest_quote.return_value = {
            "bid": 99.6, "ask": 100.4, "mid": 100.0, "spread_bps": 80.0, "symbol": "AAPL",
        }
        signal = {"action": "buy", "strength": 0.8, "reason": "test", "strategy": "mean_rev"}
        with patch("core.engine.route_signals", return_value=signal), \
             patch.object(Config, "LIQUIDITY_GATE_ENABLED", True), \
             patch.object(Config, "MAX_SPREAD_BPS_EQUITY", 50.0), \
             patch.object(Config, "VWAP_TWAP_ENABLED", False), \
             patch.object(Config, "VOLATILITY_SIZING_ENABLED", False), \
             patch.object(Config, "CORRELATION_CHECK_ENABLED", False), \
             patch.object(Config, "KELLY_SIZING_ENABLED", False), \
             patch.object(Config, "MEAN_VARIANCE_ENABLED", False), \
             patch.object(Config, "TC_ENABLED", False), \
             patch.object(Config, "MIN_SIGNAL_STRENGTH", 0.3), \
             patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False):
            engine._process_symbol_inner("AAPL", 10_000.0, _make_df())
        engine.broker.buy.assert_not_called()
        engine.alert_manager.liquidity_skip.assert_called_once()

    def test_crypto_uses_wider_limit(self):
        engine = _make_engine_for_buy()
        # 80 bps — above equity limit (50) but UNDER crypto limit (100)
        engine.broker.get_latest_quote.return_value = {
            "bid": 49_900, "ask": 50_100, "mid": 50_000, "spread_bps": 80.0, "symbol": "BTC/USD",
        }
        signal = {"action": "buy", "strength": 0.8, "reason": "test", "strategy": "mean_rev"}
        with patch("core.engine.route_signals", return_value=signal), \
             patch.object(Config, "LIQUIDITY_GATE_ENABLED", True), \
             patch.object(Config, "MAX_SPREAD_BPS_CRYPTO", 100.0), \
             patch.object(Config, "VWAP_TWAP_ENABLED", False), \
             patch.object(Config, "VOLATILITY_SIZING_ENABLED", False), \
             patch.object(Config, "CORRELATION_CHECK_ENABLED", False), \
             patch.object(Config, "KELLY_SIZING_ENABLED", False), \
             patch.object(Config, "MEAN_VARIANCE_ENABLED", False), \
             patch.object(Config, "TC_ENABLED", False), \
             patch.object(Config, "MIN_SIGNAL_STRENGTH", 0.3), \
             patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False):
            engine._process_symbol_inner("BTC/USD", 10_000.0, _make_df(base_price=50_000))
        # Crypto allowed through at 80 bps (under 100 bps crypto limit)
        engine.broker.buy.assert_called_once()

    def test_gate_disabled_allows_wide_spread(self):
        engine = _make_engine_for_buy()
        engine.broker.get_latest_quote.return_value = {
            "bid": 99, "ask": 101, "mid": 100.0, "spread_bps": 200.0, "symbol": "AAPL",
        }
        signal = {"action": "buy", "strength": 0.8, "reason": "test", "strategy": "mean_rev"}
        with patch("core.engine.route_signals", return_value=signal), \
             patch.object(Config, "LIQUIDITY_GATE_ENABLED", False), \
             patch.object(Config, "VWAP_TWAP_ENABLED", False), \
             patch.object(Config, "VOLATILITY_SIZING_ENABLED", False), \
             patch.object(Config, "CORRELATION_CHECK_ENABLED", False), \
             patch.object(Config, "KELLY_SIZING_ENABLED", False), \
             patch.object(Config, "MEAN_VARIANCE_ENABLED", False), \
             patch.object(Config, "TC_ENABLED", False), \
             patch.object(Config, "MIN_SIGNAL_STRENGTH", 0.3), \
             patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False):
            engine._process_symbol_inner("AAPL", 10_000.0, _make_df())
        engine.broker.buy.assert_called_once()  # gate off → any spread ok

    def test_quote_unavailable_allows_buy(self):
        """When the broker can't produce a quote, we proceed (graceful fallback)."""
        engine = _make_engine_for_buy()
        engine.broker.get_latest_quote.return_value = None
        signal = {"action": "buy", "strength": 0.8, "reason": "test", "strategy": "mean_rev"}
        with patch("core.engine.route_signals", return_value=signal), \
             patch.object(Config, "LIQUIDITY_GATE_ENABLED", True), \
             patch.object(Config, "VWAP_TWAP_ENABLED", False), \
             patch.object(Config, "VOLATILITY_SIZING_ENABLED", False), \
             patch.object(Config, "CORRELATION_CHECK_ENABLED", False), \
             patch.object(Config, "KELLY_SIZING_ENABLED", False), \
             patch.object(Config, "MEAN_VARIANCE_ENABLED", False), \
             patch.object(Config, "TC_ENABLED", False), \
             patch.object(Config, "MIN_SIGNAL_STRENGTH", 0.3), \
             patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False):
            engine._process_symbol_inner("AAPL", 10_000.0, _make_df())
        engine.broker.buy.assert_called_once()

    def test_quote_fetch_exception_allows_buy(self):
        """Don't let quote fetch errors block the entire buy pipeline."""
        engine = _make_engine_for_buy()
        engine.broker.get_latest_quote.side_effect = Exception("API timeout")
        signal = {"action": "buy", "strength": 0.8, "reason": "test", "strategy": "mean_rev"}
        with patch("core.engine.route_signals", return_value=signal), \
             patch.object(Config, "LIQUIDITY_GATE_ENABLED", True), \
             patch.object(Config, "VWAP_TWAP_ENABLED", False), \
             patch.object(Config, "VOLATILITY_SIZING_ENABLED", False), \
             patch.object(Config, "CORRELATION_CHECK_ENABLED", False), \
             patch.object(Config, "KELLY_SIZING_ENABLED", False), \
             patch.object(Config, "MEAN_VARIANCE_ENABLED", False), \
             patch.object(Config, "TC_ENABLED", False), \
             patch.object(Config, "MIN_SIGNAL_STRENGTH", 0.3), \
             patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False):
            engine._process_symbol_inner("AAPL", 10_000.0, _make_df())
        engine.broker.buy.assert_called_once()


class TestBrokerGetLatestQuote:
    def test_returns_dict_with_spread(self):
        from core.broker import Broker
        broker = Broker.__new__(Broker)
        broker.stock_data = MagicMock()

        fake_quote = MagicMock()
        fake_quote.bid_price = 99.95
        fake_quote.ask_price = 100.05
        broker.stock_data.get_stock_latest_quote.return_value = {"AAPL": fake_quote}

        result = broker.get_latest_quote("AAPL")
        assert result is not None
        assert result["bid"] == pytest.approx(99.95)
        assert result["ask"] == pytest.approx(100.05)
        assert result["mid"] == pytest.approx(100.0)
        assert result["spread_bps"] == pytest.approx(10.0, abs=0.1)

    def test_returns_none_on_missing_quote(self):
        from core.broker import Broker
        broker = Broker.__new__(Broker)
        broker.stock_data = MagicMock()
        broker.stock_data.get_stock_latest_quote.return_value = {}
        assert broker.get_latest_quote("AAPL") is None

    def test_returns_none_on_zero_prices(self):
        from core.broker import Broker
        broker = Broker.__new__(Broker)
        broker.stock_data = MagicMock()
        fake_quote = MagicMock()
        fake_quote.bid_price = 0
        fake_quote.ask_price = 0
        broker.stock_data.get_stock_latest_quote.return_value = {"AAPL": fake_quote}
        assert broker.get_latest_quote("AAPL") is None

    def test_returns_none_on_inverted_spread(self):
        from core.broker import Broker
        broker = Broker.__new__(Broker)
        broker.stock_data = MagicMock()
        fake_quote = MagicMock()
        fake_quote.bid_price = 100.5
        fake_quote.ask_price = 100.0  # ask < bid (bad data)
        broker.stock_data.get_stock_latest_quote.return_value = {"AAPL": fake_quote}
        assert broker.get_latest_quote("AAPL") is None

    def test_returns_none_on_exception(self):
        from core.broker import Broker
        broker = Broker.__new__(Broker)
        broker.stock_data = MagicMock()
        broker.stock_data.get_stock_latest_quote.side_effect = Exception("API down")
        assert broker.get_latest_quote("AAPL") is None
