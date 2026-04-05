"""Sprint 4 — additional engine.py tests for coverage gaps."""

import math
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

from core.engine import TradingEngine
from core.risk_manager import RiskManager, TrailingStop
from config import Config


def _make_engine(**overrides):
    """Create a bare TradingEngine without __init__."""
    engine = TradingEngine.__new__(TradingEngine)
    engine._last_trade_time = {}
    engine._daily_trade_count = {}
    engine._daily_trade_date = ""
    engine._equity_buys_today = {}
    engine.cycle_count = 0
    engine.risk = RiskManager()
    engine.broker = MagicMock()
    engine.state_store = None
    engine.order_manager = None
    engine.data_fetcher = None
    engine.config_reloader = None
    engine.drift_detector = None
    engine.position_reconciler = None
    engine.cost_model = None
    engine.db_rotator = None
    engine.execution_manager = None
    engine.portfolio_optimizer = None
    engine.alert_manager = None
    engine._shutdown_requested = False
    engine._reload_requested = False
    for k, v in overrides.items():
        setattr(engine, k, v)
    return engine


def _make_df(n=50):
    """Create a simple OHLCV DataFrame."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


# -- Daily Trade Limit --

class TestDailyTradeLimitSprint4:

    def test_fresh_engine_no_limit(self):
        engine = _make_engine()
        assert engine._is_daily_trade_limit_reached("AAPL") is False
        assert engine._is_daily_trade_limit_reached("BTC/USD") is False

    @patch.object(Config, "MAX_TRADES_PER_DAY_EQUITY", 1)
    def test_equity_limit_at_one(self):
        engine = _make_engine()
        engine._record_trade("AAPL")
        assert engine._is_daily_trade_limit_reached("NVDA") is True

    @patch.object(Config, "MAX_TRADES_PER_DAY_CRYPTO", 2)
    def test_crypto_limit_at_two(self):
        engine = _make_engine()
        engine._record_trade("BTC/USD")
        assert engine._is_daily_trade_limit_reached("ETH/USD") is False
        engine._record_trade("ETH/USD")
        assert engine._is_daily_trade_limit_reached("BTC/USD") is True

    @patch.object(Config, "MAX_TRADES_PER_DAY_EQUITY", 1)
    def test_equity_limit_doesnt_block_crypto(self):
        engine = _make_engine()
        engine._record_trade("AAPL")
        assert engine._is_daily_trade_limit_reached("AAPL") is True
        assert engine._is_daily_trade_limit_reached("BTC/USD") is False

    def test_date_rollover_resets_count(self):
        engine = _make_engine()
        engine._record_trade("AAPL")
        assert engine._daily_trade_count.get("equity") == 1
        engine._daily_trade_date = "2020-01-01"
        engine._record_trade("AAPL")
        assert engine._daily_trade_count.get("equity") == 1  # reset + 1

    def test_record_trade_tracks_both_classes(self):
        engine = _make_engine()
        engine._record_trade("AAPL")
        engine._record_trade("BTC/USD")
        engine._record_trade("NVDA")
        assert engine._daily_trade_count["equity"] == 2
        assert engine._daily_trade_count["crypto"] == 1


# -- Orphan Detection --

class TestOrphanDetection:

    def test_detect_registers_missing_stops(self):
        engine = _make_engine()
        engine.broker.get_positions.return_value = [
            {"symbol": "AAPL", "avg_entry_price": "150.0",
             "current_price": "155.0", "market_value": "1550.0"},
        ]
        engine.broker.get_recent_bars.return_value = _make_df()
        engine._detect_orphan_positions()
        assert "AAPL" in engine.risk.trailing_stops

    def test_detect_skips_already_tracked(self):
        engine = _make_engine()
        existing_stop = TrailingStop("AAPL", 150.0, 155.0, 0.02, time.time())
        engine.risk.trailing_stops["AAPL"] = existing_stop
        engine.broker.get_positions.return_value = [
            {"symbol": "AAPL", "avg_entry_price": "150.0",
             "current_price": "155.0", "market_value": "1550.0"},
        ]
        engine._detect_orphan_positions()
        assert engine.risk.trailing_stops["AAPL"] is existing_stop

    def test_resolve_symbol_crypto(self):
        engine = _make_engine()
        assert engine._resolve_symbol("BTCUSD") == "BTC/USD"
        assert engine._resolve_symbol("ETHUSD") == "ETH/USD"

    def test_resolve_symbol_equity(self):
        engine = _make_engine()
        assert engine._resolve_symbol("AAPL") == "AAPL"

    def test_detect_handles_broker_failure(self):
        engine = _make_engine()
        engine.broker.get_positions.side_effect = Exception("API down")
        engine._detect_orphan_positions()  # should not raise

    def test_orphan_gets_higher_price_updated(self):
        engine = _make_engine()
        engine.broker.get_positions.return_value = [
            {"symbol": "AAPL", "avg_entry_price": "100.0",
             "current_price": "120.0", "market_value": "1200.0"},
        ]
        engine.broker.get_recent_bars.return_value = _make_df()
        engine._detect_orphan_positions()
        stop = engine.risk.trailing_stops["AAPL"]
        assert stop.highest_price == 120.0


# -- Cooldown --

class TestCooldownSprint4:

    def test_crypto_cooldown_is_45_min(self):
        assert TradingEngine.COOLDOWN_CRYPTO == 2700

    def test_within_crypto_cooldown(self):
        engine = _make_engine()
        engine._last_trade_time = {"BTC/USD": time.time() - 1800}
        engine.cycle_count = 1
        assert engine._is_on_cooldown("BTC/USD") is True

    def test_past_crypto_cooldown(self):
        engine = _make_engine()
        engine._last_trade_time = {"BTC/USD": time.time() - 3000}
        engine.cycle_count = 1
        assert engine._is_on_cooldown("BTC/USD") is False


# -- Halt / Resume --

class TestHaltResume:

    def test_drawdown_halts_trading(self):
        engine = _make_engine()
        engine.risk.initialize(100000)
        engine.risk.daily_start_equity = 100000
        assert engine.risk.check_drawdown(85000) is True
        assert engine.risk.halted is True

    def test_halted_blocks_can_trade(self):
        engine = _make_engine()
        engine.risk.initialize(100000)
        engine.risk.halted = True
        assert engine.risk.can_trade(100000) is False

    def test_daily_reset_clears_halt(self):
        engine = _make_engine()
        engine.risk.initialize(100000)
        engine.risk.halted = True
        engine.risk.halt_reason = "test halt"
        engine.risk._last_daily_reset = time.time() - 86400 * 2
        engine.risk.check_daily_reset(99000)
        assert engine.risk.halted is False
        assert engine.risk.halt_reason == ""
