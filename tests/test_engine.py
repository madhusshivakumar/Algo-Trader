"""Unit tests for the trading engine (logic only, no broker calls)."""

import time
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from core.engine import TradingEngine
from config import Config


class TestEngineCooldown:
    """Test the cooldown mechanism."""

    def test_no_cooldown_initially(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine._last_trade_time = {}
        engine.cycle_count = 1
        assert engine._is_on_cooldown("AAPL") is False

    def test_cooldown_after_trade(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine._last_trade_time = {"AAPL": time.time()}
        engine.cycle_count = 1
        assert engine._is_on_cooldown("AAPL") is True

    def test_cooldown_expires(self):
        engine = TradingEngine.__new__(TradingEngine)
        # Set trade time to 20 min ago (equity cooldown is 5 min)
        engine._last_trade_time = {"AAPL": time.time() - 1200}
        engine.cycle_count = 1
        assert engine._is_on_cooldown("AAPL") is False

    def test_crypto_cooldown_longer(self):
        engine = TradingEngine.__new__(TradingEngine)
        # 10 min ago — within crypto cooldown (15 min) but past equity cooldown (5 min)
        engine._last_trade_time = {"BTC/USD": time.time() - 600}
        engine.cycle_count = 1
        assert engine._is_on_cooldown("BTC/USD") is True

    def test_record_trade(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine._last_trade_time = {}
        engine._daily_trade_count = {}
        engine._daily_trade_date = ""
        engine._record_trade("AAPL")
        assert "AAPL" in engine._last_trade_time
        assert time.time() - engine._last_trade_time["AAPL"] < 1
        assert engine._daily_trade_count.get("equity", 0) == 1


class TestEnginePDT:
    """Test PDT protection logic."""

    def test_record_and_check_same_day(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine._equity_buys_today = {}
        engine._record_buy("AAPL")
        assert engine._is_same_day_buy("AAPL") is True

    def test_no_buy_recorded(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine._equity_buys_today = {}
        assert engine._is_same_day_buy("AAPL") is False

    def test_clear_buy_record(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine._equity_buys_today = {"AAPL": datetime.now(Config.MARKET_TZ)}
        engine._clear_buy_record("AAPL")
        assert "AAPL" not in engine._equity_buys_today

    @patch.object(Config, "PDT_PROTECTION", False)
    def test_pdt_disabled(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine._equity_buys_today = {"AAPL": datetime.now(Config.MARKET_TZ)}
        assert engine._is_same_day_buy("AAPL") is False


class TestEngineConstants:
    """Test engine class-level constants."""

    def test_cooldown_values(self):
        assert TradingEngine.COOLDOWN_CRYPTO == 2700  # 45 min
        assert TradingEngine.COOLDOWN_EQUITY == 300   # 5 min

    def test_crypto_cooldown_longer_than_equity(self):
        assert TradingEngine.COOLDOWN_CRYPTO > TradingEngine.COOLDOWN_EQUITY


class TestDailyTradeLimit:
    """Test daily trade limit enforcement."""

    def _make_engine(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine._last_trade_time = {}
        engine._daily_trade_count = {}
        engine._daily_trade_date = ""
        engine.cycle_count = 0
        return engine

    def test_no_limit_initially(self):
        engine = self._make_engine()
        assert engine._is_daily_trade_limit_reached("AAPL") is False
        assert engine._is_daily_trade_limit_reached("BTC/USD") is False

    @patch.object(Config, "MAX_TRADES_PER_DAY_EQUITY", 2)
    def test_equity_limit_reached(self):
        engine = self._make_engine()
        engine._record_trade("AAPL")
        engine._record_trade("NVDA")
        assert engine._is_daily_trade_limit_reached("META") is True

    @patch.object(Config, "MAX_TRADES_PER_DAY_CRYPTO", 3)
    def test_crypto_limit_reached(self):
        engine = self._make_engine()
        engine._record_trade("BTC/USD")
        engine._record_trade("ETH/USD")
        engine._record_trade("BTC/USD")
        assert engine._is_daily_trade_limit_reached("BTC/USD") is True

    @patch.object(Config, "MAX_TRADES_PER_DAY_EQUITY", 2)
    @patch.object(Config, "MAX_TRADES_PER_DAY_CRYPTO", 3)
    def test_equity_limit_doesnt_affect_crypto(self):
        engine = self._make_engine()
        engine._record_trade("AAPL")
        engine._record_trade("NVDA")
        # Equity limit reached, but crypto should still be allowed
        assert engine._is_daily_trade_limit_reached("META") is True
        assert engine._is_daily_trade_limit_reached("BTC/USD") is False

    def test_record_trade_increments_count(self):
        engine = self._make_engine()
        engine._record_trade("AAPL")
        assert engine._daily_trade_count.get("equity") == 1
        engine._record_trade("BTC/USD")
        assert engine._daily_trade_count.get("crypto") == 1
        engine._record_trade("NVDA")
        assert engine._daily_trade_count.get("equity") == 2


class TestMinSignalStrength:
    """Test that MIN_SIGNAL_STRENGTH config exists and has correct default."""

    def test_default_value(self):
        assert Config.MIN_SIGNAL_STRENGTH == 0.55

    def test_max_trades_per_day_defaults(self):
        assert Config.MAX_TRADES_PER_DAY_CRYPTO == 8
        assert Config.MAX_TRADES_PER_DAY_EQUITY == 4

    def test_max_single_position_pct_lowered(self):
        assert Config.MAX_SINGLE_POSITION_PCT == 0.25
