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
        engine._record_trade("AAPL")
        assert "AAPL" in engine._last_trade_time
        assert time.time() - engine._last_trade_time["AAPL"] < 1


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
        assert TradingEngine.COOLDOWN_CRYPTO == 900   # 15 min
        assert TradingEngine.COOLDOWN_EQUITY == 300   # 5 min

    def test_crypto_cooldown_longer_than_equity(self):
        assert TradingEngine.COOLDOWN_CRYPTO > TradingEngine.COOLDOWN_EQUITY
