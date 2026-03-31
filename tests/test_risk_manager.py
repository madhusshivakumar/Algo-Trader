"""Unit tests for the risk manager."""

import pytest
from unittest.mock import patch

from core.risk_manager import RiskManager, TrailingStop
from config import Config


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

    def test_crypto_gets_tighter_stop(self):
        rm = RiskManager()
        rm.register_entry("BTC/USD", 60000)
        stop = rm.trailing_stops["BTC/USD"]
        assert stop.stop_pct == 0.015  # 1.5% for crypto

    def test_equity_gets_wider_stop(self):
        rm = RiskManager()
        rm.register_entry("AAPL", 250)
        stop = rm.trailing_stops["AAPL"]
        assert stop.stop_pct == 0.02  # 2.0% for equities

    def test_calculate_position_size(self):
        rm = RiskManager()
        size = rm.calculate_position_size(100000)
        assert size == 100000 * Config.MAX_POSITION_PCT

    def test_halt_reason_set_on_drawdown(self):
        rm = RiskManager()
        rm.initialize(100000)
        rm.can_trade(85000)
        assert "drawdown" in rm.halt_reason.lower()
