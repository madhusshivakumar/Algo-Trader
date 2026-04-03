"""Tests for v3 engine integrations: DB rotation, slippage logging, walk-forward CLI."""

import time
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock, patch, call

import pytest

from core.engine import TradingEngine
from config import Config


# ── DB Rotation Integration ─────────────────────────────────────────


class TestDBRotationIntegration:
    """Test DB rotation wiring in engine."""

    def _make_engine(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.db_rotator = MagicMock()
        engine.state_store = None
        return engine

    def test_check_db_rotation_calls_should_rotate(self):
        engine = self._make_engine()
        engine.db_rotator.should_rotate.return_value = False

        engine._check_db_rotation()

        engine.db_rotator.should_rotate.assert_called_once_with(
            max_rows=Config.DB_ROTATION_MAX_ROWS,
            max_age_days=Config.DB_ROTATION_MAX_AGE_DAYS,
        )

    def test_check_db_rotation_rotates_when_needed(self):
        engine = self._make_engine()
        engine.db_rotator.should_rotate.return_value = True

        engine._check_db_rotation()

        engine.db_rotator.rotate.assert_called_once_with(
            keep_recent_days=Config.DB_ROTATION_KEEP_DAYS
        )

    def test_check_db_rotation_skips_when_not_needed(self):
        engine = self._make_engine()
        engine.db_rotator.should_rotate.return_value = False

        engine._check_db_rotation()

        engine.db_rotator.rotate.assert_not_called()

    def test_check_db_rotation_noop_when_disabled(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.db_rotator = None
        # Should not raise
        engine._check_db_rotation()

    def test_check_db_rotation_handles_error(self):
        engine = self._make_engine()
        engine.db_rotator.should_rotate.side_effect = Exception("db locked")

        # Should not raise
        engine._check_db_rotation()

    def test_db_rotation_called_on_initialize(self):
        engine = self._make_engine()
        engine.broker = MagicMock()
        engine.broker.get_account.return_value = {"equity": 10000.0, "cash": 5000.0, "daytrade_count": 0}
        engine.risk = MagicMock()
        engine.db_rotator.should_rotate.return_value = False

        with patch.object(Config, "is_paper", return_value=True), \
             patch.object(Config, "CRYPTO_SYMBOLS", ["BTC/USD"]), \
             patch.object(Config, "EQUITY_SYMBOLS", ["AAPL"]), \
             patch.object(Config, "SYMBOLS", ["BTC/USD", "AAPL"]), \
             patch.object(Config, "PDT_PROTECTION", False), \
             patch("core.engine.get_strategy", return_value=("test", lambda df: None)):
            engine.initialize()

        engine.db_rotator.should_rotate.assert_called_once()

    def test_db_rotation_called_every_1000_cycles(self):
        engine = self._make_engine()
        engine.broker = MagicMock()
        engine.broker.get_account.return_value = {"equity": 10000.0, "cash": 5000.0}
        engine.risk = MagicMock()
        engine.risk.can_trade.return_value = False
        engine.order_manager = None
        engine.config_reloader = None
        engine.drift_detector = None
        engine.position_reconciler = None
        engine.data_fetcher = None
        engine.alert_manager = None
        engine.execution_manager = None
        engine.portfolio_optimizer = None
        engine._cached_position_dfs = None
        engine.db_rotator.should_rotate.return_value = False

        # Cycle 999 — should not trigger
        engine.cycle_count = 998
        engine.run_cycle()
        assert engine.db_rotator.should_rotate.call_count == 0

        # Cycle 1000 — should trigger
        engine.cycle_count = 999
        engine.run_cycle()
        assert engine.db_rotator.should_rotate.call_count == 1


# ── Slippage Logging Integration ────────────────────────────────────


class TestSlippageLoggingIntegration:
    """Test slippage logging for filled orders."""

    def _make_engine(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.risk = MagicMock()
        engine.broker = MagicMock()
        engine.order_manager = MagicMock()
        engine.alert_manager = None
        engine._last_trade_time = {}
        engine._equity_buys_today = {}
        engine.cycle_count = 1
        return engine

    def test_slippage_logged_for_polled_filled_order(self):
        """Slippage should be logged when _handle_filled_orders processes a fill."""
        engine = self._make_engine()

        # Create a mock filled order
        mock_order = MagicMock()
        mock_order.state = MagicMock()
        mock_order.state.value = "filled"
        # Use actual OrderState enum
        from core.order_manager import OrderState
        mock_order.state = OrderState.FILLED
        mock_order.symbol = "AAPL"
        mock_order.side = "buy"
        mock_order.expected_price = 150.00
        mock_order.filled_avg_price = 150.30
        mock_order.slippage = 0.002  # 0.2%
        mock_order.requested_notional = 1000.0

        with patch("core.engine.log") as mock_log, \
             patch("core.engine.get_strategy_key", return_value="momentum"):
            engine._handle_filled_orders([mock_order])

        mock_log.log_slippage.assert_called_once_with(
            "AAPL", 150.00, 150.30, 0.002
        )

    def test_no_slippage_logged_when_no_expected_price(self):
        """Skip slippage logging when expected_price is 0."""
        engine = self._make_engine()

        from core.order_manager import OrderState
        mock_order = MagicMock()
        mock_order.state = OrderState.FILLED
        mock_order.symbol = "AAPL"
        mock_order.side = "buy"
        mock_order.expected_price = 0
        mock_order.filled_avg_price = 150.30
        mock_order.slippage = 0.0
        mock_order.requested_notional = 1000.0

        with patch("core.engine.log") as mock_log, \
             patch("core.engine.get_strategy_key", return_value="momentum"):
            engine._handle_filled_orders([mock_order])

        mock_log.log_slippage.assert_not_called()

    def test_slippage_not_logged_for_non_filled_orders(self):
        """Only log slippage for filled orders."""
        engine = self._make_engine()

        from core.order_manager import OrderState
        mock_order = MagicMock()
        mock_order.state = OrderState.CANCELED
        mock_order.symbol = "AAPL"

        with patch("core.engine.log") as mock_log:
            engine._handle_filled_orders([mock_order])

        mock_log.log_slippage.assert_not_called()


# ── Walk-Forward CLI Integration ────────────────────────────────────


class TestWalkForwardCLI:
    """Test --walk-forward flag in backtest.py."""

    def test_walk_forward_flag_parsed(self):
        """Verify argparse recognizes --walk-forward."""
        import argparse
        # Import the module to verify it has argparse
        import backtest
        # The main function should handle --walk-forward
        with patch("sys.argv", ["backtest.py", "--walk-forward"]), \
             patch.object(Config, "validate"), \
             patch.object(Config, "SYMBOLS", ["AAPL"]), \
             patch("backtest.Broker") as MockBroker, \
             patch("backtest._run_walk_forward") as mock_wf:
            mock_broker = MagicMock()
            MockBroker.return_value = mock_broker

            backtest.main()

            mock_wf.assert_called_once_with(mock_broker, ["AAPL"])

    def test_standard_backtest_without_flag(self):
        """Without --walk-forward, should run standard backtest."""
        import backtest
        with patch("sys.argv", ["backtest.py"]), \
             patch.object(Config, "validate"), \
             patch.object(Config, "SYMBOLS", ["AAPL"]), \
             patch.object(Config, "CANDLE_HISTORY_DAYS", 30), \
             patch("backtest.Broker") as MockBroker, \
             patch("backtest._run_walk_forward") as mock_wf, \
             patch("backtest.Backtester") as MockBT:
            mock_broker = MagicMock()
            MockBroker.return_value = mock_broker
            mock_bt_instance = MagicMock()
            MockBT.return_value = mock_bt_instance

            backtest.main()

            mock_wf.assert_not_called()
            MockBT.assert_called_once()

    def test_symbols_flag(self):
        """--symbols should override Config.SYMBOLS."""
        import backtest
        with patch("sys.argv", ["backtest.py", "--symbols", "BTC/USD", "ETH/USD"]), \
             patch.object(Config, "validate"), \
             patch.object(Config, "SYMBOLS", ["AAPL"]), \
             patch.object(Config, "CANDLE_HISTORY_DAYS", 30), \
             patch("backtest.Broker") as MockBroker, \
             patch("backtest.Backtester") as MockBT:
            mock_broker = MagicMock()
            MockBroker.return_value = mock_broker
            mock_bt_instance = MagicMock()
            MockBT.return_value = mock_bt_instance

            backtest.main()

            # Should have been called for both symbols
            assert mock_bt_instance.run.call_count == 2

    def test_walk_forward_with_symbols(self):
        """--walk-forward --symbols should pass custom symbols."""
        import backtest
        with patch("sys.argv", ["backtest.py", "--walk-forward", "--symbols", "AAPL"]), \
             patch.object(Config, "validate"), \
             patch("backtest.Broker") as MockBroker, \
             patch("backtest._run_walk_forward") as mock_wf:
            mock_broker = MagicMock()
            MockBroker.return_value = mock_broker

            backtest.main()

            mock_wf.assert_called_once_with(mock_broker, ["AAPL"])

    def test_run_walk_forward_calls_backtester(self):
        """_run_walk_forward should use WalkForwardBacktester."""
        import backtest
        import pandas as pd
        import numpy as np

        mock_broker = MagicMock()
        # Create a DataFrame with enough rows
        n = 500
        df = pd.DataFrame({
            "open": np.random.uniform(90, 110, n),
            "high": np.random.uniform(100, 120, n),
            "low": np.random.uniform(80, 100, n),
            "close": np.random.uniform(90, 110, n),
            "volume": np.random.randint(1000, 10000, n),
        })
        mock_broker.get_historical_bars.return_value = df

        with patch.object(Config, "CANDLE_HISTORY_DAYS", 30), \
             patch("strategies.router.get_strategy", return_value=("test_strat", lambda df: {"action": "hold", "strength": 0.5})), \
             patch("backtest.console"):
            backtest._run_walk_forward(mock_broker, ["AAPL"])

        mock_broker.get_historical_bars.assert_called_once()
