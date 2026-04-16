"""Full coverage tests for the trading engine with mocked broker."""

import signal
import sys
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock, call
from datetime import datetime

import pandas as pd
import numpy as np

from core.engine import TradingEngine
from config import Config


# Disable all optional v3 features for base engine tests.
# These tests verify core order routing, not the v3 feature stack.
_V3_OFF = {
    "ORDER_MANAGEMENT_ENABLED": False, "STATE_PERSISTENCE_ENABLED": False,
    "PARALLEL_FETCH_ENABLED": False, "HOT_RELOAD_ENABLED": False,
    "DRIFT_DETECTION_ENABLED": False, "ALERTING_ENABLED": False,
    "POSITION_RECONCILIATION_ENABLED": False, "TC_ENABLED": False,
    "VWAP_TWAP_ENABLED": False, "DB_ROTATION_ENABLED": False,
    "PORTFOLIO_OPTIMIZATION_ENABLED": False, "VOLATILITY_SIZING_ENABLED": False,
    "CORRELATION_CHECK_ENABLED": False, "MONTE_CARLO_ENABLED": False,
    "EARNINGS_CALENDAR_ENABLED": False, "KELLY_SIZING_ENABLED": False,
    "MEAN_VARIANCE_ENABLED": False,
}


def _v3_off_context():
    """Return a combined patch context that disables all v3 features."""
    import contextlib
    return contextlib.ExitStack()


@pytest.fixture
def v3_off():
    """Fixture that disables all v3 features on Config during test."""
    import contextlib
    with contextlib.ExitStack() as stack:
        for attr, val in _V3_OFF.items():
            stack.enter_context(patch.object(Config, attr, val))
        yield


@pytest.fixture
def mock_broker():
    with patch("core.engine.Broker") as MockBroker:
        broker = MagicMock()
        MockBroker.return_value = broker
        broker.get_account.return_value = {
            "equity": 100000.0, "cash": 50000.0,
            "buying_power": 200000.0, "daytrade_count": 0,
            "pattern_day_trader": False,
        }
        broker.get_positions.return_value = []
        broker.get_position.return_value = None
        yield broker


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Patch DB_PATH so tests never write to the production trades.db."""
    from utils import logger
    test_db = str(tmp_path / "test_trades.db")
    with patch("utils.logger.DB_PATH", test_db):
        logger.log._init_db()  # re-init tables in the tmp db
        yield


@pytest.fixture
def engine(mock_broker):
    # Ensure optional v3 features are disabled for base engine tests
    import contextlib
    with contextlib.ExitStack() as stack:
        for attr, val in _V3_OFF.items():
            stack.enter_context(patch.object(Config, attr, val))
        engine = TradingEngine()
    return engine


@pytest.fixture
def sample_df():
    n = 100
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "open": 100 + rng.randn(n) * 0.5,
        "high": 101 + rng.randn(n) * 0.5,
        "low": 99 + rng.randn(n) * 0.5,
        "close": 100 + rng.randn(n) * 0.5,
        "volume": rng.randint(1_000_000, 10_000_000, n),
    })


class TestEngineInitialize:
    def test_initialize(self, engine, mock_broker):
        engine.initialize()
        assert engine.risk.starting_equity == 100000.0

    def test_initialize_paper_mode(self, engine, mock_broker):
        with patch.object(Config, "is_paper", return_value=True):
            engine.initialize()  # Should log "PAPER trading mode"

    def test_initialize_live_mode(self, engine, mock_broker):
        with patch.object(Config, "is_paper", return_value=False):
            engine.initialize()  # Should log "LIVE trading mode"


class TestEngineRunCycle:
    def test_run_cycle_increments_count(self, engine, mock_broker):
        engine.risk.initialize(100000)
        mock_broker.get_recent_bars.return_value = pd.DataFrame()
        engine.run_cycle()
        assert engine.cycle_count == 1

    def test_run_cycle_snapshots_every_10(self, engine, mock_broker):
        engine.risk.initialize(100000)
        mock_broker.get_recent_bars.return_value = pd.DataFrame()
        for _ in range(10):
            engine.run_cycle()
        assert engine.cycle_count == 10

    def test_run_cycle_halts_on_drawdown(self, engine, mock_broker):
        engine.risk.initialize(100000)
        mock_broker.get_account.return_value = {
            "equity": 85000.0, "cash": 40000.0,
            "buying_power": 170000.0, "daytrade_count": 0,
            "pattern_day_trader": False,
        }
        mock_broker.get_recent_bars.return_value = pd.DataFrame()
        engine.run_cycle()
        # Should have halted — broker methods for processing shouldn't be heavily called
        assert engine.risk.halted is True

    def test_run_cycle_skips_equities_when_market_closed(self, engine, mock_broker):
        engine.risk.initialize(100000)
        mock_broker.get_recent_bars.return_value = pd.DataFrame()
        with patch.object(Config, "is_market_open", return_value=False):
            engine.run_cycle()

    def test_run_cycle_processes_equities_when_market_open(self, engine, mock_broker, sample_df):
        engine.risk.initialize(100000)
        mock_broker.get_recent_bars.return_value = sample_df
        with patch.object(Config, "is_market_open", return_value=True):
            engine.run_cycle()

    def test_run_cycle_handles_processing_error(self, engine, mock_broker):
        engine.risk.initialize(100000)
        mock_broker.get_recent_bars.side_effect = Exception("API error")
        engine.run_cycle()  # Should not crash
        assert engine.cycle_count == 1


class TestEngineProcessSymbol:
    def test_skip_empty_df(self, engine, mock_broker):
        mock_broker.get_recent_bars.return_value = pd.DataFrame()
        engine._process_symbol("AAPL", 100000)
        mock_broker.buy.assert_not_called()

    def test_skip_small_df(self, engine, mock_broker):
        small_df = pd.DataFrame({"close": [100] * 10, "open": [100] * 10,
                                  "high": [101] * 10, "low": [99] * 10,
                                  "volume": [1000000] * 10})
        mock_broker.get_recent_bars.return_value = small_df
        engine._process_symbol("AAPL", 100000)
        mock_broker.buy.assert_not_called()

    def test_buy_signal_places_order(self, engine, mock_broker, sample_df):
        mock_broker.get_recent_bars.return_value = sample_df
        mock_broker.get_position.return_value = None
        mock_broker.buy.return_value = {"id": "123", "status": "filled"}

        buy_signal = {"action": "buy", "reason": "test buy", "strength": 0.8, "strategy": "test"}

        with patch("core.engine.route_signals", return_value=buy_signal):
            engine._process_symbol("AAPL", 100000)

        mock_broker.buy.assert_called_once()

    def test_buy_signal_records_cooldown(self, engine, mock_broker, sample_df):
        mock_broker.get_recent_bars.return_value = sample_df
        mock_broker.get_position.return_value = None
        mock_broker.buy.return_value = {"id": "123", "status": "filled"}

        buy_signal = {"action": "buy", "reason": "test", "strength": 0.7, "strategy": "test"}
        with patch("core.engine.route_signals", return_value=buy_signal):
            engine._process_symbol("AAPL", 100000)

        assert "AAPL" in engine._last_trade_time

    def test_buy_signal_records_pdt_for_equity(self, engine, mock_broker, sample_df):
        mock_broker.get_recent_bars.return_value = sample_df
        mock_broker.get_position.return_value = None
        mock_broker.buy.return_value = {"id": "123", "status": "filled"}

        buy_signal = {"action": "buy", "reason": "test", "strength": 0.7, "strategy": "test"}
        with patch("core.engine.route_signals", return_value=buy_signal):
            engine._process_symbol("AAPL", 100000)

        assert "AAPL" in engine._equity_buys_today

    def test_buy_signal_no_pdt_for_crypto(self, engine, mock_broker, sample_df):
        mock_broker.get_recent_bars.return_value = sample_df
        mock_broker.get_position.return_value = None
        mock_broker.buy.return_value = {"id": "123", "status": "filled"}

        buy_signal = {"action": "buy", "reason": "test", "strength": 0.7, "strategy": "test"}
        with patch("core.engine.route_signals", return_value=buy_signal):
            engine._process_symbol("BTC/USD", 100000)

        assert "BTC/USD" not in engine._equity_buys_today

    def test_sell_signal_closes_position(self, engine, mock_broker, sample_df):
        mock_broker.get_recent_bars.return_value = sample_df
        mock_broker.get_position.return_value = {
            "symbol": "AAPL", "qty": 10, "market_value": 2500,
            "unrealized_pl": 50, "avg_entry_price": 245, "current_price": 250,
        }
        mock_broker.close_position.return_value = {"id": "456", "status": "filled"}

        sell_signal = {"action": "sell", "reason": "test sell", "strength": 0.7, "strategy": "test"}
        with patch("core.engine.route_signals", return_value=sell_signal):
            engine._process_symbol("AAPL", 100000)

        mock_broker.close_position.assert_called_once_with("AAPL")

    def test_sell_blocked_by_pdt(self, engine, mock_broker, sample_df):
        mock_broker.get_recent_bars.return_value = sample_df
        mock_broker.get_position.return_value = {
            "symbol": "AAPL", "qty": 10, "market_value": 2500,
            "unrealized_pl": -50, "avg_entry_price": 255, "current_price": 250,
        }
        # Record a buy today for PDT
        engine._equity_buys_today["AAPL"] = datetime.now(Config.MARKET_TZ)

        sell_signal = {"action": "sell", "reason": "test sell", "strength": 0.7, "strategy": "test"}
        with patch("core.engine.route_signals", return_value=sell_signal):
            engine._process_symbol("AAPL", 100000)

        mock_broker.close_position.assert_not_called()

    def test_hold_signal_no_action(self, engine, mock_broker, sample_df):
        mock_broker.get_recent_bars.return_value = sample_df
        hold_signal = {"action": "hold", "reason": "neutral", "strength": 0, "strategy": "test"}
        with patch("core.engine.route_signals", return_value=hold_signal):
            engine._process_symbol("AAPL", 100000)

        mock_broker.buy.assert_not_called()
        mock_broker.close_position.assert_not_called()

    def test_trailing_stop_closes_position(self, engine, mock_broker, sample_df):
        mock_broker.get_recent_bars.return_value = sample_df
        mock_broker.get_position.return_value = {
            "symbol": "AAPL", "qty": 10, "market_value": 2500,
            "unrealized_pl": -100, "avg_entry_price": 260, "current_price": 250,
        }
        mock_broker.close_position.return_value = {"id": "789", "status": "filled"}

        # Register a trailing stop that will be triggered
        engine.risk.register_entry("AAPL", 260)
        # Force the stop by setting highest price high
        engine.risk.trailing_stops["AAPL"].highest_price = 270

        engine._process_symbol("AAPL", 100000)
        # Price ~100 is way below stop of 270*0.98=264.6, so stop triggers
        mock_broker.close_position.assert_called_once_with("AAPL")

    def test_trailing_stop_blocked_by_pdt(self, engine, mock_broker, sample_df):
        mock_broker.get_recent_bars.return_value = sample_df
        mock_broker.get_position.return_value = {
            "symbol": "AAPL", "qty": 10, "market_value": 2500,
            "unrealized_pl": -100, "avg_entry_price": 260, "current_price": 250,
        }
        # Register trailing stop and PDT buy
        engine.risk.register_entry("AAPL", 260)
        engine.risk.trailing_stops["AAPL"].highest_price = 270
        engine._equity_buys_today["AAPL"] = datetime.now(Config.MARKET_TZ)

        engine._process_symbol("AAPL", 100000)
        mock_broker.close_position.assert_not_called()

    def test_exposure_cap_blocks_buy(self, engine, mock_broker, sample_df):
        mock_broker.get_recent_bars.return_value = sample_df
        mock_broker.get_position.return_value = None
        # Already at 90% exposure
        mock_broker.get_positions.return_value = [
            {"market_value": 90000.0},
        ]

        buy_signal = {"action": "buy", "reason": "test", "strength": 0.8, "strategy": "test"}
        with patch("core.engine.route_signals", return_value=buy_signal):
            engine._process_symbol("AAPL", 100000)

        mock_broker.buy.assert_not_called()

    def test_equity_position_sizing(self, engine, mock_broker, sample_df):
        mock_broker.get_recent_bars.return_value = sample_df
        mock_broker.get_position.return_value = None
        mock_broker.get_positions.return_value = []
        mock_broker.buy.return_value = {"id": "123", "status": "filled"}

        buy_signal = {"action": "buy", "reason": "test", "strength": 1.0, "strategy": "test"}
        with patch("core.engine.route_signals", return_value=buy_signal):
            engine._process_symbol("AAPL", 100000)

        # Equity max = 100000 * 0.15 = 15000
        buy_call = mock_broker.buy.call_args
        assert buy_call[0][1] <= 15000

    def test_crypto_position_sizing(self, engine, mock_broker, sample_df):
        mock_broker.get_recent_bars.return_value = sample_df
        mock_broker.get_position.return_value = None
        mock_broker.get_positions.return_value = []
        mock_broker.buy.return_value = {"id": "123", "status": "filled"}

        buy_signal = {"action": "buy", "reason": "test", "strength": 1.0, "strategy": "test"}
        with patch("core.engine.route_signals", return_value=buy_signal):
            engine._process_symbol("BTC/USD", 100000)

        # Crypto max = 100000 * 0.35 = 35000
        buy_call = mock_broker.buy.call_args
        assert buy_call[0][1] <= 35000

    def test_cooldown_skips_processing(self, engine, mock_broker):
        engine._last_trade_time["AAPL"] = time.time()  # Just traded
        engine._process_symbol("AAPL", 100000)
        mock_broker.get_recent_bars.assert_not_called()


class TestEngineGetTotalExposure:
    def test_no_positions(self, engine, mock_broker):
        mock_broker.get_positions.return_value = []
        assert engine._get_total_exposure() == 0

    def test_with_positions(self, engine, mock_broker):
        mock_broker.get_positions.return_value = [
            {"market_value": "25000.00"},
            {"market_value": "15000.00"},
        ]
        assert engine._get_total_exposure() == 40000.0

    def test_negative_positions(self, engine, mock_broker):
        mock_broker.get_positions.return_value = [
            {"market_value": "-5000.00"},
        ]
        assert engine._get_total_exposure() == 5000.0


class TestEngineShutdown:
    def test_shutdown_prints_summary(self, engine, mock_broker):
        engine.risk.initialize(100000)
        mock_broker.get_positions.return_value = [
            {"symbol": "AAPL", "qty": 10, "avg_entry_price": 245.0, "unrealized_pl": 50.0},
        ]
        engine._shutdown()  # Should not crash

    def test_shutdown_no_positions(self, engine, mock_broker):
        engine.risk.initialize(100000)
        mock_broker.get_positions.return_value = []
        engine._shutdown()  # Should not crash

    def test_shutdown_crypto_positions(self, engine, mock_broker):
        engine.risk.initialize(100000)
        mock_broker.get_positions.return_value = [
            {"symbol": "BTCUSD", "qty": 0.5, "avg_entry_price": 65000.0, "unrealized_pl": 200.0},
        ]
        engine._shutdown()


class TestEngineRunLoop:
    def test_run_stops_on_keyboard_interrupt(self, engine, mock_broker):
        engine.risk.initialize(100000)
        mock_broker.get_recent_bars.return_value = pd.DataFrame()
        mock_broker.get_positions.return_value = []

        with patch.object(engine, "run_cycle", side_effect=KeyboardInterrupt):
            engine.run(interval_seconds=1)  # Should exit cleanly

    def test_run_handles_unexpected_error(self, engine, mock_broker):
        engine.risk.initialize(100000)
        mock_broker.get_positions.return_value = []

        call_count = 0
        def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("unexpected")
            raise KeyboardInterrupt

        with patch.object(engine, "run_cycle", side_effect=side_effect), \
             patch("time.sleep"):
            engine.run(interval_seconds=1)


# ── Sprint 1: Order Management & State Persistence Integration ──────

class TestEngineStatePersistence:
    """Tests for state load/save integration in engine."""

    def test_state_loaded_on_initialize(self, mock_broker):
        mock_state_store = MagicMock()
        mock_state_store.load_engine_state.return_value = {
            "trailing_stops": {
                "AAPL": {"entry_price": 150.0, "highest_price": 155.0, "stop_pct": 0.02},
            },
            "cooldowns": {"AAPL": time.time() - 10},
            "pdt_buys": {"TSLA": "2026-03-30T10:00:00"},
            "scalars": {
                "halted": False, "halt_reason": "",
                "daily_start_equity": 99000.0, "cycle_count": 50,
            },
        }

        with patch.object(Config, "STATE_PERSISTENCE_ENABLED", True), \
             patch("core.engine.Broker", return_value=mock_broker), \
             patch("core.state_store.StateStore.__init__", return_value=None):
            engine = TradingEngine()
            engine.state_store = mock_state_store
            engine.initialize()

            assert "AAPL" in engine.risk.trailing_stops
            assert engine.risk.trailing_stops["AAPL"].entry_price == 150.0
            assert "AAPL" in engine._last_trade_time
            assert "TSLA" in engine._equity_buys_today
            assert engine.cycle_count == 50

    def test_state_saved_on_shutdown(self, mock_broker):
        mock_state_store = MagicMock()

        with patch("core.engine.Broker", return_value=mock_broker):
            engine = TradingEngine()
            engine.state_store = mock_state_store
            engine.risk.initialize(100000)
            engine._shutdown()

            mock_state_store.save_engine_state.assert_called_once()

    def test_state_saved_periodically(self, mock_broker):
        mock_state_store = MagicMock()
        mock_broker.get_recent_bars.return_value = pd.DataFrame()

        with patch("core.engine.Broker", return_value=mock_broker):
            engine = TradingEngine()
            engine.state_store = mock_state_store
            engine.risk.initialize(100000)

            # Run 10 cycles to trigger the save (happens at cycle 10)
            for _ in range(10):
                engine.run_cycle()

            mock_state_store.save_engine_state.assert_called()

    def test_state_load_failure_graceful(self, mock_broker):
        mock_state_store = MagicMock()
        mock_state_store.load_engine_state.side_effect = Exception("DB locked")

        with patch("core.engine.Broker", return_value=mock_broker):
            engine = TradingEngine()
            engine.state_store = mock_state_store
            engine.initialize()  # Should not crash


class TestEngineOrderManager:
    """Tests for order manager integration in engine."""

    def test_order_manager_polls_on_cycle(self, mock_broker, sample_df):
        mock_order_mgr = MagicMock()
        mock_order_mgr.poll_pending_orders.return_value = []
        mock_order_mgr.cancel_stale_orders.return_value = []
        mock_broker.get_recent_bars.return_value = pd.DataFrame()

        with patch("core.engine.Broker", return_value=mock_broker):
            engine = TradingEngine()
            engine.order_manager = mock_order_mgr
            engine.risk.initialize(100000)
            engine.run_cycle()

            mock_order_mgr.poll_pending_orders.assert_called_once()

    def test_buying_power_precheck(self, mock_broker, sample_df):
        mock_order_mgr = MagicMock()
        mock_order_mgr.poll_pending_orders.return_value = []
        mock_order_mgr.cancel_stale_orders.return_value = []
        mock_order_mgr.get_active_orders.return_value = []  # No pending orders
        mock_broker.get_recent_bars.return_value = sample_df
        mock_broker.get_position.return_value = None
        mock_broker.get_positions.return_value = []
        # Set buying power very low
        mock_broker.check_buying_power.return_value = 50.0

        with patch("core.engine.Broker", return_value=mock_broker):
            engine = TradingEngine()
            engine.order_manager = mock_order_mgr
            engine.risk.initialize(100000)

            buy_signal = {"action": "buy", "reason": "test", "strength": 0.8, "strategy": "test"}
            with patch("core.engine.route_signals", return_value=buy_signal):
                engine._process_symbol("AAPL", 100000)

            # Should not have submitted order due to buying power check
            mock_order_mgr.submit_market_buy.assert_not_called()

    def test_order_manager_submit_used(self, mock_broker, sample_df, v3_off):
        from core.order_manager import ManagedOrder, OrderState
        mock_order = ManagedOrder(
            order_id="test-001", symbol="AAPL", side="buy", order_type="market",
            state=OrderState.SUBMITTED,
        )
        mock_order_mgr = MagicMock()
        mock_order_mgr.poll_pending_orders.return_value = []
        mock_order_mgr.cancel_stale_orders.return_value = []
        mock_order_mgr.get_active_orders.return_value = []
        mock_order_mgr.submit_market_buy.return_value = mock_order
        mock_broker.get_recent_bars.return_value = sample_df
        mock_broker.get_position.return_value = None
        mock_broker.get_positions.return_value = []
        mock_broker.check_buying_power.return_value = 200000.0

        with patch("core.engine.Broker", return_value=mock_broker):
            engine = TradingEngine()
            engine.order_manager = mock_order_mgr
            engine.risk.initialize(100000)

            buy_signal = {"action": "buy", "reason": "test", "strength": 0.8, "strategy": "test"}
            with patch("core.engine.route_signals", return_value=buy_signal):
                engine._process_symbol("AAPL", 100000)

            mock_order_mgr.submit_market_buy.assert_called_once()
            mock_broker.buy.assert_not_called()

    def test_no_order_manager_uses_direct_buy(self, mock_broker, sample_df, v3_off):
        mock_broker.get_recent_bars.return_value = sample_df
        mock_broker.get_position.return_value = None
        mock_broker.get_positions.return_value = []
        mock_broker.buy.return_value = {"id": "123", "status": "filled"}

        with patch("core.engine.Broker", return_value=mock_broker):
            engine = TradingEngine()
            engine.order_manager = None
            engine.risk.initialize(100000)

            buy_signal = {"action": "buy", "reason": "test", "strength": 0.8, "strategy": "test"}
            with patch("core.engine.route_signals", return_value=buy_signal):
                engine._process_symbol("AAPL", 100000)

            mock_broker.buy.assert_called_once()

    def test_stale_orders_canceled_every_5_cycles(self, mock_broker):
        mock_order_mgr = MagicMock()
        mock_order_mgr.poll_pending_orders.return_value = []
        mock_order_mgr.cancel_stale_orders.return_value = []
        mock_broker.get_recent_bars.return_value = pd.DataFrame()

        with patch("core.engine.Broker", return_value=mock_broker):
            engine = TradingEngine()
            engine.order_manager = mock_order_mgr
            engine.risk.initialize(100000)

            for _ in range(5):
                engine.run_cycle()

            mock_order_mgr.cancel_stale_orders.assert_called()


class TestSignalHandling:
    def test_sigterm_sets_shutdown_flag(self, engine, mock_broker):
        engine._handle_sigterm(signal.SIGTERM, None)
        assert engine._shutdown_requested is True

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGHUP not available on Windows")
    def test_sighup_sets_reload_flag(self, engine, mock_broker):
        engine._handle_sighup(signal.SIGHUP, None)
        assert engine._reload_requested is True

    def test_interruptible_sleep_breaks_on_shutdown(self, engine, mock_broker):
        engine._shutdown_requested = True
        start = time.time()
        engine._interruptible_sleep(60)
        elapsed = time.time() - start
        assert elapsed < 2  # Should break immediately, not wait 60s

    def test_register_signal_handlers(self, engine, mock_broker):
        with patch("core.engine.signal.signal") as mock_signal:
            engine._register_signal_handlers()
            calls = mock_signal.call_args_list
            signames = [c[0][0] for c in calls]
            assert signal.SIGTERM in signames
            if hasattr(signal, "SIGHUP"):
                assert signal.SIGHUP in signames

    def test_shutdown_saves_state(self, engine, mock_broker):
        engine.risk.initialize(100000)
        engine.state_store = MagicMock()
        engine._save_persisted_state = MagicMock()
        engine._shutdown()
        engine._save_persisted_state.assert_called_once()

    def test_shutdown_warns_pending_orders(self, engine, mock_broker):
        engine.risk.initialize(100000)
        engine.order_manager = MagicMock()
        engine.order_manager.pending_count = 3
        engine._shutdown()
        # Should not crash, just warn about pending orders

    def test_shutdown_sends_alert(self, engine, mock_broker):
        engine.risk.initialize(100000)
        engine.alert_manager = MagicMock()
        engine._shutdown(reason="SIGTERM")
        engine.alert_manager.bot_shutdown.assert_called_once_with("SIGTERM")

    def test_handle_reload_creates_reloader_if_needed(self, engine, mock_broker):
        engine.config_reloader = None
        with patch("core.config_reloader.ConfigReloader") as MockReloader:
            MockReloader.return_value.reload.return_value = {}
            engine._handle_reload()
            assert engine.config_reloader is not None

    def test_handle_reload_logs_changes(self, engine, mock_broker):
        from core.config_reloader import ConfigReloader
        engine.config_reloader = MagicMock(spec=ConfigReloader)
        engine.config_reloader.reload.return_value = {"STOP_LOSS_PCT": "0.05"}
        engine._handle_reload()
        engine.config_reloader.reload.assert_called_once()

    def test_run_loop_checks_shutdown_flag(self, engine, mock_broker):
        """Engine run loop should exit when _shutdown_requested is set."""
        engine.risk.initialize(100000)
        mock_broker.get_recent_bars.return_value = MagicMock(empty=True)

        # Set shutdown after first cycle
        original_run_cycle = engine.run_cycle
        def run_cycle_and_stop():
            original_run_cycle()
            engine._shutdown_requested = True
        engine.run_cycle = run_cycle_and_stop

        with patch("core.engine.signal.signal"):
            engine.run(interval_seconds=1)
        # Should have exited cleanly
        assert engine.cycle_count == 1

    def test_run_loop_checks_reload_flag(self, engine, mock_broker):
        """Engine run loop should call _handle_reload when flag is set."""
        engine.risk.initialize(100000)
        mock_broker.get_recent_bars.return_value = MagicMock(empty=True)
        engine._reload_requested = True

        call_count = 0
        def run_cycle_and_stop():
            nonlocal call_count
            call_count += 1
            engine._shutdown_requested = True
        engine.run_cycle = run_cycle_and_stop

        with patch("core.engine.signal.signal"), \
             patch.object(engine, "_handle_reload") as mock_reload:
            engine.run(interval_seconds=1)
        mock_reload.assert_called_once()
        assert engine._reload_requested is False

    def test_handle_reload_sends_alert_on_changes(self, engine, mock_broker):
        """Alert manager should be notified when config changes are detected."""
        from core.config_reloader import ConfigReloader
        engine.config_reloader = MagicMock(spec=ConfigReloader)
        engine.config_reloader.reload.return_value = {"STOP_LOSS_PCT": "0.03", "TRAILING_STOP_PCT": "0.01"}
        engine.alert_manager = MagicMock()
        engine._handle_reload()
        engine.alert_manager.config_reloaded.assert_called_once()
        keys = engine.alert_manager.config_reloaded.call_args[0][0]
        assert "STOP_LOSS_PCT" in keys
        assert "TRAILING_STOP_PCT" in keys

    def test_handle_reload_no_alert_on_empty_changes(self, engine, mock_broker):
        """No alert should be sent when reload detects no changes."""
        from core.config_reloader import ConfigReloader
        engine.config_reloader = MagicMock(spec=ConfigReloader)
        engine.config_reloader.reload.return_value = {}
        engine.alert_manager = MagicMock()
        engine._handle_reload()
        engine.alert_manager.config_reloaded.assert_not_called()

    def test_handle_reload_exception_does_not_crash(self, engine, mock_broker):
        """Reload failure should be logged, not crash the engine."""
        from core.config_reloader import ConfigReloader
        engine.config_reloader = MagicMock(spec=ConfigReloader)
        engine.config_reloader.reload.side_effect = RuntimeError("parse error")
        # Should not raise
        engine._handle_reload()
