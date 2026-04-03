"""Tests for VWAP/TWAP execution algorithms."""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest

from core.execution_algo import (
    ChildOrder, ExecutionPlan, ExecutionAlgoManager,
    build_volume_profile, compute_vwap_weights, _uniform_weights,
)
from config import Config


# ── Helpers ─────────────────────────────────────────────────────────


def _make_plan(num_children=3, algo="twap", total=1000.0, price=100.0) -> ExecutionPlan:
    """Create a test execution plan with children."""
    mgr = ExecutionAlgoManager()
    return mgr.create_plan(
        symbol="AAPL", side="buy", total_notional=total,
        algo=algo, current_price=price, num_slices=num_children,
        interval_seconds=60,
    )


# ── ChildOrder Tests ────────────────────────────────────────────────


class TestChildOrder:
    def test_default_state_is_pending(self):
        c = ChildOrder(child_id="test-1", slice_index=0, notional=100.0,
                       scheduled_at=datetime.now())
        assert c.state == "pending"
        assert c.broker_order_id is None
        assert c.filled_avg_price == 0.0

    def test_fields_settable(self):
        c = ChildOrder(child_id="test-1", slice_index=0, notional=200.0,
                       scheduled_at=datetime.now())
        c.state = "filled"
        c.filled_avg_price = 150.0
        c.filled_qty = 1.33
        assert c.state == "filled"


# ── ExecutionPlan Tests ─────────────────────────────────────────────


class TestExecutionPlan:
    def test_avg_fill_price_no_fills(self):
        plan = _make_plan()
        assert plan.avg_fill_price == 0.0

    def test_avg_fill_price_single_fill(self):
        plan = _make_plan()
        plan.children[0].state = "filled"
        plan.children[0].filled_avg_price = 150.0
        plan.children[0].filled_qty = 2.0
        assert plan.avg_fill_price == 150.0

    def test_avg_fill_price_weighted(self):
        plan = _make_plan(num_children=2, total=1000.0)
        plan.children[0].state = "filled"
        plan.children[0].filled_avg_price = 100.0
        plan.children[0].filled_qty = 3.0  # $300
        plan.children[1].state = "filled"
        plan.children[1].filled_avg_price = 200.0
        plan.children[1].filled_qty = 1.0  # $200
        # VWAP: (100*3 + 200*1) / (3+1) = 500/4 = 125
        assert plan.avg_fill_price == pytest.approx(125.0)

    def test_slippage_bps(self):
        plan = _make_plan(price=100.0)
        plan.children[0].state = "filled"
        plan.children[0].filled_avg_price = 100.50
        plan.children[0].filled_qty = 5.0
        # (100.50 - 100.0) / 100.0 * 10000 = 50 bps
        assert plan.slippage_bps == pytest.approx(50.0)

    def test_slippage_bps_zero_benchmark(self):
        plan = _make_plan(price=0.0)
        assert plan.slippage_bps == 0.0

    def test_filled_notional(self):
        plan = _make_plan()
        plan.children[0].state = "filled"
        plan.children[0].filled_avg_price = 100.0
        plan.children[0].filled_qty = 2.0
        assert plan.filled_notional == pytest.approx(200.0)

    def test_is_complete_all_filled(self):
        plan = _make_plan(num_children=2)
        plan.children[0].state = "filled"
        plan.children[1].state = "filled"
        assert plan.is_complete is True

    def test_is_complete_mixed_terminal(self):
        plan = _make_plan(num_children=2)
        plan.children[0].state = "filled"
        plan.children[1].state = "failed"
        assert plan.is_complete is True

    def test_not_complete_pending(self):
        plan = _make_plan(num_children=2)
        plan.children[0].state = "filled"
        plan.children[1].state = "pending"
        assert plan.is_complete is False

    def test_pending_children_count(self):
        plan = _make_plan(num_children=3)
        assert plan.pending_children == 3
        plan.children[0].state = "filled"
        assert plan.pending_children == 2


# ── Volume Profile & Weights ───────────────────────────────────────


class TestBuildVolumeProfile:
    def test_normal_profile(self):
        mock_broker = MagicMock()
        n = 200
        df = pd.DataFrame({
            "open": [100] * n, "high": [105] * n, "low": [95] * n,
            "close": [102] * n,
            "volume": [1000 + i * 10 for i in range(n)],
        }, index=pd.date_range("2026-01-01 09:30", periods=n, freq="5min"))

        mock_broker.get_historical_bars.return_value = df
        weights = build_volume_profile(mock_broker, "AAPL", lookback_days=5)

        assert len(weights) > 0
        assert sum(weights) == pytest.approx(1.0)

    def test_empty_bars(self):
        mock_broker = MagicMock()
        mock_broker.get_historical_bars.return_value = pd.DataFrame()
        weights = build_volume_profile(mock_broker, "AAPL")
        assert weights == []

    def test_none_bars(self):
        mock_broker = MagicMock()
        mock_broker.get_historical_bars.return_value = None
        weights = build_volume_profile(mock_broker, "AAPL")
        assert weights == []

    def test_api_error(self):
        mock_broker = MagicMock()
        mock_broker.get_historical_bars.side_effect = Exception("timeout")
        weights = build_volume_profile(mock_broker, "AAPL")
        assert weights == []


class TestComputeVwapWeights:
    def test_exact_match_slices(self):
        profile = [0.2, 0.3, 0.5]
        weights = compute_vwap_weights(profile, 3)
        assert len(weights) == 3
        assert sum(weights) == pytest.approx(1.0)

    def test_more_profile_than_slices(self):
        profile = [10, 20, 30, 40, 50, 60]
        weights = compute_vwap_weights(profile, 3)
        assert len(weights) == 3
        assert sum(weights) == pytest.approx(1.0)

    def test_fewer_profile_than_slices(self):
        profile = [0.5, 0.5]
        weights = compute_vwap_weights(profile, 5)
        assert len(weights) == 5
        assert sum(weights) == pytest.approx(1.0)

    def test_empty_profile_falls_back_to_uniform(self):
        weights = compute_vwap_weights([], 5)
        assert weights == [0.2] * 5

    def test_none_profile_falls_back_to_uniform(self):
        weights = compute_vwap_weights(None, 5)
        assert weights == [0.2] * 5

    def test_zero_profile_falls_back_to_uniform(self):
        weights = compute_vwap_weights([0, 0, 0], 3)
        assert len(weights) == 3
        assert sum(weights) == pytest.approx(1.0)

    def test_zero_slices(self):
        weights = compute_vwap_weights([1, 2, 3], 0)
        assert weights == []


class TestUniformWeights:
    def test_basic(self):
        assert _uniform_weights(4) == [0.25] * 4

    def test_single(self):
        assert _uniform_weights(1) == [1.0]

    def test_zero(self):
        assert _uniform_weights(0) == []


# ── ExecutionAlgoManager ────────────────────────────────────────────


class TestCreatePlan:
    def test_twap_equal_slices(self):
        plan = _make_plan(num_children=5, total=1000.0)
        assert len(plan.children) == 5
        for child in plan.children:
            assert child.notional == pytest.approx(200.0)

    def test_twap_scheduling(self):
        plan = _make_plan(num_children=3)
        t0 = plan.children[0].scheduled_at
        t1 = plan.children[1].scheduled_at
        t2 = plan.children[2].scheduled_at
        assert (t1 - t0).total_seconds() == pytest.approx(60.0, abs=1)
        assert (t2 - t1).total_seconds() == pytest.approx(60.0, abs=1)

    def test_vwap_weighted_slices(self):
        mgr = ExecutionAlgoManager()
        plan = mgr.create_plan(
            symbol="AAPL", side="buy", total_notional=1000.0,
            algo="vwap", current_price=100.0,
            volume_profile=[10, 30, 60],
            num_slices=3, interval_seconds=60,
        )
        notionals = [c.notional for c in plan.children]
        assert sum(notionals) == pytest.approx(1000.0)
        # First slice should be smallest (10%), last should be largest (60%)
        assert notionals[0] < notionals[2]

    def test_vwap_no_profile_falls_back(self):
        mgr = ExecutionAlgoManager()
        plan = mgr.create_plan(
            symbol="AAPL", side="buy", total_notional=1000.0,
            algo="vwap", current_price=100.0,
            volume_profile=None, num_slices=5, interval_seconds=60,
        )
        # Falls back to uniform weights
        for child in plan.children:
            assert child.notional == pytest.approx(200.0)

    def test_plan_state_is_active(self):
        plan = _make_plan()
        assert plan.state == "active"

    def test_plan_has_benchmark_price(self):
        plan = _make_plan(price=150.0)
        assert plan.benchmark_price == 150.0

    def test_single_slice_when_num_1(self):
        plan = _make_plan(num_children=1, total=500.0)
        assert len(plan.children) == 1
        assert plan.children[0].notional == pytest.approx(500.0)

    def test_child_ids_unique(self):
        plan = _make_plan(num_children=5)
        ids = [c.child_id for c in plan.children]
        assert len(set(ids)) == 5


class TestTick:
    def test_dispatches_due_children(self):
        mgr = ExecutionAlgoManager()
        plan = mgr.create_plan(
            symbol="BTC/USD", side="buy", total_notional=1000.0,
            algo="twap", current_price=100.0, num_slices=3,
            interval_seconds=0,  # All due immediately
        )
        mock_broker = MagicMock()
        mock_broker.buy.return_value = {"id": "order-123"}

        submitted = mgr.tick(mock_broker)

        assert len(submitted) == 3
        assert mock_broker.buy.call_count == 3
        for child in plan.children:
            assert child.state == "submitted"

    def test_skips_future_children(self):
        mgr = ExecutionAlgoManager()
        plan = mgr.create_plan(
            symbol="BTC/USD", side="buy", total_notional=1000.0,
            algo="twap", current_price=100.0, num_slices=3,
            interval_seconds=3600,  # 1 hour apart — only first is due
        )
        mock_broker = MagicMock()
        mock_broker.buy.return_value = {"id": "order-123"}

        submitted = mgr.tick(mock_broker)

        assert len(submitted) == 1
        assert plan.children[0].state == "submitted"
        assert plan.children[1].state == "pending"
        assert plan.children[2].state == "pending"

    def test_handles_broker_failure(self):
        mgr = ExecutionAlgoManager()
        mgr.create_plan(
            symbol="BTC/USD", side="buy", total_notional=500.0,
            algo="twap", current_price=100.0, num_slices=2,
            interval_seconds=0,
        )
        mock_broker = MagicMock()
        mock_broker.buy.return_value = None  # Broker fails

        submitted = mgr.tick(mock_broker)

        assert len(submitted) == 0

    def test_handles_broker_exception(self):
        mgr = ExecutionAlgoManager()
        mgr.create_plan(
            symbol="BTC/USD", side="buy", total_notional=500.0,
            algo="twap", current_price=100.0, num_slices=1,
            interval_seconds=0,
        )
        mock_broker = MagicMock()
        mock_broker.buy.side_effect = Exception("network error")

        submitted = mgr.tick(mock_broker)
        assert len(submitted) == 0

    def test_skips_inactive_plans(self):
        mgr = ExecutionAlgoManager()
        plan = mgr.create_plan(
            symbol="BTC/USD", side="buy", total_notional=500.0,
            algo="twap", current_price=100.0, num_slices=1,
            interval_seconds=0,
        )
        plan.state = "canceled"
        mock_broker = MagicMock()

        submitted = mgr.tick(mock_broker)
        assert len(submitted) == 0
        mock_broker.buy.assert_not_called()

    @patch.object(Config, "is_market_open", return_value=False)
    def test_skips_equity_when_market_closed(self, mock_market):
        mgr = ExecutionAlgoManager()
        mgr.create_plan(
            symbol="AAPL", side="buy", total_notional=500.0,
            algo="twap", current_price=100.0, num_slices=1,
            interval_seconds=0,
        )
        mock_broker = MagicMock()

        submitted = mgr.tick(mock_broker)
        assert len(submitted) == 0

    def test_no_plans_is_noop(self):
        mgr = ExecutionAlgoManager()
        mock_broker = MagicMock()
        submitted = mgr.tick(mock_broker)
        assert len(submitted) == 0


class TestPollChildren:
    def test_detects_filled(self):
        mgr = ExecutionAlgoManager()
        plan = mgr.create_plan(
            symbol="BTC/USD", side="buy", total_notional=500.0,
            algo="twap", current_price=100.0, num_slices=1,
            interval_seconds=0,
        )
        plan.children[0].state = "submitted"
        plan.children[0].broker_order_id = "order-1"

        mock_broker = MagicMock()
        mock_broker.get_order_by_id.return_value = {
            "status": "filled", "filled_avg_price": "101.5", "filled_qty": "5.0",
        }

        completed = mgr.poll_children(mock_broker)

        assert len(completed) == 1
        assert completed[0].state == "completed"
        assert plan.children[0].filled_avg_price == 101.5
        assert plan.children[0].filled_qty == 5.0

    def test_partial_fills_keep_active(self):
        mgr = ExecutionAlgoManager()
        plan = mgr.create_plan(
            symbol="BTC/USD", side="buy", total_notional=1000.0,
            algo="twap", current_price=100.0, num_slices=2,
            interval_seconds=0,
        )
        plan.children[0].state = "submitted"
        plan.children[0].broker_order_id = "order-1"
        plan.children[1].state = "submitted"
        plan.children[1].broker_order_id = "order-2"

        mock_broker = MagicMock()
        # Only first order filled
        mock_broker.get_order_by_id.side_effect = lambda oid: {
            "order-1": {"status": "filled", "filled_avg_price": "100", "filled_qty": "5"},
            "order-2": {"status": "new", "filled_avg_price": "0", "filled_qty": "0"},
        }.get(oid)

        completed = mgr.poll_children(mock_broker)
        assert len(completed) == 0
        assert plan.state == "active"

    def test_all_failed_marks_plan_failed(self):
        mgr = ExecutionAlgoManager()
        plan = mgr.create_plan(
            symbol="BTC/USD", side="buy", total_notional=500.0,
            algo="twap", current_price=100.0, num_slices=1,
            interval_seconds=0,
        )
        plan.children[0].state = "submitted"
        plan.children[0].broker_order_id = "order-1"

        mock_broker = MagicMock()
        mock_broker.get_order_by_id.return_value = {"status": "canceled"}

        completed = mgr.poll_children(mock_broker)
        assert len(completed) == 1
        assert completed[0].state == "failed"


class TestCancelPlan:
    def test_cancels_pending_children(self):
        mgr = ExecutionAlgoManager()
        plan = mgr.create_plan(
            symbol="BTC/USD", side="buy", total_notional=1000.0,
            algo="twap", current_price=100.0, num_slices=3,
            interval_seconds=60,
        )
        plan.children[0].state = "filled"

        result = mgr.cancel_plan(plan.plan_id)
        assert result is True
        assert plan.children[0].state == "filled"  # Untouched
        assert plan.children[1].state == "canceled"
        assert plan.children[2].state == "canceled"
        assert plan.state == "canceled"

    def test_cancel_nonexistent_plan(self):
        mgr = ExecutionAlgoManager()
        assert mgr.cancel_plan("nonexistent") is False

    def test_cancels_submitted_via_broker(self):
        mgr = ExecutionAlgoManager()
        plan = mgr.create_plan(
            symbol="BTC/USD", side="buy", total_notional=500.0,
            algo="twap", current_price=100.0, num_slices=1,
            interval_seconds=0,
        )
        plan.children[0].state = "submitted"
        plan.children[0].broker_order_id = "order-1"

        mock_broker = MagicMock()
        mock_broker.cancel_order.return_value = True

        mgr.cancel_plan(plan.plan_id, broker=mock_broker)
        mock_broker.cancel_order.assert_called_once_with("order-1")


class TestGetActivePlans:
    def test_returns_active_only(self):
        mgr = ExecutionAlgoManager()
        p1 = mgr.create_plan("AAPL", "buy", 500, "twap", 100.0, num_slices=1, interval_seconds=60)
        p2 = mgr.create_plan("TSLA", "buy", 500, "twap", 200.0, num_slices=1, interval_seconds=60)
        p1.state = "completed"

        active = mgr.get_active_plans()
        assert len(active) == 1
        assert active[0].symbol == "TSLA"

    def test_filter_by_symbol(self):
        mgr = ExecutionAlgoManager()
        mgr.create_plan("AAPL", "buy", 500, "twap", 100.0, num_slices=1, interval_seconds=60)
        mgr.create_plan("TSLA", "buy", 500, "twap", 200.0, num_slices=1, interval_seconds=60)

        assert len(mgr.get_active_plans("AAPL")) == 1
        assert len(mgr.get_active_plans("NVDA")) == 0


class TestCleanupCompleted:
    def test_removes_old_completed(self):
        mgr = ExecutionAlgoManager()
        plan = mgr.create_plan("AAPL", "buy", 500, "twap", 100.0, num_slices=1, interval_seconds=60)
        plan.state = "completed"
        plan.created_at = datetime.now() - timedelta(hours=25)

        mgr.cleanup_completed(max_age_hours=24)
        assert len(mgr._plans) == 0

    def test_keeps_recent_completed(self):
        mgr = ExecutionAlgoManager()
        plan = mgr.create_plan("AAPL", "buy", 500, "twap", 100.0, num_slices=1, interval_seconds=60)
        plan.state = "completed"

        mgr.cleanup_completed(max_age_hours=24)
        assert len(mgr._plans) == 1

    def test_keeps_active_plans(self):
        mgr = ExecutionAlgoManager()
        plan = mgr.create_plan("AAPL", "buy", 500, "twap", 100.0, num_slices=1, interval_seconds=60)
        plan.created_at = datetime.now() - timedelta(hours=48)

        mgr.cleanup_completed(max_age_hours=24)
        assert len(mgr._plans) == 1  # Active, not cleaned


# ── Execution Quality ───────────────────────────────────────────────


class TestExecutionQuality:
    def test_perfect_fill_zero_slippage(self):
        plan = _make_plan(num_children=2, price=100.0)
        for c in plan.children:
            c.state = "filled"
            c.filled_avg_price = 100.0
            c.filled_qty = 5.0
        assert plan.slippage_bps == pytest.approx(0.0)

    def test_adverse_slippage(self):
        plan = _make_plan(num_children=1, price=100.0)
        plan.children[0].state = "filled"
        plan.children[0].filled_avg_price = 100.10
        plan.children[0].filled_qty = 10.0
        assert plan.slippage_bps == pytest.approx(10.0)

    def test_favorable_slippage(self):
        plan = _make_plan(num_children=1, price=100.0)
        plan.children[0].state = "filled"
        plan.children[0].filled_avg_price = 99.90
        plan.children[0].filled_qty = 10.0
        assert plan.slippage_bps == pytest.approx(-10.0)


# ── Engine Integration ──────────────────────────────────────────────


class TestEngineIntegration:
    def _make_engine(self):
        from core.engine import TradingEngine
        engine = TradingEngine.__new__(TradingEngine)
        engine.broker = MagicMock()
        engine.risk = MagicMock()
        engine.order_manager = None
        engine.execution_manager = None
        engine.data_fetcher = None
        engine.config_reloader = None
        engine.drift_detector = None
        engine.position_reconciler = None
        engine.cost_model = None
        engine.alert_manager = None
        engine.db_rotator = None
        engine.state_store = None
        engine.portfolio_optimizer = None
        engine._last_trade_time = {}
        engine._equity_buys_today = {}
        engine._cached_position_dfs = None
        engine.cycle_count = 1
        return engine

    def test_handle_completed_executions_registers_entry(self):
        engine = self._make_engine()
        plan = _make_plan(num_children=1, price=100.0, total=500.0)
        plan.children[0].state = "filled"
        plan.children[0].filled_avg_price = 101.0
        plan.children[0].filled_qty = 5.0
        plan.state = "completed"

        with patch("core.engine.log"), \
             patch("core.engine.get_strategy_key", return_value="momentum"):
            engine._handle_completed_executions([plan])

        engine.risk.register_entry.assert_called_once_with("AAPL", 101.0)

    def test_handle_completed_logs_filled_notional_not_total(self):
        """Bug fix: log.trade should use filled_notional, not total_notional."""
        engine = self._make_engine()
        plan = _make_plan(num_children=2, price=100.0, total=1000.0)
        # Only first child filled, second failed
        plan.children[0].state = "filled"
        plan.children[0].filled_avg_price = 101.0
        plan.children[0].filled_qty = 5.0  # filled_notional = 505
        plan.children[1].state = "failed"
        plan.state = "completed"

        with patch("core.engine.log") as mock_log, \
             patch("core.engine.get_strategy_key", return_value="momentum"):
            engine._handle_completed_executions([plan])

        # Should log filled_notional (505), NOT total_notional (1000)
        mock_log.trade.assert_called_once()
        logged_notional = mock_log.trade.call_args[0][2]
        assert logged_notional == pytest.approx(505.0)

    def test_handle_completed_skips_zero_fills(self):
        engine = self._make_engine()
        plan = _make_plan(num_children=1, price=100.0)
        plan.children[0].state = "failed"
        plan.state = "failed"

        with patch("core.engine.log") as mock_log:
            engine._handle_completed_executions([plan])

        engine.risk.register_entry.assert_not_called()

    @patch.object(Config, "VWAP_TWAP_ENABLED", True)
    @patch.object(Config, "VWAP_TWAP_MIN_NOTIONAL", 100.0)
    @patch.object(Config, "VWAP_TWAP_ALGO", "twap")
    @patch.object(Config, "VWAP_TWAP_NUM_SLICES", 3)
    @patch.object(Config, "VWAP_TWAP_INTERVAL_SECONDS", 60)
    def test_buy_creates_execution_plan(self):
        import pandas as pd
        import numpy as np
        engine = self._make_engine()
        engine.execution_manager = ExecutionAlgoManager()

        engine.broker.get_position.return_value = None
        engine.risk.should_stop_loss.return_value = False
        engine.broker.check_buying_power.return_value = 50000.0

        df = pd.DataFrame({
            "close": np.linspace(100, 105, 50),
            "open": np.linspace(99, 104, 50),
            "high": np.linspace(101, 106, 50),
            "low": np.linspace(98, 103, 50),
            "volume": [1000] * 50,
        })

        with patch("core.engine.log"), \
             patch("core.engine.route_signals", return_value={
                 "action": "buy", "strength": 0.8, "reason": "test",
                 "strategy": "momentum",
             }), \
             patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False), \
             patch.object(Config, "CORRELATION_CHECK_ENABLED", False), \
             patch.object(Config, "VOLATILITY_SIZING_ENABLED", False), \
             patch.object(Config, "MAX_POSITION_PCT", 0.15):
            engine._process_symbol_inner("AAPL", 10000.0, df)

        # Should have created an execution plan
        plans = engine.execution_manager.get_active_plans("AAPL")
        assert len(plans) == 1
        assert plans[0].algo == "twap"
        assert len(plans[0].children) == 3

    @patch.object(Config, "VWAP_TWAP_ENABLED", True)
    @patch.object(Config, "VWAP_TWAP_MIN_NOTIONAL", 10000.0)  # Very high min
    def test_below_min_notional_skips_algo(self):
        import pandas as pd
        import numpy as np
        engine = self._make_engine()
        engine.execution_manager = ExecutionAlgoManager()

        engine.broker.get_position.return_value = None
        engine.risk.should_stop_loss.return_value = False
        engine.broker.buy.return_value = {"id": "order-1"}

        df = pd.DataFrame({
            "close": np.linspace(100, 105, 50),
            "open": np.linspace(99, 104, 50),
            "high": np.linspace(101, 106, 50),
            "low": np.linspace(98, 103, 50),
            "volume": [1000] * 50,
        })

        with patch("core.engine.log"), \
             patch("core.engine.route_signals", return_value={
                 "action": "buy", "strength": 0.8, "reason": "test",
                 "strategy": "momentum",
             }), \
             patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False), \
             patch.object(Config, "CORRELATION_CHECK_ENABLED", False), \
             patch.object(Config, "VOLATILITY_SIZING_ENABLED", False), \
             patch.object(Config, "MAX_POSITION_PCT", 0.01):  # Small position
            engine._process_symbol_inner("AAPL", 10000.0, df)

        # Should NOT have created an execution plan
        plans = engine.execution_manager.get_active_plans("AAPL")
        assert len(plans) == 0
        # Should have gone through direct broker.buy() instead
        engine.broker.buy.assert_called_once()

    @patch.object(Config, "VWAP_TWAP_ENABLED", True)
    @patch.object(Config, "VWAP_TWAP_MIN_NOTIONAL", 100.0)
    @patch.object(Config, "VWAP_TWAP_ALGO", "twap")
    def test_skips_duplicate_plan(self):
        import pandas as pd
        import numpy as np
        engine = self._make_engine()
        engine.execution_manager = ExecutionAlgoManager()

        # Pre-create an active plan for AAPL
        engine.execution_manager.create_plan(
            "AAPL", "buy", 1000, "twap", 100.0, num_slices=3, interval_seconds=60)

        engine.broker.get_position.return_value = None
        engine.risk.should_stop_loss.return_value = False

        df = pd.DataFrame({
            "close": np.linspace(100, 105, 50),
            "open": np.linspace(99, 104, 50),
            "high": np.linspace(101, 106, 50),
            "low": np.linspace(98, 103, 50),
            "volume": [1000] * 50,
        })

        with patch("core.engine.log"), \
             patch("core.engine.route_signals", return_value={
                 "action": "buy", "strength": 0.8, "reason": "test",
                 "strategy": "momentum",
             }), \
             patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False), \
             patch.object(Config, "CORRELATION_CHECK_ENABLED", False), \
             patch.object(Config, "VOLATILITY_SIZING_ENABLED", False), \
             patch.object(Config, "MAX_POSITION_PCT", 0.15):
            engine._process_symbol_inner("AAPL", 10000.0, df)

        # Should still have only 1 plan (not 2)
        plans = engine.execution_manager.get_active_plans("AAPL")
        assert len(plans) == 1
