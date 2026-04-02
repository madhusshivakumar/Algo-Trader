"""Tests for core/order_manager.py — order lifecycle management."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from core.order_manager import OrderManager, ManagedOrder, OrderState, TERMINAL_STATES


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.buy.return_value = {"id": "ord-001", "status": "accepted", "symbol": "AAPL"}
    broker.sell.return_value = {"id": "ord-002", "status": "accepted", "symbol": "AAPL"}
    broker.submit_limit_order.return_value = {"id": "ord-003", "status": "accepted", "symbol": "AAPL"}
    broker.submit_stop_limit_order.return_value = {"id": "ord-004", "status": "accepted", "symbol": "AAPL"}
    broker.cancel_order.return_value = True
    broker.get_order_by_id.return_value = None
    return broker


@pytest.fixture
def mgr(mock_broker):
    return OrderManager(mock_broker, state_store=None)


# ── ManagedOrder ───────────────────────────────────────────────────

class TestManagedOrder:
    def test_initial_state(self):
        order = ManagedOrder(order_id="x", symbol="AAPL", side="buy", order_type="market")
        assert order.state == OrderState.PENDING
        assert not order.is_terminal

    def test_terminal_states(self):
        for state in TERMINAL_STATES:
            order = ManagedOrder(order_id="x", symbol="AAPL", side="buy", order_type="market", state=state)
            assert order.is_terminal

    def test_non_terminal_states(self):
        for state in [OrderState.PENDING, OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED]:
            order = ManagedOrder(order_id="x", symbol="AAPL", side="buy", order_type="market", state=state)
            assert not order.is_terminal

    def test_to_dict_roundtrip(self):
        order = ManagedOrder(
            order_id="ord-001", symbol="AAPL", side="buy", order_type="market",
            requested_notional=1500.0, expected_price=150.0,
            state=OrderState.SUBMITTED, submitted_at=datetime(2026, 3, 30, 10, 0),
        )
        d = order.to_dict()
        restored = ManagedOrder.from_dict(d)
        assert restored.order_id == "ord-001"
        assert restored.state == OrderState.SUBMITTED
        assert restored.submitted_at == datetime(2026, 3, 30, 10, 0)

    def test_from_dict_minimal(self):
        d = {"order_id": "x", "symbol": "AAPL", "side": "buy", "order_type": "market", "state": "pending"}
        order = ManagedOrder.from_dict(d)
        assert order.state == OrderState.PENDING
        assert order.filled_qty == 0


# ── Submit market buy ──────────────────────────────────────────────

class TestSubmitMarketBuy:
    def test_success(self, mgr, mock_broker):
        order = mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        assert order.order_id == "ord-001"
        assert order.state == OrderState.SUBMITTED
        assert order.requested_notional == 1500.0
        assert order.expected_price == 150.0
        mock_broker.buy.assert_called_once_with("AAPL", 1500.0)

    def test_broker_exception(self, mgr, mock_broker):
        mock_broker.buy.side_effect = Exception("API timeout")
        order = mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        assert order.state == OrderState.FAILED
        assert "API timeout" in order.error

    def test_broker_returns_none(self, mgr, mock_broker):
        mock_broker.buy.return_value = None
        order = mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        assert order.state == OrderState.FAILED
        assert "None" in order.error

    def test_order_tracked(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        assert mgr.pending_count == 1
        assert mgr.get_order("ord-001") is not None


# ── Submit market sell ─────────────────────────────────────────────

class TestSubmitMarketSell:
    def test_success(self, mgr, mock_broker):
        order = mgr.submit_market_sell("AAPL", 10.0, 155.0)
        assert order.order_id == "ord-002"
        assert order.state == OrderState.SUBMITTED
        assert order.side == "sell"
        mock_broker.sell.assert_called_once_with("AAPL", 10.0)

    def test_broker_exception(self, mgr, mock_broker):
        mock_broker.sell.side_effect = Exception("Network error")
        order = mgr.submit_market_sell("AAPL", 10.0, 155.0)
        assert order.state == OrderState.FAILED

    def test_broker_returns_none(self, mgr, mock_broker):
        mock_broker.sell.return_value = None
        order = mgr.submit_market_sell("AAPL", 10.0, 155.0)
        assert order.state == OrderState.FAILED


# ── Submit limit buy ───────────────────────────────────────────────

class TestSubmitLimitBuy:
    def test_success(self, mgr, mock_broker):
        order = mgr.submit_limit_buy("AAPL", 10.0, 148.0)
        assert order.order_id == "ord-003"
        assert order.order_type == "limit"
        assert order.limit_price == 148.0
        mock_broker.submit_limit_order.assert_called_once_with("AAPL", 10.0, 148.0, side="buy")

    def test_broker_exception(self, mgr, mock_broker):
        mock_broker.submit_limit_order.side_effect = Exception("Fail")
        order = mgr.submit_limit_buy("AAPL", 10.0, 148.0)
        assert order.state == OrderState.FAILED

    def test_broker_returns_none(self, mgr, mock_broker):
        mock_broker.submit_limit_order.return_value = None
        order = mgr.submit_limit_buy("AAPL", 10.0, 148.0)
        assert order.state == OrderState.FAILED


# ── Submit stop-limit sell ─────────────────────────────────────────

class TestSubmitStopLimitSell:
    def test_success(self, mgr, mock_broker):
        order = mgr.submit_stop_limit_sell("AAPL", 10.0, 145.0, 144.50)
        assert order.order_id == "ord-004"
        assert order.order_type == "stop_limit"
        assert order.stop_price == 145.0
        assert order.limit_price == 144.50

    def test_broker_exception(self, mgr, mock_broker):
        mock_broker.submit_stop_limit_order.side_effect = Exception("Fail")
        order = mgr.submit_stop_limit_sell("AAPL", 10.0, 145.0, 144.50)
        assert order.state == OrderState.FAILED

    def test_broker_returns_none(self, mgr, mock_broker):
        mock_broker.submit_stop_limit_order.return_value = None
        order = mgr.submit_stop_limit_sell("AAPL", 10.0, 145.0, 144.50)
        assert order.state == OrderState.FAILED


# ── Poll pending orders ───────────────────────────────────────────

class TestPollPendingOrders:
    def test_poll_updates_filled(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        mock_broker.get_order_by_id.return_value = {
            "id": "ord-001", "symbol": "AAPL", "side": "buy",
            "status": "filled", "filled_qty": 10.0, "filled_avg_price": 150.20,
        }
        terminal = mgr.poll_pending_orders()
        assert len(terminal) == 1
        assert terminal[0].state == OrderState.FILLED
        assert terminal[0].filled_avg_price == 150.20

    def test_poll_updates_partially_filled(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        mock_broker.get_order_by_id.return_value = {
            "id": "ord-001", "symbol": "AAPL", "side": "buy",
            "status": "partially_filled", "filled_qty": 5.0, "filled_avg_price": 150.10,
        }
        terminal = mgr.poll_pending_orders()
        assert len(terminal) == 0  # partially filled is not terminal
        assert mgr.get_order("ord-001").state == OrderState.PARTIALLY_FILLED

    def test_poll_updates_canceled(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        mock_broker.get_order_by_id.return_value = {
            "id": "ord-001", "status": "canceled", "filled_qty": 0, "filled_avg_price": 0,
        }
        terminal = mgr.poll_pending_orders()
        assert len(terminal) == 1
        assert terminal[0].state == OrderState.CANCELED

    def test_poll_updates_rejected(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        mock_broker.get_order_by_id.return_value = {
            "id": "ord-001", "status": "rejected", "filled_qty": 0, "filled_avg_price": 0,
        }
        terminal = mgr.poll_pending_orders()
        assert len(terminal) == 1
        assert terminal[0].state == OrderState.REJECTED

    def test_poll_updates_expired(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        mock_broker.get_order_by_id.return_value = {
            "id": "ord-001", "status": "expired", "filled_qty": 0, "filled_avg_price": 0,
        }
        terminal = mgr.poll_pending_orders()
        assert len(terminal) == 1
        assert terminal[0].state == OrderState.EXPIRED

    def test_poll_skips_terminal_orders(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        # Fill it first
        mock_broker.get_order_by_id.return_value = {
            "id": "ord-001", "status": "filled", "filled_qty": 10.0, "filled_avg_price": 150.20,
        }
        mgr.poll_pending_orders()
        # Poll again - should not query broker
        mock_broker.get_order_by_id.reset_mock()
        mgr.poll_pending_orders()
        mock_broker.get_order_by_id.assert_not_called()

    def test_poll_handles_api_error(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        mock_broker.get_order_by_id.return_value = None
        terminal = mgr.poll_pending_orders()
        assert len(terminal) == 0
        assert mgr.get_order("ord-001").state == OrderState.SUBMITTED  # unchanged

    def test_poll_multiple_orders(self, mgr, mock_broker):
        mock_broker.buy.side_effect = [
            {"id": "o1", "status": "accepted", "symbol": "AAPL"},
            {"id": "o2", "status": "accepted", "symbol": "TSLA"},
        ]
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        mgr.submit_market_buy("TSLA", 2000.0, 200.0)

        def get_order(oid):
            if oid == "o1":
                return {"id": "o1", "status": "filled", "filled_qty": 10, "filled_avg_price": 150.5}
            return {"id": "o2", "status": "new", "filled_qty": 0, "filled_avg_price": 0}

        mock_broker.get_order_by_id.side_effect = get_order
        terminal = mgr.poll_pending_orders()
        assert len(terminal) == 1
        assert terminal[0].order_id == "o1"

    def test_poll_computes_slippage_on_fill(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        mock_broker.get_order_by_id.return_value = {
            "id": "ord-001", "status": "filled", "filled_qty": 10.0, "filled_avg_price": 150.30,
        }
        terminal = mgr.poll_pending_orders()
        assert abs(terminal[0].slippage - 0.002) < 0.001  # (150.30 - 150.0) / 150.0

    def test_poll_handles_orderstatus_prefix(self, mgr, mock_broker):
        """Alpaca sometimes returns 'OrderStatus.filled' instead of 'filled'."""
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        mock_broker.get_order_by_id.return_value = {
            "id": "ord-001", "status": "OrderStatus.filled", "filled_qty": 10.0, "filled_avg_price": 150.0,
        }
        terminal = mgr.poll_pending_orders()
        assert len(terminal) == 1
        assert terminal[0].state == OrderState.FILLED


# ── Cancel orders ──────────────────────────────────────────────────

class TestCancelOrder:
    def test_cancel_success(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        assert mgr.cancel_order("ord-001") is True
        assert mgr.get_order("ord-001").state == OrderState.CANCELED

    def test_cancel_already_filled(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        mgr.get_order("ord-001").state = OrderState.FILLED
        assert mgr.cancel_order("ord-001") is False

    def test_cancel_api_error(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        mock_broker.cancel_order.return_value = False
        assert mgr.cancel_order("ord-001") is False
        # State should remain submitted
        assert mgr.get_order("ord-001").state == OrderState.SUBMITTED

    def test_cancel_unknown_order(self, mgr):
        assert mgr.cancel_order("nonexistent") is False


class TestCancelStaleOrders:
    def test_cancels_old_orders(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        # Backdate the submitted_at
        mgr.get_order("ord-001").submitted_at = datetime.now() - timedelta(seconds=600)
        canceled = mgr.cancel_stale_orders(max_age_seconds=300)
        assert len(canceled) == 1

    def test_keeps_fresh_orders(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        canceled = mgr.cancel_stale_orders(max_age_seconds=300)
        assert len(canceled) == 0

    def test_skips_terminal_orders(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        mgr.get_order("ord-001").state = OrderState.FILLED
        mgr.get_order("ord-001").submitted_at = datetime.now() - timedelta(seconds=600)
        canceled = mgr.cancel_stale_orders(max_age_seconds=300)
        assert len(canceled) == 0


# ── Slippage ───────────────────────────────────────────────────────

class TestSlippage:
    def test_positive_slippage(self):
        order = ManagedOrder(
            order_id="x", symbol="AAPL", side="buy", order_type="market",
            expected_price=150.0, filled_avg_price=150.60,
        )
        assert abs(OrderManager.compute_slippage(order) - 0.004) < 0.001

    def test_negative_slippage(self):
        order = ManagedOrder(
            order_id="x", symbol="AAPL", side="buy", order_type="market",
            expected_price=150.0, filled_avg_price=149.70,
        )
        assert OrderManager.compute_slippage(order) < 0

    def test_zero_slippage(self):
        order = ManagedOrder(
            order_id="x", symbol="AAPL", side="buy", order_type="market",
            expected_price=150.0, filled_avg_price=150.0,
        )
        assert OrderManager.compute_slippage(order) == 0.0

    def test_no_expected_price(self):
        order = ManagedOrder(
            order_id="x", symbol="AAPL", side="buy", order_type="market",
            expected_price=0, filled_avg_price=150.0,
        )
        assert OrderManager.compute_slippage(order) == 0.0

    def test_no_filled_price(self):
        order = ManagedOrder(
            order_id="x", symbol="AAPL", side="buy", order_type="market",
            expected_price=150.0, filled_avg_price=0,
        )
        assert OrderManager.compute_slippage(order) == 0.0


# ── Query ──────────────────────────────────────────────────────────

class TestQuery:
    def test_get_active_orders_all(self, mgr, mock_broker):
        mock_broker.buy.side_effect = [
            {"id": "o1", "status": "accepted", "symbol": "AAPL"},
            {"id": "o2", "status": "accepted", "symbol": "TSLA"},
        ]
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        mgr.submit_market_buy("TSLA", 2000.0, 200.0)
        assert len(mgr.get_active_orders()) == 2

    def test_get_active_orders_by_symbol(self, mgr, mock_broker):
        mock_broker.buy.side_effect = [
            {"id": "o1", "status": "accepted", "symbol": "AAPL"},
            {"id": "o2", "status": "accepted", "symbol": "TSLA"},
        ]
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        mgr.submit_market_buy("TSLA", 2000.0, 200.0)
        assert len(mgr.get_active_orders("AAPL")) == 1

    def test_get_order_by_id(self, mgr, mock_broker):
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        assert mgr.get_order("ord-001").symbol == "AAPL"

    def test_get_order_not_found(self, mgr):
        assert mgr.get_order("nonexistent") is None

    def test_pending_count(self, mgr, mock_broker):
        assert mgr.pending_count == 0
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        assert mgr.pending_count == 1
        mgr.get_order("ord-001").state = OrderState.FILLED
        assert mgr.pending_count == 0


# ── Persistence ────────────────────────────────────────────────────

class TestPersistence:
    def test_orders_saved_to_state_store(self, mock_broker):
        store = MagicMock()
        store.load_active_orders.return_value = []
        mgr = OrderManager(mock_broker, state_store=store)
        mgr.submit_market_buy("AAPL", 1500.0, 150.0)
        store.save_order.assert_called_once()
        saved = store.save_order.call_args[0][0]
        assert saved["order_id"] == "ord-001"
        assert saved["state"] == "submitted"

    def test_orders_loaded_on_init(self, mock_broker):
        store = MagicMock()
        store.load_active_orders.return_value = [
            {"order_id": "old-001", "symbol": "AAPL", "side": "buy",
             "order_type": "market", "state": "submitted",
             "submitted_at": "2026-03-30T10:00:00"},
        ]
        mgr = OrderManager(mock_broker, state_store=store)
        assert mgr.get_order("old-001") is not None
        assert mgr.pending_count == 1

    def test_state_store_error_doesnt_crash(self, mock_broker):
        store = MagicMock()
        store.load_active_orders.side_effect = Exception("DB locked")
        mgr = OrderManager(mock_broker, state_store=store)
        assert mgr.pending_count == 0  # graceful fallback
