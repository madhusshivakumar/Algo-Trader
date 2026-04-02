"""Order lifecycle management — tracks orders from submission to fill/cancel.

Wraps Broker to provide:
  - Order state tracking (pending → filled/canceled/rejected)
  - Buying power pre-checks
  - Slippage computation (expected vs actual fill price)
  - Stale order cancellation
  - Persistence via StateStore (optional)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from utils.logger import log


class OrderState(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


# Terminal states — once reached, no further transitions
TERMINAL_STATES = {
    OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED,
    OrderState.EXPIRED, OrderState.FAILED,
}

# Map Alpaca status strings to our OrderState
_ALPACA_STATUS_MAP = {
    "new": OrderState.SUBMITTED,
    "accepted": OrderState.SUBMITTED,
    "partially_filled": OrderState.PARTIALLY_FILLED,
    "filled": OrderState.FILLED,
    "canceled": OrderState.CANCELED,
    "expired": OrderState.EXPIRED,
    "rejected": OrderState.REJECTED,
    "pending_new": OrderState.PENDING,
    "pending_cancel": OrderState.SUBMITTED,
    "pending_replace": OrderState.SUBMITTED,
    "held": OrderState.SUBMITTED,
    "replaced": OrderState.SUBMITTED,
}


@dataclass
class ManagedOrder:
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: str  # "market", "limit", "stop_limit"
    requested_notional: Optional[float] = None
    requested_qty: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    state: OrderState = OrderState.PENDING
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    expected_price: float = 0.0
    slippage: float = 0.0
    error: str = ""
    last_checked: Optional[datetime] = None

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "requested_notional": self.requested_notional,
            "requested_qty": self.requested_qty,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "state": self.state.value,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_qty": self.filled_qty,
            "filled_avg_price": self.filled_avg_price,
            "expected_price": self.expected_price,
            "slippage": self.slippage,
            "error": self.error,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ManagedOrder":
        return cls(
            order_id=d["order_id"],
            symbol=d["symbol"],
            side=d["side"],
            order_type=d["order_type"],
            requested_notional=d.get("requested_notional"),
            requested_qty=d.get("requested_qty"),
            limit_price=d.get("limit_price"),
            stop_price=d.get("stop_price"),
            state=OrderState(d["state"]),
            submitted_at=datetime.fromisoformat(d["submitted_at"]) if d.get("submitted_at") else None,
            filled_at=datetime.fromisoformat(d["filled_at"]) if d.get("filled_at") else None,
            filled_qty=d.get("filled_qty", 0),
            filled_avg_price=d.get("filled_avg_price", 0),
            expected_price=d.get("expected_price", 0),
            slippage=d.get("slippage", 0),
            error=d.get("error", ""),
            last_checked=datetime.fromisoformat(d["last_checked"]) if d.get("last_checked") else None,
        )


class OrderManager:
    def __init__(self, broker, state_store=None):
        self.broker = broker
        self.state_store = state_store
        self._orders: dict[str, ManagedOrder] = {}
        self._load_persisted_orders()

    def _load_persisted_orders(self):
        if self.state_store:
            try:
                active = self.state_store.load_active_orders()
                for od in active:
                    self._orders[od["order_id"]] = ManagedOrder.from_dict(od)
            except Exception:
                pass

    def _persist_order(self, order: ManagedOrder):
        if self.state_store:
            try:
                self.state_store.save_order(order.to_dict())
            except Exception:
                pass

    # ── Submit orders ───────────────────────────────────────────────

    def submit_market_buy(self, symbol: str, notional: float,
                          expected_price: float) -> Optional[ManagedOrder]:
        """Submit a market buy order and track it."""
        try:
            result = self.broker.buy(symbol, notional)
        except Exception as e:
            order = ManagedOrder(
                order_id=f"failed-{datetime.now().timestamp():.0f}",
                symbol=symbol, side="buy", order_type="market",
                requested_notional=notional, expected_price=expected_price,
                state=OrderState.FAILED, error=str(e),
            )
            self._orders[order.order_id] = order
            self._persist_order(order)
            log.error(f"Order failed for {symbol}: {e}")
            return order

        if not result:
            order = ManagedOrder(
                order_id=f"failed-{datetime.now().timestamp():.0f}",
                symbol=symbol, side="buy", order_type="market",
                requested_notional=notional, expected_price=expected_price,
                state=OrderState.FAILED, error="Broker returned None",
            )
            self._orders[order.order_id] = order
            self._persist_order(order)
            return order

        order = ManagedOrder(
            order_id=result["id"],
            symbol=symbol, side="buy", order_type="market",
            requested_notional=notional, expected_price=expected_price,
            state=OrderState.SUBMITTED,
            submitted_at=datetime.now(),
        )
        self._orders[order.order_id] = order
        self._persist_order(order)
        return order

    def submit_market_sell(self, symbol: str, qty: float,
                           expected_price: float) -> Optional[ManagedOrder]:
        """Submit a market sell order and track it."""
        try:
            result = self.broker.sell(symbol, qty)
        except Exception as e:
            order = ManagedOrder(
                order_id=f"failed-{datetime.now().timestamp():.0f}",
                symbol=symbol, side="sell", order_type="market",
                requested_qty=qty, expected_price=expected_price,
                state=OrderState.FAILED, error=str(e),
            )
            self._orders[order.order_id] = order
            self._persist_order(order)
            log.error(f"Sell order failed for {symbol}: {e}")
            return order

        if not result:
            order = ManagedOrder(
                order_id=f"failed-{datetime.now().timestamp():.0f}",
                symbol=symbol, side="sell", order_type="market",
                requested_qty=qty, expected_price=expected_price,
                state=OrderState.FAILED, error="Broker returned None",
            )
            self._orders[order.order_id] = order
            self._persist_order(order)
            return order

        order = ManagedOrder(
            order_id=result["id"],
            symbol=symbol, side="sell", order_type="market",
            requested_qty=qty, expected_price=expected_price,
            state=OrderState.SUBMITTED,
            submitted_at=datetime.now(),
        )
        self._orders[order.order_id] = order
        self._persist_order(order)
        return order

    def submit_limit_buy(self, symbol: str, qty: float,
                         limit_price: float) -> Optional[ManagedOrder]:
        """Submit a limit buy order."""
        try:
            result = self.broker.submit_limit_order(symbol, qty, limit_price, side="buy")
        except Exception as e:
            order = ManagedOrder(
                order_id=f"failed-{datetime.now().timestamp():.0f}",
                symbol=symbol, side="buy", order_type="limit",
                requested_qty=qty, limit_price=limit_price,
                expected_price=limit_price,
                state=OrderState.FAILED, error=str(e),
            )
            self._orders[order.order_id] = order
            self._persist_order(order)
            return order

        if not result:
            order = ManagedOrder(
                order_id=f"failed-{datetime.now().timestamp():.0f}",
                symbol=symbol, side="buy", order_type="limit",
                requested_qty=qty, limit_price=limit_price,
                expected_price=limit_price,
                state=OrderState.FAILED, error="Broker returned None",
            )
            self._orders[order.order_id] = order
            self._persist_order(order)
            return order

        order = ManagedOrder(
            order_id=result["id"],
            symbol=symbol, side="buy", order_type="limit",
            requested_qty=qty, limit_price=limit_price,
            expected_price=limit_price,
            state=OrderState.SUBMITTED,
            submitted_at=datetime.now(),
        )
        self._orders[order.order_id] = order
        self._persist_order(order)
        return order

    def submit_stop_limit_sell(self, symbol: str, qty: float,
                               stop_price: float, limit_price: float) -> Optional[ManagedOrder]:
        """Submit a stop-limit sell order (protective stop)."""
        try:
            result = self.broker.submit_stop_limit_order(symbol, qty, stop_price, limit_price, side="sell")
        except Exception as e:
            order = ManagedOrder(
                order_id=f"failed-{datetime.now().timestamp():.0f}",
                symbol=symbol, side="sell", order_type="stop_limit",
                requested_qty=qty, stop_price=stop_price, limit_price=limit_price,
                expected_price=stop_price,
                state=OrderState.FAILED, error=str(e),
            )
            self._orders[order.order_id] = order
            self._persist_order(order)
            return order

        if not result:
            order = ManagedOrder(
                order_id=f"failed-{datetime.now().timestamp():.0f}",
                symbol=symbol, side="sell", order_type="stop_limit",
                requested_qty=qty, stop_price=stop_price, limit_price=limit_price,
                expected_price=stop_price,
                state=OrderState.FAILED, error="Broker returned None",
            )
            self._orders[order.order_id] = order
            self._persist_order(order)
            return order

        order = ManagedOrder(
            order_id=result["id"],
            symbol=symbol, side="sell", order_type="stop_limit",
            requested_qty=qty, stop_price=stop_price, limit_price=limit_price,
            expected_price=stop_price,
            state=OrderState.SUBMITTED,
            submitted_at=datetime.now(),
        )
        self._orders[order.order_id] = order
        self._persist_order(order)
        return order

    # ── Poll & lifecycle ────────────────────────────────────────────

    def poll_pending_orders(self) -> list[ManagedOrder]:
        """Check status of all non-terminal orders. Returns list of newly-terminal orders."""
        newly_terminal = []
        for order in list(self._orders.values()):
            if order.is_terminal:
                continue

            broker_order = self.broker.get_order_by_id(order.order_id)
            order.last_checked = datetime.now()

            if broker_order is None:
                continue

            status_str = broker_order["status"].lower().replace("orderstatus.", "")
            new_state = _ALPACA_STATUS_MAP.get(status_str, order.state)

            if new_state != order.state:
                old_state = order.state
                order.state = new_state
                order.filled_qty = broker_order.get("filled_qty", order.filled_qty)
                order.filled_avg_price = broker_order.get("filled_avg_price", order.filled_avg_price)

                if new_state == OrderState.FILLED:
                    order.filled_at = datetime.now()
                    order.slippage = self.compute_slippage(order)
                    log.info(
                        f"Order {order.order_id[:8]} filled: {order.symbol} {order.side} "
                        f"@ ${order.filled_avg_price:.2f} "
                        f"(slippage: {order.slippage:+.4f})"
                    )

                if order.is_terminal:
                    newly_terminal.append(order)

                self._persist_order(order)

        return newly_terminal

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        order = self._orders.get(order_id)
        if not order or order.is_terminal:
            return False

        success = self.broker.cancel_order(order_id)
        if success:
            order.state = OrderState.CANCELED
            self._persist_order(order)
        return success

    def cancel_stale_orders(self, max_age_seconds: int = 300) -> list[ManagedOrder]:
        """Cancel orders older than max_age_seconds that haven't filled."""
        canceled = []
        now = datetime.now()
        for order in list(self._orders.values()):
            if order.is_terminal or not order.submitted_at:
                continue
            age = (now - order.submitted_at).total_seconds()
            if age > max_age_seconds:
                if self.cancel_order(order.order_id):
                    canceled.append(order)
                    log.warning(f"Canceled stale order {order.order_id[:8]} for {order.symbol} (age: {age:.0f}s)")
        return canceled

    # ── Query ───────────────────────────────────────────────────────

    def get_active_orders(self, symbol: str = None) -> list[ManagedOrder]:
        """Get all non-terminal orders, optionally filtered by symbol."""
        orders = [o for o in self._orders.values() if not o.is_terminal]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_order(self, order_id: str) -> Optional[ManagedOrder]:
        return self._orders.get(order_id)

    @property
    def pending_count(self) -> int:
        return sum(1 for o in self._orders.values() if not o.is_terminal)

    # ── Slippage ────────────────────────────────────────────────────

    @staticmethod
    def compute_slippage(order: ManagedOrder) -> float:
        """Compute slippage as fraction: (filled - expected) / expected.
        Positive = paid more than expected (bad for buys).
        Negative = paid less than expected (good for buys).
        """
        if not order.expected_price or order.expected_price == 0:
            return 0.0
        if not order.filled_avg_price or order.filled_avg_price == 0:
            return 0.0
        return (order.filled_avg_price - order.expected_price) / order.expected_price
