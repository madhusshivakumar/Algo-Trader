"""VWAP/TWAP execution algorithms — split large orders into smaller child orders.

Instead of sending one large market order, the execution manager creates a plan
that dispatches smaller child orders over time. Two algorithms are supported:

- TWAP: Equal-sized child orders at regular intervals.
- VWAP: Child order sizes weighted by historical intraday volume profile.

The engine's main loop drives child order dispatch by calling tick() each cycle.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from config import Config
from utils.logger import log


@dataclass
class ChildOrder:
    """One slice of a parent execution plan."""
    child_id: str
    slice_index: int
    notional: float
    scheduled_at: datetime
    broker_order_id: Optional[str] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_avg_price: float = 0.0
    filled_qty: float = 0.0
    state: str = "pending"  # pending, submitted, filled, failed, canceled


@dataclass
class ExecutionPlan:
    """Parent container for a VWAP/TWAP execution."""
    plan_id: str
    symbol: str
    side: str
    algo: str  # "vwap" or "twap"
    total_notional: float
    benchmark_price: float  # price at plan creation
    created_at: datetime
    children: list[ChildOrder] = field(default_factory=list)
    state: str = "active"  # active, completed, canceled, failed

    @property
    def filled_notional(self) -> float:
        return sum(c.filled_avg_price * c.filled_qty for c in self.children if c.state == "filled")

    @property
    def filled_children(self) -> int:
        return sum(1 for c in self.children if c.state == "filled")

    @property
    def pending_children(self) -> int:
        return sum(1 for c in self.children if c.state in ("pending", "submitted"))

    @property
    def avg_fill_price(self) -> float:
        """Volume-weighted average fill price across all filled children."""
        total_qty = sum(c.filled_qty for c in self.children if c.state == "filled")
        if total_qty == 0:
            return 0.0
        total_cost = sum(c.filled_avg_price * c.filled_qty for c in self.children if c.state == "filled")
        return total_cost / total_qty

    @property
    def slippage_bps(self) -> float:
        """Slippage vs arrival price in basis points."""
        avg = self.avg_fill_price
        if avg == 0 or self.benchmark_price == 0:
            return 0.0
        return ((avg - self.benchmark_price) / self.benchmark_price) * 10000

    @property
    def is_complete(self) -> bool:
        return all(c.state in ("filled", "failed", "canceled") for c in self.children)


def build_volume_profile(broker, symbol: str, lookback_days: int = 5) -> list[float]:
    """Build normalized intraday volume weights from historical data.

    Returns a list of weights (summing to 1.0) representing relative volume
    at each time slice of the trading day. Used by VWAP to size child orders.
    """
    try:
        df = broker.get_historical_bars(symbol, days=lookback_days)
        if df is None or df.empty or "volume" not in df.columns:
            return []

        # Group by hour-of-day and average volumes
        if hasattr(df.index, 'hour'):
            df = df.copy()
            df["hour"] = df.index.hour
        elif "timestamp" in df.columns:
            df = df.copy()
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        else:
            return []

        hourly_volume = df.groupby("hour")["volume"].mean()
        if hourly_volume.sum() == 0:
            return []

        weights = (hourly_volume / hourly_volume.sum()).tolist()
        return weights
    except Exception:
        return []


def compute_vwap_weights(volume_profile: list[float], num_slices: int) -> list[float]:
    """Map a volume profile to N slice weights.

    If the volume profile has more entries than slices, it's bucketed.
    If fewer, it's interpolated. Always returns weights summing to 1.0.
    """
    if not volume_profile or num_slices <= 0:
        return _uniform_weights(num_slices)

    if len(volume_profile) == num_slices:
        total = sum(volume_profile)
        if total == 0:
            return _uniform_weights(num_slices)
        return [w / total for w in volume_profile]

    # Bucket or interpolate
    weights = []
    bucket_size = len(volume_profile) / num_slices
    for i in range(num_slices):
        start = int(i * bucket_size)
        end = int((i + 1) * bucket_size)
        end = max(end, start + 1)  # at least one element per bucket
        bucket_sum = sum(volume_profile[start:min(end, len(volume_profile))])
        weights.append(bucket_sum)

    total = sum(weights)
    if total == 0:
        return _uniform_weights(num_slices)
    return [w / total for w in weights]


def _uniform_weights(num_slices: int) -> list[float]:
    """Equal weights for TWAP or fallback."""
    if num_slices <= 0:
        return []
    return [1.0 / num_slices] * num_slices


class ExecutionAlgoManager:
    """Manages VWAP/TWAP execution plans across engine cycles."""

    def __init__(self):
        self._plans: dict[str, ExecutionPlan] = {}

    def create_plan(self, symbol: str, side: str, total_notional: float,
                    algo: str, current_price: float,
                    volume_profile: list[float] | None = None,
                    num_slices: int | None = None,
                    interval_seconds: int | None = None) -> ExecutionPlan:
        """Create a new execution plan with scheduled child orders.

        Args:
            symbol: Trading symbol.
            side: "buy" (sell support can be added later).
            total_notional: Total dollar amount to execute.
            algo: "vwap" or "twap".
            current_price: Current market price (benchmark).
            volume_profile: Intraday volume weights (for VWAP).
            num_slices: Number of child orders (default from config).
            interval_seconds: Seconds between slices (default from config).

        Returns:
            ExecutionPlan with scheduled children.
        """
        if num_slices is None:
            num_slices = Config.VWAP_TWAP_NUM_SLICES
        if interval_seconds is None:
            interval_seconds = Config.VWAP_TWAP_INTERVAL_SECONDS

        num_slices = max(1, num_slices)

        # Compute weights
        if algo == "vwap" and volume_profile:
            weights = compute_vwap_weights(volume_profile, num_slices)
        else:
            weights = _uniform_weights(num_slices)

        plan_id = f"exec-{uuid.uuid4().hex[:8]}"
        now = datetime.now()

        children = []
        for i, weight in enumerate(weights):
            child_notional = total_notional * weight
            scheduled_at = now + timedelta(seconds=i * interval_seconds)
            child = ChildOrder(
                child_id=f"{plan_id}-slice-{i}",
                slice_index=i,
                notional=child_notional,
                scheduled_at=scheduled_at,
            )
            children.append(child)

        plan = ExecutionPlan(
            plan_id=plan_id,
            symbol=symbol,
            side=side,
            algo=algo,
            total_notional=total_notional,
            benchmark_price=current_price,
            created_at=now,
            children=children,
        )
        self._plans[plan_id] = plan
        return plan

    def tick(self, broker) -> list[ChildOrder]:
        """Dispatch due child orders. Called once per engine cycle.

        Returns list of newly submitted children.
        """
        now = datetime.now()
        submitted = []

        for plan in list(self._plans.values()):
            if plan.state != "active":
                continue

            # Skip equity orders when market is closed
            if not Config.is_crypto(plan.symbol) and not Config.is_market_open():
                continue

            for child in plan.children:
                if child.state != "pending":
                    continue
                if child.scheduled_at > now:
                    continue

                # Submit child order
                try:
                    result = broker.buy(plan.symbol, child.notional)
                    if result:
                        child.broker_order_id = result.get("id")
                        child.submitted_at = datetime.now()
                        child.state = "submitted"
                        submitted.append(child)
                    else:
                        child.state = "failed"
                        log.warning(f"Child order {child.child_id} failed to submit")
                except Exception as e:
                    child.state = "failed"
                    log.error(f"Child order {child.child_id} error: {e}")

        return submitted

    def poll_children(self, broker) -> list[ExecutionPlan]:
        """Poll submitted child orders for fill status.

        Returns list of plans that just completed (all children terminal).
        """
        newly_completed = []

        for plan in list(self._plans.values()):
            if plan.state != "active":
                continue

            for child in plan.children:
                if child.state != "submitted" or not child.broker_order_id:
                    continue

                try:
                    order_info = broker.get_order_by_id(child.broker_order_id)
                    if not order_info:
                        continue

                    status = order_info.get("status", "")
                    if status in ("filled", "partially_filled"):
                        child.filled_avg_price = float(order_info.get("filled_avg_price", 0))
                        child.filled_qty = float(order_info.get("filled_qty", 0))
                        child.filled_at = datetime.now()
                        child.state = "filled"
                    elif status in ("canceled", "expired", "rejected"):
                        child.state = "failed"
                except Exception as e:
                    log.error(f"Error polling child {child.child_id}: {e}")

            # Check if plan is complete
            if plan.is_complete and plan.state == "active":
                plan.state = "completed" if plan.filled_children > 0 else "failed"
                newly_completed.append(plan)

        return newly_completed

    def cancel_plan(self, plan_id: str, broker=None) -> bool:
        """Cancel remaining pending children in a plan."""
        plan = self._plans.get(plan_id)
        if not plan:
            return False

        for child in plan.children:
            if child.state == "pending":
                child.state = "canceled"
            elif child.state == "submitted" and child.broker_order_id and broker:
                try:
                    broker.cancel_order(child.broker_order_id)
                    child.state = "canceled"
                except Exception:
                    pass

        plan.state = "canceled"
        return True

    def get_active_plans(self, symbol: str | None = None) -> list[ExecutionPlan]:
        """Get active execution plans, optionally filtered by symbol."""
        plans = [p for p in self._plans.values() if p.state == "active"]
        if symbol:
            plans = [p for p in plans if p.symbol == symbol]
        return plans

    def cleanup_completed(self, max_age_hours: int = 24):
        """Remove old completed plans to prevent memory growth."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = [
            pid for pid, p in self._plans.items()
            if p.state in ("completed", "canceled", "failed") and p.created_at < cutoff
        ]
        for pid in to_remove:
            del self._plans[pid]
