"""Tests for core/state_store.py — state persistence via SQLite."""

import os
import sqlite3
import pytest

from core.state_store import StateStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_state.db")
    return StateStore(db_path=db_path)


# ── Table initialization ───────────────────────────────────────────

class TestInit:
    def test_creates_tables(self, store):
        conn = sqlite3.connect(store.db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "runtime_state" in tables
        assert "persisted_trailing_stops" in tables
        assert "persisted_cooldowns" in tables
        assert "persisted_pdt_buys" in tables
        assert "managed_orders" in tables

    def test_idempotent_init(self, store):
        # Calling _init_tables again should not error
        store._init_tables()
        store._init_tables()


# ── Scalar state ───────────────────────────────────────────────────

class TestScalarState:
    def test_save_and_load_string(self, store):
        store.save_scalar("test_key", "hello")
        assert store.load_scalar("test_key") == "hello"

    def test_save_and_load_float(self, store):
        store.save_scalar("equity", 99500.50)
        assert store.load_scalar("equity") == 99500.50

    def test_save_and_load_bool(self, store):
        store.save_scalar("halted", True)
        assert store.load_scalar("halted") is True

    def test_save_and_load_int(self, store):
        store.save_scalar("cycle_count", 42)
        assert store.load_scalar("cycle_count") == 42

    def test_load_missing_returns_default(self, store):
        assert store.load_scalar("nonexistent") is None
        assert store.load_scalar("nonexistent", "fallback") == "fallback"

    def test_overwrite_existing(self, store):
        store.save_scalar("key", "old")
        store.save_scalar("key", "new")
        assert store.load_scalar("key") == "new"

    def test_save_and_load_dict(self, store):
        store.save_scalar("complex", {"a": 1, "b": [2, 3]})
        assert store.load_scalar("complex") == {"a": 1, "b": [2, 3]}

    def test_save_and_load_none(self, store):
        store.save_scalar("nullable", None)
        assert store.load_scalar("nullable") is None


# ── Trailing stops ─────────────────────────────────────────────────

class TestTrailingStops:
    def test_save_and_load_empty(self, store):
        store.save_trailing_stops({})
        assert store.load_trailing_stops() == {}

    def test_save_and_load_multiple(self, store):
        stops = {
            "AAPL": {"entry_price": 150.0, "highest_price": 160.0, "stop_pct": 0.02, "entry_time": 0.0},
            "BTC/USD": {"entry_price": 60000.0, "highest_price": 65000.0, "stop_pct": 0.015, "entry_time": 0.0},
        }
        store.save_trailing_stops(stops)
        loaded = store.load_trailing_stops()
        assert loaded == stops

    def test_full_replace_on_save(self, store):
        store.save_trailing_stops({"AAPL": {"entry_price": 150.0, "highest_price": 160.0, "stop_pct": 0.02, "entry_time": 0.0}})
        store.save_trailing_stops({"TSLA": {"entry_price": 200.0, "highest_price": 210.0, "stop_pct": 0.02, "entry_time": 0.0}})
        loaded = store.load_trailing_stops()
        assert "AAPL" not in loaded
        assert "TSLA" in loaded

    def test_single_stop(self, store):
        stops = {"SPY": {"entry_price": 500.0, "highest_price": 505.0, "stop_pct": 0.01, "entry_time": 0.0}}
        store.save_trailing_stops(stops)
        assert store.load_trailing_stops() == stops


# ── Cooldowns ──────────────────────────────────────────────────────

class TestCooldowns:
    def test_save_and_load(self, store):
        cooldowns = {"AAPL": 1700000000.0, "BTC/USD": 1700000900.0}
        store.save_cooldowns(cooldowns)
        assert store.load_cooldowns() == cooldowns

    def test_empty_cooldowns(self, store):
        store.save_cooldowns({})
        assert store.load_cooldowns() == {}

    def test_overwrite(self, store):
        store.save_cooldowns({"AAPL": 1000.0})
        store.save_cooldowns({"TSLA": 2000.0})
        loaded = store.load_cooldowns()
        assert "AAPL" not in loaded
        assert loaded["TSLA"] == 2000.0


# ── PDT buys ──────────────────────────────────────────────────────

class TestPDTBuys:
    def test_save_and_load(self, store):
        buys = {"AAPL": "2026-03-30T10:30:00", "TSLA": "2026-03-30T11:00:00"}
        store.save_pdt_buys(buys)
        assert store.load_pdt_buys() == buys

    def test_empty_buys(self, store):
        store.save_pdt_buys({})
        assert store.load_pdt_buys() == {}

    def test_date_string_roundtrip(self, store):
        buys = {"NVDA": "2026-03-30T09:35:00-04:00"}
        store.save_pdt_buys(buys)
        assert store.load_pdt_buys()["NVDA"] == "2026-03-30T09:35:00-04:00"


# ── Managed orders ─────────────────────────────────────────────────

class TestManagedOrders:
    def _make_order(self, **overrides):
        base = {
            "order_id": "ord-001",
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "market",
            "requested_notional": 1500.0,
            "requested_qty": None,
            "limit_price": None,
            "stop_price": None,
            "expected_price": 150.0,
            "state": "submitted",
            "filled_qty": 0,
            "filled_avg_price": 0,
            "slippage": 0,
            "submitted_at": "2026-03-30T10:00:00",
            "filled_at": None,
            "last_checked": None,
            "error": "",
        }
        base.update(overrides)
        return base

    def test_save_new_order(self, store):
        store.save_order(self._make_order())
        orders = store.load_active_orders()
        assert len(orders) == 1
        assert orders[0]["order_id"] == "ord-001"

    def test_update_existing_order(self, store):
        store.save_order(self._make_order(state="submitted"))
        store.save_order(self._make_order(state="filled", filled_qty=10, filled_avg_price=150.5))
        # Should have 0 active (filled is terminal)
        assert len(store.load_active_orders()) == 0

    def test_load_active_excludes_terminal(self, store):
        store.save_order(self._make_order(order_id="o1", state="submitted"))
        store.save_order(self._make_order(order_id="o2", state="filled"))
        store.save_order(self._make_order(order_id="o3", state="canceled"))
        store.save_order(self._make_order(order_id="o4", state="rejected"))
        store.save_order(self._make_order(order_id="o5", state="partially_filled"))
        active = store.load_active_orders()
        active_ids = {o["order_id"] for o in active}
        assert active_ids == {"o1", "o5"}

    def test_load_active_empty(self, store):
        assert store.load_active_orders() == []

    def test_order_fields_roundtrip(self, store):
        order = self._make_order(
            order_type="limit",
            limit_price=149.50,
            requested_qty=10.0,
            error="test error",
        )
        store.save_order(order)
        loaded = store.load_active_orders()[0]
        assert loaded["order_type"] == "limit"
        assert loaded["limit_price"] == 149.50
        assert loaded["error"] == "test error"


# ── Bulk engine state ──────────────────────────────────────────────

class TestBulkOperations:
    def test_save_engine_state_all_components(self, store):
        stops = {"AAPL": {"entry_price": 150.0, "highest_price": 155.0, "stop_pct": 0.02, "entry_time": 0.0}}
        cooldowns = {"AAPL": 1700000000.0}
        pdt = {"TSLA": "2026-03-30T10:30:00"}
        scalars = {"halted": True, "halt_reason": "drawdown", "daily_start_equity": 99000.0, "cycle_count": 100}

        store.save_engine_state(stops, cooldowns, pdt, scalars)

        state = store.load_engine_state()
        assert state["trailing_stops"] == stops
        assert state["cooldowns"] == cooldowns
        assert state["pdt_buys"] == pdt
        assert state["scalars"]["halted"] is True
        assert state["scalars"]["halt_reason"] == "drawdown"
        assert state["scalars"]["daily_start_equity"] == 99000.0
        assert state["scalars"]["cycle_count"] == 100

    def test_load_engine_state_empty_db(self, store):
        state = store.load_engine_state()
        assert state["trailing_stops"] == {}
        assert state["cooldowns"] == {}
        assert state["pdt_buys"] == {}
        assert state["scalars"]["halted"] is False
        assert state["scalars"]["halt_reason"] == ""
        assert state["scalars"]["daily_start_equity"] == 0.0

    def test_save_load_roundtrip(self, store):
        stops = {
            "BTC/USD": {"entry_price": 60000.0, "highest_price": 62000.0, "stop_pct": 0.015, "entry_time": 0.0},
            "ETH/USD": {"entry_price": 3000.0, "highest_price": 3100.0, "stop_pct": 0.015, "entry_time": 0.0},
        }
        cooldowns = {"BTC/USD": 1700000000.0, "AAPL": 1700000500.0}
        pdt = {"AAPL": "2026-03-30T10:00:00", "NVDA": "2026-03-30T11:00:00"}
        scalars = {"halted": False, "halt_reason": "", "daily_start_equity": 100000.0, "cycle_count": 50}

        store.save_engine_state(stops, cooldowns, pdt, scalars)
        state = store.load_engine_state()

        assert state["trailing_stops"] == stops
        assert state["cooldowns"] == cooldowns
        assert state["pdt_buys"] == pdt
        assert state["scalars"] == scalars

    def test_save_engine_state_replaces_previous(self, store):
        store.save_engine_state(
            {"AAPL": {"entry_price": 100.0, "highest_price": 105.0, "stop_pct": 0.02, "entry_time": 0.0}},
            {"AAPL": 1000.0}, {}, {"halted": False, "halt_reason": "", "daily_start_equity": 50000.0, "cycle_count": 1},
        )
        store.save_engine_state(
            {"TSLA": {"entry_price": 200.0, "highest_price": 210.0, "stop_pct": 0.02, "entry_time": 0.0}},
            {"TSLA": 2000.0}, {}, {"halted": True, "halt_reason": "test", "daily_start_equity": 48000.0, "cycle_count": 2},
        )
        state = store.load_engine_state()
        assert "AAPL" not in state["trailing_stops"]
        assert "TSLA" in state["trailing_stops"]
        assert state["scalars"]["halted"] is True
