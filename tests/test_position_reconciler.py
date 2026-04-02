"""Tests for core/position_reconciler.py — position reconciliation."""

import pytest
from unittest.mock import patch

from core.position_reconciler import (
    PositionReconciler,
    ReconciliationResult,
    Mismatch,
    _normalize_symbol,
    _build_broker_map,
    _build_engine_map,
)
from core.risk_manager import TrailingStop


# ── Helpers ──────────────────────────────────────────────────────────

def make_broker_position(symbol: str, qty: float = 1.0, market_value: float = 100.0,
                         avg_entry_price: float = 100.0, **kwargs) -> dict:
    return {
        "symbol": symbol,
        "qty": qty,
        "market_value": market_value,
        "unrealized_pl": kwargs.get("unrealized_pl", 0.0),
        "unrealized_plpc": kwargs.get("unrealized_plpc", 0.0),
        "avg_entry_price": avg_entry_price,
        "current_price": kwargs.get("current_price", market_value / qty if qty else 100.0),
        "asset_class": kwargs.get("asset_class", "equity"),
    }


def make_trailing_stop(symbol: str, entry_price: float = 100.0,
                       highest_price: float = 105.0, stop_pct: float = 0.02) -> TrailingStop:
    return TrailingStop(
        symbol=symbol,
        entry_price=entry_price,
        highest_price=highest_price,
        stop_pct=stop_pct,
    )


# ── Unit Tests: Helpers ──────────────────────────────────────────────

class TestNormalizeSymbol:
    def test_strips_slash(self):
        assert _normalize_symbol("BTC/USD") == "BTCUSD"

    def test_no_slash_unchanged(self):
        assert _normalize_symbol("AAPL") == "AAPL"

    def test_multiple_slashes(self):
        assert _normalize_symbol("A/B/C") == "ABC"

    def test_empty_string(self):
        assert _normalize_symbol("") == ""


class TestBuildBrokerMap:
    def test_builds_normalized_map(self):
        positions = [
            make_broker_position("BTCUSD", qty=0.5, market_value=15000.0),
            make_broker_position("AAPL", qty=10, market_value=1500.0),
        ]
        result = _build_broker_map(positions)
        assert "BTCUSD" in result
        assert "AAPL" in result
        assert result["BTCUSD"]["qty"] == 0.5

    def test_empty_positions(self):
        assert _build_broker_map([]) == {}


class TestBuildEngineMap:
    def test_builds_normalized_map(self):
        stops = {
            "BTC/USD": make_trailing_stop("BTC/USD", entry_price=30000.0),
            "AAPL": make_trailing_stop("AAPL", entry_price=150.0),
        }
        result = _build_engine_map(stops)
        assert "BTCUSD" in result
        assert "AAPL" in result
        assert result["BTCUSD"].entry_price == 30000.0

    def test_empty_stops(self):
        assert _build_engine_map({}) == {}


# ── ReconciliationResult ─────────────────────────────────────────────

class TestReconciliationResult:
    def test_ok_when_no_mismatches(self):
        r = ReconciliationResult()
        assert r.ok is True
        assert r.count == 0

    def test_not_ok_with_mismatches(self):
        r = ReconciliationResult(mismatches=[Mismatch("AAPL", "orphaned_stop")])
        assert r.ok is False
        assert r.count == 1


# ── Core Reconciliation ──────────────────────────────────────────────

class TestReconcile:
    def setup_method(self):
        self.reconciler = PositionReconciler(entry_price_tolerance=0.02)

    def test_perfect_match_no_mismatches(self):
        """Engine and broker agree on all positions — no mismatches."""
        stops = {"AAPL": make_trailing_stop("AAPL", entry_price=150.0)}
        positions = [make_broker_position("AAPL", avg_entry_price=150.0)]
        result = self.reconciler.reconcile(stops, positions)
        assert result.ok

    def test_perfect_match_crypto_symbol(self):
        """Crypto symbols with '/' should match broker's slash-stripped format."""
        stops = {"BTC/USD": make_trailing_stop("BTC/USD", entry_price=30000.0)}
        positions = [make_broker_position("BTCUSD", avg_entry_price=30000.0)]
        result = self.reconciler.reconcile(stops, positions)
        assert result.ok

    def test_orphaned_stop_detected(self):
        """Engine has a stop but broker has no position."""
        stops = {"AAPL": make_trailing_stop("AAPL", entry_price=150.0)}
        positions = []  # Broker has nothing
        result = self.reconciler.reconcile(stops, positions)
        assert not result.ok
        assert result.count == 1
        assert result.mismatches[0].issue == "orphaned_stop"
        assert result.mismatches[0].symbol == "AAPL"

    def test_untracked_position_detected(self):
        """Broker has a position the engine doesn't track."""
        stops = {}
        positions = [make_broker_position("TSLA", qty=5, market_value=1000.0)]
        result = self.reconciler.reconcile(stops, positions)
        assert not result.ok
        assert result.count == 1
        assert result.mismatches[0].issue == "untracked_position"

    def test_external_position_when_tracked_symbols_provided(self):
        """Positions outside the symbol list are flagged as external."""
        stops = {}
        positions = [make_broker_position("GOOG", qty=2, market_value=500.0)]
        result = self.reconciler.reconcile(stops, positions, tracked_symbols=["AAPL", "TSLA"])
        assert not result.ok
        assert result.mismatches[0].issue == "external_position"

    def test_untracked_within_symbol_list(self):
        """Position in symbol list but no trailing stop → untracked, not external."""
        stops = {}
        positions = [make_broker_position("AAPL", qty=10, market_value=1500.0)]
        result = self.reconciler.reconcile(stops, positions, tracked_symbols=["AAPL", "TSLA"])
        assert result.mismatches[0].issue == "untracked_position"

    def test_entry_price_drift_detected(self):
        """Entry price drift exceeding tolerance is flagged."""
        stops = {"AAPL": make_trailing_stop("AAPL", entry_price=150.0)}
        # Broker says avg_entry was 160 — that's 6.7% drift, above 2% tolerance
        positions = [make_broker_position("AAPL", avg_entry_price=160.0)]
        result = self.reconciler.reconcile(stops, positions)
        assert not result.ok
        assert result.mismatches[0].issue == "entry_price_drift"

    def test_entry_price_within_tolerance(self):
        """Small entry price difference within tolerance is OK."""
        stops = {"AAPL": make_trailing_stop("AAPL", entry_price=150.0)}
        # Broker says 151.0 — 0.67% drift, within 2% tolerance
        positions = [make_broker_position("AAPL", avg_entry_price=151.0)]
        result = self.reconciler.reconcile(stops, positions)
        assert result.ok

    def test_custom_tolerance(self):
        """Custom entry price tolerance is respected."""
        reconciler = PositionReconciler(entry_price_tolerance=0.10)  # 10%
        stops = {"AAPL": make_trailing_stop("AAPL", entry_price=150.0)}
        # 6.7% drift — within 10% tolerance
        positions = [make_broker_position("AAPL", avg_entry_price=160.0)]
        result = reconciler.reconcile(stops, positions)
        assert result.ok

    def test_multiple_mismatches(self):
        """Multiple issues detected in a single reconciliation run."""
        stops = {
            "AAPL": make_trailing_stop("AAPL", entry_price=150.0),
            "NVDA": make_trailing_stop("NVDA", entry_price=300.0),  # orphaned
        }
        positions = [
            make_broker_position("AAPL", avg_entry_price=200.0),  # drift
            make_broker_position("TSLA", qty=5, market_value=1000.0),  # untracked
        ]
        result = self.reconciler.reconcile(stops, positions)
        assert result.count == 3  # drift + orphaned + untracked
        issues = {m.issue for m in result.mismatches}
        assert issues == {"entry_price_drift", "orphaned_stop", "untracked_position"}

    def test_empty_both_sides(self):
        """No positions and no stops — perfect match."""
        result = self.reconciler.reconcile({}, [])
        assert result.ok

    def test_broker_and_engine_symbols_tracked(self):
        """Result tracks which symbols each side has."""
        stops = {"BTC/USD": make_trailing_stop("BTC/USD")}
        positions = [make_broker_position("AAPL")]
        result = self.reconciler.reconcile(stops, positions)
        assert "BTCUSD" in result.engine_symbols
        assert "AAPL" in result.broker_symbols

    def test_last_result_stored(self):
        """Reconciler stores the most recent result."""
        assert self.reconciler.last_result is None
        result = self.reconciler.reconcile({}, [])
        assert self.reconciler.last_result is result

    def test_zero_broker_entry_price_skips_drift_check(self):
        """Zero avg_entry_price from broker should not cause a false drift alert."""
        stops = {"AAPL": make_trailing_stop("AAPL", entry_price=150.0)}
        positions = [make_broker_position("AAPL", avg_entry_price=0.0)]
        result = self.reconciler.reconcile(stops, positions)
        # Should not flag drift — broker entry is 0 (unknown)
        assert result.ok

    def test_zero_engine_entry_price_skips_drift_check(self):
        """Zero engine entry price should not cause false drift."""
        stops = {"AAPL": make_trailing_stop("AAPL", entry_price=0.0)}
        positions = [make_broker_position("AAPL", avg_entry_price=150.0)]
        result = self.reconciler.reconcile(stops, positions)
        assert result.ok

    def test_multiple_crypto_symbols(self):
        """Multiple crypto symbols reconcile correctly."""
        stops = {
            "BTC/USD": make_trailing_stop("BTC/USD", entry_price=30000.0),
            "ETH/USD": make_trailing_stop("ETH/USD", entry_price=2000.0),
        }
        positions = [
            make_broker_position("BTCUSD", avg_entry_price=30000.0),
            make_broker_position("ETHUSD", avg_entry_price=2000.0),
        ]
        result = self.reconciler.reconcile(stops, positions)
        assert result.ok

    def test_entry_price_drift_at_exact_boundary(self):
        """Drift exactly at tolerance boundary should NOT be flagged (strict >)."""
        # 2% tolerance: entry 100 vs broker 102 = exactly 2% drift
        stops = {"AAPL": make_trailing_stop("AAPL", entry_price=100.0)}
        positions = [make_broker_position("AAPL", avg_entry_price=102.0)]
        result = self.reconciler.reconcile(stops, positions)
        assert result.ok  # Exactly at boundary, not exceeded

    def test_entry_price_drift_just_above_boundary(self):
        """Drift just above tolerance should be flagged."""
        # 2% tolerance: entry 100 vs broker 102.1 = 2.06% drift
        stops = {"AAPL": make_trailing_stop("AAPL", entry_price=100.0)}
        positions = [make_broker_position("AAPL", avg_entry_price=102.1)]
        result = self.reconciler.reconcile(stops, positions)
        assert not result.ok
        assert result.mismatches[0].issue == "entry_price_drift"


# ── Auto-Fix Orphaned Stops ──────────────────────────────────────────

class TestAutoFixOrphanedStops:
    def setup_method(self):
        self.reconciler = PositionReconciler()

    def test_removes_orphaned_stops(self):
        stops = {
            "AAPL": make_trailing_stop("AAPL"),
            "NVDA": make_trailing_stop("NVDA"),
        }
        result = ReconciliationResult(mismatches=[
            Mismatch("AAPL", "orphaned_stop", "Engine has stop but broker has no position"),
        ])
        cleaned = self.reconciler.auto_fix_orphaned_stops(stops, result)
        assert "AAPL" in cleaned
        assert "AAPL" not in stops
        assert "NVDA" in stops  # untouched

    def test_removes_crypto_orphaned_stop(self):
        stops = {"BTC/USD": make_trailing_stop("BTC/USD")}
        result = ReconciliationResult(mismatches=[
            Mismatch("BTC/USD", "orphaned_stop"),
        ])
        cleaned = self.reconciler.auto_fix_orphaned_stops(stops, result)
        assert "BTC/USD" in cleaned
        assert len(stops) == 0

    def test_ignores_non_orphaned_mismatches(self):
        stops = {"AAPL": make_trailing_stop("AAPL")}
        result = ReconciliationResult(mismatches=[
            Mismatch("AAPL", "entry_price_drift", "drift details"),
        ])
        cleaned = self.reconciler.auto_fix_orphaned_stops(stops, result)
        assert cleaned == []
        assert "AAPL" in stops  # untouched

    def test_empty_mismatches(self):
        stops = {"AAPL": make_trailing_stop("AAPL")}
        result = ReconciliationResult()
        cleaned = self.reconciler.auto_fix_orphaned_stops(stops, result)
        assert cleaned == []

    def test_multiple_orphans(self):
        stops = {
            "AAPL": make_trailing_stop("AAPL"),
            "TSLA": make_trailing_stop("TSLA"),
            "NVDA": make_trailing_stop("NVDA"),
        }
        result = ReconciliationResult(mismatches=[
            Mismatch("AAPL", "orphaned_stop"),
            Mismatch("TSLA", "orphaned_stop"),
        ])
        cleaned = self.reconciler.auto_fix_orphaned_stops(stops, result)
        assert set(cleaned) == {"AAPL", "TSLA"}
        assert list(stops.keys()) == ["NVDA"]


# ── Engine Integration Tests ─────────────────────────────────────────

class TestEngineReconciliation:
    """Test reconciliation integration in engine.py."""

    @pytest.fixture
    def engine(self):
        from unittest.mock import MagicMock, patch, PropertyMock
        with patch.object(__import__("config", fromlist=["Config"]).Config,
                          "POSITION_RECONCILIATION_ENABLED", True), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "RECONCILIATION_INTERVAL_CYCLES", 10), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "RECONCILIATION_AUTO_FIX", False), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "RECONCILIATION_ENTRY_TOLERANCE", 0.02), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "STATE_PERSISTENCE_ENABLED", False), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "ORDER_MANAGEMENT_ENABLED", False), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "PARALLEL_FETCH_ENABLED", False), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "HOT_RELOAD_ENABLED", False), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "DRIFT_DETECTION_ENABLED", False), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "ALERTING_ENABLED", False), \
             patch("core.engine.Broker") as MockBroker:
            from core.engine import TradingEngine
            engine = TradingEngine()
            mock_broker = MockBroker.return_value
            mock_broker.get_account.return_value = {"equity": 100000.0, "cash": 50000.0}
            mock_broker.get_positions.return_value = []
            engine.broker = mock_broker
            yield engine

    def test_reconciler_created_when_enabled(self, engine):
        assert engine.position_reconciler is not None

    def test_reconciler_not_created_when_disabled(self):
        from unittest.mock import patch, MagicMock
        with patch.object(__import__("config", fromlist=["Config"]).Config,
                          "POSITION_RECONCILIATION_ENABLED", False), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "STATE_PERSISTENCE_ENABLED", False), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "ORDER_MANAGEMENT_ENABLED", False), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "PARALLEL_FETCH_ENABLED", False), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "HOT_RELOAD_ENABLED", False), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "DRIFT_DETECTION_ENABLED", False), \
             patch.object(__import__("config", fromlist=["Config"]).Config,
                          "ALERTING_ENABLED", False), \
             patch("core.engine.Broker"):
            from core.engine import TradingEngine
            engine = TradingEngine()
            assert engine.position_reconciler is None

    def test_run_reconciliation_called_at_interval(self, engine):
        """Reconciliation runs at the configured cycle interval."""
        from unittest.mock import MagicMock, patch
        engine.risk.initialize(100000)
        engine.broker.get_recent_bars.return_value = MagicMock(empty=True)

        with patch.object(engine, "_run_reconciliation") as mock_recon:
            # Cycle 10 should trigger
            engine.cycle_count = 9  # will become 10 in run_cycle
            engine.run_cycle()
            mock_recon.assert_called_once()

    def test_run_reconciliation_skipped_between_intervals(self, engine):
        """Reconciliation is not run between interval cycles."""
        from unittest.mock import MagicMock, patch
        engine.risk.initialize(100000)
        engine.broker.get_recent_bars.return_value = MagicMock(empty=True)

        with patch.object(engine, "_run_reconciliation") as mock_recon:
            engine.cycle_count = 4  # will become 5 — not a multiple of 10
            engine.run_cycle()
            mock_recon.assert_not_called()

    def test_run_reconciliation_logs_mismatches(self, engine):
        """Mismatches are logged as warnings."""
        from unittest.mock import patch
        engine.risk.trailing_stops = {
            "AAPL": make_trailing_stop("AAPL", entry_price=150.0),
        }
        engine.broker.get_positions.return_value = []  # orphaned

        with patch("core.engine.log") as mock_log:
            engine._run_reconciliation()
            # Should have warned about orphaned stop
            assert any("orphaned_stop" in str(call)
                       for call in mock_log.warning.call_args_list)

    def test_run_reconciliation_sends_alert(self, engine):
        """Mismatches trigger an alert when alert_manager is set."""
        from unittest.mock import MagicMock
        engine.alert_manager = MagicMock()
        engine.risk.trailing_stops = {
            "AAPL": make_trailing_stop("AAPL"),
        }
        engine.broker.get_positions.return_value = []
        engine._run_reconciliation()
        engine.alert_manager.position_mismatch.assert_called_once()

    def test_run_reconciliation_no_alert_when_ok(self, engine):
        """No alert sent when reconciliation finds no issues."""
        from unittest.mock import MagicMock
        engine.alert_manager = MagicMock()
        engine.risk.trailing_stops = {}
        engine.broker.get_positions.return_value = []
        engine._run_reconciliation()
        engine.alert_manager.position_mismatch.assert_not_called()

    def test_run_reconciliation_auto_fix(self, engine):
        """Auto-fix removes orphaned stops when enabled."""
        from unittest.mock import patch
        engine.risk.trailing_stops = {
            "AAPL": make_trailing_stop("AAPL"),
        }
        engine.broker.get_positions.return_value = []

        with patch.object(__import__("config", fromlist=["Config"]).Config,
                          "RECONCILIATION_AUTO_FIX", True):
            engine._run_reconciliation()
        assert "AAPL" not in engine.risk.trailing_stops

    def test_run_reconciliation_no_auto_fix_by_default(self, engine):
        """Auto-fix is disabled by default — orphaned stops remain."""
        engine.risk.trailing_stops = {
            "AAPL": make_trailing_stop("AAPL"),
        }
        engine.broker.get_positions.return_value = []
        engine._run_reconciliation()
        assert "AAPL" in engine.risk.trailing_stops

    def test_run_reconciliation_exception_handled(self, engine):
        """Reconciliation errors don't crash the engine."""
        engine.broker.get_positions.side_effect = Exception("API timeout")
        # Should not raise
        engine._run_reconciliation()

    def test_alert_sent_before_auto_fix(self, engine):
        """Alert should report orphaned stops before auto-fix removes them."""
        from unittest.mock import MagicMock, patch, call
        engine.alert_manager = MagicMock()
        engine.risk.trailing_stops = {
            "AAPL": make_trailing_stop("AAPL"),
        }
        engine.broker.get_positions.return_value = []

        call_order = []
        original_mismatch = engine.alert_manager.position_mismatch
        def track_alert(*args, **kwargs):
            # Record that alert was called while AAPL still in stops
            call_order.append(("alert", "AAPL" in engine.risk.trailing_stops))
            return original_mismatch(*args, **kwargs)
        engine.alert_manager.position_mismatch = track_alert

        with patch.object(__import__("config", fromlist=["Config"]).Config,
                          "RECONCILIATION_AUTO_FIX", True):
            engine._run_reconciliation()
        # Alert was called while AAPL was still in trailing_stops (before auto-fix)
        assert call_order == [("alert", True)]
        # But after reconciliation, AAPL should be removed
        assert "AAPL" not in engine.risk.trailing_stops
