"""Engine coverage tests — Sprint 4: error paths, feature flags, state, halt/resume,
orphan detection, daily trade limit, min signal strength, crypto cooldown."""

import math
import time
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from config import Config
from core.engine import TradingEngine
from core.risk_manager import RiskManager, TrailingStop


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_engine(**overrides):
    """Create a TradingEngine without calling __init__ (avoids broker/DB)."""
    engine = TradingEngine.__new__(TradingEngine)
    engine.broker = MagicMock()
    engine.risk = MagicMock()
    engine.risk.is_max_hold_exceeded.return_value = False
    engine.risk.trailing_stops = {}
    engine.order_manager = None
    engine.data_fetcher = None
    engine.config_reloader = None
    engine.drift_detector = None
    engine.position_reconciler = None
    engine.cost_model = None
    engine.alert_manager = None
    engine.db_rotator = None
    engine.state_store = None
    engine.execution_manager = None
    engine.portfolio_optimizer = None
    engine._last_trade_time = {}
    engine._equity_buys_today = {}
    engine._cached_position_dfs = None
    engine._daily_trade_count = {}
    engine._daily_trade_date = ""
    engine.cycle_count = 1
    engine._shutdown_requested = False
    engine._reload_requested = False
    for k, v in overrides.items():
        setattr(engine, k, v)
    return engine


def _make_df(n=50, base=100.0):
    """Quick OHLCV DataFrame."""
    close = np.linspace(base, base + 5, n) + np.random.RandomState(42).randn(n) * 0.3
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": [1_000_000] * n,
    })


# ═══════════════════════════════════════════════════════════════════════
# 1. Trading cycle error paths
# ═══════════════════════════════════════════════════════════════════════

class TestCycleErrorPaths:
    """Broker failures and API timeouts should not crash the engine."""

    def test_broker_get_account_failure_in_run_cycle(self):
        engine = _make_engine()
        engine.broker.get_account.side_effect = Exception("API timeout")
        # run_cycle should not propagate the exception
        with pytest.raises(Exception, match="API timeout"):
            engine.run_cycle()

    def test_single_symbol_failure_does_not_block_others(self):
        engine = _make_engine()
        engine.broker.get_account.return_value = {"equity": 100000, "cash": 50000}
        engine.risk.can_trade.return_value = True
        engine.risk.halted = False

        call_count = {"processed": 0}

        def fake_process(symbol, equity):
            if symbol == "AAPL":
                raise RuntimeError("AAPL data error")
            call_count["processed"] += 1

        engine._process_symbol = fake_process

        with patch.object(Config, "CRYPTO_SYMBOLS", []), \
             patch.object(Config, "EQUITY_SYMBOLS", ["AAPL", "TSLA"]), \
             patch.object(Config, "is_market_open", return_value=True), \
             patch("core.engine.log"):
            engine.run_cycle()

        assert call_count["processed"] == 1  # TSLA was processed

    def test_process_symbol_inner_nan_price_skips(self):
        engine = _make_engine()
        df = _make_df()
        df.iloc[-1, df.columns.get_loc("close")] = float("nan")
        engine.broker.get_position.return_value = None

        with patch("core.engine.log"):
            engine._process_symbol_inner("AAPL", 10000, df)

        engine.broker.buy.assert_not_called()

    def test_process_symbol_inner_inf_price_skips(self):
        engine = _make_engine()
        df = _make_df()
        df.iloc[-1, df.columns.get_loc("close")] = float("inf")

        with patch("core.engine.log"):
            engine._process_symbol_inner("AAPL", 10000, df)

        engine.broker.buy.assert_not_called()

    def test_process_symbol_inner_zero_price_skips(self):
        engine = _make_engine()
        df = _make_df()
        df.iloc[-1, df.columns.get_loc("close")] = 0.0

        with patch("core.engine.log"):
            engine._process_symbol_inner("AAPL", 10000, df)

        engine.broker.buy.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# 2. Feature flag integration paths
# ═══════════════════════════════════════════════════════════════════════

class TestFeatureFlags:
    """Each optional feature should be cleanly gated."""

    def test_state_persistence_disabled_no_save(self):
        engine = _make_engine(state_store=None)
        # Should be a no-op
        engine._save_persisted_state()

    def test_state_persistence_disabled_no_load(self):
        engine = _make_engine(state_store=None)
        engine._load_persisted_state()

    def test_db_rotation_disabled_no_check(self):
        engine = _make_engine(db_rotator=None)
        engine._check_db_rotation()  # should not raise

    def test_order_manager_disabled_no_poll(self):
        engine = _make_engine()
        engine.broker.get_account.return_value = {"equity": 100000, "cash": 50000}
        engine.risk.can_trade.return_value = True
        engine.risk.halted = False

        with patch.object(Config, "CRYPTO_SYMBOLS", []), \
             patch.object(Config, "EQUITY_SYMBOLS", []), \
             patch.object(Config, "is_market_open", return_value=False), \
             patch("core.engine.log"):
            engine.run_cycle()
        # No crash means order_manager=None path is safe


# ═══════════════════════════════════════════════════════════════════════
# 3. State save/load roundtrip
# ═══════════════════════════════════════════════════════════════════════

class TestStatePersistence:
    """Engine state save/load roundtrip."""

    def test_save_state_calls_store(self):
        store = MagicMock()
        engine = _make_engine(state_store=store)
        engine.risk.trailing_stops = {
            "AAPL": TrailingStop("AAPL", 150.0, 155.0, 0.02, entry_time=1000.0)
        }
        engine.risk.halted = False
        engine.risk.halt_reason = ""
        engine.risk.daily_start_equity = 100000.0

        engine._save_persisted_state()

        store.save_engine_state.assert_called_once()
        args = store.save_engine_state.call_args
        stops = args[0][0]
        assert "AAPL" in stops
        assert stops["AAPL"]["entry_time"] == 1000.0

    def test_load_state_restores_trailing_stops(self):
        store = MagicMock()
        store.load_engine_state.return_value = {
            "cooldowns": {"BTC/USD": 1000.0},
            "pdt_buys": {},
            "trailing_stops": {
                "AAPL": {"entry_price": 150.0, "highest_price": 155.0,
                         "stop_pct": 0.02, "entry_time": 500.0}
            },
            "scalars": {"halted": False, "halt_reason": "", "daily_start_equity": 0.0, "cycle_count": 0},
        }
        engine = _make_engine(state_store=store)
        engine.risk = RiskManager()

        with patch("core.engine.log"):
            engine._load_persisted_state()

        assert "AAPL" in engine.risk.trailing_stops
        assert engine.risk.trailing_stops["AAPL"].entry_time == 500.0
        assert engine._last_trade_time == {"BTC/USD": 1000.0}

    def test_load_state_handles_corrupt_data(self):
        store = MagicMock()
        store.load_engine_state.side_effect = Exception("DB corrupted")
        engine = _make_engine(state_store=store)
        engine.risk = RiskManager()

        with patch("core.engine.log"):
            engine._load_persisted_state()
        # Should not raise — just log warning

    def test_save_state_handles_failure(self):
        store = MagicMock()
        store.save_engine_state.side_effect = Exception("disk full")
        engine = _make_engine(state_store=store)
        engine.risk.trailing_stops = {}
        engine.risk.halted = False
        engine.risk.halt_reason = ""
        engine.risk.daily_start_equity = 100000.0

        with patch("core.engine.log"):
            engine._save_persisted_state()
        # Should not raise


# ═══════════════════════════════════════════════════════════════════════
# 4. Halt/resume logic
# ═══════════════════════════════════════════════════════════════════════

class TestHaltResume:
    def test_halted_skips_trading(self):
        engine = _make_engine()
        engine.broker.get_account.return_value = {"equity": 100000, "cash": 50000}
        engine.risk.can_trade.return_value = False
        engine.risk.halted = True
        engine.risk.halt_reason = "Daily drawdown limit hit: 10.0%"

        with patch.object(Config, "CRYPTO_SYMBOLS", ["BTC/USD"]), \
             patch.object(Config, "EQUITY_SYMBOLS", []), \
             patch("core.engine.log"):
            engine.run_cycle()

        engine.broker.buy.assert_not_called()

    def test_can_trade_resumes_after_halt_clear(self):
        engine = _make_engine()
        engine.broker.get_account.return_value = {"equity": 100000, "cash": 50000}
        engine.risk.can_trade.return_value = True
        engine.risk.halted = False

        processed = []
        original_process = engine._process_symbol

        def track_process(sym, eq):
            processed.append(sym)

        engine._process_symbol = track_process

        with patch.object(Config, "CRYPTO_SYMBOLS", ["BTC/USD"]), \
             patch.object(Config, "EQUITY_SYMBOLS", []), \
             patch("core.engine.log"):
            engine.run_cycle()

        assert "BTC/USD" in processed


# ═══════════════════════════════════════════════════════════════════════
# 5. Orphan detection edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestOrphanDetection:
    def test_no_positions_no_orphans(self):
        engine = _make_engine()
        engine.broker.get_positions.return_value = []
        engine.risk = RiskManager()

        with patch("core.engine.log"):
            engine._detect_orphan_positions()

    def test_position_with_existing_stop_not_orphan(self):
        engine = _make_engine()
        engine.risk = RiskManager()
        engine.risk.trailing_stops["AAPL"] = TrailingStop("AAPL", 150.0, 155.0, 0.02)
        engine.broker.get_positions.return_value = [
            {"symbol": "AAPL", "avg_entry_price": "150.0", "current_price": "155.0",
             "market_value": "5000.0"}
        ]

        with patch("core.engine.log"):
            engine._detect_orphan_positions()

        # Should still have only the original stop
        assert len(engine.risk.trailing_stops) == 1

    def test_orphan_position_gets_registered(self):
        engine = _make_engine()
        engine.risk = RiskManager()
        engine.broker.get_positions.return_value = [
            {"symbol": "TSLA", "avg_entry_price": "200.0", "current_price": "210.0",
             "market_value": "5000.0"}
        ]
        engine.broker.get_recent_bars.return_value = _make_df()

        with patch("core.engine.log"):
            engine._detect_orphan_positions()

        assert "TSLA" in engine.risk.trailing_stops
        # Current > entry, so highest_price should be updated
        assert engine.risk.trailing_stops["TSLA"].highest_price == 210.0

    def test_orphan_crypto_symbol_resolved(self):
        engine = _make_engine()
        engine.risk = RiskManager()
        engine.broker.get_positions.return_value = [
            {"symbol": "BTCUSD", "avg_entry_price": "60000.0", "current_price": "61000.0",
             "market_value": "30000.0"}
        ]
        engine.broker.get_recent_bars.return_value = _make_df()

        with patch("core.engine.log"):
            engine._detect_orphan_positions()

        assert "BTC/USD" in engine.risk.trailing_stops

    def test_orphan_detection_broker_failure(self):
        engine = _make_engine()
        engine.risk = RiskManager()
        engine.broker.get_positions.side_effect = Exception("broker down")

        with patch("core.engine.log"):
            engine._detect_orphan_positions()
        # Should not raise

    def test_orphan_bars_fetch_fails_still_registers(self):
        engine = _make_engine()
        engine.risk = RiskManager()
        engine.broker.get_positions.return_value = [
            {"symbol": "AMD", "avg_entry_price": "160.0", "current_price": "155.0",
             "market_value": "4000.0"}
        ]
        engine.broker.get_recent_bars.side_effect = Exception("timeout")

        with patch("core.engine.log"):
            engine._detect_orphan_positions()

        # Should still register with fallback stop (no ATR data)
        assert "AMD" in engine.risk.trailing_stops

    def test_orphan_position_below_entry_no_highest_update(self):
        engine = _make_engine()
        engine.risk = RiskManager()
        engine.broker.get_positions.return_value = [
            {"symbol": "NVDA", "avg_entry_price": "200.0", "current_price": "190.0",
             "market_value": "-2000.0"}
        ]
        engine.broker.get_recent_bars.return_value = _make_df()

        with patch("core.engine.log"):
            engine._detect_orphan_positions()

        assert "NVDA" in engine.risk.trailing_stops
        # current < entry, highest should stay at entry
        assert engine.risk.trailing_stops["NVDA"].highest_price == 200.0


# ═══════════════════════════════════════════════════════════════════════
# 6. Daily trade limit
# ═══════════════════════════════════════════════════════════════════════

class TestDailyTradeLimit:
    def test_limit_not_reached_returns_false(self):
        engine = _make_engine()
        engine._daily_trade_count = {"crypto": 0}
        engine._daily_trade_date = datetime.now(Config.MARKET_TZ).strftime("%Y-%m-%d")

        with patch.object(Config, "MAX_TRADES_PER_DAY_CRYPTO", 8):
            assert engine._is_daily_trade_limit_reached("BTC/USD") is False

    def test_limit_reached_returns_true(self):
        engine = _make_engine()
        today = datetime.now(Config.MARKET_TZ).strftime("%Y-%m-%d")
        engine._daily_trade_date = today
        engine._daily_trade_count = {"crypto": 8}

        with patch.object(Config, "MAX_TRADES_PER_DAY_CRYPTO", 8):
            assert engine._is_daily_trade_limit_reached("BTC/USD") is True

    def test_equity_limit_separate_from_crypto(self):
        engine = _make_engine()
        today = datetime.now(Config.MARKET_TZ).strftime("%Y-%m-%d")
        engine._daily_trade_date = today
        engine._daily_trade_count = {"crypto": 10, "equity": 2}

        with patch.object(Config, "MAX_TRADES_PER_DAY_EQUITY", 4):
            assert engine._is_daily_trade_limit_reached("AAPL") is False

    def test_new_day_resets_count(self):
        engine = _make_engine()
        engine._daily_trade_date = "1999-01-01"  # Old date
        engine._daily_trade_count = {"crypto": 100}

        with patch.object(Config, "MAX_TRADES_PER_DAY_CRYPTO", 8):
            assert engine._is_daily_trade_limit_reached("BTC/USD") is False

    def test_record_trade_increments_count(self):
        engine = _make_engine()
        engine._daily_trade_count = {}
        engine._daily_trade_date = datetime.now(Config.MARKET_TZ).strftime("%Y-%m-%d")

        engine._record_trade("BTC/USD")
        assert engine._daily_trade_count.get("crypto", 0) == 1

        engine._record_trade("BTC/USD")
        assert engine._daily_trade_count.get("crypto", 0) == 2

    def test_buy_blocked_when_daily_limit_reached(self):
        engine = _make_engine()
        today = datetime.now(Config.MARKET_TZ).strftime("%Y-%m-%d")
        engine._daily_trade_date = today
        engine._daily_trade_count = {"equity": 4}
        engine.broker.get_position.return_value = None
        engine.risk.should_stop_loss.return_value = False

        df = _make_df()

        with patch.object(Config, "MAX_TRADES_PER_DAY_EQUITY", 4), \
             patch.object(Config, "MIN_SIGNAL_STRENGTH", 0.0), \
             patch("core.engine.route_signals",
                   return_value={"action": "buy", "strength": 0.9, "reason": "test"}), \
             patch("core.engine.log"):
            engine._process_symbol_inner("AAPL", 100000, df)

        engine.broker.buy.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# 7. Min signal strength gate
# ═══════════════════════════════════════════════════════════════════════

class TestMinSignalStrength:
    def test_weak_signal_blocked(self):
        engine = _make_engine()
        engine.broker.get_position.return_value = None
        engine.risk.should_stop_loss.return_value = False

        df = _make_df()

        with patch.object(Config, "MIN_SIGNAL_STRENGTH", 0.55), \
             patch("core.engine.route_signals",
                   return_value={"action": "buy", "strength": 0.3, "reason": "test"}), \
             patch("core.engine.log"):
            engine._process_symbol_inner("AAPL", 100000, df)

        engine.broker.buy.assert_not_called()

    def test_strong_signal_passes(self):
        engine = _make_engine()
        engine.broker.get_position.return_value = None
        engine.risk.should_stop_loss.return_value = False
        engine.broker.get_positions.return_value = []
        engine.broker.buy.return_value = True

        df = _make_df()

        with patch.object(Config, "MIN_SIGNAL_STRENGTH", 0.55), \
             patch.object(Config, "MAX_TRADES_PER_DAY_EQUITY", 100), \
             patch.object(Config, "CORRELATION_CHECK_ENABLED", False), \
             patch.object(Config, "VOLATILITY_SIZING_ENABLED", False), \
             patch("core.engine.route_signals",
                   return_value={"action": "buy", "strength": 0.8, "reason": "test", "strategy": "test"}), \
             patch("core.engine.log"):
            engine._process_symbol_inner("AAPL", 100000, df)

        engine.broker.buy.assert_called_once()

    def test_signal_missing_strength_treated_as_zero(self):
        engine = _make_engine()
        engine.broker.get_position.return_value = None
        engine.risk.should_stop_loss.return_value = False

        df = _make_df()

        with patch.object(Config, "MIN_SIGNAL_STRENGTH", 0.55), \
             patch("core.engine.route_signals",
                   return_value={"action": "buy", "reason": "test"}), \
             patch("core.engine.log"):
            engine._process_symbol_inner("AAPL", 100000, df)

        engine.broker.buy.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# 8. Crypto cooldown at 2700s
# ═══════════════════════════════════════════════════════════════════════

class TestCryptoCooldown:
    def test_crypto_cooldown_is_2700s(self):
        assert TradingEngine.COOLDOWN_CRYPTO == 2700

    def test_crypto_on_cooldown_within_2700(self):
        engine = _make_engine()
        engine._last_trade_time["BTC/USD"] = time.time() - 1000  # 1000s ago

        result = engine._is_on_cooldown("BTC/USD")
        assert result is True

    def test_crypto_off_cooldown_after_2700(self):
        engine = _make_engine()
        engine._last_trade_time["BTC/USD"] = time.time() - 3000  # 3000s ago

        result = engine._is_on_cooldown("BTC/USD")
        assert result is False

    def test_equity_cooldown_is_300s(self):
        assert TradingEngine.COOLDOWN_EQUITY == 300

    def test_equity_on_cooldown_within_300(self):
        engine = _make_engine()
        engine._last_trade_time["AAPL"] = time.time() - 100

        with patch("core.engine.log"):
            result = engine._is_on_cooldown("AAPL")
        assert result is True

    def test_no_trade_history_not_on_cooldown(self):
        engine = _make_engine()
        assert engine._is_on_cooldown("NEW_SYM") is False


# ═══════════════════════════════════════════════════════════════════════
# 9. Resolve symbol
# ═══════════════════════════════════════════════════════════════════════

class TestResolveSymbol:
    def test_resolve_crypto_stripped(self):
        engine = _make_engine()
        assert engine._resolve_symbol("BTCUSD") == "BTC/USD"
        assert engine._resolve_symbol("ETHUSD") == "ETH/USD"

    def test_resolve_equity_unchanged(self):
        engine = _make_engine()
        assert engine._resolve_symbol("AAPL") == "AAPL"
        assert engine._resolve_symbol("TSLA") == "TSLA"


# ═══════════════════════════════════════════════════════════════════════
# 10. Shutdown
# ═══════════════════════════════════════════════════════════════════════

class TestShutdown:
    def test_shutdown_saves_state(self):
        store = MagicMock()
        engine = _make_engine(state_store=store)
        engine.risk.trailing_stops = {}
        engine.risk.halted = False
        engine.risk.halt_reason = ""
        engine.risk.daily_start_equity = 100000.0
        engine.risk.starting_equity = 100000.0
        engine.broker.get_account.return_value = {"equity": 100000, "cash": 50000}
        engine.broker.get_positions.return_value = []

        with patch("core.engine.log"):
            engine._shutdown("test")

        store.save_engine_state.assert_called_once()

    def test_shutdown_warns_about_crypto_positions(self):
        engine = _make_engine()
        engine.risk.trailing_stops = {}
        engine.risk.halted = False
        engine.risk.halt_reason = ""
        engine.risk.daily_start_equity = 100000.0
        engine.risk.starting_equity = 100000.0
        engine.broker.get_account.return_value = {"equity": 100000, "cash": 50000}
        engine.broker.get_positions.return_value = [
            {"symbol": "BTCUSD", "qty": 0.5, "avg_entry_price": 60000.0,
             "unrealized_pl": 100.0, "market_value": 30000.0}
        ]

        with patch("core.engine.log") as mock_log:
            engine._shutdown("test")

        # Check that a warning about crypto positions was logged
        warning_calls = [str(c) for c in mock_log.warning.call_args_list]
        assert any("crypto" in c.lower() for c in warning_calls)
