"""Sprint 5D: Intraday P&L circuit breaker tests.

Verifies the dollar-aware halt kicks in BEFORE the 10% pct drawdown on small
accounts. A $800 Beginner account should halt at ~$24 loss, not $80.
"""

from unittest.mock import MagicMock, patch

import pytest

from config import Config
from core.risk_manager import RiskManager
from core.user_profile import BEGINNER, HOBBYIST, LEARNER


class TestCheckIntradayHalt:
    def _make_rm(self, start_equity: float) -> RiskManager:
        rm = RiskManager()
        rm.initialize(start_equity)
        return rm

    def test_tiny_loss_does_not_halt(self):
        rm = self._make_rm(1_000)
        # $1K account, profile threshold $30 (3%)
        # Lost $10 — should not halt
        halted = rm.check_intraday_halt(current_equity=990, max_daily_loss_usd=30,
                                        unrealized_pnl=0.0)
        assert halted is False
        assert rm.halted is False

    def test_breach_threshold_halts(self):
        rm = self._make_rm(1_000)
        # Lost $35, threshold $30
        halted = rm.check_intraday_halt(current_equity=965, max_daily_loss_usd=30,
                                        unrealized_pnl=0.0)
        assert halted is True
        assert rm.halted is True
        assert "Intraday realized loss" in rm.halt_reason

    def test_unrealized_early_warn(self):
        rm = self._make_rm(1_000)
        # Realized = 0 (current_equity = start - unrealized + realized)
        # equity=950, unrealized=-50 → realized = 950 - (-50) - 1000 = 0 (no realized loss)
        # But unrealized -50 is 1.67x of $30 threshold → early warning halt
        halted = rm.check_intraday_halt(current_equity=950, max_daily_loss_usd=30,
                                        unrealized_pnl=-50.0)
        assert halted is True
        assert "early-warning" in rm.halt_reason

    def test_unrealized_within_warn_threshold(self):
        rm = self._make_rm(1_000)
        # Unrealized -40, early-warn threshold = 30 * 1.5 = 45 → still under
        halted = rm.check_intraday_halt(current_equity=960, max_daily_loss_usd=30,
                                        unrealized_pnl=-40.0)
        assert halted is False

    def test_zero_start_equity_no_halt(self):
        rm = self._make_rm(0)
        assert rm.check_intraday_halt(0, 50, 0) is False

    def test_zero_threshold_disables_check(self):
        rm = self._make_rm(1_000)
        # threshold=0 means this check is effectively off
        assert rm.check_intraday_halt(800, 0, 0) is False

    def test_profile_threshold_wiring(self):
        """Beginner $800 account with BEGINNER profile: halt at $24 (3% × 800), not $80."""
        rm = self._make_rm(800)
        threshold = BEGINNER.max_daily_loss_usd(800)  # min(24, 50) = 24
        # Lose $25 — should halt
        halted = rm.check_intraday_halt(775, threshold, 0)
        assert halted is True

    def test_learner_threshold_wider(self):
        """A $50K Learner account halts at 5% ($2,500), not $24 — proper headroom."""
        rm = self._make_rm(50_000)
        threshold = LEARNER.max_daily_loss_usd(50_000)  # 5% × 50K = $2,500
        # Lose $500 — should NOT halt (well under $2,500)
        halted = rm.check_intraday_halt(49_500, threshold, 0)
        assert halted is False

    def test_hobbyist_cap_dominates(self):
        """$8K Hobbyist: pct=$320, cap=$200 → cap dominates. Halt at $210."""
        rm = self._make_rm(8_000)
        threshold = HOBBYIST.max_daily_loss_usd(8_000)  # min(320, 200) = 200
        assert threshold == pytest.approx(200.0)
        # Lose $210 — should halt
        halted = rm.check_intraday_halt(7_790, threshold, 0)
        assert halted is True


class TestIntradayHaltInEngine:
    """End-to-end: engine.run_cycle respects the profile-aware intraday halt."""

    def test_beginner_small_loss_triggers_halt(self):
        """$800 Beginner account loses $25 → halt + alert via engine cycle."""
        from core.engine import TradingEngine

        engine = TradingEngine.__new__(TradingEngine)
        engine.broker = MagicMock()
        engine.risk = RiskManager()
        engine.risk.initialize(800)
        engine.cycle_count = 0
        engine.alert_manager = MagicMock()
        engine.order_manager = None
        engine.execution_manager = None
        engine.config_reloader = None
        engine.drift_detector = None
        engine.position_reconciler = None
        engine.cost_model = None
        engine.data_fetcher = None
        engine.portfolio_optimizer = None
        engine.db_rotator = None
        engine.state_store = None
        engine._cached_position_dfs = None
        engine._consecutive_broker_failures = {}
        engine._last_trade_time = {}
        engine._equity_buys_today = {}
        engine._daily_trade_count = {}
        engine._daily_trade_date = ""
        engine._alerted_drawdown_halt = False
        engine._alerted_intraday_halt = False
        engine._alerted_external_positions = set()
        engine._alerted_degraded = set()

        # Activate Beginner profile
        Config._PROFILE = BEGINNER

        # Account equity dropped to $770 (= -$30 realized loss, threshold $24)
        engine.broker.get_account.return_value = {
            "equity": 770, "cash": 770, "buying_power": 770,
            "daytrade_count": 0, "pattern_day_trader": False,
            "unrealized_pl": 0.0,
        }

        try:
            engine.run_cycle()
        finally:
            Config._PROFILE = None

        assert engine.risk.halted is True
        # Intraday halt alert fired once
        engine.alert_manager.intraday_halt.assert_called_once()

    def test_learner_same_loss_does_not_halt(self):
        """$50K Learner loses $100 → does NOT halt (well under $2,500 threshold)."""
        from core.engine import TradingEngine

        engine = TradingEngine.__new__(TradingEngine)
        engine.broker = MagicMock()
        engine.broker.get_positions.return_value = []
        engine.broker.get_recent_bars.return_value = None
        engine.risk = RiskManager()
        engine.risk.initialize(50_000)
        engine.cycle_count = 0
        engine.alert_manager = MagicMock()
        engine.order_manager = None
        engine.execution_manager = None
        engine.config_reloader = None
        engine.drift_detector = None
        engine.position_reconciler = None
        engine.cost_model = None
        engine.data_fetcher = None
        engine.portfolio_optimizer = None
        engine.db_rotator = None
        engine.state_store = None
        engine._cached_position_dfs = None
        engine._consecutive_broker_failures = {}
        engine._last_trade_time = {}
        engine._equity_buys_today = {}
        engine._daily_trade_count = {}
        engine._daily_trade_date = ""
        engine._alerted_drawdown_halt = False
        engine._alerted_intraday_halt = False
        engine._alerted_external_positions = set()
        engine._alerted_degraded = set()

        Config._PROFILE = LEARNER

        engine.broker.get_account.return_value = {
            "equity": 49_900, "cash": 49_900, "buying_power": 49_900,
            "daytrade_count": 0, "pattern_day_trader": False,
            "unrealized_pl": 0.0,
        }
        # Skip the symbol processing loop
        with patch.object(Config, "CRYPTO_SYMBOLS", []), \
             patch.object(Config, "EQUITY_SYMBOLS", []):
            try:
                engine.run_cycle()
            finally:
                Config._PROFILE = None

        assert engine.risk.halted is False
        engine.alert_manager.intraday_halt.assert_not_called()

    def test_alert_fires_once_per_halt(self):
        """Once halted, we shouldn't spam alerts every cycle."""
        from core.engine import TradingEngine

        engine = TradingEngine.__new__(TradingEngine)
        engine.broker = MagicMock()
        engine.risk = RiskManager()
        engine.risk.initialize(1_000)
        engine.cycle_count = 0
        engine.alert_manager = MagicMock()
        engine.order_manager = None
        engine.execution_manager = None
        engine.config_reloader = None
        engine.drift_detector = None
        engine.position_reconciler = None
        engine.cost_model = None
        engine.data_fetcher = None
        engine.portfolio_optimizer = None
        engine.db_rotator = None
        engine.state_store = None
        engine._cached_position_dfs = None
        engine._consecutive_broker_failures = {}
        engine._alerted_drawdown_halt = False
        engine._alerted_intraday_halt = False
        engine._alerted_external_positions = set()
        engine._alerted_degraded = set()

        Config._PROFILE = BEGINNER

        engine.broker.get_account.return_value = {
            "equity": 900, "cash": 900, "buying_power": 900,
            "daytrade_count": 0, "pattern_day_trader": False,
            "unrealized_pl": 0.0,
        }

        try:
            # 3 cycles with same halted state
            for _ in range(3):
                engine.run_cycle()
        finally:
            Config._PROFILE = None

        # Alert fired exactly once despite 3 cycles
        assert engine.alert_manager.intraday_halt.call_count == 1
