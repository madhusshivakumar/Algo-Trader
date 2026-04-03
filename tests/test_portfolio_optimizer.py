"""Tests for portfolio optimization module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.portfolio_optimizer import (
    KellyResult,
    AllocationResult,
    compute_kelly_fraction,
    compute_kelly_position_size,
    compute_mean_variance_weights,
    get_optimal_position_pct,
    PortfolioOptimizer,
)
from config import Config


# ── Helpers ─────────────────────────────────────────────────────────


def _make_trades(wins: int = 15, losses: int = 5,
                 avg_win: float = 100.0, avg_loss: float = -50.0) -> list[float]:
    """Generate synthetic trade P&L list."""
    return [avg_win] * wins + [avg_loss] * losses


def _make_returns_df(n_days: int = 60, n_assets: int = 3,
                     seed: int = 42) -> pd.DataFrame:
    """Generate synthetic multi-asset daily returns."""
    rng = np.random.default_rng(seed)
    data = {}
    symbols = ["AAPL", "TSLA", "NVDA", "AMD", "META"][:n_assets]
    for sym in symbols:
        data[sym] = rng.normal(0.001, 0.02, n_days)
    return pd.DataFrame(data)


# ── KellyResult Tests ──────────────────────────────────────────────


class TestKellyResult:
    def test_default_values(self):
        r = KellyResult()
        assert r.full_kelly_fraction == 0.0
        assert r.adjusted_fraction == 0.0
        assert r.reliable is False

    def test_fields_settable(self):
        r = KellyResult(win_rate=0.6, avg_win=100.0, avg_loss=50.0,
                        num_trades=30, reliable=True)
        assert r.win_rate == 0.6


# ── compute_kelly_fraction Tests ──────────────────────────────────


class TestComputeKellyFraction:
    def test_profitable_strategy(self):
        trades = _make_trades(wins=15, losses=5, avg_win=100, avg_loss=-50)
        result = compute_kelly_fraction(trades, kelly_fraction=0.25, min_trades=10)
        assert result.win_rate == pytest.approx(0.75)
        assert result.avg_win == pytest.approx(100.0)
        assert result.avg_loss == pytest.approx(50.0)
        assert result.full_kelly_fraction > 0
        assert result.adjusted_fraction > 0
        assert result.adjusted_fraction < result.full_kelly_fraction
        assert result.reliable is True

    def test_kelly_formula_correctness(self):
        """Verify Kelly formula: f* = p - (1-p) / (W/L)"""
        trades = _make_trades(wins=60, losses=40, avg_win=150, avg_loss=-100)
        result = compute_kelly_fraction(trades, kelly_fraction=1.0, min_trades=10)
        # p = 0.6, W/L = 150/100 = 1.5
        # f* = 0.6 - 0.4/1.5 = 0.6 - 0.2667 = 0.3333
        expected_kelly = 0.6 - 0.4 / 1.5
        assert result.full_kelly_fraction == pytest.approx(expected_kelly, abs=0.01)

    def test_fractional_kelly_scaling(self):
        trades = _make_trades(wins=60, losses=40, avg_win=150, avg_loss=-100)
        full = compute_kelly_fraction(trades, kelly_fraction=1.0, min_trades=10)
        quarter = compute_kelly_fraction(trades, kelly_fraction=0.25, min_trades=10)
        assert quarter.adjusted_fraction == pytest.approx(
            full.full_kelly_fraction * 0.25, abs=0.001)

    def test_losing_strategy_zero_kelly(self):
        # More losses than wins, avg loss > avg win → negative Kelly → clamped to 0
        trades = _make_trades(wins=3, losses=17, avg_win=50, avg_loss=-100)
        result = compute_kelly_fraction(trades, kelly_fraction=0.25, min_trades=10)
        assert result.full_kelly_fraction == 0.0
        assert result.adjusted_fraction == 0.0

    def test_empty_trades(self):
        result = compute_kelly_fraction([], kelly_fraction=0.25, min_trades=10)
        assert result.num_trades == 0
        assert result.full_kelly_fraction == 0.0

    def test_all_wins_no_losses(self):
        result = compute_kelly_fraction([100, 200, 50], kelly_fraction=0.25, min_trades=1)
        assert result.full_kelly_fraction == 0.0  # Can't compute without losses

    def test_all_losses_no_wins(self):
        result = compute_kelly_fraction([-100, -200], kelly_fraction=0.25, min_trades=1)
        assert result.full_kelly_fraction == 0.0  # Can't compute without wins

    def test_unreliable_below_min_trades(self):
        trades = _make_trades(wins=3, losses=2)
        result = compute_kelly_fraction(trades, kelly_fraction=0.25, min_trades=20)
        assert result.reliable is False
        assert result.num_trades == 5

    def test_reliable_above_min_trades(self):
        trades = _make_trades(wins=15, losses=10)
        result = compute_kelly_fraction(trades, kelly_fraction=0.25, min_trades=20)
        assert result.reliable is True

    def test_uses_config_defaults(self):
        trades = _make_trades(wins=15, losses=5)
        with patch.object(Config, "KELLY_FRACTION", 0.5), \
             patch.object(Config, "KELLY_MIN_TRADES", 10):
            result = compute_kelly_fraction(trades)
        assert result.adjusted_fraction == pytest.approx(
            result.full_kelly_fraction * 0.5, abs=0.001)
        assert result.reliable is True

    def test_breakeven_trades_ignored(self):
        """Zero P&L trades don't count as wins or losses."""
        trades = [100, -50, 0, 0, 100, -50, 100, -50, 0, 100, -50,
                  100, -50, 100, -50, 100, -50, 100, -50, 100, -50]
        result = compute_kelly_fraction(trades, kelly_fraction=0.25, min_trades=10)
        # Breakeven trades are in total count but not in win/loss
        assert result.win_rate > 0


# ── compute_kelly_position_size Tests ──────────────────────────────


class TestComputeKellyPositionSize:
    def test_reliable_kelly(self):
        trades = _make_trades(wins=15, losses=5, avg_win=100, avg_loss=-50)
        size = compute_kelly_position_size(
            equity=100000.0, trades_pnl=trades, base_pct=0.15,
            kelly_fraction=0.25, max_pct=0.35)
        # Should use Kelly sizing, not base
        assert size > 0
        assert size <= 100000 * 0.35

    def test_unreliable_falls_back(self):
        trades = _make_trades(wins=3, losses=2)  # Only 5 trades
        size = compute_kelly_position_size(
            equity=100000.0, trades_pnl=trades, base_pct=0.15,
            kelly_fraction=0.25, max_pct=0.35)
        # Should fall back to base_pct
        assert size == pytest.approx(100000 * 0.15)

    def test_empty_trades_falls_back(self):
        size = compute_kelly_position_size(
            equity=50000.0, trades_pnl=[], base_pct=0.10, max_pct=0.35)
        assert size == pytest.approx(50000 * 0.10)

    def test_respects_max_pct(self):
        # Very profitable strategy with high Kelly
        trades = _make_trades(wins=19, losses=1, avg_win=200, avg_loss=-10)
        size = compute_kelly_position_size(
            equity=100000.0, trades_pnl=trades, base_pct=0.15,
            kelly_fraction=1.0, max_pct=0.20)
        assert size <= 100000 * 0.20 + 0.01

    def test_base_pct_capped_by_max(self):
        size = compute_kelly_position_size(
            equity=100000.0, trades_pnl=[], base_pct=0.50, max_pct=0.35)
        # base_pct > max_pct, should use max_pct
        assert size == pytest.approx(100000 * 0.35)


# ── compute_mean_variance_weights Tests ────────────────────────────


class TestComputeMeanVarianceWeights:
    def test_basic_allocation(self):
        df = _make_returns_df(n_days=60, n_assets=3)
        result = compute_mean_variance_weights(df, risk_free_rate=0.05, max_weight=0.5)
        assert result.num_assets == 3
        assert len(result.weights) == 3
        total_weight = sum(result.weights.values())
        assert total_weight == pytest.approx(1.0, abs=0.01)

    def test_no_negative_weights(self):
        df = _make_returns_df(n_days=100, n_assets=4)
        result = compute_mean_variance_weights(df, max_weight=0.5)
        for w in result.weights.values():
            assert w >= 0.0

    def test_max_weight_constraint(self):
        df = _make_returns_df(n_days=100, n_assets=3)
        result = compute_mean_variance_weights(df, max_weight=0.40)
        for w in result.weights.values():
            assert w <= 0.40 + 0.01  # small tolerance

    def test_sharpe_ratio_computed(self):
        df = _make_returns_df(n_days=100, n_assets=3)
        result = compute_mean_variance_weights(df)
        assert isinstance(result.sharpe_ratio, float)

    def test_expected_return_and_vol(self):
        df = _make_returns_df(n_days=100, n_assets=3)
        result = compute_mean_variance_weights(df)
        assert result.expected_volatility > 0

    def test_empty_df(self):
        result = compute_mean_variance_weights(pd.DataFrame())
        assert result.weights == {}
        assert result.num_assets == 0

    def test_none_df(self):
        result = compute_mean_variance_weights(None)
        assert result.weights == {}

    def test_single_asset(self):
        df = _make_returns_df(n_days=60, n_assets=1)
        result = compute_mean_variance_weights(df)
        # Need >= 2 assets
        assert result.weights == {}

    def test_insufficient_rows(self):
        df = _make_returns_df(n_days=3, n_assets=3)
        result = compute_mean_variance_weights(df)
        assert result.weights == {}

    def test_column_with_zero_variance_dropped(self):
        df = _make_returns_df(n_days=60, n_assets=3)
        df["FLAT"] = 0.0  # Zero variance column
        result = compute_mean_variance_weights(df, max_weight=0.5)
        assert "FLAT" not in result.weights
        assert result.num_assets == 3  # Only the 3 valid columns

    def test_column_with_nan_dropped(self):
        df = _make_returns_df(n_days=60, n_assets=3)
        df["BAD"] = np.nan
        result = compute_mean_variance_weights(df, max_weight=0.5)
        assert "BAD" not in result.weights

    def test_uses_config_defaults(self):
        df = _make_returns_df(n_days=60, n_assets=3)
        with patch.object(Config, "MEAN_VARIANCE_RISK_FREE_RATE", 0.03), \
             patch.object(Config, "MAX_SINGLE_POSITION_PCT", 0.40):
            result = compute_mean_variance_weights(df)
        assert result.num_assets == 3


# ── get_optimal_position_pct Tests ─────────────────────────────────


class TestGetOptimalPositionPct:
    def test_returns_weight_when_present(self):
        alloc = AllocationResult(weights={"AAPL": 0.25, "TSLA": 0.35})
        assert get_optimal_position_pct("AAPL", alloc, 0.15) == 0.25

    def test_falls_back_when_missing(self):
        alloc = AllocationResult(weights={"AAPL": 0.25})
        assert get_optimal_position_pct("NVDA", alloc, 0.15) == 0.15

    def test_falls_back_on_zero_weight(self):
        alloc = AllocationResult(weights={"AAPL": 0.0})
        assert get_optimal_position_pct("AAPL", alloc, 0.15) == 0.15

    def test_empty_allocation(self):
        alloc = AllocationResult()
        assert get_optimal_position_pct("AAPL", alloc, 0.20) == 0.20


# ── PortfolioOptimizer Tests ──────────────────────────────────────


class TestPortfolioOptimizer:
    def test_initial_state(self):
        opt = PortfolioOptimizer()
        assert opt.allocation is None
        assert opt.should_recompute(0) is True

    def test_should_recompute_after_interval(self):
        opt = PortfolioOptimizer()
        df = _make_returns_df(60, 3)
        opt.update_allocation(df, cycle_count=100)
        assert opt.should_recompute(100) is False
        assert opt.should_recompute(149) is False
        assert opt.should_recompute(150) is True

    def test_update_allocation(self):
        opt = PortfolioOptimizer()
        df = _make_returns_df(60, 3)
        result = opt.update_allocation(df, cycle_count=10)
        assert result.num_assets == 3
        assert opt.allocation is not None

    def test_get_position_pct_no_allocation(self):
        opt = PortfolioOptimizer()
        assert opt.get_position_pct("AAPL", 0.15) == 0.15

    def test_get_position_pct_with_allocation(self):
        opt = PortfolioOptimizer()
        df = _make_returns_df(60, 3)
        opt.update_allocation(df, cycle_count=10)
        pct = opt.get_position_pct("AAPL", 0.15)
        assert pct > 0  # Should have some allocation

    def test_update_kelly(self):
        opt = PortfolioOptimizer()
        trades = _make_trades(wins=15, losses=5)
        result = opt.update_kelly("AAPL", trades)
        assert result.num_trades == 20
        assert result.reliable is True

    def test_get_kelly_size_no_data(self):
        opt = PortfolioOptimizer()
        size = opt.get_kelly_size("AAPL", 100000.0, 0.15)
        assert size == pytest.approx(100000 * 0.15)

    def test_get_kelly_size_with_data(self):
        opt = PortfolioOptimizer()
        trades = _make_trades(wins=15, losses=5, avg_win=100, avg_loss=-50)
        opt.update_kelly("AAPL", trades)
        size = opt.get_kelly_size("AAPL", 100000.0, 0.15)
        assert size > 0
        assert size <= 100000 * Config.MAX_SINGLE_POSITION_PCT + 0.01

    def test_get_kelly_size_unreliable(self):
        opt = PortfolioOptimizer()
        trades = _make_trades(wins=3, losses=2)  # Only 5 trades
        opt.update_kelly("AAPL", trades)
        size = opt.get_kelly_size("AAPL", 100000.0, 0.15)
        assert size == pytest.approx(100000 * 0.15)


# ── Engine Integration Tests ──────────────────────────────────────


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

    @patch.object(Config, "PORTFOLIO_OPTIMIZATION_ENABLED", True)
    @patch.object(Config, "KELLY_SIZING_ENABLED", True)
    @patch.object(Config, "MEAN_VARIANCE_ENABLED", False)
    def test_kelly_sizing_in_buy_path(self):
        engine = self._make_engine()
        opt = PortfolioOptimizer()
        # Pre-load Kelly data
        trades = _make_trades(wins=15, losses=5, avg_win=100, avg_loss=-50)
        opt.update_kelly("AAPL", trades)
        engine.portfolio_optimizer = opt

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
             patch.object(Config, "VWAP_TWAP_ENABLED", False):
            engine._process_symbol_inner("AAPL", 10000.0, df)

        engine.broker.buy.assert_called_once()
        # Size should reflect Kelly, not the default 15%
        called_size = engine.broker.buy.call_args[0][1]
        assert called_size > 0

    @patch.object(Config, "PORTFOLIO_OPTIMIZATION_ENABLED", True)
    @patch.object(Config, "MEAN_VARIANCE_ENABLED", True)
    @patch.object(Config, "KELLY_SIZING_ENABLED", False)
    def test_mean_variance_sizing_in_buy_path(self):
        engine = self._make_engine()
        opt = PortfolioOptimizer()
        # Pre-load allocation
        df_returns = _make_returns_df(60, 3)
        opt.update_allocation(df_returns, cycle_count=1)
        engine.portfolio_optimizer = opt

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
             patch.object(Config, "VWAP_TWAP_ENABLED", False):
            engine._process_symbol_inner("AAPL", 10000.0, df)

        engine.broker.buy.assert_called_once()

    @patch.object(Config, "PORTFOLIO_OPTIMIZATION_ENABLED", True)
    @patch.object(Config, "MEAN_VARIANCE_ENABLED", True)
    @patch.object(Config, "KELLY_SIZING_ENABLED", False)
    def test_recompute_allocation_in_run_cycle(self):
        engine = self._make_engine()
        opt = PortfolioOptimizer()
        engine.portfolio_optimizer = opt

        engine.broker.get_account.return_value = {"equity": 10000.0, "cash": 5000.0}
        engine.risk.can_trade.return_value = False  # Skip symbol processing

        # Mock _recompute_allocation
        engine._recompute_allocation = MagicMock()

        engine.run_cycle()

        # Should have been called since no prior allocation
        engine._recompute_allocation.assert_called_once()

    def test_no_optimizer_no_effect(self):
        """When portfolio_optimizer is None, sizing uses base_pct."""
        engine = self._make_engine()
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
                 "action": "buy", "strength": 1.0, "reason": "test",
                 "strategy": "momentum",
             }), \
             patch.object(Config, "EARNINGS_CALENDAR_ENABLED", False), \
             patch.object(Config, "CORRELATION_CHECK_ENABLED", False), \
             patch.object(Config, "VOLATILITY_SIZING_ENABLED", False), \
             patch.object(Config, "VWAP_TWAP_ENABLED", False):
            engine._process_symbol_inner("AAPL", 10000.0, df)

        engine.broker.buy.assert_called_once()
        # Default 15% of 10000 * strength 1.0 = 1500
        called_size = engine.broker.buy.call_args[0][1]
        assert called_size == pytest.approx(1500.0, abs=1.0)
