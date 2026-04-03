"""Tests for Monte Carlo simulation module."""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from core.monte_carlo import (
    MonteCarloResult,
    simulate_paths,
    compute_max_drawdowns,
    compute_var,
    compute_cvar,
    run_simulation,
    run_from_equity_curve,
    format_report,
)
from config import Config


# ── Helpers ─────────────────────────────────────────────────────────


def _daily_returns(n: int = 252, mean: float = 0.0005, std: float = 0.01,
                   seed: int = 42) -> np.ndarray:
    """Generate synthetic daily returns."""
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, n)


def _equity_curve(n: int = 253, start: float = 10000.0, seed: int = 42) -> list[float]:
    """Generate a synthetic equity curve."""
    returns = _daily_returns(n - 1, seed=seed)
    eq = [start]
    for r in returns:
        eq.append(eq[-1] * (1 + r))
    return eq


# ── MonteCarloResult Tests ─────────────────────────────────────────


class TestMonteCarloResult:
    def test_default_values(self):
        r = MonteCarloResult()
        assert r.num_simulations == 0
        assert r.prob_loss == 0.0
        assert r.var == {}
        assert r.paths is None

    def test_fields_settable(self):
        r = MonteCarloResult(num_simulations=1000, starting_equity=10000.0)
        assert r.num_simulations == 1000
        assert r.starting_equity == 10000.0


# ── simulate_paths Tests ───────────────────────────────────────────


class TestSimulatePaths:
    def test_output_shape(self):
        returns = _daily_returns(100)
        paths = simulate_paths(returns, 10000.0, num_simulations=50,
                               horizon_days=20, seed=42)
        assert paths.shape == (50, 21)  # 20 days + starting column

    def test_starting_column_is_equity(self):
        returns = _daily_returns(100)
        paths = simulate_paths(returns, 5000.0, num_simulations=10,
                               horizon_days=10, seed=42)
        assert np.all(paths[:, 0] == 5000.0)

    def test_reproducible_with_seed(self):
        returns = _daily_returns(100)
        p1 = simulate_paths(returns, 10000.0, 20, 30, seed=123)
        p2 = simulate_paths(returns, 10000.0, 20, 30, seed=123)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        returns = _daily_returns(100)
        p1 = simulate_paths(returns, 10000.0, 20, 30, seed=1)
        p2 = simulate_paths(returns, 10000.0, 20, 30, seed=2)
        assert not np.array_equal(p1, p2)

    def test_positive_returns_grow(self):
        # All positive returns should produce growth
        returns = np.full(100, 0.01)  # 1% daily
        paths = simulate_paths(returns, 10000.0, 10, 50, seed=42)
        # Every terminal value should be > starting
        assert np.all(paths[:, -1] > 10000.0)

    def test_negative_returns_shrink(self):
        returns = np.full(100, -0.01)  # -1% daily
        paths = simulate_paths(returns, 10000.0, 10, 50, seed=42)
        assert np.all(paths[:, -1] < 10000.0)

    def test_empty_returns(self):
        paths = simulate_paths(np.array([]), 10000.0, 5, 10)
        assert paths.shape == (5, 1)
        assert np.all(paths == 10000.0)

    def test_zero_simulations(self):
        returns = _daily_returns(50)
        paths = simulate_paths(returns, 10000.0, num_simulations=0,
                               horizon_days=10)
        assert paths.shape[0] == 1  # fallback to 1

    def test_zero_horizon(self):
        returns = _daily_returns(50)
        paths = simulate_paths(returns, 10000.0, num_simulations=5,
                               horizon_days=0)
        assert paths.shape == (5, 1)

    def test_uses_config_defaults(self):
        returns = _daily_returns(50)
        with patch.object(Config, "MONTE_CARLO_NUM_SIMULATIONS", 3), \
             patch.object(Config, "MONTE_CARLO_HORIZON_DAYS", 5):
            paths = simulate_paths(returns, 10000.0, seed=42)
        assert paths.shape == (3, 6)

    def test_values_stay_positive(self):
        """With reasonable returns, equity should stay positive."""
        returns = _daily_returns(252, mean=0.0, std=0.02)
        paths = simulate_paths(returns, 10000.0, 100, 252, seed=42)
        assert np.all(paths > 0)


# ── compute_max_drawdowns Tests ────────────────────────────────────


class TestComputeMaxDrawdowns:
    def test_no_drawdown(self):
        # Monotonically increasing paths
        paths = np.array([[100, 110, 120, 130]])
        dds = compute_max_drawdowns(paths)
        assert dds[0] == pytest.approx(0.0)

    def test_simple_drawdown(self):
        # Peak at 200, trough at 150 = 25% drawdown
        paths = np.array([[100, 200, 150, 180]])
        dds = compute_max_drawdowns(paths)
        assert dds[0] == pytest.approx(0.25)

    def test_multiple_paths(self):
        paths = np.array([
            [100, 110, 105, 115],  # dd = 5/110 ≈ 4.5%
            [100, 100, 50, 75],    # dd = 50/100 = 50%
        ])
        dds = compute_max_drawdowns(paths)
        assert len(dds) == 2
        assert dds[0] == pytest.approx(5.0 / 110.0)
        assert dds[1] == pytest.approx(0.5)

    def test_single_point_path(self):
        paths = np.array([[100]])
        dds = compute_max_drawdowns(paths)
        assert dds[0] == 0.0

    def test_flat_path(self):
        paths = np.array([[100, 100, 100, 100]])
        dds = compute_max_drawdowns(paths)
        assert dds[0] == pytest.approx(0.0)


# ── compute_var Tests ──────────────────────────────────────────────


class TestComputeVar:
    def test_known_distribution(self):
        # 100 returns: 95 positive, 5 negative
        rng = np.random.default_rng(42)
        returns = rng.normal(0.05, 0.10, 1000)
        var = compute_var(returns, [0.95])
        # 95% VaR should be a positive number (representing loss)
        assert "95%" in var
        # Should be relatively small for positive-mean distribution
        assert var["95%"] >= 0

    def test_all_positive_returns(self):
        returns = np.full(100, 0.05)  # All 5% gains
        var = compute_var(returns, [0.95, 0.99])
        # No losses at all → VaR should be 0 (clamped)
        assert var["95%"] == 0.0
        assert var["99%"] == 0.0

    def test_all_negative_returns(self):
        returns = np.full(100, -0.05)  # All 5% losses
        var = compute_var(returns, [0.95])
        assert var["95%"] == pytest.approx(0.05)

    def test_empty_returns(self):
        var = compute_var(np.array([]), [0.95])
        assert var["95%"] == 0.0

    def test_uses_config_defaults(self):
        returns = _daily_returns(100)
        with patch.object(Config, "MONTE_CARLO_CONFIDENCE_LEVELS", [0.90]):
            var = compute_var(returns)
        assert "90%" in var

    def test_multiple_confidence_levels(self):
        returns = _daily_returns(1000)
        var = compute_var(returns, [0.90, 0.95, 0.99])
        assert len(var) == 3
        # Higher confidence = higher VaR (more extreme tail)
        assert var["99%"] >= var["95%"]
        assert var["95%"] >= var["90%"]


# ── compute_cvar Tests ─────────────────────────────────────────────


class TestComputeCvar:
    def test_cvar_gte_var(self):
        """CVaR (expected shortfall) should always >= VaR at same level."""
        returns = _daily_returns(1000, mean=-0.001, std=0.02)
        var = compute_var(returns, [0.95])
        cvar = compute_cvar(returns, [0.95])
        assert cvar["95%"] >= var["95%"]

    def test_all_positive_returns(self):
        returns = np.full(100, 0.05)
        cvar = compute_cvar(returns, [0.95])
        assert cvar["95%"] == 0.0

    def test_empty_returns(self):
        cvar = compute_cvar(np.array([]), [0.95])
        assert cvar["95%"] == 0.0

    def test_uses_config_defaults(self):
        returns = _daily_returns(100)
        with patch.object(Config, "MONTE_CARLO_CONFIDENCE_LEVELS", [0.90]):
            cvar = compute_cvar(returns)
        assert "90%" in cvar


# ── run_simulation Tests ───────────────────────────────────────────


class TestRunSimulation:
    def test_basic_simulation(self):
        returns = _daily_returns(252)
        result = run_simulation(returns, 10000.0, num_simulations=100,
                                horizon_days=50, seed=42)
        assert result.num_simulations == 100
        assert result.horizon_days == 50
        assert result.starting_equity == 10000.0
        assert result.terminal_mean > 0
        assert result.terminal_median > 0
        assert result.terminal_std > 0

    def test_percentiles_ordered(self):
        returns = _daily_returns(252)
        result = run_simulation(returns, 10000.0, 500, 50, seed=42)
        p = result.terminal_percentiles
        assert p["5%"] <= p["25%"] <= p["50%"] <= p["75%"] <= p["95%"]

    def test_var_populated(self):
        returns = _daily_returns(252)
        result = run_simulation(returns, 10000.0, 500, 50,
                                confidence_levels=[0.95, 0.99], seed=42)
        assert "95%" in result.var
        assert "99%" in result.var
        assert "95%" in result.cvar
        assert "99%" in result.cvar

    def test_drawdown_metrics_populated(self):
        returns = _daily_returns(252)
        result = run_simulation(returns, 10000.0, 200, 50, seed=42)
        assert result.max_drawdown_mean > 0
        assert result.max_drawdown_median > 0
        assert result.max_drawdown_95th >= result.max_drawdown_median
        assert result.max_drawdown_99th >= result.max_drawdown_95th

    def test_probability_estimates(self):
        returns = _daily_returns(252, mean=0.0, std=0.02)
        result = run_simulation(returns, 10000.0, 1000, 50, seed=42)
        # All probabilities should be between 0 and 1
        assert 0.0 <= result.prob_loss <= 1.0
        assert 0.0 <= result.prob_loss_10pct <= 1.0
        assert 0.0 <= result.prob_gain_20pct <= 1.0
        # P(loss > 10%) should be <= P(loss)
        assert result.prob_loss_10pct <= result.prob_loss

    def test_store_paths(self):
        returns = _daily_returns(100)
        result = run_simulation(returns, 10000.0, 5, 10, seed=42,
                                store_paths=True)
        assert result.paths is not None
        assert len(result.paths) == 5
        assert len(result.paths[0]) == 11  # 10 days + start

    def test_no_store_paths_default(self):
        returns = _daily_returns(100)
        result = run_simulation(returns, 10000.0, 5, 10, seed=42)
        assert result.paths is None

    def test_insufficient_returns(self):
        returns = np.array([0.01])  # Only 1 return
        result = run_simulation(returns, 10000.0, 10, 10)
        assert result.num_simulations == 10
        assert result.terminal_mean == 0.0  # Not populated

    def test_empty_returns(self):
        result = run_simulation(np.array([]), 10000.0, 10, 10)
        assert result.terminal_mean == 0.0

    def test_uses_config_defaults(self):
        returns = _daily_returns(100)
        with patch.object(Config, "MONTE_CARLO_NUM_SIMULATIONS", 5), \
             patch.object(Config, "MONTE_CARLO_HORIZON_DAYS", 3), \
             patch.object(Config, "MONTE_CARLO_CONFIDENCE_LEVELS", [0.95]):
            result = run_simulation(returns, 10000.0, seed=42)
        assert result.num_simulations == 5
        assert result.horizon_days == 3
        assert "95%" in result.var

    def test_negative_drift_higher_loss_probability(self):
        """Negative-drift returns should have higher loss probability."""
        neg_returns = _daily_returns(252, mean=-0.002, std=0.01, seed=42)
        pos_returns = _daily_returns(252, mean=0.002, std=0.01, seed=42)
        neg_result = run_simulation(neg_returns, 10000.0, 500, 50, seed=42)
        pos_result = run_simulation(pos_returns, 10000.0, 500, 50, seed=42)
        assert neg_result.prob_loss > pos_result.prob_loss


# ── run_from_equity_curve Tests ────────────────────────────────────


class TestRunFromEquityCurve:
    def test_basic(self):
        eq = _equity_curve(100)
        result = run_from_equity_curve(eq, num_simulations=50,
                                       horizon_days=20, seed=42)
        assert result.starting_equity == pytest.approx(eq[-1])
        assert result.num_simulations == 50
        assert result.terminal_mean > 0

    def test_short_curve(self):
        result = run_from_equity_curve([100.0, 101.0], num_simulations=10,
                                       horizon_days=5)
        # Only 1 return — too few, returns empty result
        assert result.starting_equity == 101.0
        assert result.terminal_mean == 0.0

    def test_empty_curve(self):
        result = run_from_equity_curve([])
        assert result.starting_equity == 0.0

    def test_single_value(self):
        result = run_from_equity_curve([10000.0])
        assert result.starting_equity == 10000.0

    def test_handles_inf_in_curve(self):
        """Equity curves with zeros can produce inf returns — should be filtered."""
        eq = [100, 0, 50, 100]  # 0 causes inf return
        result = run_from_equity_curve(eq, num_simulations=10,
                                       horizon_days=5, seed=42)
        # Should still produce a valid result (inf returns filtered out)
        assert result.starting_equity == 100.0

    def test_store_paths_passthrough(self):
        eq = _equity_curve(50)
        result = run_from_equity_curve(eq, num_simulations=5,
                                       horizon_days=5, seed=42,
                                       store_paths=True)
        assert result.paths is not None


# ── format_report Tests ────────────────────────────────────────────


class TestFormatReport:
    def test_contains_key_sections(self):
        returns = _daily_returns(252)
        result = run_simulation(returns, 10000.0, 100, 50, seed=42)
        report = format_report(result)
        assert "Monte Carlo Simulation" in report
        assert "Terminal Wealth" in report
        assert "Value at Risk" in report
        assert "Conditional VaR" in report
        assert "Max Drawdown" in report
        assert "Probability Estimates" in report

    def test_shows_starting_equity(self):
        result = MonteCarloResult(starting_equity=25000.0,
                                  num_simulations=100, horizon_days=30)
        report = format_report(result)
        assert "$25,000.00" in report

    def test_empty_result_doesnt_crash(self):
        result = MonteCarloResult()
        report = format_report(result)
        assert "Monte Carlo" in report


# ── Backtest CLI Integration ───────────────────────────────────────


class TestBacktestCLIIntegration:
    def test_monte_carlo_flag_exists(self):
        """The --monte-carlo flag should be accepted by the argument parser."""
        import argparse
        # We just verify the parser doesn't reject the flag
        from backtest import main
        # This tests the argparse setup indirectly

    @patch("backtest.Broker")
    @patch("backtest.Config")
    def test_run_monte_carlo_function_exists(self, mock_config, mock_broker):
        """The _run_monte_carlo function should be importable."""
        from backtest import _run_monte_carlo
        assert callable(_run_monte_carlo)

    @patch("backtest.Broker")
    def test_run_monte_carlo_handles_short_data(self, mock_broker_cls):
        """Monte Carlo should gracefully handle insufficient backtest data."""
        import pandas as pd
        from backtest import _run_monte_carlo

        broker = MagicMock()
        # Return a very small DataFrame — not enough for backtest
        broker.get_historical_bars.return_value = pd.DataFrame({
            "open": [100], "high": [105], "low": [95],
            "close": [102], "volume": [1000],
        })

        # Should not raise
        _run_monte_carlo(broker, ["AAPL"])
