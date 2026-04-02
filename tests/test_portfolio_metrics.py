"""Tests for core/portfolio_metrics.py — advanced portfolio metrics."""

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from core.portfolio_metrics import (
    compute_returns,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    profit_factor,
    expectancy,
    max_drawdown,
    analyze_drawdowns,
    consecutive_streaks,
    monthly_returns,
    compute_all,
    compute_from_db,
    PortfolioMetrics,
    DrawdownInfo,
)


# ── compute_returns ──────────────────────────────────────────────────

class TestComputeReturns:
    def test_simple_returns(self):
        eq = [100, 110, 105]
        r = compute_returns(eq)
        assert len(r) == 2
        assert abs(r[0] - 0.1) < 1e-10
        assert abs(r[1] - (-5 / 110)) < 1e-10

    def test_single_element(self):
        assert len(compute_returns([100])) == 0

    def test_empty(self):
        assert len(compute_returns([])) == 0

    def test_handles_zero_in_curve(self):
        # Zero in equity should produce inf, which gets replaced with 0
        r = compute_returns([100, 0, 50])
        assert np.isfinite(r).all()

    def test_numpy_input(self):
        eq = np.array([100.0, 110.0, 121.0])
        r = compute_returns(eq)
        assert abs(r[0] - 0.1) < 1e-10
        assert abs(r[1] - 0.1) < 1e-10


# ── sharpe_ratio ─────────────────────────────────────────────────────

class TestSharpeRatio:
    def test_positive_returns(self):
        # Consistent positive returns = high Sharpe
        returns = np.array([0.01] * 100)
        s = sharpe_ratio(returns, annualize=252)
        assert s > 0

    def test_zero_std(self):
        # All same return = zero std = zero Sharpe
        returns = np.array([0.0] * 100)
        assert sharpe_ratio(returns) == 0.0

    def test_insufficient_data(self):
        assert sharpe_ratio(np.array([0.01])) == 0.0
        assert sharpe_ratio(np.array([])) == 0.0

    def test_negative_returns(self):
        returns = np.array([-0.01] * 100)
        s = sharpe_ratio(returns, annualize=252)
        assert s < 0

    def test_annualization_scales(self):
        returns = np.array([0.001] * 100)
        s1 = sharpe_ratio(returns, annualize=252)
        s2 = sharpe_ratio(returns, annualize=52)
        # Higher annualization factor = larger absolute Sharpe
        assert abs(s1) > abs(s2)


# ── sortino_ratio ────────────────────────────────────────────────────

class TestSortinoRatio:
    def test_all_positive_returns(self):
        # No downside returns = zero downside std = returns 0
        returns = np.array([0.01] * 100)
        assert sortino_ratio(returns) == 0.0

    def test_mixed_returns(self):
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.01])
        s = sortino_ratio(returns, annualize=252)
        assert s != 0  # Should produce a meaningful value

    def test_all_negative_returns(self):
        returns = np.array([-0.01] * 100)
        s = sortino_ratio(returns, annualize=252)
        assert s < 0

    def test_insufficient_data(self):
        assert sortino_ratio(np.array([])) == 0.0
        assert sortino_ratio(np.array([0.01])) == 0.0

    def test_sortino_gte_sharpe_for_skewed_positive(self):
        """Sortino should >= Sharpe when positive returns dominate (less downside)."""
        np.random.seed(42)
        returns = np.abs(np.random.randn(200) * 0.01) + 0.001  # mostly positive
        returns[::10] = -0.005  # occasional small negatives
        sh = sharpe_ratio(returns, annualize=252)
        so = sortino_ratio(returns, annualize=252)
        # Sortino uses only downside dev, so should be >= Sharpe for positive skew
        assert so >= sh or abs(so - sh) < 1  # allow small tolerance


# ── calmar_ratio ─────────────────────────────────────────────────────

class TestCalmarRatio:
    def test_positive_return_positive_dd(self):
        assert calmar_ratio(0.20, 0.10) == 2.0

    def test_zero_drawdown(self):
        assert calmar_ratio(0.20, 0.0) == 0.0

    def test_negative_return(self):
        assert calmar_ratio(-0.10, 0.15) < 0


# ── profit_factor ────────────────────────────────────────────────────

class TestProfitFactor:
    def test_basic(self):
        pnl = [100, -50, 200, -30]
        pf = profit_factor(pnl)
        assert abs(pf - (300 / 80)) < 1e-10

    def test_no_losses(self):
        assert profit_factor([100, 50]) == 999.0  # capped for JSON safety

    def test_no_profits(self):
        assert profit_factor([-100, -50]) == 0.0

    def test_empty(self):
        assert profit_factor([]) == 0.0

    def test_breakeven_ignored(self):
        pnl = [100, 0, -50]
        pf = profit_factor(pnl)
        assert abs(pf - (100 / 50)) < 1e-10


# ── expectancy ───────────────────────────────────────────────────────

class TestExpectancy:
    def test_positive(self):
        assert expectancy([100, -20, 50]) == pytest.approx(130 / 3)

    def test_negative(self):
        assert expectancy([-100, -20]) == pytest.approx(-60)

    def test_empty(self):
        assert expectancy([]) == 0.0


# ── max_drawdown ─────────────────────────────────────────────────────

class TestMaxDrawdown:
    def test_simple_drawdown(self):
        eq = [100, 110, 90, 95, 105]
        dd = max_drawdown(eq)
        # Peak=110, trough=90, dd=20/110 ≈ 0.1818
        assert abs(dd - 20 / 110) < 1e-10

    def test_no_drawdown(self):
        eq = [100, 110, 120, 130]
        assert max_drawdown(eq) == 0.0

    def test_all_decline(self):
        eq = [100, 80, 60, 40]
        assert abs(max_drawdown(eq) - 0.6) < 1e-10

    def test_single_point(self):
        assert max_drawdown([100]) == 0.0

    def test_empty(self):
        assert max_drawdown([]) == 0.0


# ── analyze_drawdowns ────────────────────────────────────────────────

class TestAnalyzeDrawdowns:
    def test_single_drawdown(self):
        eq = [100, 110, 95, 100, 115]
        dds = analyze_drawdowns(eq, min_depth=0.01)
        assert len(dds) == 1
        assert dds[0].depth == pytest.approx(15 / 110, abs=1e-10)
        assert dds[0].end_idx is not None  # recovered

    def test_open_drawdown(self):
        eq = [100, 110, 95, 90]
        dds = analyze_drawdowns(eq, min_depth=0.01)
        assert len(dds) == 1
        assert dds[0].end_idx is None  # not recovered
        assert dds[0].recovery_bars is None

    def test_min_depth_filter(self):
        eq = [100, 100.5, 100.3, 100.6, 100.4, 101]
        # Very small drawdowns — should be filtered at 1% threshold
        dds = analyze_drawdowns(eq, min_depth=0.01)
        assert len(dds) == 0

    def test_multiple_drawdowns(self):
        eq = [100, 110, 95, 115, 100, 120]
        dds = analyze_drawdowns(eq, min_depth=0.01)
        assert len(dds) == 2
        # Sorted by depth (deepest first)
        assert dds[0].depth >= dds[1].depth

    def test_empty_curve(self):
        assert analyze_drawdowns([]) == []

    def test_recovery_bars_computed(self):
        eq = [100, 110, 95, 100, 110, 115]
        dds = analyze_drawdowns(eq, min_depth=0.01)
        assert len(dds) == 1
        assert dds[0].recovery_bars is not None
        assert dds[0].recovery_bars > 0


# ── consecutive_streaks ──────────────────────────────────────────────

class TestConsecutiveStreaks:
    def test_basic(self):
        pnl = [10, 20, -5, -10, -15, 30]
        w, l = consecutive_streaks(pnl)
        assert w == 2
        assert l == 3

    def test_all_wins(self):
        w, l = consecutive_streaks([10, 20, 30])
        assert w == 3
        assert l == 0

    def test_all_losses(self):
        w, l = consecutive_streaks([-10, -20, -30])
        assert w == 0
        assert l == 3

    def test_empty(self):
        assert consecutive_streaks([]) == (0, 0)

    def test_breakeven_resets(self):
        pnl = [10, 10, 0, 10]
        w, l = consecutive_streaks(pnl)
        assert w == 2  # First two, then breakeven resets, then 1

    def test_single_trade(self):
        assert consecutive_streaks([10]) == (1, 0)
        assert consecutive_streaks([-10]) == (0, 1)


# ── monthly_returns ──────────────────────────────────────────────────

class TestMonthlyReturns:
    def test_two_months(self):
        timestamps = [
            datetime(2026, 1, 1),
            datetime(2026, 1, 15),
            datetime(2026, 1, 31),
            datetime(2026, 2, 1),
            datetime(2026, 2, 15),
            datetime(2026, 2, 28),
        ]
        eq = [100, 105, 110, 110, 108, 115]
        result = monthly_returns(eq, timestamps)
        assert "2026-01" in result
        assert "2026-02" in result
        assert abs(result["2026-01"] - 0.1) < 1e-10  # 100 -> 110
        assert abs(result["2026-02"] - (115 - 110) / 110) < 1e-10

    def test_no_timestamps(self):
        assert monthly_returns([100, 110]) == {}

    def test_mismatched_lengths(self):
        assert monthly_returns([100, 110, 120], [datetime(2026, 1, 1)]) == {}

    def test_empty(self):
        assert monthly_returns([], []) == {}


# ── compute_all ──────────────────────────────────────────────────────

class TestComputeAll:
    def test_basic_metrics(self):
        eq = [100, 110, 105, 115, 120]
        pnl = [10, -5, 10, 5]
        m = compute_all(eq, pnl)
        assert m.total_return == pytest.approx(0.2)
        assert m.total_pnl == pytest.approx(20.0)
        assert m.wins == 3
        assert m.losses == 1
        assert m.win_rate == pytest.approx(0.75)
        assert m.total_trades == 4
        assert m.max_consecutive_wins == 2  # last two: 10, 5
        assert m.max_consecutive_losses == 1

    def test_sharpe_and_sortino_computed(self):
        np.random.seed(42)
        eq = 100 + np.cumsum(np.random.randn(500) * 0.1)
        eq = np.maximum(eq, 1)  # avoid negative equity
        m = compute_all(eq.tolist())
        # Should be finite numbers
        assert np.isfinite(m.sharpe_ratio)
        assert np.isfinite(m.sortino_ratio)

    def test_drawdown_metrics(self):
        eq = [100, 110, 90, 95, 110, 120]
        m = compute_all(eq)
        assert m.max_drawdown > 0
        assert m.drawdown_count > 0
        assert m.max_drawdown_duration > 0

    def test_profit_factor_computed(self):
        pnl = [100, -30, 50, -20]
        m = compute_all([100, 200], pnl)
        assert m.profit_factor == pytest.approx(150 / 50)

    def test_expectancy_computed(self):
        pnl = [100, -50, 200]
        m = compute_all([100, 200], pnl)
        assert m.expectancy == pytest.approx(250 / 3)

    def test_recovery_factor(self):
        eq = [100, 110, 90, 120]
        pnl = [10, -20, 30]
        m = compute_all(eq, pnl)
        # Total PnL = 20, dollar drawdown = max_dd_frac * starting_equity
        dollar_dd = m.max_drawdown * eq[0]
        expected_rf = sum(pnl) / dollar_dd if dollar_dd > 0 else 0
        assert m.recovery_factor == pytest.approx(expected_rf)

    def test_empty_curve(self):
        m = compute_all([])
        assert m.total_return == 0.0
        assert m.sharpe_ratio == 0.0

    def test_single_point_curve(self):
        m = compute_all([100])
        assert m.total_return == 0.0

    def test_no_trades(self):
        m = compute_all([100, 110, 120])
        assert m.total_trades == 0
        assert m.wins == 0
        assert m.win_rate == 0.0

    def test_monthly_returns_populated(self):
        timestamps = [
            datetime(2026, 1, 1) + timedelta(days=i) for i in range(60)
        ]
        eq = [100 + i * 0.5 for i in range(60)]
        m = compute_all(eq, timestamps=timestamps)
        assert len(m.monthly_returns) > 0

    def test_calmar_ratio_computed(self):
        eq = [100, 110, 90, 120]
        m = compute_all(eq)
        assert m.calmar_ratio == pytest.approx(
            m.total_return / m.max_drawdown if m.max_drawdown > 0 else 0)


# ── compute_from_db ──────────────────────────────────────────────────

class TestComputeFromDB:
    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a temporary trades.db with test data."""
        path = str(tmp_path / "trades.db")
        conn = sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE equity_snapshots (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                equity REAL,
                cash REAL
            )
        """)
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                amount REAL,
                price REAL,
                reason TEXT,
                pnl REAL,
                strategy TEXT
            )
        """)
        # Insert equity snapshots
        base = datetime(2026, 1, 1)
        for i in range(100):
            ts = (base + timedelta(minutes=i)).isoformat()
            equity = 100000 + i * 10 + ((-1) ** i) * 5
            conn.execute(
                "INSERT INTO equity_snapshots (timestamp, equity, cash) VALUES (?, ?, ?)",
                (ts, equity, 50000),
            )
        # Insert some trades
        conn.execute(
            "INSERT INTO trades (timestamp, symbol, side, amount, price, reason, pnl, strategy) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("2026-01-01T01:00:00", "AAPL", "sell", 1000, 150, "signal", 50.0, "momentum"),
        )
        conn.execute(
            "INSERT INTO trades (timestamp, symbol, side, amount, price, reason, pnl, strategy) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("2026-01-01T02:00:00", "TSLA", "sell", 500, 200, "signal", -20.0, "momentum"),
        )
        conn.commit()
        conn.close()
        return path

    def test_computes_from_db(self, db_path):
        m = compute_from_db(db_path)
        assert m.total_trades == 2
        assert m.wins == 1
        assert m.losses == 1
        assert m.sharpe_ratio != 0  # Should have a value from equity curve

    def test_missing_db_returns_empty(self, tmp_path):
        m = compute_from_db(str(tmp_path / "nonexistent.db"))
        assert m.total_return == 0.0
        assert m.total_trades == 0

    def test_empty_db(self, tmp_path):
        path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE equity_snapshots (id INTEGER, timestamp TEXT, equity REAL, cash REAL)")
        conn.execute("CREATE TABLE trades (id INTEGER, timestamp TEXT, symbol TEXT, side TEXT, amount REAL, price REAL, reason TEXT, pnl REAL, strategy TEXT)")
        conn.commit()
        conn.close()
        m = compute_from_db(path)
        assert m.total_return == 0.0


# ── PortfolioMetrics dataclass ───────────────────────────────────────

class TestPortfolioMetricsDataclass:
    def test_defaults(self):
        m = PortfolioMetrics()
        assert m.total_return == 0.0
        assert m.monthly_returns == {}
        assert m.total_trades == 0

    def test_fields_writable(self):
        m = PortfolioMetrics()
        m.sharpe_ratio = 1.5
        m.sortino_ratio = 2.0
        assert m.sharpe_ratio == 1.5
