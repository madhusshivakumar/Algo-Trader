"""Tests for core/drift_detector.py — live performance drift detection."""

import sqlite3
import pytest
from datetime import datetime, timedelta

from core.drift_detector import DriftDetector, DriftMetrics


def _create_test_db(db_path, trades):
    """Create a test DB with given trades list: (timestamp, symbol, strategy, side, pnl)."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, symbol TEXT, side TEXT,
            amount REAL, price REAL, reason TEXT,
            pnl REAL DEFAULT 0, strategy TEXT DEFAULT ''
        )
    """)
    for ts, symbol, strategy, side, pnl in trades:
        conn.execute(
            "INSERT INTO trades (timestamp, symbol, strategy, side, amount, price, reason, pnl) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, symbol, strategy, side, 1000, 100, "test", pnl),
        )
    conn.commit()
    conn.close()


def _ts(days_ago):
    """Helper to create ISO timestamp N days ago."""
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


@pytest.fixture
def good_baseline_db(tmp_path):
    """DB with good baseline performance and degraded recent performance."""
    db_path = str(tmp_path / "trades.db")
    trades = []
    # Baseline: 14-28 days ago, good performance
    for i in range(10):
        trades.append((_ts(20 + i), "AAPL", "momentum", "buy", 0))
        trades.append((_ts(20 + i), "AAPL", "momentum", "sell", 50.0))  # winning
    # Recent: last 7 days, poor performance
    for i in range(10):
        trades.append((_ts(i + 1), "AAPL", "momentum", "buy", 0))
        trades.append((_ts(i + 1), "AAPL", "momentum", "sell", -30.0))  # losing
    _create_test_db(db_path, trades)
    return db_path


@pytest.fixture
def stable_db(tmp_path):
    """DB with consistent performance."""
    db_path = str(tmp_path / "trades.db")
    trades = []
    for i in range(30):
        trades.append((_ts(i + 1), "AAPL", "mean_reversion", "buy", 0))
        trades.append((_ts(i + 1), "AAPL", "mean_reversion", "sell", 20.0))
    _create_test_db(db_path, trades)
    return db_path


@pytest.fixture
def empty_db(tmp_path):
    db_path = str(tmp_path / "trades.db")
    _create_test_db(db_path, [])
    return db_path


class TestCheckDrift:
    def test_detects_degradation(self, good_baseline_db):
        detector = DriftDetector(db_path=good_baseline_db, lookback_days=7, min_trades=3)
        metrics = detector.check_drift()
        aapl = [m for m in metrics if m.symbol == "AAPL"]
        assert len(aapl) == 1
        assert aapl[0].is_degraded is True
        assert aapl[0].degradation_reason != ""

    def test_stable_not_degraded(self, stable_db):
        detector = DriftDetector(db_path=stable_db, lookback_days=7, min_trades=3)
        metrics = detector.check_drift()
        for m in metrics:
            assert m.is_degraded is False

    def test_empty_db(self, empty_db):
        detector = DriftDetector(db_path=empty_db)
        metrics = detector.check_drift()
        assert metrics == []

    def test_filter_by_symbol(self, good_baseline_db):
        detector = DriftDetector(db_path=good_baseline_db, lookback_days=7)
        metrics = detector.check_drift(symbol="AAPL")
        assert all(m.symbol == "AAPL" for m in metrics)

    def test_filter_nonexistent_symbol(self, good_baseline_db):
        detector = DriftDetector(db_path=good_baseline_db, lookback_days=7)
        metrics = detector.check_drift(symbol="ZZZZ")
        assert metrics == []

    def test_insufficient_trades_not_flagged(self, tmp_path):
        """With min_trades=20 and only 10 trades, should not flag as degraded."""
        db_path = str(tmp_path / "trades.db")
        trades = []
        for i in range(5):
            trades.append((_ts(i + 1), "AAPL", "test", "sell", -100.0))
        for i in range(5):
            trades.append((_ts(20 + i), "AAPL", "test", "sell", 100.0))
        _create_test_db(db_path, trades)
        detector = DriftDetector(db_path=db_path, lookback_days=7, min_trades=20)
        metrics = detector.check_drift()
        for m in metrics:
            assert m.is_degraded is False


class TestDriftMetrics:
    def test_metrics_fields(self, good_baseline_db):
        detector = DriftDetector(db_path=good_baseline_db, lookback_days=7)
        metrics = detector.check_drift()
        m = metrics[0]
        assert isinstance(m, DriftMetrics)
        assert isinstance(m.win_rate, float)
        assert isinstance(m.avg_pnl, float)
        assert isinstance(m.trade_count, int)


class TestIsDegraded:
    def test_win_rate_drop(self):
        detector = DriftDetector()
        recent = DriftMetrics(symbol="X", strategy="", window_start="", window_end="",
                              trade_count=10, win_rate=0.3, avg_pnl=-5, total_pnl=-50)
        baseline = DriftMetrics(symbol="X", strategy="", window_start="", window_end="",
                                trade_count=10, win_rate=0.7, avg_pnl=10, total_pnl=100)
        degraded, reason = detector._is_degraded(recent, baseline)
        assert degraded is True
        assert "Win rate" in reason

    def test_avg_pnl_turned_negative(self):
        detector = DriftDetector()
        recent = DriftMetrics(symbol="X", strategy="", window_start="", window_end="",
                              trade_count=10, win_rate=0.5, avg_pnl=-10, total_pnl=-100)
        baseline = DriftMetrics(symbol="X", strategy="", window_start="", window_end="",
                                trade_count=10, win_rate=0.5, avg_pnl=10, total_pnl=100)
        degraded, reason = detector._is_degraded(recent, baseline)
        assert degraded is True
        assert "negative" in reason.lower()

    def test_no_degradation(self):
        detector = DriftDetector()
        recent = DriftMetrics(symbol="X", strategy="", window_start="", window_end="",
                              trade_count=10, win_rate=0.6, avg_pnl=15, total_pnl=150)
        baseline = DriftMetrics(symbol="X", strategy="", window_start="", window_end="",
                                trade_count=10, win_rate=0.55, avg_pnl=10, total_pnl=100)
        degraded, _ = detector._is_degraded(recent, baseline)
        assert degraded is False


class TestGetReport:
    def test_report_structure(self, good_baseline_db):
        detector = DriftDetector(db_path=good_baseline_db, lookback_days=7, min_trades=3)
        report = detector.get_report()
        assert "timestamp" in report
        assert "total_symbols" in report
        assert "degraded_count" in report
        assert "degraded_symbols" in report
        assert "all_metrics" in report
        assert report["degraded_count"] > 0

    def test_report_empty_db(self, empty_db):
        detector = DriftDetector(db_path=empty_db)
        report = detector.get_report()
        assert report["total_symbols"] == 0
        assert report["degraded_count"] == 0


class TestShouldAlert:
    def test_alerts_for_degraded(self, good_baseline_db):
        detector = DriftDetector(db_path=good_baseline_db, lookback_days=7, min_trades=3)
        alerts = detector.should_alert()
        assert len(alerts) > 0
        assert all("DRIFT ALERT" in a for a in alerts)

    def test_no_alerts_for_stable(self, stable_db):
        detector = DriftDetector(db_path=stable_db, lookback_days=7, min_trades=3)
        alerts = detector.should_alert()
        assert alerts == []
