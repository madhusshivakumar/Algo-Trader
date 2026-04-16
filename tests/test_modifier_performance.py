"""Tests for analytics/modifier_performance.py — the modifier scorer.

Sprint 6D. Covers:
    - _load_trades: SQLite read, missing file, SQLite error path
    - _attach_deltas_to_trades: window matching, most-recent rule, bad timestamps
    - _contribution: Σ(delta × pnl) with malformed data coerced to zero
    - compute_report: no_data vs keep vs disable classification
    - compute_all_reports: one per MODIFIER_REGISTRY entry
    - auto_disable_negative_modifiers: mutates Config, alerts, dry_run
    - format_report_text: readable output
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from analytics import modifier_ab, modifier_performance
from analytics.modifier_performance import (
    MODIFIER_REGISTRY, ModifierReport, _attach_deltas_to_trades,
    _contribution, _load_trades, auto_disable_negative_modifiers,
    compute_all_reports, compute_report, format_report_text,
)


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_log(tmp_path):
    return str(tmp_path / "ab_log.jsonl")


@pytest.fixture
def tmp_db(tmp_path):
    """Create a SQLite DB with a trades table and hand back the path."""
    path = str(tmp_path / "trades.db")
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE trades (
            timestamp TEXT,
            symbol TEXT,
            side TEXT,
            pnl REAL
        )
    """)
    conn.commit()
    conn.close()
    return path


def _insert_trade(db_path: str, ts: str, symbol: str, side: str, pnl: float):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO trades (timestamp, symbol, side, pnl) VALUES (?, ?, ?, ?)",
        (ts, symbol, side, pnl),
    )
    conn.commit()
    conn.close()


# ── _load_trades ─────────────────────────────────────────────────────────

class TestLoadTrades:
    def test_missing_file(self, tmp_path):
        path = str(tmp_path / "no_such.db")
        assert _load_trades(path, datetime.now() - timedelta(days=30)) == []

    def test_returns_rows_since(self, tmp_db):
        old_ts = (datetime.now() - timedelta(days=60)).isoformat()
        new_ts = (datetime.now() - timedelta(days=1)).isoformat()
        _insert_trade(tmp_db, old_ts, "AAPL", "buy", 10.0)
        _insert_trade(tmp_db, new_ts, "AAPL", "buy", 20.0)
        rows = _load_trades(tmp_db, datetime.now() - timedelta(days=30))
        assert len(rows) == 1
        assert rows[0]["pnl"] == 20.0

    def test_handles_sqlite_error(self, tmp_path):
        # Create a file that isn't a real SQLite DB
        path = str(tmp_path / "corrupt.db")
        with open(path, "w") as f:
            f.write("not a database")
        assert _load_trades(path, datetime.now() - timedelta(days=30)) == []


# ── _attach_deltas_to_trades ─────────────────────────────────────────────

class TestAttachDeltas:
    def test_pairs_within_window(self):
        trade_ts = datetime(2026, 4, 1, 10, 0, 0)
        delta_ts = trade_ts - timedelta(minutes=5)  # within default 10-min
        trade = {"symbol": "AAPL",
                 "timestamp": trade_ts.isoformat(),
                 "pnl": 5.0}
        delta = {"symbol": "AAPL",
                 "ts": delta_ts.isoformat(),
                 "delta": 0.1}
        pairs = _attach_deltas_to_trades([trade], [delta])
        assert len(pairs) == 1

    def test_drops_if_outside_window(self):
        trade_ts = datetime(2026, 4, 1, 10, 0, 0)
        delta_ts = trade_ts - timedelta(minutes=20)  # outside default 10-min
        trade = {"symbol": "AAPL",
                 "timestamp": trade_ts.isoformat(),
                 "pnl": 5.0}
        delta = {"symbol": "AAPL",
                 "ts": delta_ts.isoformat(),
                 "delta": 0.1}
        pairs = _attach_deltas_to_trades([trade], [delta])
        assert pairs == []

    def test_picks_most_recent_delta(self):
        trade_ts = datetime(2026, 4, 1, 10, 0, 0)
        trade = {"symbol": "AAPL",
                 "timestamp": trade_ts.isoformat(),
                 "pnl": 5.0}
        deltas = [
            {"symbol": "AAPL",
             "ts": (trade_ts - timedelta(minutes=9)).isoformat(),
             "delta": 0.1},
            {"symbol": "AAPL",
             "ts": (trade_ts - timedelta(minutes=1)).isoformat(),
             "delta": 0.2},  # most recent — should win
        ]
        pairs = _attach_deltas_to_trades([trade], deltas)
        assert len(pairs) == 1
        assert pairs[0][1]["delta"] == 0.2

    def test_delta_after_trade_ignored(self):
        trade_ts = datetime(2026, 4, 1, 10, 0, 0)
        trade = {"symbol": "AAPL",
                 "timestamp": trade_ts.isoformat(),
                 "pnl": 5.0}
        delta = {"symbol": "AAPL",
                 "ts": (trade_ts + timedelta(minutes=1)).isoformat(),
                 "delta": 0.2}
        pairs = _attach_deltas_to_trades([trade], [delta])
        assert pairs == []

    def test_symbol_mismatch_drops(self):
        trade_ts = datetime(2026, 4, 1, 10, 0, 0)
        trade = {"symbol": "AAPL",
                 "timestamp": trade_ts.isoformat(),
                 "pnl": 5.0}
        delta = {"symbol": "TSLA",
                 "ts": (trade_ts - timedelta(minutes=1)).isoformat(),
                 "delta": 0.1}
        pairs = _attach_deltas_to_trades([trade], [delta])
        assert pairs == []

    def test_bad_timestamps_skipped(self):
        trade = {"symbol": "AAPL", "timestamp": "nonsense", "pnl": 5.0}
        delta = {"symbol": "AAPL", "ts": "also-nonsense", "delta": 0.1}
        assert _attach_deltas_to_trades([trade], [delta]) == []

    def test_missing_fields_skipped(self):
        assert _attach_deltas_to_trades(
            [{"pnl": 5.0}],  # no symbol or timestamp
            [{"symbol": "AAPL", "ts": "2026-04-01T10:00:00", "delta": 0.1}],
        ) == []


# ── _contribution ────────────────────────────────────────────────────────

class TestContribution:
    def test_sum_positive(self):
        pairs = [
            ({"pnl": 5.0}, {"delta": 0.1}),
            ({"pnl": 10.0}, {"delta": 0.2}),
        ]
        total, n = _contribution(pairs)
        assert total == pytest.approx(0.5 + 2.0, abs=1e-4)
        assert n == 2

    def test_sum_negative(self):
        pairs = [
            ({"pnl": -5.0}, {"delta": 0.1}),  # modifier boosted, lost money
            ({"pnl": 10.0}, {"delta": -0.2}),  # modifier cut, we still made money
        ]
        total, n = _contribution(pairs)
        # -0.5 + -2.0 = -2.5 → both "bad signals from this modifier"
        assert total == pytest.approx(-2.5, abs=1e-4)
        assert n == 2

    def test_malformed_skipped(self):
        pairs = [
            ({"pnl": "oops"}, {"delta": 0.1}),
            ({"pnl": 10.0}, {"delta": None}),
            ({"pnl": 5.0}, {"delta": 0.2}),  # good one
        ]
        total, n = _contribution(pairs)
        # None → 0.0 so (pnl=10, delta=0) contributes 0
        # "oops" raises on float() → skipped
        # good pair contributes 5 × 0.2 = 1.0
        assert total == pytest.approx(1.0, abs=1e-4)
        # n counts every pair that didn't raise
        assert n == 2

    def test_empty(self):
        assert _contribution([]) == (0.0, 0)


# ── compute_report ───────────────────────────────────────────────────────

class TestComputeReport:
    def test_no_data_when_no_records(self, tmp_log, tmp_db):
        rep = compute_report("sentiment", "SENTIMENT_ENABLED",
                             days=30, db_path=tmp_db, log_path=tmp_log)
        assert rep.recommendation == "no_data"
        assert rep.contribution == 0.0
        assert rep.n_trades_matched == 0
        assert rep.summary["count"] == 0

    def test_no_data_when_under_min_trades(self, tmp_log, tmp_db):
        # Log 2 deltas but only match 2 trades (< min of 5)
        ts = datetime.now() - timedelta(days=1)
        trade_ts = ts + timedelta(minutes=1)  # after the delta
        for i in range(2):
            modifier_ab.log_delta(
                "AAPL", "sentiment",
                {"action": "buy", "strength": 0.5},
                {"action": "buy", "strength": 0.6},
                timestamp=ts + timedelta(seconds=i),
                log_path=tmp_log,
            )
            _insert_trade(tmp_db,
                          (trade_ts + timedelta(seconds=i)).isoformat(),
                          "AAPL", "buy", 10.0)

        rep = compute_report("sentiment", "SENTIMENT_ENABLED",
                             days=30, db_path=tmp_db, log_path=tmp_log)
        assert rep.summary["count"] == 2
        # n_matched will be 2 which is < 5 threshold → no_data
        assert rep.recommendation == "no_data"

    def test_keep_when_contribution_positive(self, tmp_log, tmp_db):
        # 6 trades all with positive delta × positive pnl = positive contribution
        for i in range(6):
            ts = datetime.now() - timedelta(days=1, hours=i)
            modifier_ab.log_delta(
                "AAPL", "sentiment",
                {"action": "buy", "strength": 0.5},
                {"action": "buy", "strength": 0.6},
                timestamp=ts, log_path=tmp_log,
            )
            _insert_trade(tmp_db,
                          (ts + timedelta(minutes=1)).isoformat(),
                          "AAPL", "buy", 20.0)

        rep = compute_report("sentiment", "SENTIMENT_ENABLED",
                             days=30, db_path=tmp_db, log_path=tmp_log)
        assert rep.n_trades_matched == 6
        assert rep.contribution > 0
        assert rep.recommendation == "keep"

    def test_disable_when_contribution_negative(self, tmp_log, tmp_db):
        # 6 trades, delta positive but pnl negative
        for i in range(6):
            ts = datetime.now() - timedelta(days=1, hours=i)
            modifier_ab.log_delta(
                "AAPL", "sentiment",
                {"action": "buy", "strength": 0.5},
                {"action": "buy", "strength": 0.7},
                timestamp=ts, log_path=tmp_log,
            )
            _insert_trade(tmp_db,
                          (ts + timedelta(minutes=1)).isoformat(),
                          "AAPL", "buy", -15.0)

        rep = compute_report("sentiment", "SENTIMENT_ENABLED",
                             days=30, db_path=tmp_db, log_path=tmp_log)
        assert rep.n_trades_matched == 6
        assert rep.contribution < 0
        assert rep.recommendation == "disable"

    def test_disable_when_contribution_zero(self, tmp_log, tmp_db):
        # 6 trades, delta = 0 (noop modifier) → contribution = 0 → disable
        for i in range(6):
            ts = datetime.now() - timedelta(days=1, hours=i)
            modifier_ab.log_delta(
                "AAPL", "sentiment",
                {"action": "buy", "strength": 0.5},
                {"action": "buy", "strength": 0.5},
                timestamp=ts, log_path=tmp_log,
            )
            _insert_trade(tmp_db,
                          (ts + timedelta(minutes=1)).isoformat(),
                          "AAPL", "buy", 10.0)

        rep = compute_report("sentiment", "SENTIMENT_ENABLED",
                             days=30, db_path=tmp_db, log_path=tmp_log)
        assert rep.n_trades_matched == 6
        assert rep.contribution == 0.0
        assert rep.recommendation == "disable"  # <= 0 is disable


# ── compute_all_reports ──────────────────────────────────────────────────

class TestComputeAllReports:
    def test_one_per_registry_entry(self, tmp_log, tmp_db):
        reports = compute_all_reports(days=30,
                                      db_path=tmp_db, log_path=tmp_log)
        assert len(reports) == len(MODIFIER_REGISTRY)
        names = {r.modifier for r in reports}
        expected = {name for name, _ in MODIFIER_REGISTRY}
        assert names == expected


# ── auto_disable_negative_modifiers ──────────────────────────────────────

class TestAutoDisable:
    def _make_report(self, name, flag, rec, contrib=-1.0, n=10):
        return ModifierReport(
            modifier=name, config_flag=flag, days=30,
            contribution=contrib, n_trades_matched=n,
            summary={"count": 20}, recommendation=rec,
        )

    def test_disables_negative(self, monkeypatch):
        """Mutates Config attribute when recommendation=='disable'."""
        from config import Config
        # Use a dedicated test attr to avoid mutating real flags
        monkeypatch.setattr(Config, "_TEST_FLAG", True, raising=False)
        rep = self._make_report("dummy", "_TEST_FLAG", "disable")
        disabled = auto_disable_negative_modifiers([rep])
        assert disabled == ["dummy"]
        assert Config._TEST_FLAG is False

    def test_ignores_keep(self, monkeypatch):
        from config import Config
        monkeypatch.setattr(Config, "_TEST_FLAG2", True, raising=False)
        rep = self._make_report("other", "_TEST_FLAG2", "keep",
                                contrib=1.5, n=20)
        disabled = auto_disable_negative_modifiers([rep])
        assert disabled == []
        assert Config._TEST_FLAG2 is True

    def test_ignores_no_data(self, monkeypatch):
        from config import Config
        monkeypatch.setattr(Config, "_TEST_FLAG3", True, raising=False)
        rep = self._make_report("third", "_TEST_FLAG3", "no_data")
        disabled = auto_disable_negative_modifiers([rep])
        assert disabled == []
        assert Config._TEST_FLAG3 is True

    def test_dry_run_does_not_mutate(self, monkeypatch):
        from config import Config
        monkeypatch.setattr(Config, "_TEST_FLAG4", True, raising=False)
        rep = self._make_report("drytest", "_TEST_FLAG4", "disable")
        disabled = auto_disable_negative_modifiers([rep], dry_run=True)
        assert disabled == ["drytest"]
        # Config should NOT have been flipped
        assert Config._TEST_FLAG4 is True

    def test_alerts_dispatched(self, monkeypatch):
        from config import Config
        monkeypatch.setattr(Config, "_TEST_FLAG5", True, raising=False)
        am = MagicMock()
        rep = self._make_report("alerttest", "_TEST_FLAG5", "disable")
        auto_disable_negative_modifiers([rep], alert_manager=am)
        assert am.alert.called
        args, kwargs = am.alert.call_args
        assert args[0] == "warning"
        assert "alerttest" in args[1]

    def test_alert_failure_does_not_block_disable(self, monkeypatch):
        from config import Config
        monkeypatch.setattr(Config, "_TEST_FLAG6", True, raising=False)
        am = MagicMock()
        am.alert.side_effect = RuntimeError("boom")
        rep = self._make_report("robust", "_TEST_FLAG6", "disable")
        # Should not raise
        disabled = auto_disable_negative_modifiers([rep], alert_manager=am)
        assert disabled == ["robust"]
        assert Config._TEST_FLAG6 is False


# ── format_report_text ───────────────────────────────────────────────────

class TestFormatReportText:
    def test_contains_header(self):
        out = format_report_text([])
        assert "Modifier A/B Report" in out

    def test_contains_modifier_line(self):
        rep = ModifierReport(
            modifier="sentiment", config_flag="SENTIMENT_ENABLED",
            days=30, contribution=1.5, n_trades_matched=12,
            summary={"count": 50, "action_flip_count": 3},
            recommendation="keep",
        )
        out = format_report_text([rep])
        assert "sentiment" in out
        assert "+1.5000" in out
        assert "keep" in out
        assert "12" in out

    def test_contains_negative_format(self):
        rep = ModifierReport(
            modifier="llm", config_flag="LLM_ANALYST_ENABLED",
            days=30, contribution=-2.3, n_trades_matched=7,
            summary={"count": 30, "action_flip_count": 1},
            recommendation="disable",
        )
        out = format_report_text([rep])
        assert "llm" in out
        assert "-2.3000" in out
        assert "disable" in out
