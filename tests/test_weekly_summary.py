"""Tests for agents/weekly_summary.py — Bug #10 (May 3)."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


def _make_db(tmp_path) -> str:
    """Spin up a tiny trades db with the schema the agent reads."""
    db = str(tmp_path / "trades.db")
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, symbol TEXT, side TEXT, amount REAL,
            price REAL, reason TEXT, pnl REAL DEFAULT 0,
            strategy TEXT DEFAULT ''
        );
    """)
    conn.commit()
    conn.close()
    return db


def _insert(db: str, rows):
    conn = sqlite3.connect(db)
    conn.executemany(
        "INSERT INTO trades (timestamp, symbol, side, amount, price, "
        "reason, pnl, strategy) VALUES (?,?,?,?,?,?,?,?)", rows,
    )
    conn.commit()
    conn.close()


def _today_iso():
    return datetime.now().date().isoformat()


def _week_ago_iso():
    return (datetime.now().date() - timedelta(days=6)).isoformat()


class TestQueryWeek:
    def test_empty_db_returns_zero_summary(self, tmp_path):
        from agents.weekly_summary import _query_week
        db = _make_db(tmp_path)
        s = _query_week(db, _week_ago_iso(), _today_iso())
        assert s["trades"] == 0
        assert s["pnl"] == 0.0
        assert s["wins"] == 0
        assert s["losses"] == 0

    def test_missing_db_returns_zero_summary(self, tmp_path):
        from agents.weekly_summary import _query_week
        s = _query_week(str(tmp_path / "nope.db"), "2026-01-01", "2026-01-07")
        assert s["trades"] == 0

    def test_aggregates_realized_pnl(self, tmp_path):
        from agents.weekly_summary import _query_week
        db = _make_db(tmp_path)
        today = _today_iso()
        _insert(db, [
            (f"{today} 09:00:00", "AAPL", "buy", 100, 150, "buy", 0, "mr"),
            (f"{today} 10:00:00", "AAPL", "sell", 100, 152, "sell", 200.0, "mr"),
            (f"{today} 11:00:00", "TSLA", "sell", 100, 200, "sell", -50.0, "mr"),
        ])
        s = _query_week(db, _week_ago_iso(), today)
        assert s["trades"] == 2  # only sells (closed)
        assert s["wins"] == 1
        assert s["losses"] == 1
        assert s["pnl"] == 150.0

    def test_per_day_breakdown(self, tmp_path):
        from agents.weekly_summary import _query_week
        db = _make_db(tmp_path)
        d1 = (datetime.now().date() - timedelta(days=2)).isoformat()
        d2 = _today_iso()
        _insert(db, [
            (f"{d1} 10:00:00", "AAPL", "sell", 100, 150, "x", 100.0, "mr"),
            (f"{d2} 10:00:00", "TSLA", "sell", 100, 200, "x", -25.0, "mr"),
        ])
        s = _query_week(db, _week_ago_iso(), d2)
        assert len(s["by_day"]) == 2
        days = {d["date"]: d["pnl"] for d in s["by_day"]}
        assert days[d1] == 100.0
        assert days[d2] == -25.0

    def test_top_and_bottom_symbols(self, tmp_path):
        from agents.weekly_summary import _query_week
        db = _make_db(tmp_path)
        today = _today_iso()
        _insert(db, [
            (f"{today} 10:00:00", "AAPL", "sell", 100, 150, "x", 500.0, "mr"),
            (f"{today} 11:00:00", "TSLA", "sell", 100, 200, "x", 200.0, "mr"),
            (f"{today} 12:00:00", "INTC", "sell", 100, 30, "x", -150.0, "mr"),
        ])
        s = _query_week(db, _week_ago_iso(), today)
        assert s["by_symbol_top"][0]["sym"] == "AAPL"
        assert s["by_symbol_top"][0]["pnl"] == 500.0
        # INTC is the only loser → goes in bottom bucket
        assert any(b["sym"] == "INTC" for b in s["by_symbol_bottom"])


class TestFormatSummary:
    def test_zero_trade_message(self):
        from agents.weekly_summary import _format_summary
        msg = _format_summary(
            {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0,
             "by_day": [], "by_symbol_top": [], "by_symbol_bottom": []},
            "2026-04-27", "2026-05-03", 0,
        )
        assert "NO closed trades" in msg

    def test_normal_summary_includes_key_fields(self):
        from agents.weekly_summary import _format_summary
        msg = _format_summary(
            {"trades": 10, "wins": 7, "losses": 3, "pnl": 1234.56,
             "by_day": [
                 {"date": "2026-04-30", "n": 5, "pnl": 800.00},
                 {"date": "2026-05-01", "n": 5, "pnl": 434.56},
             ],
             "by_symbol_top": [{"sym": "AAPL", "n": 3, "pnl": 600.0}],
             "by_symbol_bottom": []},
            "2026-04-27", "2026-05-03", 0,
        )
        assert "Trades: 10" in msg
        assert "70% WR" in msg
        assert "$+1,234.56" in msg
        assert "Best day" in msg
        assert "AAPL" in msg

    def test_heartbeat_alarms_surfaced(self):
        from agents.weekly_summary import _format_summary
        msg = _format_summary(
            {"trades": 5, "wins": 3, "losses": 2, "pnl": 100.0,
             "by_day": [{"date": "2026-05-03", "n": 5, "pnl": 100.0}],
             "by_symbol_top": [], "by_symbol_bottom": []},
            "2026-04-27", "2026-05-03", 12,
        )
        assert "Heartbeat-stale alarms this week: 12" in msg


class TestMain:
    def test_dry_run_no_alert(self, tmp_path, monkeypatch):
        import agents.weekly_summary as mod
        db = _make_db(tmp_path)
        monkeypatch.setattr(mod, "_DEFAULT_DB", db)
        # Make AlertManager unreachable so a non-dry call would crash
        with patch("core.alerting.AlertManager",
                   side_effect=RuntimeError("should not be called")):
            rc = mod.main(["--dry-run", "--db", db])
        assert rc == 0

    def test_live_run_calls_alert_and_flush(self, tmp_path, monkeypatch):
        import agents.weekly_summary as mod
        db = _make_db(tmp_path)
        today = _today_iso()
        _insert(db, [
            (f"{today} 10:00:00", "AAPL", "sell", 100, 150, "x", 50.0, "mr"),
        ])

        fake_mgr = MagicMock()
        fake_mgr.channels = ["dummy"]
        fake_mgr.flush.return_value = 0
        with patch("core.alerting.AlertManager", return_value=fake_mgr):
            rc = mod.main(["--db", db])
        assert rc == 0
        fake_mgr.alert.assert_called_once()
        # Bug #1a discipline: flush must be called before exit
        fake_mgr.flush.assert_called_once()

    def test_no_channels_returns_0_with_warning(self, tmp_path):
        import agents.weekly_summary as mod
        db = _make_db(tmp_path)
        fake_mgr = MagicMock()
        fake_mgr.channels = []
        with patch("core.alerting.AlertManager", return_value=fake_mgr):
            rc = mod.main(["--db", db])
        assert rc == 0
        fake_mgr.alert.assert_not_called()

    def test_flush_timeout_returns_1(self, tmp_path):
        import agents.weekly_summary as mod
        db = _make_db(tmp_path)
        fake_mgr = MagicMock()
        fake_mgr.channels = ["dummy"]
        fake_mgr.flush.return_value = 1  # one thread didn't finish
        with patch("core.alerting.AlertManager", return_value=fake_mgr):
            rc = mod.main(["--db", db])
        assert rc == 1
