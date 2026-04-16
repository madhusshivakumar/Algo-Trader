"""Sprint 5G: paper→live promotion gate tests.

Verifies that TRADING_MODE=live is refused unless 30+ days of paper trading
have a worst day-loss >= -3%. Verifies the override (I_UNDERSTAND_THE_RISK)
is awkward enough that a beginner won't accidentally trip it.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from core.live_gate import (
    MAX_DRAWDOWN_THRESHOLD,
    MIN_PAPER_DAYS,
    OVERRIDE_ENV,
    OVERRIDE_VALUE,
    GateResult,
    check_live_mode_allowed,
    evaluate_gate,
    is_override_active,
)


def _seed_equity(db_path: str, days: int, equity_per_day: list[float]):
    """Write `equity_snapshots` rows simulating `days` days of trading."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            equity REAL,
            cash REAL
        )
    """)
    base_date = datetime.now() - timedelta(days=days)
    for i, equity in enumerate(equity_per_day):
        ts = (base_date + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO equity_snapshots (timestamp, equity, cash) VALUES (?, ?, ?)",
            (ts, equity, equity),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def tmp_db():
    """Provide an isolated SQLite path for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


class TestEvaluateGate:
    def test_no_history_blocks_promotion(self, tmp_db):
        result = evaluate_gate(db_path=tmp_db)
        assert result.can_promote is False
        assert result.days_observed == 0
        assert "history" in result.reason.lower()

    def test_too_few_days_blocks(self, tmp_db):
        # 10 days of stable equity — too few
        _seed_equity(tmp_db, days=10, equity_per_day=[100_000.0] * 10)
        result = evaluate_gate(db_path=tmp_db, min_days=30)
        assert result.can_promote is False
        assert result.days_observed == 10
        assert "Only 10 days" in result.reason

    def test_passes_with_steady_equity(self, tmp_db):
        # 35 days of perfectly stable equity → worst drawdown 0
        _seed_equity(tmp_db, days=35, equity_per_day=[100_000.0] * 35)
        result = evaluate_gate(db_path=tmp_db, min_days=30, max_drawdown=-0.03)
        assert result.can_promote is True
        assert result.days_observed == 35
        assert result.worst_drawdown == 0.0

    def test_blocks_on_big_single_day_loss(self, tmp_db):
        # 35 days, but day 5 has a -10% loss — fails the -3% threshold
        equity = [100_000.0] * 35
        equity[5] = 90_000.0  # -10% from day 4
        equity[6] = 90_000.0  # back to flat
        _seed_equity(tmp_db, days=35, equity_per_day=equity)
        result = evaluate_gate(db_path=tmp_db, min_days=30, max_drawdown=-0.03)
        assert result.can_promote is False
        assert result.worst_drawdown < -0.05  # at least -10%
        assert "breaches" in result.reason

    def test_passes_with_small_drawdowns(self, tmp_db):
        # 35 days with worst -2% (under -3% threshold)
        equity = [100_000.0] * 35
        equity[10] = 98_000.0  # -2%
        equity[11] = 99_000.0
        _seed_equity(tmp_db, days=35, equity_per_day=equity)
        result = evaluate_gate(db_path=tmp_db, min_days=30, max_drawdown=-0.03)
        assert result.can_promote is True

    def test_missing_db_returns_no_history(self):
        result = evaluate_gate(db_path="/nonexistent/path.db")
        assert result.can_promote is False
        assert result.days_observed == 0


class TestIsOverrideActive:
    def test_no_env_returns_false(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(OVERRIDE_ENV, None)
            assert is_override_active() is False

    def test_wrong_value_returns_false(self):
        with patch.dict(os.environ, {OVERRIDE_ENV: "yes"}):
            assert is_override_active() is False

    def test_correct_value_returns_true(self):
        with patch.dict(os.environ, {OVERRIDE_ENV: OVERRIDE_VALUE}):
            assert is_override_active() is True

    def test_case_insensitive(self):
        with patch.dict(os.environ, {OVERRIDE_ENV: "YES-REALLY"}):
            assert is_override_active() is True


class TestCheckLiveModeAllowed:
    def test_paper_mode_passes_through(self, tmp_db):
        mode, gate, override = check_live_mode_allowed("paper", db_path=tmp_db)
        assert mode == "paper"
        assert gate is None
        assert override is False

    def test_live_with_no_history_downgrades(self, tmp_db):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(OVERRIDE_ENV, None)
            mode, gate, override = check_live_mode_allowed("live", db_path=tmp_db)
        assert mode == "paper"
        assert gate is not None
        assert gate.can_promote is False
        assert override is False

    def test_live_with_validation_passes(self, tmp_db):
        _seed_equity(tmp_db, days=35, equity_per_day=[100_000.0] * 35)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(OVERRIDE_ENV, None)
            mode, gate, override = check_live_mode_allowed("live", db_path=tmp_db)
        assert mode == "live"
        assert gate is not None and gate.can_promote is True
        assert override is False

    def test_override_bypasses_gate(self, tmp_db):
        # Empty DB — would normally fail. Override should succeed anyway.
        with patch.dict(os.environ, {OVERRIDE_ENV: OVERRIDE_VALUE}):
            mode, gate, override = check_live_mode_allowed("live", db_path=tmp_db)
        assert mode == "live"
        assert gate is None  # short-circuited
        assert override is True


class TestGateResult:
    def test_truthy_when_can_promote(self):
        r = GateResult(can_promote=True, days_observed=30, worst_drawdown=-0.01,
                      reason="ok")
        assert bool(r) is True

    def test_falsy_when_blocked(self):
        r = GateResult(can_promote=False, days_observed=5, worst_drawdown=0.0,
                      reason="too few days")
        assert bool(r) is False
