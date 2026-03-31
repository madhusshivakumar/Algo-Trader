"""Unit tests for the logger and database layer."""

import sqlite3
import os
import pytest
from unittest.mock import patch

from utils.logger import Logger, DB_PATH


class TestLoggerDB:
    """Test database operations."""

    def test_db_creates_tables(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        with patch("utils.logger.DB_PATH", db_path):
            logger = Logger()

        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        conn.close()

        assert "trades" in table_names
        assert "equity_snapshots" in table_names

    def test_trade_inserts_record(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        with patch("utils.logger.DB_PATH", db_path):
            logger = Logger()
            logger.trade("AAPL", "buy", 5000, 248.50, "test buy", strategy="mean_reversion")

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT * FROM trades").fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][2] == "AAPL"       # symbol
        assert rows[0][3] == "buy"          # side
        assert rows[0][4] == 5000           # amount
        assert rows[0][5] == 248.50         # price
        assert rows[0][6] == "test buy"     # reason
        assert rows[0][8] == "mean_reversion"  # strategy

    def test_trade_with_pnl(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        with patch("utils.logger.DB_PATH", db_path):
            logger = Logger()
            logger.trade("AAPL", "sell", 5000, 250, "test sell", pnl=32.50, strategy="mr")

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT pnl FROM trades").fetchone()
        conn.close()
        assert row[0] == 32.50

    def test_snapshot_inserts_record(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        with patch("utils.logger.DB_PATH", db_path):
            logger = Logger()
            logger.snapshot(100000, 50000)

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT * FROM equity_snapshots").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][2] == 100000  # equity
        assert rows[0][3] == 50000   # cash

    def test_strategy_column_exists(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        with patch("utils.logger.DB_PATH", db_path):
            logger = Logger()

        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()]
        conn.close()
        assert "strategy" in cols

    def test_migration_adds_strategy_column(self, tmp_path):
        """Test that the migration adds 'strategy' to an old DB without it."""
        db_path = str(tmp_path / "old.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY, timestamp TEXT, symbol TEXT,
                side TEXT, amount REAL, price REAL, reason TEXT, pnl REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE equity_snapshots (
                id INTEGER PRIMARY KEY, timestamp TEXT, equity REAL, cash REAL
            )
        """)
        conn.commit()
        conn.close()

        with patch("utils.logger.DB_PATH", db_path):
            logger = Logger()

        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()]
        conn.close()
        assert "strategy" in cols


class TestLoggerOutput:
    """Test that logging methods don't crash."""

    def test_info_no_crash(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        with patch("utils.logger.DB_PATH", db_path):
            logger = Logger()
            logger.info("test message")  # Should not raise

    def test_warning_no_crash(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        with patch("utils.logger.DB_PATH", db_path):
            logger = Logger()
            logger.warning("test warning")

    def test_error_no_crash(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        with patch("utils.logger.DB_PATH", db_path):
            logger = Logger()
            logger.error("test error")
