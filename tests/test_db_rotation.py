"""Tests for utils/db_rotation.py — SQLite log rotation."""

import os
import sqlite3
import pytest
from datetime import datetime, timedelta

from utils.db_rotation import DBRotator


def _create_test_db(db_path, num_trades=100, start_days_ago=60):
    """Create a test DB with trades spanning a date range."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, symbol TEXT, side TEXT,
            amount REAL, price REAL, reason TEXT, pnl REAL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, equity REAL, cash REAL
        )
    """)

    now = datetime.now()
    for i in range(num_trades):
        ts = (now - timedelta(days=start_days_ago - (i * start_days_ago / num_trades))).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO trades (timestamp, symbol, side, amount, price, reason) VALUES (?, ?, ?, ?, ?, ?)",
            (ts, "AAPL", "buy", 1000.0, 150.0, "test"),
        )

    for i in range(20):
        ts = (now - timedelta(days=start_days_ago - (i * start_days_ago / 20))).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO equity_snapshots (timestamp, equity, cash) VALUES (?, ?, ?)",
            (ts, 100000.0, 50000.0),
        )

    conn.commit()
    conn.close()


@pytest.fixture
def test_db(tmp_path):
    db_path = str(tmp_path / "test_trades.db")
    _create_test_db(db_path, num_trades=100, start_days_ago=60)
    return db_path


@pytest.fixture
def rotator(test_db, tmp_path):
    archive_dir = str(tmp_path / "archives")
    return DBRotator(test_db, archive_dir=archive_dir)


class TestShouldRotate:
    def test_by_row_count(self, test_db, tmp_path):
        rotator = DBRotator(test_db, archive_dir=str(tmp_path / "archives"))
        assert rotator.should_rotate(max_rows=50) is True

    def test_not_by_row_count(self, test_db, tmp_path):
        rotator = DBRotator(test_db, archive_dir=str(tmp_path / "archives"))
        assert rotator.should_rotate(max_rows=1000) is False

    def test_by_age(self, test_db, tmp_path):
        rotator = DBRotator(test_db, archive_dir=str(tmp_path / "archives"))
        assert rotator.should_rotate(max_rows=1000, max_age_days=30) is True

    def test_not_by_age(self, test_db, tmp_path):
        rotator = DBRotator(test_db, archive_dir=str(tmp_path / "archives"))
        assert rotator.should_rotate(max_rows=1000, max_age_days=365) is False

    def test_nonexistent_db(self, tmp_path):
        rotator = DBRotator(str(tmp_path / "nonexistent.db"))
        assert rotator.should_rotate() is False


class TestRotate:
    def test_creates_archive(self, rotator):
        archive_path = rotator.rotate(keep_recent_days=30)
        assert os.path.exists(archive_path)
        assert archive_path.endswith(".db")

    def test_removes_old_rows(self, rotator, test_db):
        conn = sqlite3.connect(test_db)
        before = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        conn.close()

        rotator.rotate(keep_recent_days=30)

        conn = sqlite3.connect(test_db)
        after = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        conn.close()

        assert after < before

    def test_keeps_recent(self, rotator, test_db):
        rotator.rotate(keep_recent_days=30)

        conn = sqlite3.connect(test_db)
        count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        conn.close()
        assert count > 0  # Recent trades preserved

    def test_archive_contains_full_data(self, rotator, test_db):
        conn = sqlite3.connect(test_db)
        original_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        conn.close()

        archive_path = rotator.rotate(keep_recent_days=30)

        conn = sqlite3.connect(archive_path)
        archive_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        conn.close()

        assert archive_count == original_count  # Full copy

    def test_archive_dir_created(self, test_db, tmp_path):
        archive_dir = str(tmp_path / "new_archives")
        rotator = DBRotator(test_db, archive_dir=archive_dir)
        rotator.rotate(keep_recent_days=30)
        assert os.path.exists(archive_dir)


class TestListArchives:
    def test_no_archives(self, rotator):
        assert rotator.list_archives() == []

    def test_after_rotation(self, rotator):
        rotator.rotate(keep_recent_days=30)
        archives = rotator.list_archives()
        assert len(archives) == 1
        assert archives[0].startswith("trades_")

    def test_multiple_rotations(self, rotator):
        rotator.rotate(keep_recent_days=30)
        # Re-populate and rotate again
        _create_test_db(rotator.db_path, num_trades=50, start_days_ago=60)
        rotator.rotate(keep_recent_days=30)
        archives = rotator.list_archives()
        assert len(archives) == 2


class TestGetDBStats:
    def test_stats(self, rotator):
        stats = rotator.get_db_stats()
        assert stats["trade_count"] == 100
        assert stats["snapshot_count"] == 20
        assert stats["size_bytes"] > 0
        assert stats["oldest_trade"] is not None
        assert stats["newest_trade"] is not None

    def test_empty_db(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, timestamp TEXT)")
        conn.execute("CREATE TABLE equity_snapshots (id INTEGER PRIMARY KEY, timestamp TEXT)")
        conn.commit()
        conn.close()

        rotator = DBRotator(db_path)
        stats = rotator.get_db_stats()
        assert stats["trade_count"] == 0

    def test_nonexistent_db(self, tmp_path):
        rotator = DBRotator(str(tmp_path / "nonexistent.db"))
        stats = rotator.get_db_stats()
        assert stats["trade_count"] == 0
