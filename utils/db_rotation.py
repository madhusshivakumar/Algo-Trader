"""SQLite database rotation — archives old trades and vacuums the main DB.

Prevents trades.db from growing unbounded. Archives old rows to
logs/archives/trades_YYYYMMDD.db, then removes them from the main DB.
"""

import os
import shutil
import sqlite3
from datetime import datetime, timedelta

from utils.logger import log


class DBRotator:
    def __init__(self, db_path: str, archive_dir: str = None):
        self.db_path = db_path
        if archive_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(db_path)))
            self.archive_dir = os.path.join(base_dir, "logs", "archives")
        else:
            self.archive_dir = archive_dir

    def should_rotate(self, max_rows: int = 50000, max_age_days: int = 90) -> bool:
        """Check if rotation is needed based on row count or oldest record age."""
        if not os.path.exists(self.db_path):
            return False

        conn = sqlite3.connect(self.db_path)
        try:
            # Check row count
            row_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            if row_count >= max_rows:
                return True

            # Check oldest record age
            oldest = conn.execute("SELECT MIN(timestamp) FROM trades").fetchone()[0]
            if oldest:
                try:
                    oldest_dt = datetime.fromisoformat(oldest)
                    age_days = (datetime.now() - oldest_dt).days
                    if age_days >= max_age_days:
                        return True
                except (ValueError, TypeError):
                    pass

            return False
        finally:
            conn.close()

    def rotate(self, keep_recent_days: int = 30) -> str:
        """Archive old rows and vacuum. Returns archive file path."""
        os.makedirs(self.archive_dir, exist_ok=True)

        cutoff = datetime.now() - timedelta(days=keep_recent_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        archive_name = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db"
        archive_path = os.path.join(self.archive_dir, archive_name)

        # Copy full DB as archive
        shutil.copy2(self.db_path, archive_path)

        # Delete old rows from main DB
        conn = sqlite3.connect(self.db_path)
        try:
            deleted = conn.execute(
                "DELETE FROM trades WHERE timestamp < ?", (cutoff_str,)
            ).rowcount
            conn.execute(
                "DELETE FROM equity_snapshots WHERE timestamp < ?", (cutoff_str,)
            )
            conn.commit()
            conn.execute("VACUUM")
            log.info(f"DB rotation: archived {deleted} rows to {archive_name}")
        finally:
            conn.close()

        return archive_path

    def list_archives(self) -> list[str]:
        """List all archive files."""
        if not os.path.exists(self.archive_dir):
            return []
        return sorted([
            f for f in os.listdir(self.archive_dir)
            if f.startswith("trades_") and f.endswith(".db")
        ])

    def get_db_stats(self) -> dict:
        """Get current database statistics."""
        if not os.path.exists(self.db_path):
            return {"trade_count": 0, "snapshot_count": 0, "size_bytes": 0}

        conn = sqlite3.connect(self.db_path)
        try:
            trade_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            snapshot_count = conn.execute("SELECT COUNT(*) FROM equity_snapshots").fetchone()[0]
            size_bytes = os.path.getsize(self.db_path)
            oldest = conn.execute("SELECT MIN(timestamp) FROM trades").fetchone()[0]
            newest = conn.execute("SELECT MAX(timestamp) FROM trades").fetchone()[0]
            return {
                "trade_count": trade_count,
                "snapshot_count": snapshot_count,
                "size_bytes": size_bytes,
                "oldest_trade": oldest,
                "newest_trade": newest,
            }
        finally:
            conn.close()
