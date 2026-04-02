"""State persistence — saves/loads engine runtime state to SQLite.

Persists data that would otherwise be lost on restart:
  - Trailing stops (symbol → entry_price, highest_price, stop_pct)
  - Cooldown timestamps (symbol → last_trade_time)
  - PDT buy records (symbol → buy_datetime)
  - Risk manager state (halted, halt_reason, daily_start_equity)
  - Managed order records (for order lifecycle tracking)
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Optional

from utils.logger import DB_PATH


class StateStore:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._init_tables()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runtime_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS persisted_trailing_stops (
                symbol TEXT PRIMARY KEY,
                entry_price REAL,
                highest_price REAL,
                stop_pct REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS persisted_cooldowns (
                symbol TEXT PRIMARY KEY,
                last_trade_time REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS persisted_pdt_buys (
                symbol TEXT PRIMARY KEY,
                buy_datetime TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS managed_orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                order_type TEXT,
                requested_notional REAL,
                requested_qty REAL,
                limit_price REAL,
                stop_price REAL,
                expected_price REAL,
                state TEXT,
                filled_qty REAL DEFAULT 0,
                filled_avg_price REAL DEFAULT 0,
                slippage REAL DEFAULT 0,
                submitted_at TEXT,
                filled_at TEXT,
                last_checked TEXT,
                error TEXT DEFAULT ''
            )
        """)
        conn.commit()
        conn.close()

    # ── Scalar state (key-value) ────────────────────────────────────

    def save_scalar(self, key: str, value: Any) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO runtime_state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def load_scalar(self, key: str, default: Any = None) -> Any:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT value FROM runtime_state WHERE key = ?", (key,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return default
        return json.loads(row[0])

    # ── Trailing stops ──────────────────────────────────────────────

    def save_trailing_stops(self, stops: dict) -> None:
        """Save trailing stops. stops = {symbol: {entry_price, highest_price, stop_pct}}"""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM persisted_trailing_stops")
            for symbol, data in stops.items():
                conn.execute(
                    "INSERT INTO persisted_trailing_stops (symbol, entry_price, highest_price, stop_pct) "
                    "VALUES (?, ?, ?, ?)",
                    (symbol, data["entry_price"], data["highest_price"], data["stop_pct"]),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def load_trailing_stops(self) -> dict:
        """Returns {symbol: {entry_price, highest_price, stop_pct}}"""
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT symbol, entry_price, highest_price, stop_pct FROM persisted_trailing_stops").fetchall()
        finally:
            conn.close()
        return {
            row[0]: {"entry_price": row[1], "highest_price": row[2], "stop_pct": row[3]}
            for row in rows
        }

    # ── Cooldowns ───────────────────────────────────────────────────

    def save_cooldowns(self, cooldowns: dict[str, float]) -> None:
        """Save cooldown timestamps. cooldowns = {symbol: last_trade_time_epoch}"""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM persisted_cooldowns")
            for symbol, ts in cooldowns.items():
                conn.execute(
                    "INSERT INTO persisted_cooldowns (symbol, last_trade_time) VALUES (?, ?)",
                    (symbol, ts),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def load_cooldowns(self) -> dict[str, float]:
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT symbol, last_trade_time FROM persisted_cooldowns").fetchall()
        finally:
            conn.close()
        return {row[0]: row[1] for row in rows}

    # ── PDT buy records ─────────────────────────────────────────────

    def save_pdt_buys(self, buys: dict[str, str]) -> None:
        """Save PDT records. buys = {symbol: iso_datetime_string}"""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM persisted_pdt_buys")
            for symbol, dt_str in buys.items():
                conn.execute(
                    "INSERT INTO persisted_pdt_buys (symbol, buy_datetime) VALUES (?, ?)",
                    (symbol, dt_str),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def load_pdt_buys(self) -> dict[str, str]:
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT symbol, buy_datetime FROM persisted_pdt_buys").fetchall()
        finally:
            conn.close()
        return {row[0]: row[1] for row in rows}

    # ── Managed orders ──────────────────────────────────────────────

    def save_order(self, order: dict) -> None:
        """Upsert a managed order record."""
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO managed_orders
                (order_id, symbol, side, order_type, requested_notional, requested_qty,
                 limit_price, stop_price, expected_price, state, filled_qty, filled_avg_price,
                 slippage, submitted_at, filled_at, last_checked, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    order["order_id"], order["symbol"], order["side"], order["order_type"],
                    order.get("requested_notional"), order.get("requested_qty"),
                    order.get("limit_price"), order.get("stop_price"),
                    order.get("expected_price", 0), order["state"],
                    order.get("filled_qty", 0), order.get("filled_avg_price", 0),
                    order.get("slippage", 0), order.get("submitted_at"),
                    order.get("filled_at"), order.get("last_checked"),
                    order.get("error", ""),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def load_active_orders(self) -> list[dict]:
        """Load non-terminal orders."""
        terminal = ("filled", "canceled", "rejected", "expired", "failed")
        conn = self._get_conn()
        try:
            placeholders = ",".join("?" for _ in terminal)
            cursor = conn.execute(
                f"SELECT * FROM managed_orders WHERE state NOT IN ({placeholders})",
                terminal,
            )
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        finally:
            conn.close()
        return [dict(zip(columns, row)) for row in rows]

    # ── Bulk engine state ───────────────────────────────────────────

    def save_engine_state(self, trailing_stops: dict, cooldowns: dict[str, float],
                          pdt_buys: dict[str, str], scalars: dict) -> None:
        """Atomically save all engine state in one transaction."""
        conn = self._get_conn()
        try:
            # Scalars
            for key, value in scalars.items():
                conn.execute(
                    "INSERT OR REPLACE INTO runtime_state (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, json.dumps(value), datetime.now().isoformat()),
                )

            # Trailing stops
            conn.execute("DELETE FROM persisted_trailing_stops")
            for symbol, data in trailing_stops.items():
                conn.execute(
                    "INSERT INTO persisted_trailing_stops (symbol, entry_price, highest_price, stop_pct) "
                    "VALUES (?, ?, ?, ?)",
                    (symbol, data["entry_price"], data["highest_price"], data["stop_pct"]),
                )

            # Cooldowns
            conn.execute("DELETE FROM persisted_cooldowns")
            for symbol, ts in cooldowns.items():
                conn.execute(
                    "INSERT INTO persisted_cooldowns (symbol, last_trade_time) VALUES (?, ?)",
                    (symbol, ts),
                )

            # PDT buys
            conn.execute("DELETE FROM persisted_pdt_buys")
            for symbol, dt_str in pdt_buys.items():
                conn.execute(
                    "INSERT INTO persisted_pdt_buys (symbol, buy_datetime) VALUES (?, ?)",
                    (symbol, dt_str),
                )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def load_engine_state(self) -> dict:
        """Load all engine state. Returns dict with keys: trailing_stops, cooldowns, pdt_buys, scalars."""
        return {
            "trailing_stops": self.load_trailing_stops(),
            "cooldowns": self.load_cooldowns(),
            "pdt_buys": self.load_pdt_buys(),
            "scalars": {
                "halted": self.load_scalar("halted", False),
                "halt_reason": self.load_scalar("halt_reason", ""),
                "daily_start_equity": self.load_scalar("daily_start_equity", 0.0),
                "cycle_count": self.load_scalar("cycle_count", 0),
            },
        }
