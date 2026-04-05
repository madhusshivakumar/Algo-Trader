"""Simple logging with Rich for nice terminal output."""

import sqlite3
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table

console = Console()

DB_PATH = os.environ.get("DB_PATH",
                         os.path.join(os.path.dirname(os.path.dirname(__file__)), "trades.db"))


class Logger:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        # WAL mode: allows concurrent reads from dashboard/agents while engine writes
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                amount REAL,
                price REAL,
                reason TEXT,
                pnl REAL DEFAULT 0,
                strategy TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
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
                last_updated TEXT,
                error TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                equity REAL,
                cash REAL
            )
        """)
        conn.commit()
        # Migrate: add columns if missing (existing DBs)
        for col, col_type in [
            ("strategy", "TEXT DEFAULT ''"),
            ("sentiment_score", "REAL"),
            ("llm_conviction", "REAL"),
            ("rl_selected", "TEXT DEFAULT ''"),
        ]:
            try:
                conn.execute(f"SELECT {col} FROM trades LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {col_type}")
                conn.commit()
        conn.close()

    def info(self, msg: str):
        console.print(f"[dim]{_ts()}[/dim] [blue]INFO[/blue]  {msg}")

    def success(self, msg: str):
        console.print(f"[dim]{_ts()}[/dim] [green]OK[/green]    {msg}")

    def warning(self, msg: str):
        console.print(f"[dim]{_ts()}[/dim] [yellow]WARN[/yellow]  {msg}")

    def error(self, msg: str):
        console.print(f"[dim]{_ts()}[/dim] [red]ERROR[/red] {msg}")

    def trade(self, symbol: str, side: str, amount: float, price: float,
              reason: str, pnl: float = 0, strategy: str = "",
              sentiment_score: float = None, llm_conviction: float = None,
              rl_selected: str = ""):
        self.success(
            f"{'BUY' if side == 'buy' else 'SELL'} {symbol} "
            f"${amount:.2f} @ ${price:.2f} | {reason}"
            + (f" | PnL: ${pnl:.2f}" if pnl is not None else "")
        )
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                "INSERT INTO trades (timestamp, symbol, side, amount, price, reason, pnl, "
                "strategy, sentiment_score, llm_conviction, rl_selected) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (_ts(), symbol, side, amount, price, reason, pnl, strategy,
                 sentiment_score, llm_conviction, rl_selected),
            )
            conn.commit()
        finally:
            conn.close()

    def log_order(self, order_dict: dict):
        """Log or update a managed order in the orders table."""
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                """INSERT OR REPLACE INTO orders
                (order_id, symbol, side, order_type, requested_notional, requested_qty,
                 limit_price, stop_price, expected_price, state, filled_qty, filled_avg_price,
                 slippage, submitted_at, filled_at, last_updated, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    order_dict["order_id"], order_dict.get("symbol"), order_dict.get("side"),
                    order_dict.get("order_type"), order_dict.get("requested_notional"),
                    order_dict.get("requested_qty"), order_dict.get("limit_price"),
                    order_dict.get("stop_price"), order_dict.get("expected_price", 0),
                    order_dict.get("state", ""), order_dict.get("filled_qty", 0),
                    order_dict.get("filled_avg_price", 0), order_dict.get("slippage", 0),
                    order_dict.get("submitted_at"), order_dict.get("filled_at"),
                    _ts(), order_dict.get("error", ""),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def log_slippage(self, symbol: str, expected: float, actual: float, slippage_pct: float):
        """Log slippage for a filled order."""
        direction = "over" if slippage_pct > 0 else "under"
        self.info(
            f"Slippage {symbol}: expected ${expected:.2f} → actual ${actual:.2f} "
            f"({direction} by {abs(slippage_pct):.4%})"
        )

    def snapshot(self, equity: float, cash: float):
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                "INSERT INTO equity_snapshots (timestamp, equity, cash) VALUES (?,?,?)",
                (_ts(), equity, cash),
            )
            conn.commit()
        finally:
            conn.close()

    def print_summary(self):
        conn = sqlite3.connect(DB_PATH)
        trades = conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 20").fetchall()
        snapshots = conn.execute(
            "SELECT * FROM equity_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()

        if snapshots:
            console.print(f"\n[bold]Current Equity:[/bold] ${snapshots[2]:.2f}  |  Cash: ${snapshots[3]:.2f}")

        if trades:
            table = Table(title="Recent Trades")
            table.add_column("Time")
            table.add_column("Symbol")
            table.add_column("Side")
            table.add_column("Amount")
            table.add_column("Price")
            table.add_column("Reason")
            table.add_column("PnL")
            for t in trades:
                table.add_row(
                    t[1], t[2], t[3],
                    f"${t[4]:.2f}", f"${t[5]:.2f}",
                    t[6], f"${t[7]:.2f}" if t[7] else "-",
                )
            console.print(table)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


log = Logger()
