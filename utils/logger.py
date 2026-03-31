"""Simple logging with Rich for nice terminal output."""

import sqlite3
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table

console = Console()

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trades.db")


class Logger:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
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
            + (f" | PnL: ${pnl:.2f}" if pnl else "")
        )
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO trades (timestamp, symbol, side, amount, price, reason, pnl, "
            "strategy, sentiment_score, llm_conviction, rl_selected) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (_ts(), symbol, side, amount, price, reason, pnl, strategy,
             sentiment_score, llm_conviction, rl_selected),
        )
        conn.commit()
        conn.close()

    def snapshot(self, equity: float, cash: float):
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO equity_snapshots (timestamp, equity, cash) VALUES (?,?,?)",
            (_ts(), equity, cash),
        )
        conn.commit()
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
