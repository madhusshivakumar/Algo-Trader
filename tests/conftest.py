"""Shared fixtures for all tests."""

import os
import sys
import sqlite3
import json
import tempfile

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ── Market Data Fixtures ──────────────────────────────────────────────


def _make_ohlcv(n: int, base_price: float = 100.0, seed: int = 42,
                trend: str = "flat") -> pd.DataFrame:
    """Generate synthetic OHLCV data.

    trend: "flat", "up", "down", "volatile"
    """
    rng = np.random.RandomState(seed)

    if trend == "up":
        drift = np.linspace(0, 15, n)
    elif trend == "down":
        drift = np.linspace(0, -15, n)
    elif trend == "volatile":
        drift = np.cumsum(rng.randn(n) * 2)
    else:
        drift = np.zeros(n)

    noise = rng.randn(n) * 0.5
    close = base_price + drift + noise
    close = np.maximum(close, 1.0)  # No negative prices

    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    low = np.maximum(low, 0.5)
    opn = close + rng.uniform(-0.5, 0.5, n)

    volume = rng.uniform(1_000_000, 10_000_000, n)

    return pd.DataFrame({
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def flat_market():
    """200 bars of sideways market data."""
    return _make_ohlcv(200, trend="flat")


@pytest.fixture
def uptrend_market():
    """200 bars of uptrending market data."""
    return _make_ohlcv(200, trend="up")


@pytest.fixture
def downtrend_market():
    """200 bars of downtrending market data."""
    return _make_ohlcv(200, trend="down")


@pytest.fixture
def volatile_market():
    """200 bars of volatile market data."""
    return _make_ohlcv(200, trend="volatile")


@pytest.fixture
def small_df():
    """Only 10 bars — too few for any strategy."""
    return _make_ohlcv(10)


@pytest.fixture
def overbought_df():
    """Strongly uptrending to trigger overbought signals."""
    return _make_ohlcv(200, base_price=50, trend="up", seed=99)


@pytest.fixture
def oversold_df():
    """Strongly downtrending to trigger oversold signals."""
    return _make_ohlcv(200, base_price=150, trend="down", seed=99)


# ── Database Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary trades database with sample data."""
    db_path = str(tmp_path / "test_trades.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE trades (
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
        CREATE TABLE equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            equity REAL,
            cash REAL
        )
    """)

    # Insert sample trades for today
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    trades = [
        (f"{today} 09:35:00", "AAPL", "buy", 5000, 248.50, "Mean rev buy", 0, "mean_reversion_aggressive"),
        (f"{today} 10:15:00", "AAPL", "sell", 5000, 250.10, "Mean rev sell", 32.0, "mean_reversion_aggressive"),
        (f"{today} 09:40:00", "NVDA", "buy", 8000, 880.00, "Extreme oversold", 0, "mean_reversion"),
        (f"{today} 11:30:00", "NVDA", "sell", 8000, 875.00, "trailing stop", -45.50, "mean_reversion"),
        (f"{today} 12:00:00", "BTC/USD", "buy", 25000, 66000, "Mean rev buy", 0, "mean_reversion_aggressive"),
        (f"{today} 13:00:00", "BTC/USD", "sell", 25000, 66200, "Mean rev sell", 75.75, "mean_reversion_aggressive"),
        (f"{today} 14:00:00", "AMD", "buy", 6000, 160, "Volume spike", 0, "volume_profile"),
        (f"{today} 15:30:00", "AMD", "sell", 6000, 157, "OBV breakdown", -112.50, "volume_profile"),
    ]
    conn.executemany(
        "INSERT INTO trades (timestamp, symbol, side, amount, price, reason, pnl, strategy) "
        "VALUES (?,?,?,?,?,?,?,?)", trades
    )
    conn.execute(
        "INSERT INTO equity_snapshots (timestamp, equity, cash) VALUES (?,?,?)",
        (f"{today} 09:30:00", 100000, 50000),
    )
    conn.commit()
    conn.close()
    return db_path


# ── Config/Data Fixtures ──────────────────────────────────────────────


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    dirs = ["scanner", "optimizer", "analyzer/reports", "history/symbols", "history/assignments"]
    for d in dirs:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def sample_agent_state(tmp_data_dir):
    """Write sample agent_state.json."""
    state = {
        "strategy_optimizer": {"last_run": "2026-03-28T05:30:00", "status": "success",
                               "duration_seconds": 300, "symbols_tested": 8, "error": None},
        "market_scanner": {"last_run": "2026-03-28T06:00:00", "status": "success",
                           "duration_seconds": 5, "symbols_selected": 15, "error": None},
        "trade_analyzer": {"last_run": "2026-03-27T17:00:00", "status": "success",
                           "duration_seconds": 2, "trades_analyzed": 20, "error": None},
    }
    path = tmp_data_dir / "agent_state.json"
    path.write_text(json.dumps(state, indent=2))
    return path


@pytest.fixture
def sample_learnings(tmp_data_dir):
    """Write sample learnings.json."""
    learnings = {
        "version": 1,
        "entries": [{
            "date": "2026-03-27",
            "source": "trade_analyzer",
            "findings": {
                "best_performing_symbols": ["NVDA", "AMD"],
                "worst_performing_symbols": ["META"],
                "strategy_notes": {"mean_reversion_aggressive": "Strong on high-vol tech"},
                "risk_observations": {"overall_win_rate": 55.0, "total_pnl": 42.30},
                "symbols_to_avoid": ["META"],
                "symbols_to_favor": ["NVDA", "AMD"],
            }
        }]
    }
    path = tmp_data_dir / "learnings.json"
    path.write_text(json.dumps(learnings, indent=2))
    return path


@pytest.fixture
def sample_assignments(tmp_data_dir):
    """Write sample strategy_assignments.json."""
    assignments = {
        "run_date": "2026-03-28",
        "assignments": {
            "BTC/USD": {"strategy": "mean_reversion_aggressive", "reason": "Best Sharpe"},
            "ETH/USD": {"strategy": "volume_profile", "reason": "Best return"},
            "TSLA": {"strategy": "mean_reversion", "reason": "Highest composite"},
        }
    }
    path = tmp_data_dir / "optimizer" / "strategy_assignments.json"
    path.write_text(json.dumps(assignments, indent=2))
    return path
