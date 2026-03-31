#!/usr/bin/env python3
"""Agent 2: Market Scanner — selects the day's equity universe (10-20 symbols).

Runs daily at 6:00 AM weekdays.
Reads optimizer assignments, applies screening criteria, writes .env and router config.
Must complete by 6:25 AM (bot starts at 6:30 AM).
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame

from config import Config

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
AGENT_STATE_FILE = os.path.join(DATA_DIR, "agent_state.json")
ASSIGNMENTS_FILE = os.path.join(DATA_DIR, "optimizer", "strategy_assignments.json")
FALLBACK_FILE = os.path.join(DATA_DIR, "fallback_config.json")
LEARNINGS_FILE = os.path.join(DATA_DIR, "learnings.json")
SELECTED_FILE = os.path.join(DATA_DIR, "scanner", "selected_symbols.json")
CANDIDATES_FILE = os.path.join(DATA_DIR, "scanner", "candidates.json")
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

# Screening parameters
MIN_SYMBOLS = 6
MAX_SYMBOLS = 20
TARGET_SYMBOLS = 15
MIN_AVG_VOLUME = 3_000_000      # 3M shares daily
MIN_PRICE = 10.0
MAX_PRICE = 500.0
MIN_ATR_PCT = 0.01              # 1% minimum volatility
MAX_ATR_PCT = 0.08              # 8% max volatility
MAX_PER_SECTOR = 4

# Well-known liquid stocks to scan (top ~80 by typical volume)
SCAN_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSM", "AVGO", "ORCL", "CRM",
    "AMD", "INTC", "QCOM", "AMAT", "MU", "LRCX", "KLAC", "MRVL", "SNPS", "CDNS",
    # Consumer/EV
    "TSLA", "NFLX", "DIS", "BABA", "NKE", "SBUX", "MCD", "COST", "WMT", "TGT",
    "AMGN", "GILD", "ISRG", "DXCM", "VRTX", "REGN", "MRNA", "BIIB",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "AXP", "V", "MA",
    # Energy
    "XOM", "CVX", "COP", "SLB", "OXY", "EOG", "MPC", "VLO", "PSX",
    # Industrial/Aerospace
    "BA", "CAT", "DE", "HON", "GE", "RTX", "LMT", "UNP",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "ARKK",
    # Misc high-volume
    "COIN", "SQ", "SHOP", "PLTR", "SOFI", "RIVN", "LCID", "UBER", "LYFT", "SNAP",
]

# Rough sector mapping for diversity
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech", "META": "Tech",
    "NVDA": "Tech", "TSM": "Tech", "AVGO": "Tech", "ORCL": "Tech", "CRM": "Tech",
    "AMD": "Tech", "INTC": "Tech", "QCOM": "Tech", "AMAT": "Tech", "MU": "Tech",
    "LRCX": "Tech", "KLAC": "Tech", "MRVL": "Tech", "SNPS": "Tech", "CDNS": "Tech",
    "TSLA": "Consumer", "NFLX": "Consumer", "DIS": "Consumer", "BABA": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "MCD": "Consumer", "COST": "Consumer",
    "WMT": "Consumer", "TGT": "Consumer",
    "AMGN": "Healthcare", "GILD": "Healthcare", "ISRG": "Healthcare", "DXCM": "Healthcare",
    "VRTX": "Healthcare", "REGN": "Healthcare", "MRNA": "Healthcare", "BIIB": "Healthcare",
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "MS": "Finance", "WFC": "Finance",
    "C": "Finance", "BLK": "Finance", "SCHW": "Finance", "AXP": "Finance", "V": "Finance", "MA": "Finance",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "OXY": "Energy",
    "EOG": "Energy", "MPC": "Energy", "VLO": "Energy", "PSX": "Energy",
    "BA": "Industrial", "CAT": "Industrial", "DE": "Industrial", "HON": "Industrial",
    "GE": "Industrial", "RTX": "Industrial", "LMT": "Industrial", "UNP": "Industrial",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF", "XLF": "ETF",
    "XLE": "ETF", "XLK": "ETF", "ARKK": "ETF",
    "COIN": "Fintech", "SQ": "Fintech", "SHOP": "Fintech", "PLTR": "Fintech",
    "SOFI": "Fintech", "RIVN": "Consumer", "LCID": "Consumer", "UBER": "Consumer",
    "LYFT": "Consumer", "SNAP": "Tech",
}

DEFAULT_STRATEGY = "mean_reversion_aggressive"


def load_optimizer_assignments() -> dict:
    """Load strategy assignments from the optimizer, or fallback."""
    # Check agent state to see if optimizer ran today
    if os.path.exists(AGENT_STATE_FILE):
        try:
            with open(AGENT_STATE_FILE) as f:
                state = json.load(f)
            opt = state.get("strategy_optimizer", {})
            today = datetime.now().strftime("%Y-%m-%d")
            if opt.get("status") == "success" and opt.get("last_run", "").startswith(today):
                print("  Optimizer ran successfully today — using its assignments")
            else:
                print(f"  Optimizer status: {opt.get('status', 'unknown')} — will use fallback for new symbols")
        except (json.JSONDecodeError, KeyError):
            pass

    # Load assignments
    if os.path.exists(ASSIGNMENTS_FILE):
        try:
            with open(ASSIGNMENTS_FILE) as f:
                data = json.load(f)
            return {sym: info["strategy"] if isinstance(info, dict) else info
                    for sym, info in data.get("assignments", {}).items()}
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback
    if os.path.exists(FALLBACK_FILE):
        try:
            with open(FALLBACK_FILE) as f:
                data = json.load(f)
            return data.get("strategy_map", {})
        except (json.JSONDecodeError, KeyError):
            pass

    return {}


def load_learnings() -> tuple[list[str], list[str]]:
    """Load symbols to favor and avoid from learnings."""
    favor, avoid = [], []
    if os.path.exists(LEARNINGS_FILE):
        try:
            with open(LEARNINGS_FILE) as f:
                data = json.load(f)
            # Use last 7 days of learnings
            for entry in data.get("entries", [])[-7:]:
                findings = entry.get("findings", {})
                favor.extend(findings.get("symbols_to_favor", []))
                avoid.extend(findings.get("symbols_to_avoid", []))
        except (json.JSONDecodeError, KeyError):
            pass
    return list(set(favor)), list(set(avoid))


def check_deadline():
    """Abort if past 6:25 AM to avoid interfering with bot startup at 6:30."""
    now = datetime.now()
    deadline = now.replace(hour=6, minute=25, second=0, microsecond=0)
    if now > deadline and now.hour < 12:  # Only enforce in the morning
        print("  ⚠️ Past 6:25 AM deadline — aborting to avoid interfering with bot startup")
        return True
    return False


def fetch_stock_data(stock_client: StockHistoricalDataClient, symbols: list[str]) -> dict:
    """Fetch 5-day daily bars for screening. Returns {symbol: DataFrame}."""
    results = {}
    # Batch in groups of 20 to avoid API limits
    for i in range(0, len(symbols), 20):
        batch = symbols[i:i + 20]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=10),
            )
            bars = stock_client.get_stock_bars(req)
            df = bars.df.reset_index()
            for sym in batch:
                sym_df = df[df["symbol"] == sym] if "symbol" in df.columns else df
                if not sym_df.empty:
                    results[sym] = sym_df
        except Exception as e:
            print(f"  Warning: Failed to fetch batch {batch[:3]}...: {e}")
            continue
    return results


def score_candidate(sym: str, df: pd.DataFrame, backtest_score: float,
                    favor: list, avoid: list) -> dict | None:
    """Score a single stock candidate. Returns scoring dict or None if filtered out."""
    if len(df) < 3:
        return None

    # Basic metrics
    latest = df.iloc[-1]
    price = float(latest["close"])
    avg_volume = float(df["volume"].mean())

    # Price filter
    if price < MIN_PRICE or price > MAX_PRICE:
        return None

    # Volume filter
    if avg_volume < MIN_AVG_VOLUME:
        return None

    # ATR as percentage of price (volatility)
    df = df.copy()
    df["tr"] = df.apply(
        lambda r: max(
            float(r["high"]) - float(r["low"]),
            abs(float(r["high"]) - float(r["close"])),
            abs(float(r["low"]) - float(r["close"]))
        ), axis=1
    )
    atr = float(df["tr"].mean())
    atr_pct = atr / price if price > 0 else 0

    if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
        return None

    # 5-day momentum (price change)
    if len(df) >= 5:
        momentum = (price - float(df.iloc[-5]["close"])) / float(df.iloc[-5]["close"])
    else:
        momentum = (price - float(df.iloc[0]["close"])) / float(df.iloc[0]["close"])

    # Composite score
    vol_score = min(avg_volume / 20_000_000, 1.0)  # Normalize to 20M
    volatility_score = min(atr_pct / 0.04, 1.0)     # Normalize to 4%
    momentum_score = max(min((momentum + 0.05) / 0.10, 1.0), 0)  # -5% to +5% → 0-1
    bt_score = max(min(backtest_score, 1.0), 0)

    composite = 0.3 * vol_score + 0.3 * volatility_score + 0.2 * momentum_score + 0.2 * bt_score

    # Learnings adjustments
    if sym in avoid:
        composite *= 0.5  # Heavy penalty
    if sym in favor:
        composite *= 1.3  # Boost

    sector = SECTOR_MAP.get(sym, "Other")

    return {
        "symbol": sym,
        "price": round(price, 2),
        "avg_volume": int(avg_volume),
        "atr_pct": round(atr_pct, 4),
        "momentum_5d": round(momentum, 4),
        "vol_score": round(vol_score, 3),
        "volatility_score": round(volatility_score, 3),
        "momentum_score": round(momentum_score, 3),
        "backtest_score": round(bt_score, 3),
        "composite_score": round(composite, 4),
        "sector": sector,
        "favored": sym in favor,
        "avoided": sym in avoid,
    }


def apply_sector_diversity(ranked: list[dict], max_per_sector: int = MAX_PER_SECTOR) -> list[dict]:
    """Enforce sector diversity — no more than max_per_sector from any one sector."""
    selected = []
    sector_counts = {}
    for candidate in ranked:
        sector = candidate["sector"]
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected.append(candidate)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    return selected


def update_env_file(symbols: list[str]):
    """Update EQUITY_SYMBOLS in .env file (atomic write)."""
    if not os.path.exists(ENV_FILE):
        print("  ERROR: .env file not found!")
        return

    with open(ENV_FILE) as f:
        content = f.read()

    # Replace EQUITY_SYMBOLS line
    new_value = ",".join(symbols)
    if "EQUITY_SYMBOLS=" in content:
        content = re.sub(r'EQUITY_SYMBOLS=.*', f'EQUITY_SYMBOLS={new_value}', content)
    else:
        content += f"\nEQUITY_SYMBOLS={new_value}\n"

    # Atomic write
    tmp = ENV_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.rename(tmp, ENV_FILE)
    print(f"  Updated .env: EQUITY_SYMBOLS={new_value}")


def update_agent_state(status: str, symbols_selected: int = 0, error: str = None, duration: float = 0):
    """Update agent_state.json."""
    state = {}
    if os.path.exists(AGENT_STATE_FILE):
        try:
            with open(AGENT_STATE_FILE) as f:
                state = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    state["market_scanner"] = {
        "last_run": datetime.now().isoformat(),
        "status": status,
        "duration_seconds": round(duration),
        "symbols_selected": symbols_selected,
        "error": error,
    }

    with open(AGENT_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def main():
    start_time = time.time()
    print(f"[Market Scanner] Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        Config.validate()

        # Load optimizer assignments and learnings
        assignments = load_optimizer_assignments()
        favor, avoid = load_learnings()

        print(f"  Optimizer assignments: {len(assignments)} symbols")
        print(f"  Learnings: {len(favor)} favored, {len(avoid)} avoided")

        # Fetch screening data
        stock_client = StockHistoricalDataClient(Config.ALPACA_API_KEY, Config.ALPACA_SECRET_KEY)
        print(f"  Fetching data for {len(SCAN_UNIVERSE)} candidates...")
        stock_data = fetch_stock_data(stock_client, SCAN_UNIVERSE)
        print(f"  Got data for {len(stock_data)} symbols")

        if check_deadline():
            update_agent_state("aborted", error="Past 6:25 AM deadline")
            return

        # Score all candidates
        candidates = []
        for sym in SCAN_UNIVERSE:
            if sym not in stock_data:
                continue
            # Get backtest score from optimizer (normalize return to 0-1)
            bt_info = assignments.get(sym, {})
            bt_score = 0.5  # Default for unknown
            if isinstance(bt_info, dict) and not bt_info.get("difficult", False):
                bt_score = 0.7
            elif isinstance(bt_info, str):
                bt_score = 0.6

            result = score_candidate(sym, stock_data[sym], bt_score, favor, avoid)
            if result:
                candidates.append(result)

        print(f"  {len(candidates)} candidates passed filters")

        if check_deadline():
            update_agent_state("aborted", error="Past 6:25 AM deadline")
            return

        # Rank and select
        candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        # Apply sector diversity
        selected = apply_sector_diversity(candidates)[:TARGET_SYMBOLS]

        # Enforce minimum
        if len(selected) < MIN_SYMBOLS:
            print(f"  WARNING: Only {len(selected)} candidates — keeping previous config")
            update_agent_state("success", symbols_selected=0,
                               error="Too few candidates, kept previous config")
            return

        selected_symbols = [c["symbol"] for c in selected]

        # Build strategy map for selected symbols
        strategy_map = {}
        for sym in selected_symbols:
            if sym in assignments:
                strat = assignments[sym]
                strategy_map[sym] = strat if isinstance(strat, str) else strat.get("strategy", DEFAULT_STRATEGY)
            else:
                strategy_map[sym] = DEFAULT_STRATEGY

        # Always include crypto assignments
        for sym in Config.CRYPTO_SYMBOLS:
            if sym in assignments:
                strat = assignments[sym]
                strategy_map[sym] = strat if isinstance(strat, str) else strat.get("strategy", DEFAULT_STRATEGY)
            else:
                strategy_map[sym] = DEFAULT_STRATEGY

        if check_deadline():
            update_agent_state("aborted", error="Past 6:25 AM deadline")
            return

        # Write .env
        update_env_file(selected_symbols)

        # Write strategy assignments for router to pick up
        assignments_output = {
            "run_date": datetime.now().strftime("%Y-%m-%d"),
            "run_timestamp": datetime.now().isoformat(),
            "assignments": {sym: {"strategy": strat} for sym, strat in strategy_map.items()},
        }
        tmp = os.path.join(DATA_DIR, "optimizer", "strategy_assignments.json.tmp")
        with open(tmp, "w") as f:
            json.dump(assignments_output, f, indent=2)
        os.rename(tmp, os.path.join(DATA_DIR, "optimizer", "strategy_assignments.json"))

        # Write scanner outputs
        today = datetime.now().strftime("%Y-%m-%d")

        # Selected symbols
        with open(SELECTED_FILE, "w") as f:
            json.dump({
                "date": today,
                "symbols": selected_symbols,
                "strategy_map": strategy_map,
                "selection": selected,
            }, f, indent=2)

        # Full candidates
        with open(CANDIDATES_FILE, "w") as f:
            json.dump({"date": today, "candidates": candidates}, f, indent=2)

        # Archive
        history_sym = os.path.join(DATA_DIR, "history", "symbols", f"{today}.json")
        with open(history_sym, "w") as f:
            json.dump({"date": today, "symbols": selected_symbols}, f, indent=2)

        history_assign = os.path.join(DATA_DIR, "history", "assignments", f"{today}.json")
        with open(history_assign, "w") as f:
            json.dump({"date": today, "strategy_map": strategy_map}, f, indent=2)

        # Validate
        print("\n  Validating...")
        try:
            # Re-import to verify router can load the new assignments
            import importlib
            import strategies.router as router_mod
            importlib.reload(router_mod)
            print(f"  ✓ Router loaded {len(router_mod.STRATEGY_MAP)} assignments")
        except Exception as e:
            print(f"  ⚠️ Router validation warning: {e}")

        duration = time.time() - start_time
        update_agent_state("success", symbols_selected=len(selected_symbols), duration=duration)

        print(f"\n[Market Scanner] Complete in {duration:.0f}s")
        print(f"  Selected {len(selected_symbols)} symbols:")
        for c in selected:
            strat = strategy_map.get(c["symbol"], DEFAULT_STRATEGY)
            print(f"    {c['symbol']:6s} | {c['sector']:10s} | score: {c['composite_score']:.3f} "
                  f"| vol: {c['avg_volume']/1e6:.1f}M | ATR: {c['atr_pct']:.1%} "
                  f"| mom: {c['momentum_5d']:+.1%} | strat: {strat}")

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n[Market Scanner] FAILED: {e}")
        update_agent_state("failed", error=str(e), duration=duration)
        raise


if __name__ == "__main__":
    main()
