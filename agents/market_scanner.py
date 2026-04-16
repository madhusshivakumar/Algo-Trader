#!/usr/bin/env python3
"""Agent 2: Market Scanner — selects the day's equity universe (10-20 symbols).

Runs daily at 6:00 AM weekdays.
Reads optimizer assignments, applies screening criteria, writes .env and router config.
Must complete by 6:45 AM (bot starts at 6:30 AM).

v2: Expanded to 1000+ stock universe via Alpaca Assets API + snapshot pre-filtering.
    Research picks from src/research/output/final_portfolio.json are priority-boosted.
"""

import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus, AssetClass
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
RESEARCH_PORTFOLIO_FILE = os.path.join(PROJECT_ROOT, "src", "research", "output", "final_portfolio.json")

# Screening parameters
MIN_SYMBOLS = 6
MAX_SYMBOLS = 20
TARGET_SYMBOLS = 20
MIN_AVG_VOLUME = 3_000_000       # 3M shares daily
MIN_PRICE = 10.0
MAX_PRICE = 1_000.0              # Raised to capture AVGO, COST, etc.
MIN_ATR_PCT = 0.01               # 1% minimum volatility
MAX_ATR_PCT = 0.08               # 8% max volatility
MAX_PER_SECTOR = 5               # Max per known sector
SNAPSHOT_BATCH_SIZE = 1_000      # Alpaca snapshot API limit per request
BARS_BATCH_SIZE = 50             # Alpaca bars API batch size
SNAPSHOT_WORKERS = 4             # Parallel snapshot fetch workers
RESEARCH_BUY_BOOST = 1.4        # Score multiplier for research BUY picks
RESEARCH_SELL_PENALTY = 0.3     # Score multiplier for research SELL picks

# Exchanges to include (excludes OTC/pink sheets)
MAJOR_EXCHANGES = {"NYSE", "NASDAQ", "ARCA", "BATS", "NYSEARCA", "AMEX"}

# Always-include ETFs (appended after equity selection)
ALWAYS_INCLUDE_ETFS = ["SPY", "QQQ", "IWM"]

# Sector map for known symbols — extended to cover common large-caps.
# Unknown symbols fall back to "Other" (uncapped).
SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "GOOG": "Tech",
    "AMZN": "Tech", "META": "Tech", "NVDA": "Tech", "TSM": "Tech",
    "AVGO": "Tech", "ORCL": "Tech", "CRM": "Tech", "AMD": "Tech",
    "INTC": "Tech", "QCOM": "Tech", "AMAT": "Tech", "MU": "Tech",
    "LRCX": "Tech", "KLAC": "Tech", "MRVL": "Tech", "SNPS": "Tech",
    "CDNS": "Tech", "ADBE": "Tech", "NOW": "Tech", "INTU": "Tech",
    "PANW": "Tech", "CRWD": "Tech", "SNOW": "Tech", "DDOG": "Tech",
    "ZS": "Tech", "NET": "Tech", "MDB": "Tech", "TEAM": "Tech",
    "WDAY": "Tech", "VEEV": "Tech", "TTD": "Tech", "OKTA": "Tech",
    "FTNT": "Tech", "ANSS": "Tech", "EPAM": "Tech", "CTSH": "Tech",
    "ACN": "Tech", "IBM": "Tech", "HPQ": "Tech", "DELL": "Tech",
    "STX": "Tech", "WDC": "Tech", "TXN": "Tech", "ADI": "Tech",
    "MCHP": "Tech", "NXPI": "Tech", "ON": "Tech", "SWKS": "Tech",
    "SNAP": "Tech", "PINS": "Tech", "TWTR": "Tech",
    # Consumer Discretionary
    "TSLA": "Consumer", "NFLX": "Consumer", "DIS": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "MCD": "Consumer",
    "COST": "Consumer", "WMT": "Consumer", "TGT": "Consumer",
    "BABA": "Consumer", "AMGN": "Consumer", "RIVN": "Consumer",
    "LCID": "Consumer", "UBER": "Consumer", "LYFT": "Consumer",
    "ABNB": "Consumer", "BKNG": "Consumer", "EXPE": "Consumer",
    "MAR": "Consumer", "HLT": "Consumer", "RCL": "Consumer",
    "CCL": "Consumer", "F": "Consumer", "GM": "Consumer",
    "FORD": "Consumer", "HD": "Consumer", "LOW": "Consumer",
    "ETSY": "Consumer", "EBAY": "Consumer", "ROST": "Consumer",
    "TJX": "Consumer", "LULU": "Consumer", "VFC": "Consumer",
    "PVH": "Consumer", "RL": "Consumer", "TPR": "Consumer",
    # Healthcare
    "GILD": "Healthcare", "ISRG": "Healthcare", "DXCM": "Healthcare",
    "VRTX": "Healthcare", "REGN": "Healthcare", "MRNA": "Healthcare",
    "BIIB": "Healthcare", "BMY": "Healthcare", "PFE": "Healthcare",
    "JNJ": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare",
    "MRK": "Healthcare", "TMO": "Healthcare", "DHR": "Healthcare",
    "ABT": "Healthcare", "MDT": "Healthcare", "SYK": "Healthcare",
    "ZBH": "Healthcare", "EW": "Healthcare", "IDXX": "Healthcare",
    "MTD": "Healthcare", "WAT": "Healthcare", "A": "Healthcare",
    "HOLX": "Healthcare", "PODD": "Healthcare", "ALGN": "Healthcare",
    "INCY": "Healthcare", "EXEL": "Healthcare", "ALNY": "Healthcare",
    "SRPT": "Healthcare", "IONS": "Healthcare", "BMRN": "Healthcare",
    "RARE": "Healthcare", "HALO": "Healthcare", "RCKT": "Healthcare",
    "ACAD": "Healthcare", "FOLD": "Healthcare",
    # Finance
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "MS": "Finance",
    "WFC": "Finance", "C": "Finance", "BLK": "Finance", "SCHW": "Finance",
    "AXP": "Finance", "V": "Finance", "MA": "Finance", "PYPL": "Finance",
    "COF": "Finance", "USB": "Finance", "PNC": "Finance", "TFC": "Finance",
    "FITB": "Finance", "RF": "Finance", "KEY": "Finance", "CFG": "Finance",
    "HBAN": "Finance", "MTB": "Finance", "NTRS": "Finance", "STT": "Finance",
    "BK": "Finance", "ICE": "Finance", "CME": "Finance", "CBOE": "Finance",
    "MKTX": "Finance", "NDAQ": "Finance", "MCO": "Finance", "SPGI": "Finance",
    "FDS": "Finance", "MSCI": "Finance", "VRSK": "Finance",
    # Fintech
    "COIN": "Fintech", "SQ": "Fintech", "SHOP": "Fintech", "PLTR": "Fintech",
    "SOFI": "Fintech", "HOOD": "Fintech", "AFRM": "Fintech", "UPST": "Fintech",
    "LC": "Fintech", "NU": "Fintech", "PAGS": "Fintech", "STNE": "Fintech",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "OXY": "Energy", "EOG": "Energy", "MPC": "Energy", "VLO": "Energy",
    "PSX": "Energy", "PXD": "Energy", "DVN": "Energy", "HAL": "Energy",
    "BKR": "Energy", "FANG": "Energy", "MRO": "Energy", "APA": "Energy",
    "HES": "Energy", "NOV": "Energy", "WHR": "Energy",
    # Industrials
    "BA": "Industrial", "CAT": "Industrial", "DE": "Industrial",
    "HON": "Industrial", "GE": "Industrial", "RTX": "Industrial",
    "LMT": "Industrial", "UNP": "Industrial", "UPS": "Industrial",
    "FDX": "Industrial", "CSX": "Industrial", "NSC": "Industrial",
    "WM": "Industrial", "RSG": "Industrial", "FAST": "Industrial",
    "GWW": "Industrial", "ETN": "Industrial", "EMR": "Industrial",
    "ROP": "Industrial", "IR": "Industrial", "AME": "Industrial",
    "PH": "Industrial", "DOV": "Industrial", "XYL": "Industrial",
    "ITW": "Industrial", "SWK": "Industrial", "BALL": "Industrial",
    # Real Estate
    "AMT": "RealEstate", "PLD": "RealEstate", "EQIX": "RealEstate",
    "SPG": "RealEstate", "O": "RealEstate", "WELL": "RealEstate",
    "DLR": "RealEstate", "PSA": "RealEstate", "EXR": "RealEstate",
    "AVB": "RealEstate", "EQR": "RealEstate", "VTR": "RealEstate",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "EXC": "Utilities", "AEP": "Utilities",
    "XEL": "Utilities", "PPL": "Utilities", "ED": "Utilities",
    # Materials
    "LIN": "Materials", "APD": "Materials", "ECL": "Materials",
    "NEM": "Materials", "FCX": "Materials", "NUE": "Materials",
    "VMC": "Materials", "MLM": "Materials", "CF": "Materials",
    "MOS": "Materials", "ALB": "Materials", "CTVA": "Materials",
    # Communication Services
    "GOOGL": "Comms", "META": "Comms", "NFLX": "Comms", "DIS": "Comms",
    "CMCSA": "Comms", "T": "Comms", "VZ": "Comms", "TMUS": "Comms",
    "CHTR": "Comms", "WBD": "Comms", "PARA": "Comms", "FOX": "Comms",
    # ETFs (uncapped)
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "XLF": "ETF", "XLE": "ETF", "XLK": "ETF", "ARKK": "ETF",
    "XLV": "ETF", "XLI": "ETF", "XLB": "ETF", "XLRE": "ETF",
    "XLU": "ETF", "XLY": "ETF", "XLP": "ETF", "GLD": "ETF",
    "SLV": "ETF", "TLT": "ETF", "HYG": "ETF", "LQD": "ETF",
}

DEFAULT_STRATEGY = "mean_reversion_aggressive"


# ── Universe Building ──────────────────────────────────────────────────────────

def build_dynamic_universe(trading_client: TradingClient) -> list[str]:
    """Fetch all active, tradeable US equities from Alpaca Assets API.

    Returns a deduplicated list of ticker symbols on major exchanges.
    Typically yields 4,000-8,000 symbols before screening.
    """
    print("  Fetching full US equity universe from Alpaca...")
    try:
        req = GetAssetsRequest(
            status=AssetStatus.ACTIVE,
            asset_class=AssetClass.US_EQUITY,
        )
        assets = trading_client.get_all_assets(req)
    except Exception as e:
        print(f"  WARNING: Failed to fetch dynamic universe: {e}")
        print("  Falling back to built-in sector map symbols")
        return list(SECTOR_MAP.keys())

    symbols = []
    for asset in assets:
        sym = asset.symbol
        # Skip symbols with special characters (warrants, preferred, etc.)
        if any(c in sym for c in [".", "/", "+", "-", " "]):
            continue
        # Skip overly long tickers (typically SPACs or unit trusts)
        if len(sym) > 5:
            continue
        if not asset.tradable:
            continue
        if str(asset.exchange) not in MAJOR_EXCHANGES:
            continue
        symbols.append(sym)

    symbols = list(dict.fromkeys(symbols))  # deduplicate, preserve order
    print(f"  Found {len(symbols):,} tradeable symbols on major exchanges")
    return symbols


def prefilter_with_snapshots(
    stock_client: StockHistoricalDataClient,
    symbols: list[str],
) -> list[str]:
    """Use Alpaca snapshot API for fast price/volume pre-filtering.

    Snapshots are a single API call per batch — much faster than fetching bars
    for thousands of symbols. Filters down to viable candidates before bar fetch.
    """
    passed: list[str] = []

    def fetch_snapshot_batch(batch: list[str]) -> list[str]:
        try:
            req = StockSnapshotRequest(symbol_or_symbols=batch)
            snapshots = stock_client.get_stock_snapshot(req)
            result = []
            for sym, snap in snapshots.items():
                if snap.latest_trade is None or snap.daily_bar is None:
                    continue
                price = float(snap.latest_trade.price)
                volume = float(snap.daily_bar.volume)
                if MIN_PRICE <= price <= MAX_PRICE and volume >= MIN_AVG_VOLUME:
                    result.append(sym)
            return result
        except Exception as e:
            print(f"  Warning: snapshot batch failed: {e}")
            return []

    batches = [
        symbols[i: i + SNAPSHOT_BATCH_SIZE]
        for i in range(0, len(symbols), SNAPSHOT_BATCH_SIZE)
    ]
    print(f"  Pre-filtering {len(symbols):,} symbols via snapshots ({len(batches)} batches, "
          f"{SNAPSHOT_WORKERS} workers)...")

    with ThreadPoolExecutor(max_workers=SNAPSHOT_WORKERS) as executor:
        futures = {executor.submit(fetch_snapshot_batch, b): b for b in batches}
        for future in as_completed(futures):
            passed.extend(future.result())

    print(f"  {len(passed):,} symbols passed price/volume pre-filter")
    return passed


# ── Research Integration ───────────────────────────────────────────────────────

def load_research_picks() -> tuple[set[str], set[str]]:
    """Load BUY and SELL recommendations from the research swarm portfolio.

    Returns (buy_picks, sell_picks) sets of ticker symbols.
    Gracefully returns empty sets if the file is missing or stale.
    """
    buy_picks: set[str] = set()
    sell_picks: set[str] = set()

    if not os.path.exists(RESEARCH_PORTFOLIO_FILE):
        return buy_picks, sell_picks

    try:
        with open(RESEARCH_PORTFOLIO_FILE) as f:
            portfolio = json.load(f)

        # Top picks (strong buys / buys)
        for pick in portfolio.get("top_picks", []):
            rating = pick.get("rating", "").upper()
            ticker = pick.get("ticker", "")
            if not ticker:
                continue
            if rating in ("STRONG_BUY", "BUY"):
                buy_picks.add(ticker)
            elif rating in ("SELL", "STRONG_SELL", "UNDERPERFORM"):
                sell_picks.add(ticker)

        # Sector-level picks
        for sector_data in portfolio.get("sectors", {}).values():
            for pick in sector_data.get("picks", []):
                ticker = pick.get("ticker", "")
                rating = pick.get("rating", "").upper()
                if not ticker:
                    continue
                if rating in ("STRONG_BUY", "BUY"):
                    buy_picks.add(ticker)
                elif rating in ("SELL", "STRONG_SELL"):
                    sell_picks.add(ticker)

        print(f"  Research picks: {len(buy_picks)} buys, {len(sell_picks)} sells")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  Warning: Could not load research picks: {e}")

    return buy_picks, sell_picks


# ── Existing Helpers (unchanged) ───────────────────────────────────────────────

def load_optimizer_assignments() -> dict:
    """Load strategy assignments from the optimizer, or fallback."""
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

    if os.path.exists(ASSIGNMENTS_FILE):
        try:
            with open(ASSIGNMENTS_FILE) as f:
                data = json.load(f)
            return {sym: info["strategy"] if isinstance(info, dict) else info
                    for sym, info in data.get("assignments", {}).items()}
        except (json.JSONDecodeError, KeyError):
            pass

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
            for entry in data.get("entries", [])[-7:]:
                findings = entry.get("findings", {})
                favor.extend(findings.get("symbols_to_favor", []))
                avoid.extend(findings.get("symbols_to_avoid", []))
        except (json.JSONDecodeError, KeyError):
            pass
    return list(set(favor)), list(set(avoid))


def check_deadline() -> bool:
    """Abort if past 6:45 AM to avoid interfering with bot startup."""
    now = datetime.now()
    deadline = now.replace(hour=6, minute=45, second=0, microsecond=0)
    if now > deadline and now.hour < 12:
        print("  ⚠️ Past 6:45 AM deadline — aborting to avoid interfering with bot startup")
        return True
    return False


def fetch_stock_data(stock_client: StockHistoricalDataClient, symbols: list[str]) -> dict:
    """Fetch 5-day daily bars for scoring. Returns {symbol: DataFrame}."""
    results = {}
    for i in range(0, len(symbols), BARS_BATCH_SIZE):
        batch = symbols[i: i + BARS_BATCH_SIZE]
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
            print(f"  Warning: Failed to fetch bars batch starting {batch[0]}: {e}")
            continue
    return results


def score_candidate(
    sym: str,
    df: pd.DataFrame,
    backtest_score: float,
    favor: list[str],
    avoid: list[str],
    research_buys: set[str],
    research_sells: set[str],
) -> dict | None:
    """Score a single stock candidate. Returns scoring dict or None if filtered out."""
    if len(df) < 3:
        return None

    latest = df.iloc[-1]
    price = float(latest["close"])
    avg_volume = float(df["volume"].mean())

    if price < MIN_PRICE or price > MAX_PRICE:
        return None
    if avg_volume < MIN_AVG_VOLUME:
        return None

    df = df.copy()
    df["tr"] = df.apply(
        lambda r: max(
            float(r["high"]) - float(r["low"]),
            abs(float(r["high"]) - float(r["close"])),
            abs(float(r["low"]) - float(r["close"])),
        ),
        axis=1,
    )
    atr = float(df["tr"].mean())
    atr_pct = atr / price if price > 0 else 0

    if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
        return None

    momentum = (
        (price - float(df.iloc[-5]["close"])) / float(df.iloc[-5]["close"])
        if len(df) >= 5
        else (price - float(df.iloc[0]["close"])) / float(df.iloc[0]["close"])
    )

    vol_score = min(avg_volume / 20_000_000, 1.0)
    volatility_score = min(atr_pct / 0.04, 1.0)
    momentum_score = max(min((momentum + 0.05) / 0.10, 1.0), 0)
    bt_score = max(min(backtest_score, 1.0), 0)

    composite = (
        0.3 * vol_score
        + 0.3 * volatility_score
        + 0.2 * momentum_score
        + 0.2 * bt_score
    )

    # Learnings adjustments
    if sym in avoid:
        composite *= 0.5
    if sym in favor:
        composite *= 1.3

    # Research swarm adjustments
    if sym in research_buys:
        composite *= RESEARCH_BUY_BOOST
    if sym in research_sells:
        composite *= RESEARCH_SELL_PENALTY

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
        "research_buy": sym in research_buys,
        "research_sell": sym in research_sells,
    }


def apply_sector_diversity(ranked: list[dict], max_per_sector: int = MAX_PER_SECTOR) -> list[dict]:
    """Enforce sector diversity — no more than max_per_sector from any one known sector.

    Symbols with sector "Other" (dynamic universe without known sector) share a
    higher cap so the expanded universe has room to contribute.
    """
    selected = []
    sector_counts: dict[str, int] = {}
    other_cap = max_per_sector * 3  # Generous cap for "Other" bucket

    for candidate in ranked:
        sector = candidate["sector"]
        cap = other_cap if sector == "Other" else max_per_sector
        if sector_counts.get(sector, 0) >= cap:
            continue
        selected.append(candidate)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    return selected


def update_env_file(symbols: list[str]) -> None:
    """Update EQUITY_SYMBOLS in .env file (atomic write)."""
    if not os.path.exists(ENV_FILE):
        print("  ERROR: .env file not found!")
        return

    with open(ENV_FILE) as f:
        content = f.read()

    new_value = ",".join(symbols)
    if "EQUITY_SYMBOLS=" in content:
        content = re.sub(r"EQUITY_SYMBOLS=.*", f"EQUITY_SYMBOLS={new_value}", content)
    else:
        content += f"\nEQUITY_SYMBOLS={new_value}\n"

    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(ENV_FILE), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, ENV_FILE)
    except Exception:
        os.unlink(tmp_path)
        raise
    print(f"  Updated .env: EQUITY_SYMBOLS={new_value}")


def update_agent_state(
    status: str,
    symbols_selected: int = 0,
    universe_size: int = 0,
    error: str | None = None,
    duration: float = 0,
) -> None:
    """Update agent_state.json."""
    state: dict = {}
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
        "universe_size": universe_size,
        "error": error,
    }

    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(AGENT_STATE_FILE), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, AGENT_STATE_FILE)
    except Exception:
        os.unlink(tmp_path)
        raise


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    start_time = time.time()
    print(f"[Market Scanner v2] Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        Config.validate()

        assignments = load_optimizer_assignments()
        favor, avoid = load_learnings()
        research_buys, research_sells = load_research_picks()

        print(f"  Optimizer assignments: {len(assignments)} symbols")
        print(f"  Learnings: {len(favor)} favored, {len(avoid)} avoided")

        trading_client = TradingClient(Config.ALPACA_API_KEY, Config.ALPACA_SECRET_KEY, paper=Config.is_paper())
        stock_client = StockHistoricalDataClient(Config.ALPACA_API_KEY, Config.ALPACA_SECRET_KEY)

        # ── Stage 1: Build 1000+ universe dynamically ──────────────────────────
        raw_universe = build_dynamic_universe(trading_client)

        if check_deadline():
            update_agent_state("aborted", error="Past deadline after universe build")
            return

        # ── Stage 2: Snapshot pre-filter (price + volume) ─────────────────────
        # Reduces 4000-8000 → ~500-1500 candidates before expensive bar fetches
        prefiltered = prefilter_with_snapshots(stock_client, raw_universe)

        if check_deadline():
            update_agent_state("aborted", error="Past deadline after snapshot filter",
                               universe_size=len(prefiltered))
            return

        # ── Stage 3: Fetch 5-day bars for detailed scoring ────────────────────
        print(f"  Fetching 5-day bars for {len(prefiltered):,} candidates "
              f"({len(prefiltered) // BARS_BATCH_SIZE + 1} batches)...")
        stock_data = fetch_stock_data(stock_client, prefiltered)
        print(f"  Got bar data for {len(stock_data):,} symbols")

        if check_deadline():
            update_agent_state("aborted", error="Past deadline after bar fetch",
                               universe_size=len(prefiltered))
            return

        # ── Stage 4: Score all candidates ─────────────────────────────────────
        candidates = []
        for sym in prefiltered:
            if sym not in stock_data:
                continue
            bt_info = assignments.get(sym, {})
            if isinstance(bt_info, dict):
                bt_score = 0.5 if bt_info.get("difficult") else 0.7
            elif isinstance(bt_info, str):
                bt_score = 0.6
            else:
                bt_score = 0.5

            result = score_candidate(
                sym, stock_data[sym], bt_score,
                favor, avoid, research_buys, research_sells,
            )
            if result:
                candidates.append(result)

        print(f"  {len(candidates):,} candidates passed ATR/momentum filters")

        if check_deadline():
            update_agent_state("aborted", error="Past deadline after scoring",
                               universe_size=len(candidates))
            return

        # ── Stage 5: Rank, diversify, select ──────────────────────────────────
        candidates.sort(key=lambda x: x["composite_score"], reverse=True)
        selected = apply_sector_diversity(candidates)[:TARGET_SYMBOLS]

        if len(selected) < MIN_SYMBOLS:
            print(f"  WARNING: Only {len(selected)} candidates — keeping previous config")
            update_agent_state("success", symbols_selected=0,
                               universe_size=len(candidates),
                               error="Too few candidates, kept previous config")
            return

        selected_symbols = [c["symbol"] for c in selected]

        # ── Stage 6: Build strategy map ────────────────────────────────────────
        strategy_map: dict[str, str] = {}
        for sym in selected_symbols:
            if sym in assignments:
                strat = assignments[sym]
                strategy_map[sym] = strat if isinstance(strat, str) else strat.get("strategy", DEFAULT_STRATEGY)
            else:
                strategy_map[sym] = DEFAULT_STRATEGY

        for sym in Config.CRYPTO_SYMBOLS:
            if sym in assignments:
                strat = assignments[sym]
                strategy_map[sym] = strat if isinstance(strat, str) else strat.get("strategy", DEFAULT_STRATEGY)
            else:
                strategy_map[sym] = DEFAULT_STRATEGY

        # ── Stage 7: Write outputs ─────────────────────────────────────────────
        update_env_file(selected_symbols)

        today = datetime.now().strftime("%Y-%m-%d")
        assignments_output = {
            "run_date": today,
            "run_timestamp": datetime.now().isoformat(),
            "universe_size": len(candidates),
            "assignments": {sym: {"strategy": strat} for sym, strat in strategy_map.items()},
        }

        for path, data in [
            (os.path.join(DATA_DIR, "optimizer", "strategy_assignments.json"), assignments_output),
            (SELECTED_FILE, {"date": today, "symbols": selected_symbols,
                             "universe_size": len(candidates),
                             "strategy_map": strategy_map, "selection": selected}),
            (CANDIDATES_FILE, {"date": today, "universe_size": len(prefiltered),
                               "candidates": candidates}),
            (os.path.join(DATA_DIR, "history", "symbols", f"{today}.json"),
             {"date": today, "symbols": selected_symbols}),
            (os.path.join(DATA_DIR, "history", "assignments", f"{today}.json"),
             {"date": today, "strategy_map": strategy_map}),
        ]:
            tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, path)
            except Exception:
                os.unlink(tmp_path)
                raise

        # ── Stage 8: Validate router ───────────────────────────────────────────
        try:
            import importlib
            import strategies.router as router_mod
            importlib.reload(router_mod)
            print(f"  ✓ Router loaded {len(router_mod.STRATEGY_MAP)} assignments")
        except Exception as e:
            print(f"  ⚠️ Router validation warning: {e}")

        duration = time.time() - start_time
        update_agent_state("success", symbols_selected=len(selected_symbols),
                           universe_size=len(candidates), duration=duration)

        print(f"\n[Market Scanner v2] Complete in {duration:.0f}s")
        print(f"  Universe scanned: {len(raw_universe):,} → {len(prefiltered):,} "
              f"→ {len(candidates):,} → {len(selected)} selected")
        for c in selected:
            strat = strategy_map.get(c["symbol"], DEFAULT_STRATEGY)
            flags = []
            if c.get("research_buy"):
                flags.append("📊BUY")
            if c.get("favored"):
                flags.append("⭐")
            flag_str = " ".join(flags)
            print(f"    {c['symbol']:6s} | {c['sector']:12s} | score: {c['composite_score']:.3f} "
                  f"| vol: {c['avg_volume']/1e6:.1f}M | ATR: {c['atr_pct']:.1%} "
                  f"| mom: {c['momentum_5d']:+.1%} | strat: {strat} {flag_str}")

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n[Market Scanner v2] FAILED: {e}")
        update_agent_state("failed", error=str(e), duration=duration)
        raise


if __name__ == "__main__":
    main()
