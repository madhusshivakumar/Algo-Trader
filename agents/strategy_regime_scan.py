"""Strategy × regime sensitivity scan agent — Sprint 6F.

Runs `analytics.strategy_regime_matrix.run()` over the current universe
and persists:
    data/strategy_regime/matrix.csv       — full table (every strategy × regime)
    data/strategy_regime/hard_skip.json   — empirical blocklist used by the
                                             router's regime filter

Designed to run weekly (launchd / cron). Safe to re-run; each pass overwrites
both files atomically from the fresh measurements.

Usage:
    python -m agents.strategy_regime_scan                 # run + persist
    python -m agents.strategy_regime_scan --dry-run       # compute but don't write
    python -m agents.strategy_regime_scan --days 365      # shorter window
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import pandas as pd

from analytics.strategy_regime_matrix import (
    build_matrix,
    derive_hard_skip,
    write_hard_skip,
    write_matrix,
)
from core.transaction_costs import TransactionCostModel
from utils.logger import log


# Default universe — overlap with router defaults. Keeps the scan hermetic
# to what the bot actually trades.
_DEFAULT_UNIVERSE = (
    "BTC/USD", "ETH/USD", "TSLA", "NVDA", "AMD", "AAPL", "META", "SPY",
)

_DEFAULT_DAYS = 365 * 2  # 2 years — matches the Sprint 6A window.


def _load_universe_bars(symbols: Sequence[str], days: int, timeframe: str,
                        use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch bars for every symbol. Empty frames on failure."""
    # Lazy import so the tests can monkey-patch this function cleanly without
    # dragging in the scripts/ directory unless we actually execute the CLI.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    import full_oos_backtest  # type: ignore

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = full_oos_backtest.load_or_fetch_bars(
            symbol=sym, days=days, timeframe=timeframe, use_cache=use_cache,
        )
        if df is not None and not df.empty:
            out[sym] = df
        else:
            log.warning(f"[regime-scan] No bars for {sym}; skipping")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build strategy × regime sensitivity matrix.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute matrix + overrides but don't write files.")
    parser.add_argument("--days", type=int, default=_DEFAULT_DAYS,
                        help="Bar lookback window in days.")
    parser.add_argument("--timeframe", choices=("minute", "hour", "day"),
                        default="hour")
    parser.add_argument("--symbols", nargs="*", default=list(_DEFAULT_UNIVERSE),
                        help="Override the universe.")
    parser.add_argument("--min-trades", type=int, default=20,
                        help="Minimum observations per cell before we block.")
    parser.add_argument("--max-avg-pnl", type=float, default=0.0,
                        help="Threshold for 'bad' cells (avg_pnl ≤ this).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the parquet cache (force refetch).")
    args = parser.parse_args(argv)

    log.info(f"[regime-scan] Loading {len(args.symbols)} symbol(s), "
             f"{args.days}d, {args.timeframe} bars…")
    bars_by_symbol = _load_universe_bars(
        args.symbols, args.days, args.timeframe, use_cache=not args.no_cache,
    )
    if not bars_by_symbol:
        log.error("[regime-scan] No bars loaded — aborting.")
        return 1

    cost_model = TransactionCostModel()
    matrix = build_matrix(bars_by_symbol, cost_model=cost_model)
    pairs = derive_hard_skip(matrix, min_trades=args.min_trades,
                             max_avg_pnl=args.max_avg_pnl)

    log.info(f"[regime-scan] Matrix: {len(matrix)} rows, "
             f"{len(pairs)} hard-skip override(s).")

    if args.dry_run:
        log.info("[regime-scan] --dry-run: skipping writes.")
        _render(matrix, pairs)
        return 0

    write_matrix(matrix)
    write_hard_skip(pairs, metadata={
        "symbols": sorted(bars_by_symbol.keys()),
        "days": args.days, "timeframe": args.timeframe,
        "min_trades": args.min_trades, "max_avg_pnl": args.max_avg_pnl,
    })
    _render(matrix, pairs)
    return 0


def _render(matrix: pd.DataFrame, pairs: list[tuple[str, str]]) -> None:
    """Pretty-print summary to the log (the JSON + CSV carry the detail)."""
    if matrix.empty:
        log.info("[regime-scan] (empty matrix)")
        return
    # Keep the log compact — show only cells with trades
    filtered = matrix[matrix["n_trades"] > 0]
    if filtered.empty:
        log.info("[regime-scan] No cells had any trades; check warmup / data.")
        return
    log.info("[regime-scan] Strategy × regime sample:\n"
             + filtered.to_string(index=False))
    if pairs:
        log.warning(f"[regime-scan] Hard-skip pairs: {pairs}")


if __name__ == "__main__":
    sys.exit(main())
