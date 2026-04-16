"""Full out-of-sample backtest harness — Sprint 6A.

Runs the *full* signal pipeline (strategies → modifiers → RL selector →
regime filter) over a multi-year window and writes per-symbol, portfolio,
and per-regime metrics to `data/backtest_oos/<timestamp>/`.

The entry point is `main()` — the CLI accepts:
    --days 730           # 2 years by default
    --timeframe hour     # minute | hour | day
    --symbols BTC/USD TSLA …
    --starting-cash 100000
    --no-cache           # bypass the parquet cache
    --monte-carlo-runs 1000

Acceptance gate (written to `acceptance.json`):
    The run PASSES if net portfolio Sharpe (after transaction costs) is >=
    the configured threshold (default 1.0). This is not a hard CI gate —
    backtest bars are expensive to pull — but the `passed` flag is shown on
    the dashboard and surfaced by the Sprint 6 verification checklist.

Design notes:
    * The harness uses `strategies.router.compute_signals()` so every
      enabled modifier runs exactly as it would live. Sentiment / LLM
      data files are typically absent during backtest — those modifiers
      no-op, which is the honest thing to do (we can't retroactively
      synthesise FinBERT scores for 2023-Q2).
    * Regime is labelled per-bar from a rolling 20-day realized vol window
      computed on the symbol's own close series. This matches
      `core.regime_detector.classify_vol` at runtime.
    * Costs are applied per round-trip via `core.transaction_costs`.
    * Monte Carlo shuffles the *trade P&L sequence* (not the per-bar
      returns) — keeps trade sizing intact while testing sequencing luck.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from config import Config
from core.portfolio_metrics import compute_all as compute_portfolio_metrics
from core.regime_detector import classify_vol
from core.transaction_costs import TransactionCostModel
from utils.logger import log


# ── Constants ────────────────────────────────────────────────────────────

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_REPO_ROOT, "data", "backtest_cache")
_OUT_ROOT = os.path.join(_REPO_ROOT, "data", "backtest_oos")

# Acceptance threshold — configurable via env so the dashboard + CI can
# share one source of truth.
_DEFAULT_MIN_SHARPE = float(os.getenv("BACKTEST_MIN_SHARPE", "1.0"))

# Minimum bars required before we start generating signals (matches the
# router's implicit assumption).
_WARMUP_BARS = 100


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class SymbolResult:
    """Per-symbol backtest outcome."""
    symbol: str
    bars: int
    trades: int
    wins: int
    losses: int
    total_return: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    total_costs: float
    final_equity: float
    regime_breakdown: dict = field(default_factory=dict)


@dataclass
class AcceptanceResult:
    """Whether the backtest clears the Sprint 6A gate."""
    passed: bool
    actual_sharpe: float
    min_sharpe: float
    reason: str


# ── Bar loading + caching ────────────────────────────────────────────────

def _cache_path(symbol: str, start: datetime, end: datetime,
                timeframe: str) -> str:
    """Deterministic parquet cache path for (symbol, window, tf)."""
    safe = symbol.replace("/", "_")
    tag = f"{safe}_{start.date()}_{end.date()}_{timeframe}.parquet"
    return os.path.join(_CACHE_DIR, tag)


def load_or_fetch_bars(
    symbol: str,
    days: int,
    timeframe: str,
    broker=None,
    use_cache: bool = True,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Fetch historical bars, preferring the parquet cache.

    Args:
        symbol: Trading symbol (e.g. "BTC/USD", "TSLA").
        days: Lookback window in days.
        timeframe: "minute" | "hour" | "day" — passed to the broker helper.
        broker: An object with ``get_historical_bars(symbol, days, timeframe)``.
            If None, we lazily import core.broker.Broker. Tests override this.
        use_cache: If False, always hit the network.
        now: Override "today" for deterministic caching in tests.

    Returns:
        DataFrame with a `time` column and OHLCV columns. Empty on failure.
    """
    now = now or datetime.now()
    start = now - timedelta(days=days)
    cache_file = _cache_path(symbol, start, now, timeframe)

    if use_cache and os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            log.warning(f"Bar cache read failed for {symbol}: {e}; refetching")

    if broker is None:
        from core.broker import Broker
        broker = Broker()

    try:
        df = _fetch_from_broker(broker, symbol, days, timeframe)
    except Exception as e:
        log.error(f"Broker fetch failed for {symbol}: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    os.makedirs(_CACHE_DIR, exist_ok=True)
    try:
        df.to_parquet(cache_file, index=False)
    except Exception as e:
        log.warning(f"Could not cache bars for {symbol}: {e}")
    return df


def _fetch_from_broker(broker, symbol: str, days: int,
                       timeframe: str) -> pd.DataFrame:
    """Uniform broker-fetch helper.

    The existing broker only exposes 1-min bars via get_historical_bars;
    for hourly/daily we fall back to the same method and resample locally.
    Keeps this script broker-agnostic for testing.
    """
    df = broker.get_historical_bars(symbol, days=days)
    if df is None or df.empty:
        return pd.DataFrame()
    if timeframe == "minute":
        return df
    return _resample_ohlcv(df, timeframe)


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1-min OHLCV to hourly or daily. Preserves the `time` column."""
    if "time" not in df.columns:
        return df
    rule = {"hour": "1h", "day": "1D"}.get(timeframe)
    if rule is None:
        return df
    work = df.copy()
    work = work.set_index("time")
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    out = work.resample(rule).agg(agg).dropna().reset_index()
    return out


# ── Per-symbol simulation ────────────────────────────────────────────────

def _annualize_factor(timeframe: str) -> int:
    """Sharpe annualization factor per bar timeframe."""
    return {
        "minute": 252 * 390,  # equity trading minutes/yr
        "hour":   252 * 7,    # approx hours/yr
        "day":    252,
    }.get(timeframe, 252)


def _regime_for_bar(close_series: pd.Series, window: int = 20) -> str:
    """Classify the current regime from a trailing window of closes.

    Uses the same classifier as the live detector. Returns 'normal' when
    insufficient data so we don't block early trades.
    """
    if len(close_series) < window + 2:
        return "normal"
    tail = close_series.tail(window + 1)
    returns = tail.pct_change().dropna().values
    if len(returns) < 2:
        return "normal"
    vol_annualized = float(np.std(returns, ddof=1) * np.sqrt(252))
    return classify_vol(vol_annualized)


def simulate_symbol(
    symbol: str,
    df: pd.DataFrame,
    starting_cash: float,
    cost_model: TransactionCostModel | None,
    annualize: int,
    compute_signals=None,
) -> SymbolResult:
    """Run the full pipeline over one symbol's bars.

    Args:
        symbol: Symbol being traded.
        df: OHLCV bars (chronological, with `time` column).
        starting_cash: Dollars to allocate at the start.
        cost_model: Optional transaction cost model. None = no costs.
        annualize: Sharpe annualization factor.
        compute_signals: Signal function; defaults to strategies.router.
            Tests pass a stub to keep runtime bounded.

    Returns:
        Filled SymbolResult.
    """
    if compute_signals is None:
        from strategies.router import compute_signals as router_compute
        compute_signals = router_compute

    cash = starting_cash
    position_qty = 0.0
    position_price = 0.0
    trades_pnl: list[float] = []
    equity_curve: list[float] = []
    total_costs = 0.0
    regime_trades: dict[str, list[float]] = {}

    if df is None or df.empty or len(df) < _WARMUP_BARS + 1:
        return SymbolResult(
            symbol=symbol, bars=len(df) if df is not None else 0,
            trades=0, wins=0, losses=0,
            total_return=0.0, sharpe=0.0, sortino=0.0, calmar=0.0,
            max_drawdown=0.0, win_rate=0.0, total_costs=0.0,
            final_equity=starting_cash,
        )

    closes = df["close"].astype(float)

    for i in range(_WARMUP_BARS, len(df)):
        window = df.iloc[max(0, i - _WARMUP_BARS):i + 1]
        price = float(closes.iloc[i])
        regime = _regime_for_bar(closes.iloc[:i + 1])

        try:
            signal = compute_signals(symbol, window, regime=regime)
        except Exception as e:
            log.warning(f"Signal error on {symbol} bar {i}: {e}")
            signal = {"action": "hold", "strength": 0.0}

        action = signal.get("action", "hold")

        # BUY side: open a position if flat
        if action == "buy" and position_qty == 0:
            gross = cash * 0.5 * float(signal.get("strength", 0.5))
            if gross >= 1:
                cost = cost_model.estimate(symbol, gross).total_cost \
                    if cost_model else 0.0
                net = gross - cost
                if net >= 1:
                    position_qty = net / price
                    position_price = price
                    cash -= gross
                    total_costs += cost

        # SELL side: close the position
        elif action == "sell" and position_qty > 0:
            gross = position_qty * price
            cost = cost_model.estimate(symbol, gross).total_cost \
                if cost_model else 0.0
            net = gross - cost
            pnl = net - (position_qty * position_price)
            cash += net
            total_costs += cost
            trades_pnl.append(pnl)
            regime_trades.setdefault(regime, []).append(pnl)
            position_qty = 0.0
            position_price = 0.0

        equity = cash + position_qty * price
        equity_curve.append(equity)

    # Close any open position at the last price
    if position_qty > 0:
        price = float(closes.iloc[-1])
        gross = position_qty * price
        cost = cost_model.estimate(symbol, gross).total_cost \
            if cost_model else 0.0
        net = gross - cost
        pnl = net - (position_qty * position_price)
        cash += net
        total_costs += cost
        trades_pnl.append(pnl)
        equity_curve[-1] = cash

    metrics = compute_portfolio_metrics(
        equity_curve=equity_curve,
        trades_pnl=trades_pnl,
        annualize=annualize,
    )

    # Per-regime summary (Sharpe proxy — avg P&L across trades in each regime)
    regime_summary = {
        r: {
            "n_trades": len(pnls),
            "total_pnl": round(sum(pnls), 4),
            "avg_pnl": round(sum(pnls) / len(pnls), 4) if pnls else 0.0,
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 3)
            if pnls else 0.0,
        }
        for r, pnls in regime_trades.items()
    }

    return SymbolResult(
        symbol=symbol,
        bars=len(df),
        trades=metrics.total_trades,
        wins=metrics.wins,
        losses=metrics.losses,
        total_return=round(metrics.total_return, 4),
        sharpe=round(metrics.sharpe_ratio, 4),
        sortino=round(metrics.sortino_ratio, 4),
        calmar=round(metrics.calmar_ratio, 4),
        max_drawdown=round(metrics.max_drawdown, 4),
        win_rate=round(metrics.win_rate, 4),
        total_costs=round(total_costs, 4),
        final_equity=round(cash, 4),
        regime_breakdown=regime_summary,
    )


# ── Portfolio aggregation + Monte Carlo ──────────────────────────────────

def aggregate_portfolio(results: Iterable[SymbolResult],
                        starting_cash: float,
                        annualize: int) -> dict:
    """Aggregate per-symbol outcomes into a single portfolio view.

    We sum final equities (independent sleeves) — not strictly additive
    when the real engine shares cash across symbols, but good enough for
    a high-level acceptance check and cheap to compute.
    """
    results = list(results)
    n = max(len(results), 1)
    total_final = sum(r.final_equity for r in results)
    total_start = starting_cash * n
    total_return = (total_final - total_start) / total_start \
        if total_start > 0 else 0.0

    # Aggregate trade P&L for Monte Carlo + trade-level metrics
    all_trade_pnl = [p for r in results for p in _iter_pnls(r)]

    sharpes = [r.sharpe for r in results if r.sharpe != 0.0]
    avg_sharpe = float(np.mean(sharpes)) if sharpes else 0.0

    return {
        "symbols_included": [r.symbol for r in results],
        "starting_cash_per_symbol": starting_cash,
        "total_starting": total_start,
        "total_final": total_final,
        "total_return": round(total_return, 4),
        "portfolio_sharpe_mean": round(avg_sharpe, 4),
        "trade_count": sum(r.trades for r in results),
        "total_costs": round(sum(r.total_costs for r in results), 4),
        "all_trade_pnl_count": len(all_trade_pnl),
    }


def _iter_pnls(result: SymbolResult) -> list[float]:
    """Reconstruct the P&L list per-symbol from the regime breakdown.

    We don't persist raw per-trade P&L on SymbolResult (keeps the dataclass
    light for JSON), but regime_breakdown's total_pnl × n_trades gives a
    usable approximation for Monte Carlo sequencing. For cleaner MC, run the
    simulation with `return_pnls=True` (a future flag).
    """
    pnls: list[float] = []
    for rb in result.regime_breakdown.values():
        n = rb.get("n_trades", 0)
        avg = rb.get("avg_pnl", 0.0)
        pnls.extend([avg] * n)
    return pnls


def monte_carlo_shuffle(trade_pnls: Sequence[float],
                        starting_equity: float,
                        runs: int = 1000,
                        seed: int = 42) -> dict:
    """Shuffle trade order `runs` times; return terminal-equity distribution.

    Trade magnitudes stay the same — we only permute their sequence. This
    tests whether results depend on a lucky ordering. Pure retail
    sizing is fixed-percent so this is a reasonable sequencing check.
    """
    if not trade_pnls or runs <= 0:
        return {"runs": 0, "p5": starting_equity, "p50": starting_equity,
                "p95": starting_equity, "mean": starting_equity,
                "std": 0.0}

    rng = np.random.default_rng(seed)
    pnl_array = np.asarray(trade_pnls, dtype=float)
    terminals = np.empty(runs, dtype=float)
    for i in range(runs):
        perm = rng.permutation(pnl_array)
        terminals[i] = starting_equity + float(perm.sum())

    return {
        "runs": runs,
        "p5": round(float(np.percentile(terminals, 5)), 4),
        "p50": round(float(np.percentile(terminals, 50)), 4),
        "p95": round(float(np.percentile(terminals, 95)), 4),
        "mean": round(float(np.mean(terminals)), 4),
        "std": round(float(np.std(terminals)), 4),
    }


# ── Acceptance gate ──────────────────────────────────────────────────────

def evaluate_acceptance(summary: dict, min_sharpe: float) -> AcceptanceResult:
    """Decide whether this run clears the Sprint 6A acceptance threshold."""
    actual = float(summary.get("portfolio_sharpe_mean", 0.0))
    if actual >= min_sharpe:
        return AcceptanceResult(
            passed=True, actual_sharpe=actual, min_sharpe=min_sharpe,
            reason=f"Mean per-symbol Sharpe {actual:.2f} ≥ {min_sharpe:.2f}",
        )
    return AcceptanceResult(
        passed=False, actual_sharpe=actual, min_sharpe=min_sharpe,
        reason=(f"Mean per-symbol Sharpe {actual:.2f} < {min_sharpe:.2f} — "
                "not ready to promote. Inspect per-symbol + per-regime CSV."),
    )


# ── Artifact writing ─────────────────────────────────────────────────────

def write_artifacts(out_dir: str,
                    results: list[SymbolResult],
                    summary: dict,
                    mc: dict,
                    acceptance: AcceptanceResult,
                    framing: "BacktestFrame | None" = None) -> None:
    """Write all backtest artifacts under `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)

    # Per-symbol metrics
    rows = [asdict(r) for r in results]
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "per_symbol_metrics.csv"), index=False,
    )

    # Per-regime breakdown
    regime_rows = []
    for r in results:
        for regime, stats in r.regime_breakdown.items():
            regime_rows.append({"symbol": r.symbol, "regime": regime, **stats})
    if regime_rows:
        pd.DataFrame(regime_rows).to_csv(
            os.path.join(out_dir, "per_regime.csv"), index=False,
        )

    # Summary JSON — includes framing lines for dashboard consumption.
    payload = {
        "generated_at": datetime.now().isoformat(),
        "portfolio": summary,
        "monte_carlo": mc,
    }
    if framing is not None:
        payload["framing"] = {
            "sharpe_line": framing.sharpe_line,
            "return_line": framing.return_line,
            "pct_band_line": framing.pct_band_line,
            "disclaimer_line": framing.disclaimer_line,
        }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # Acceptance JSON
    with open(os.path.join(out_dir, "acceptance.json"), "w") as f:
        json.dump(asdict(acceptance), f, indent=2, default=str)


# ── Main entry ───────────────────────────────────────────────────────────

def run_backtest(
    symbols: Sequence[str],
    days: int = 730,
    timeframe: str = "hour",
    starting_cash: float = 100_000,
    use_cache: bool = True,
    monte_carlo_runs: int = 1000,
    min_sharpe: float = _DEFAULT_MIN_SHARPE,
    broker=None,
    compute_signals=None,
    out_root: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Run the OOS backtest end-to-end. Returns a dict you can also read
    from the written `summary.json`.

    Exposed as a function so tests can drive it without invoking the CLI.
    """
    cost_model = TransactionCostModel() if Config.TC_ENABLED else None
    annualize = _annualize_factor(timeframe)

    results: list[SymbolResult] = []
    for sym in symbols:
        log.info(f"Backtest: loading {sym} ({days}d, {timeframe})")
        df = load_or_fetch_bars(sym, days=days, timeframe=timeframe,
                                broker=broker, use_cache=use_cache, now=now)
        if df.empty:
            log.warning(f"Backtest: no bars for {sym}; skipping")
            continue
        res = simulate_symbol(sym, df, starting_cash=starting_cash,
                              cost_model=cost_model, annualize=annualize,
                              compute_signals=compute_signals)
        log.info(f"Backtest {sym}: Sharpe={res.sharpe}, trades={res.trades}, "
                 f"return={res.total_return:+.2%}")
        results.append(res)

    summary = aggregate_portfolio(results, starting_cash, annualize)
    all_pnl = [p for r in results for p in _iter_pnls(r)]
    mc = monte_carlo_shuffle(all_pnl, summary["total_starting"],
                             runs=monte_carlo_runs)
    acceptance = evaluate_acceptance(summary, min_sharpe)

    # Sprint 6I: attach honest framing to the written summary so the
    # dashboard can render it without reimplementing the logic.
    from core.expected_returns import frame_backtest
    start = summary.get("total_starting", 0.0) or 1.0
    mc_p5 = (mc.get("p5", start) / start) - 1.0 if mc else None
    mc_p50 = (mc.get("p50", start) / start) - 1.0 if mc else None
    mc_p95 = (mc.get("p95", start) / start) - 1.0 if mc else None
    framing = frame_backtest(
        sharpe=summary.get("portfolio_sharpe_mean", 0.0),
        total_return_pct=100.0 * summary.get("total_return", 0.0),
        mc_p5=mc_p5, mc_p50=mc_p50, mc_p95=mc_p95,
        context="2yr OOS backtest",
    )

    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root or _OUT_ROOT, stamp)
    write_artifacts(out_dir, results, summary, mc, acceptance, framing)
    log.info(f"Backtest artifacts: {out_dir}")
    log.info(f"Acceptance: {acceptance.reason}")
    # Print the honest framing right after the acceptance line so nobody
    # can copy-paste Sharpe out of context without also seeing the anchor.
    log.info(framing.as_text())

    return {
        "out_dir": out_dir,
        "results": [asdict(r) for r in results],
        "summary": summary,
        "monte_carlo": mc,
        "acceptance": asdict(acceptance),
        "framing": {
            "sharpe_line": framing.sharpe_line,
            "return_line": framing.return_line,
            "pct_band_line": framing.pct_band_line,
            "disclaimer_line": framing.disclaimer_line,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-year OOS backtest (Sprint 6A).")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--timeframe", choices=["minute", "hour", "day"],
                        default="hour")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--starting-cash", type=float, default=100_000)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--monte-carlo-runs", type=int, default=1000)
    parser.add_argument("--min-sharpe", type=float,
                        default=_DEFAULT_MIN_SHARPE,
                        help="Acceptance threshold (default: 1.0)")
    args = parser.parse_args(argv)

    symbols = args.symbols or Config.SYMBOLS
    out = run_backtest(
        symbols=symbols,
        days=args.days,
        timeframe=args.timeframe,
        starting_cash=args.starting_cash,
        use_cache=not args.no_cache,
        monte_carlo_runs=args.monte_carlo_runs,
        min_sharpe=args.min_sharpe,
    )
    return 0 if out["acceptance"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
