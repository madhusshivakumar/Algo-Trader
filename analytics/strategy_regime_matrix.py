"""Strategy × regime sensitivity matrix — Sprint 6F.

Runs every strategy in ``strategies.router.STRATEGY_REGISTRY`` over a supplied
bar history, tags each trade with the regime at entry, and produces a
`(strategy, regime) → metrics` table. The output drives a data-driven
override on top of the Sprint 6C hardcoded regime filter in
``core.regime_detector.is_strategy_allowed``.

Two persisted artifacts:
    data/strategy_regime/matrix.csv       — full table (strategy × regime)
    data/strategy_regime/hard_skip.json   — (strategy, regime) pairs with
                                             empirically negative edge; read
                                             by the router at startup.

The matrix is deliberately cheap: we simulate each strategy *independently*
(no modifiers, no RL selector, no pipeline) so we isolate the raw signal
quality in each regime. The full-pipeline OOS backtest lives in
``scripts/full_oos_backtest.py`` (Sprint 6A).

Design notes:
    * Each strategy is evaluated on every symbol individually. That gives us
      cross-section data without cross-sleeve cash assumptions.
    * Regime is labelled per-bar from a rolling 20-day realized-vol window.
    * We run the bare ``strategies.<name>.compute_signals(df)`` function —
      *not* the router — so the hardcoded regime filter doesn't suppress
      exactly the observations we want to measure.
    * ``derive_hard_skip`` returns (strategy, regime) pairs where we have
      enough data (default ≥ 20 trades) AND empirical edge is ≤ a threshold
      (default avg_pnl ≤ 0). Under-sampled cells are ignored — we'd rather
      fall back to the hardcoded Sprint 6C rules than trust 3-trade noise.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.regime_detector import classify_vol
from core.transaction_costs import TransactionCostModel
from utils.logger import log


# ── Paths ──────────────────────────────────────────────────────────────────

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_REPO_ROOT, "data", "strategy_regime")
_MATRIX_CSV = os.path.join(_DATA_DIR, "matrix.csv")
_HARD_SKIP_JSON = os.path.join(_DATA_DIR, "hard_skip.json")

# Warmup — strategies need this many bars before we trust their signals.
_WARMUP_BARS = 100

# Default edge cutoffs for derive_hard_skip.
_DEFAULT_MIN_TRADES = 20
_DEFAULT_MAX_AVG_PNL = 0.0


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class CellResult:
    """One (strategy, regime) cell of the matrix."""
    strategy: str
    regime: str
    n_trades: int
    total_pnl: float
    avg_pnl: float
    win_rate: float
    sharpe_proxy: float  # mean/std of trade-level PnL (not annualized)

    def to_row(self) -> dict:
        return {
            "strategy": self.strategy,
            "regime": self.regime,
            "n_trades": self.n_trades,
            "total_pnl": round(self.total_pnl, 6),
            "avg_pnl": round(self.avg_pnl, 6),
            "win_rate": round(self.win_rate, 4),
            "sharpe_proxy": round(self.sharpe_proxy, 4),
        }


# ── Regime labeling (mirrors Sprint 6A logic) ─────────────────────────────

def _regime_for_bar(close_series: pd.Series, window: int = 20) -> str:
    """Classify regime from a trailing window of close prices."""
    if len(close_series) < window + 2:
        return "normal"
    tail = close_series.tail(window + 1)
    returns = tail.pct_change().dropna().values
    if len(returns) < 2:
        return "normal"
    vol_annualized = float(np.std(returns, ddof=1) * np.sqrt(252))
    return classify_vol(vol_annualized)


# ── Core simulation ────────────────────────────────────────────────────────

def _simulate_strategy_on_symbol(
    strategy_name: str,
    compute_signals: Callable,
    symbol: str,
    df: pd.DataFrame,
    cost_model: TransactionCostModel | None = None,
    starting_cash: float = 10_000.0,
) -> list[tuple[str, float]]:
    """Run one strategy over one symbol's bars.

    Returns a list of ``(regime_at_entry, trade_pnl)`` tuples — one per
    closed trade. Caller aggregates across strategies + symbols.
    """
    if df is None or df.empty or len(df) < _WARMUP_BARS + 1:
        return []

    closes = df["close"].astype(float)
    cash = starting_cash
    position_qty = 0.0
    position_price = 0.0
    entry_regime: str | None = None
    trades: list[tuple[str, float]] = []

    for i in range(_WARMUP_BARS, len(df)):
        window = df.iloc[max(0, i - _WARMUP_BARS):i + 1]
        price = float(closes.iloc[i])
        regime = _regime_for_bar(closes.iloc[:i + 1])

        try:
            signal = compute_signals(window)
        except Exception as e:
            log.warning(f"[regime-matrix] {strategy_name}/{symbol} bar {i}: {e}")
            continue

        action = signal.get("action", "hold") if isinstance(signal, Mapping) \
            else "hold"

        if action == "buy" and position_qty == 0:
            gross = cash * 0.5 * float(signal.get("strength", 0.5))
            if gross < 1:
                continue
            cost = cost_model.estimate(symbol, gross).total_cost \
                if cost_model else 0.0
            net = gross - cost
            if net < 1:
                continue
            position_qty = net / price
            position_price = price
            cash -= gross
            entry_regime = regime

        elif action == "sell" and position_qty > 0:
            gross = position_qty * price
            cost = cost_model.estimate(symbol, gross).total_cost \
                if cost_model else 0.0
            net = gross - cost
            pnl = net - (position_qty * position_price)
            cash += net
            trades.append((entry_regime or "normal", pnl))
            position_qty = 0.0
            position_price = 0.0
            entry_regime = None

    # Close open position at the final price
    if position_qty > 0:
        price = float(closes.iloc[-1])
        gross = position_qty * price
        cost = cost_model.estimate(symbol, gross).total_cost \
            if cost_model else 0.0
        net = gross - cost
        pnl = net - (position_qty * position_price)
        trades.append((entry_regime or "normal", pnl))

    return trades


# ── Matrix builder ─────────────────────────────────────────────────────────

def _cell_from_pnls(strategy: str, regime: str,
                    pnls: Sequence[float]) -> CellResult:
    """Aggregate a PnL list into a CellResult."""
    if not pnls:
        return CellResult(strategy=strategy, regime=regime, n_trades=0,
                          total_pnl=0.0, avg_pnl=0.0, win_rate=0.0,
                          sharpe_proxy=0.0)
    arr = np.asarray(pnls, dtype=float)
    total = float(arr.sum())
    avg = float(arr.mean())
    wins = int((arr > 0).sum())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sharpe = avg / std if std > 1e-9 else 0.0
    return CellResult(
        strategy=strategy, regime=regime,
        n_trades=len(pnls), total_pnl=total, avg_pnl=avg,
        win_rate=wins / len(pnls), sharpe_proxy=sharpe,
    )


def build_matrix(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    strategies: Mapping[str, Callable] | None = None,
    cost_model: TransactionCostModel | None = None,
    starting_cash: float = 10_000.0,
) -> pd.DataFrame:
    """Compute the full (strategy × regime) matrix.

    Args:
        bars_by_symbol: ``{symbol: OHLCV DataFrame}``.
        strategies: ``{name: compute_signals_fn}``. Defaults to the live
            ``STRATEGY_REGISTRY``.
        cost_model: Optional TransactionCostModel. None = no costs.
        starting_cash: Per-(symbol, strategy) starting cash. Only affects
            PnL scale — ranking across regimes is unchanged.

    Returns:
        DataFrame with columns:
            strategy, regime, n_trades, total_pnl, avg_pnl, win_rate,
            sharpe_proxy
    """
    if strategies is None:
        from strategies.router import STRATEGY_REGISTRY
        strategies = STRATEGY_REGISTRY

    regimes: list[str] = ["low_vol", "normal", "high_vol", "crisis"]
    # strategy → regime → list[pnl]
    per_strategy_regime: dict[str, dict[str, list[float]]] = {
        name: {r: [] for r in regimes} for name in strategies
    }

    for symbol, df in bars_by_symbol.items():
        if df is None or df.empty:
            continue
        for name, fn in strategies.items():
            trades = _simulate_strategy_on_symbol(
                strategy_name=name, compute_signals=fn,
                symbol=symbol, df=df,
                cost_model=cost_model, starting_cash=starting_cash,
            )
            for regime, pnl in trades:
                per_strategy_regime[name].setdefault(regime, []).append(pnl)

    rows: list[dict] = []
    for name, by_regime in per_strategy_regime.items():
        for regime in regimes:
            cell = _cell_from_pnls(name, regime, by_regime.get(regime, []))
            rows.append(cell.to_row())

    return pd.DataFrame(rows, columns=[
        "strategy", "regime", "n_trades", "total_pnl", "avg_pnl",
        "win_rate", "sharpe_proxy",
    ])


# ── Hard-skip derivation ───────────────────────────────────────────────────

def derive_hard_skip(
    matrix: pd.DataFrame,
    min_trades: int = _DEFAULT_MIN_TRADES,
    max_avg_pnl: float = _DEFAULT_MAX_AVG_PNL,
) -> list[tuple[str, str]]:
    """Return (strategy, regime) pairs that should be forced to ``hold``.

    A pair is included iff we have at least ``min_trades`` observations AND
    the average trade PnL is at most ``max_avg_pnl``. Under-sampled cells
    are ignored — they fall through to the hardcoded Sprint 6C rules.
    """
    if matrix is None or matrix.empty:
        return []

    pairs: list[tuple[str, str]] = []
    for _, row in matrix.iterrows():
        if int(row["n_trades"]) < min_trades:
            continue
        if float(row["avg_pnl"]) > max_avg_pnl:
            continue
        pairs.append((str(row["strategy"]), str(row["regime"])))
    return pairs


# ── Persistence ────────────────────────────────────────────────────────────

def write_matrix(matrix: pd.DataFrame,
                 path: str | None = None) -> str:
    """Write matrix to CSV. Returns the path written."""
    path = path or _MATRIX_CSV
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        matrix.to_csv(path, index=False)
        log.info(f"[regime-matrix] Wrote matrix to {path} "
                 f"({len(matrix)} rows)")
    except OSError as e:
        log.warning(f"[regime-matrix] Could not write matrix CSV: {e}")
    return path


def write_hard_skip(
    pairs: Sequence[tuple[str, str]],
    path: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Write hard-skip overrides as JSON. Returns the path written."""
    path = path or _HARD_SKIP_JSON
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overrides": [list(p) for p in pairs],
        "metadata": metadata or {},
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        log.info(f"[regime-matrix] Wrote {len(pairs)} hard-skip override(s) "
                 f"to {path}")
    except OSError as e:
        log.warning(f"[regime-matrix] Could not write hard_skip JSON: {e}")
    return path


def load_hard_skip(path: str | None = None) -> set[tuple[str, str]]:
    """Load hard-skip overrides. Returns empty set on any error.

    This is the hook the router calls at startup.
    """
    path = path or _HARD_SKIP_JSON
    if not os.path.exists(path):
        return set()
    try:
        with open(path) as f:
            payload = json.load(f)
        raw = payload.get("overrides", [])
        result: set[tuple[str, str]] = set()
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                result.add((str(item[0]), str(item[1])))
        return result
    except (OSError, json.JSONDecodeError, TypeError) as e:
        log.warning(f"[regime-matrix] Could not load hard_skip overrides "
                    f"from {path}: {e}")
        return set()


# ── Module-level entrypoint ────────────────────────────────────────────────

def run(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    min_trades: int = _DEFAULT_MIN_TRADES,
    max_avg_pnl: float = _DEFAULT_MAX_AVG_PNL,
    cost_model: TransactionCostModel | None = None,
    matrix_path: str | None = None,
    hard_skip_path: str | None = None,
    strategies: Mapping[str, Callable] | None = None,
) -> dict:
    """Build matrix + derive overrides + persist. Returns a summary dict."""
    matrix = build_matrix(
        bars_by_symbol, strategies=strategies, cost_model=cost_model,
    )
    pairs = derive_hard_skip(matrix, min_trades=min_trades,
                             max_avg_pnl=max_avg_pnl)

    write_matrix(matrix, matrix_path)
    metadata = {
        "symbols": sorted(bars_by_symbol.keys()),
        "min_trades": min_trades,
        "max_avg_pnl": max_avg_pnl,
    }
    write_hard_skip(pairs, hard_skip_path, metadata=metadata)

    return {
        "matrix_rows": len(matrix),
        "overrides": [list(p) for p in pairs],
        "metadata": metadata,
    }
