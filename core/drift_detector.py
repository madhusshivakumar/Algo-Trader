"""Drift detection — monitors live trading performance vs baseline.

Reads from the trades SQLite table, compares recent performance against
historical baseline, and flags strategies or symbols that are degrading.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from utils.logger import DB_PATH, log


@dataclass
class DriftMetrics:
    symbol: str
    strategy: str
    window_start: str
    window_end: str
    trade_count: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    is_degraded: bool = False
    degradation_reason: str = ""


class DriftDetector:
    def __init__(self, db_path: str = None, lookback_days: int = 7, min_trades: int = 5):
        self.db_path = db_path or DB_PATH
        self.lookback_days = lookback_days
        self.min_trades = min_trades

    def check_drift(self, symbol: str = None) -> list[DriftMetrics]:
        """Check for performance drift. Returns metrics for each symbol/strategy pair."""
        conn = sqlite3.connect(self.db_path)
        try:
            recent_cutoff = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
            baseline_cutoff = (datetime.now() - timedelta(days=self.lookback_days * 4)).strftime("%Y-%m-%d")

            # Get recent trades
            query = "SELECT symbol, strategy, side, pnl, timestamp FROM trades WHERE timestamp >= ?"
            params = [recent_cutoff]
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            recent_trades = conn.execute(query, params).fetchall()

            # Get baseline trades (earlier period)
            baseline_query = ("SELECT symbol, strategy, side, pnl, timestamp FROM trades "
                              "WHERE timestamp >= ? AND timestamp < ?")
            baseline_params = [baseline_cutoff, recent_cutoff]
            if symbol:
                baseline_query += " AND symbol = ?"
                baseline_params.append(symbol)
            baseline_trades = conn.execute(baseline_query, baseline_params).fetchall()

        finally:
            conn.close()

        # Group by symbol
        recent_by_sym = self._group_trades(recent_trades)
        baseline_by_sym = self._group_trades(baseline_trades)

        results = []
        all_symbols = set(recent_by_sym.keys()) | set(baseline_by_sym.keys())

        for sym in sorted(all_symbols):
            recent = recent_by_sym.get(sym, [])
            baseline = baseline_by_sym.get(sym, [])

            recent_metrics = self._calculate_metrics(sym, recent, recent_cutoff, "now")
            baseline_metrics = self._calculate_metrics(sym, baseline, baseline_cutoff, recent_cutoff)

            # Check for degradation
            if recent_metrics.trade_count >= self.min_trades and baseline_metrics.trade_count >= self.min_trades:
                is_degraded, reason = self._is_degraded(recent_metrics, baseline_metrics)
                recent_metrics.is_degraded = is_degraded
                recent_metrics.degradation_reason = reason

            results.append(recent_metrics)

        return results

    def _group_trades(self, trades: list) -> dict[str, list]:
        """Group trade rows by symbol."""
        grouped = {}
        for row in trades:
            sym = row[0]
            if sym not in grouped:
                grouped[sym] = []
            grouped[sym].append(row)
        return grouped

    def _calculate_metrics(self, symbol: str, trades: list,
                           window_start: str, window_end: str) -> DriftMetrics:
        """Calculate performance metrics for a set of trades."""
        if not trades:
            return DriftMetrics(
                symbol=symbol, strategy="", window_start=window_start,
                window_end=window_end, trade_count=0, win_rate=0.0,
                avg_pnl=0.0, total_pnl=0.0,
            )

        # Extract sell trades (which have PnL)
        sells = [t for t in trades if t[2] == "sell"]
        pnls = [t[3] for t in sells if t[3] is not None]
        strategy = trades[0][1] if trades[0][1] else ""

        trade_count = len(sells)
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0.0
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / len(pnls) if pnls else 0.0

        return DriftMetrics(
            symbol=symbol, strategy=strategy,
            window_start=window_start, window_end=window_end,
            trade_count=trade_count, win_rate=win_rate,
            avg_pnl=avg_pnl, total_pnl=total_pnl,
        )

    def _is_degraded(self, recent: DriftMetrics, baseline: DriftMetrics) -> tuple[bool, str]:
        """Check if recent performance is significantly worse than baseline."""
        reasons = []

        # Win rate dropped by more than 15 percentage points
        if baseline.win_rate > 0 and (baseline.win_rate - recent.win_rate) > 0.15:
            reasons.append(
                f"Win rate dropped: {baseline.win_rate:.0%} → {recent.win_rate:.0%}"
            )

        # Average PnL turned negative when baseline was positive
        if baseline.avg_pnl > 0 and recent.avg_pnl < 0:
            reasons.append(
                f"Avg PnL turned negative: ${baseline.avg_pnl:.2f} → ${recent.avg_pnl:.2f}"
            )

        # Total PnL significantly worse
        if baseline.total_pnl > 0 and recent.total_pnl < -abs(baseline.total_pnl) * 0.5:
            reasons.append(
                f"Total PnL degraded: ${baseline.total_pnl:.2f} → ${recent.total_pnl:.2f}"
            )

        if reasons:
            return True, "; ".join(reasons)
        return False, ""

    def get_report(self) -> dict:
        """Generate a full drift report."""
        metrics = self.check_drift()
        degraded = [m for m in metrics if m.is_degraded]

        return {
            "timestamp": datetime.now().isoformat(),
            "total_symbols": len(metrics),
            "degraded_count": len(degraded),
            "degraded_symbols": [
                {"symbol": m.symbol, "reason": m.degradation_reason}
                for m in degraded
            ],
            "all_metrics": [
                {
                    "symbol": m.symbol, "strategy": m.strategy,
                    "trade_count": m.trade_count, "win_rate": m.win_rate,
                    "avg_pnl": m.avg_pnl, "total_pnl": m.total_pnl,
                    "is_degraded": m.is_degraded,
                }
                for m in metrics
            ],
        }

    def should_alert(self) -> list[str]:
        """Return human-readable alerts for degraded symbols."""
        metrics = self.check_drift()
        alerts = []
        for m in metrics:
            if m.is_degraded:
                alerts.append(f"DRIFT ALERT: {m.symbol} — {m.degradation_reason}")
        return alerts
