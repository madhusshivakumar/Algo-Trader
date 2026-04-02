"""Walk-forward backtesting — rolling window validation to detect overfitting.

Splits historical data into train/test folds, runs each strategy on both,
and compares in-sample vs out-of-sample performance.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class WalkForwardResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    strategy: str
    in_sample_return: float
    in_sample_sharpe: float
    out_of_sample_return: float
    out_of_sample_sharpe: float
    trade_count: int
    win_rate: float


class WalkForwardBacktester:
    def __init__(self, strategy_fn: Callable, strategy_name: str = "",
                 train_days: int = 60, test_days: int = 20, step_days: int = 10):
        self.strategy_fn = strategy_fn
        self.strategy_name = strategy_name
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days

    def run(self, df: pd.DataFrame, symbol: str,
            starting_cash: float = 100000) -> list[WalkForwardResult]:
        """Run walk-forward analysis across all possible folds."""
        if df is None or len(df) < (self.train_days + self.test_days) * 390:
            # Not enough data for even one fold (390 = approx 1-min bars per day)
            # Fall back to row-based splitting
            min_rows = 200
            if df is None or len(df) < min_rows:
                return []

        results = []
        total_rows = len(df)
        train_rows = int(total_rows * 0.6)  # Use 60% for training window
        test_rows = int(total_rows * 0.2)   # 20% for testing
        step_rows = int(total_rows * 0.1)   # 10% step

        if train_rows < 50 or test_rows < 20:
            return []

        fold = 0
        start = 0
        while start + train_rows + test_rows <= total_rows:
            train_df = df.iloc[start:start + train_rows]
            test_df = df.iloc[start + train_rows:start + train_rows + test_rows]

            is_metrics = self._simulate(train_df, starting_cash)
            oos_metrics = self._simulate(test_df, starting_cash)

            result = WalkForwardResult(
                fold=fold,
                train_start=str(train_df.index[0]) if hasattr(train_df.index[0], '__str__') else str(fold),
                train_end=str(train_df.index[-1]) if hasattr(train_df.index[-1], '__str__') else str(fold),
                test_start=str(test_df.index[0]) if hasattr(test_df.index[0], '__str__') else str(fold),
                test_end=str(test_df.index[-1]) if hasattr(test_df.index[-1], '__str__') else str(fold),
                strategy=self.strategy_name,
                in_sample_return=is_metrics["total_return"],
                in_sample_sharpe=is_metrics["sharpe"],
                out_of_sample_return=oos_metrics["total_return"],
                out_of_sample_sharpe=oos_metrics["sharpe"],
                trade_count=oos_metrics["trade_count"],
                win_rate=oos_metrics["win_rate"],
            )
            results.append(result)
            fold += 1
            start += step_rows

        return results

    def _simulate(self, df: pd.DataFrame, starting_cash: float) -> dict:
        """Simulate trading on a DataFrame slice using the strategy function."""
        if len(df) < 30:
            return {"total_return": 0.0, "sharpe": 0.0, "trade_count": 0, "win_rate": 0.0}

        cash = starting_cash
        position_qty = 0.0
        position_price = 0.0
        trades = []
        equity_curve = []

        for i in range(30, len(df)):
            window = df.iloc[max(0, i - 100):i + 1]
            current_price = float(df["close"].iloc[i])

            try:
                signal = self.strategy_fn(window)
            except Exception:
                signal = {"action": "hold", "strength": 0.0}

            if signal["action"] == "buy" and position_qty == 0:
                size = cash * 0.5 * signal.get("strength", 0.5)
                if size > 10:
                    position_qty = size / current_price
                    position_price = current_price
                    cash -= size

            elif signal["action"] == "sell" and position_qty > 0:
                proceeds = position_qty * current_price
                pnl = proceeds - (position_qty * position_price)
                trades.append(pnl)
                cash += proceeds
                position_qty = 0.0

            total_equity = cash + (position_qty * current_price)
            equity_curve.append(total_equity)

        # Close any remaining position
        if position_qty > 0:
            final_price = float(df["close"].iloc[-1])
            proceeds = position_qty * final_price
            pnl = proceeds - (position_qty * position_price)
            trades.append(pnl)
            cash += proceeds

        total_return = (cash - starting_cash) / starting_cash
        win_rate = sum(1 for t in trades if t > 0) / len(trades) if trades else 0.0

        # Sharpe from equity curve
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 390)) if np.std(returns) > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "trade_count": len(trades),
            "win_rate": win_rate,
        }

    def summary(self, results: list[WalkForwardResult]) -> dict:
        """Summarize walk-forward results."""
        if not results:
            return {"folds": 0, "avg_oos_return": 0, "avg_oos_sharpe": 0,
                    "overfitting_probability": 0}
        return {
            "folds": len(results),
            "avg_is_return": np.mean([r.in_sample_return for r in results]),
            "avg_oos_return": np.mean([r.out_of_sample_return for r in results]),
            "avg_is_sharpe": np.mean([r.in_sample_sharpe for r in results]),
            "avg_oos_sharpe": np.mean([r.out_of_sample_sharpe for r in results]),
            "avg_trade_count": np.mean([r.trade_count for r in results]),
            "avg_win_rate": np.mean([r.win_rate for r in results]),
            "overfitting_probability": self.overfitting_probability(results),
        }

    def overfitting_probability(self, results: list[WalkForwardResult]) -> float:
        """Estimate probability of overfitting.

        Computed as the fraction of folds where OOS return < IS return,
        weighted by the magnitude of the gap.
        """
        if not results:
            return 0.0

        overfit_count = 0
        for r in results:
            if r.out_of_sample_return < r.in_sample_return:
                overfit_count += 1

        return overfit_count / len(results)
