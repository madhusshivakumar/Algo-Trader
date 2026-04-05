"""Backtester — test strategies against historical data before going live."""

import argparse

import pandas as pd
from rich.console import Console
from rich.table import Table

from core.broker import Broker
from strategies.ensemble import compute_signals
from config import Config
from core.transaction_costs import TransactionCostModel

console = Console()


class Backtester:
    def __init__(self, starting_cash: float = 100.0):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.position_qty = 0.0
        self.position_avg_price = 0.0
        self.trades: list[dict] = []
        self.equity_curve: list[float] = []
        self.cost_model = TransactionCostModel() if Config.TC_ENABLED else None
        self.total_costs = 0.0

    def run(self, df: pd.DataFrame, symbol: str):
        """Run backtest on historical data."""
        console.print(f"\n[bold]Backtesting {symbol}[/bold] | {len(df)} bars | Starting cash: ${self.starting_cash:.2f}\n")

        window = 100  # bars needed for indicators

        for i in range(window, len(df)):
            # Use bars [i-window, i) for signals — bar i is NOT included in indicator calc
            slice_df = df.iloc[i - window:i].copy()
            # Current price is the LAST bar in the slice (bar i-1), which we're "reacting to"
            current_price = float(slice_df["close"].iloc[-1])

            signal = compute_signals(slice_df)

            # Track equity
            equity = self.cash + (self.position_qty * current_price)
            self.equity_curve.append(equity)

            if signal["action"] == "buy" and self.position_qty == 0:
                # Buy with configured position size
                size = self.cash * Config.MAX_POSITION_PCT * signal["strength"]
                size = min(size, self.cash - 0.01)
                if size < 1:
                    continue
                # Apply transaction costs
                if self.cost_model:
                    costs = self.cost_model.estimate(symbol, size)
                    net_size = size - costs.total_cost
                    self.total_costs += costs.total_cost
                    if net_size < 1:
                        continue
                else:
                    net_size = size
                qty = net_size / current_price
                self.position_qty = qty
                self.position_avg_price = current_price
                self.cash -= size  # Pay full amount (costs included)
                self.trades.append({
                    "bar": i, "side": "buy", "price": current_price,
                    "qty": qty, "amount": net_size, "reason": signal["reason"],
                })

            elif signal["action"] == "sell" and self.position_qty > 0:
                # Sell entire position
                proceeds = self.position_qty * current_price
                if self.cost_model:
                    costs = self.cost_model.estimate(symbol, proceeds)
                    net_proceeds = proceeds - costs.total_cost
                    self.total_costs += costs.total_cost
                else:
                    net_proceeds = proceeds
                pnl = net_proceeds - (self.position_qty * self.position_avg_price)
                self.cash += net_proceeds
                self.trades.append({
                    "bar": i, "side": "sell", "price": current_price,
                    "qty": self.position_qty, "amount": net_proceeds,
                    "reason": signal["reason"], "pnl": pnl,
                })
                self.position_qty = 0
                self.position_avg_price = 0

            # Trailing stop check
            elif self.position_qty > 0:
                loss_pct = (current_price - self.position_avg_price) / self.position_avg_price
                if loss_pct <= -Config.STOP_LOSS_PCT:
                    proceeds = self.position_qty * current_price
                    if self.cost_model:
                        costs = self.cost_model.estimate(symbol, proceeds)
                        net_proceeds = proceeds - costs.total_cost
                        self.total_costs += costs.total_cost
                    else:
                        net_proceeds = proceeds
                    pnl = net_proceeds - (self.position_qty * self.position_avg_price)
                    self.cash += net_proceeds
                    self.trades.append({
                        "bar": i, "side": "sell", "price": current_price,
                        "qty": self.position_qty, "amount": net_proceeds,
                        "reason": "stop-loss", "pnl": pnl,
                    })
                    self.position_qty = 0
                    self.position_avg_price = 0

        # Close any remaining position at last price
        if self.position_qty > 0:
            last_price = float(df["close"].iloc[-1])
            proceeds = self.position_qty * last_price
            if self.cost_model:
                costs = self.cost_model.estimate(symbol, proceeds)
                net_proceeds = proceeds - costs.total_cost
                self.total_costs += costs.total_cost
            else:
                net_proceeds = proceeds
            pnl = net_proceeds - (self.position_qty * self.position_avg_price)
            self.cash += net_proceeds
            self.trades.append({
                "bar": len(df), "side": "sell", "price": last_price,
                "qty": self.position_qty, "amount": net_proceeds,
                "reason": "backtest end", "pnl": pnl,
            })
            self.position_qty = 0

        self._print_results(symbol)

    def _print_results(self, symbol: str):
        final_equity = self.cash
        total_return = (final_equity - self.starting_cash) / self.starting_cash * 100
        num_trades = len([t for t in self.trades if t["side"] == "buy"])
        wins = len([t for t in self.trades if t.get("pnl", 0) > 0])
        losses = len([t for t in self.trades if t.get("pnl", 0) < 0])
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

        total_pnl = sum(t.get("pnl", 0) for t in self.trades)
        avg_win = 0
        avg_loss = 0
        if wins:
            avg_win = sum(t["pnl"] for t in self.trades if t.get("pnl", 0) > 0) / wins
        if losses:
            avg_loss = sum(t["pnl"] for t in self.trades if t.get("pnl", 0) < 0) / losses

        # Max drawdown
        max_dd = 0
        peak = self.starting_cash
        for eq in self.equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        console.print(f"\n[bold]{'='*50}[/bold]")
        console.print(f"[bold]Backtest Results: {symbol}[/bold]")
        console.print(f"[bold]{'='*50}[/bold]")
        console.print(f"Final Equity:    ${final_equity:.2f}")
        console.print(f"Total Return:    {total_return:+.2f}%")
        console.print(f"Total PnL:       ${total_pnl:+.2f}")
        console.print(f"Trades:          {num_trades}")
        console.print(f"Win Rate:        {win_rate:.0f}% ({wins}W / {losses}L)")
        console.print(f"Avg Win:         ${avg_win:+.2f}")
        console.print(f"Avg Loss:        ${avg_loss:+.2f}")
        console.print(f"Max Drawdown:    {max_dd:.1%}")
        if self.total_costs > 0:
            console.print(f"Total Costs:     ${self.total_costs:.2f}")

        # Buy and hold comparison
        if len(self.equity_curve) > 1:
            # Not directly available, but we can estimate
            console.print(f"\n[dim]Equity curve: {len(self.equity_curve)} data points[/dim]")

        # Show last 10 trades
        sell_trades = [t for t in self.trades if t["side"] == "sell"]
        if sell_trades:
            table = Table(title="Recent Trades (last 10)")
            table.add_column("Side")
            table.add_column("Price")
            table.add_column("Amount")
            table.add_column("PnL")
            table.add_column("Reason")
            for t in sell_trades[-10:]:
                pnl_str = f"${t.get('pnl', 0):+.2f}" if "pnl" in t else "-"
                table.add_row(t["side"], f"${t['price']:.2f}", f"${t['amount']:.2f}", pnl_str, t["reason"])
            console.print(table)


def _run_walk_forward(broker: Broker, symbols: list[str]):
    """Run walk-forward analysis for the given symbols."""
    from core.walk_forward import WalkForwardBacktester
    from strategies.router import get_strategy

    for symbol in symbols:
        console.print(f"\nFetching historical data for {symbol}...")
        df = broker.get_historical_bars(symbol, days=Config.CANDLE_HISTORY_DAYS)
        console.print(f"Got {len(df)} bars")

        strategy_name, strategy_fn = get_strategy(symbol)
        wf = WalkForwardBacktester(
            strategy_fn=strategy_fn,
            strategy_name=strategy_name,
        )
        try:
            results = wf.run(df, symbol)
        except Exception as e:
            console.print(f"[red]Error running walk-forward on {symbol}: {e}[/red]")
            continue
        if not results:
            console.print(f"[yellow]Not enough data for walk-forward on {symbol}[/yellow]")
            continue

        summary = wf.summary(results)

        console.print(f"\n[bold]Walk-Forward Results: {symbol} ({strategy_name})[/bold]")
        console.print(f"Folds:                   {summary['folds']}")
        console.print(f"Avg In-Sample Return:    {summary['avg_is_return']:+.2%}")
        console.print(f"Avg Out-of-Sample Return:{summary['avg_oos_return']:+.2%}")
        console.print(f"Avg IS Sharpe:           {summary['avg_is_sharpe']:.2f}")
        console.print(f"Avg OOS Sharpe:          {summary['avg_oos_sharpe']:.2f}")
        console.print(f"Avg Trade Count:         {summary['avg_trade_count']:.0f}")
        console.print(f"Avg Win Rate:            {summary['avg_win_rate']:.0%}")
        console.print(f"Overfitting Probability: {summary['overfitting_probability']:.0%}")

        # Show per-fold table
        table = Table(title=f"Walk-Forward Folds: {symbol}")
        table.add_column("Fold")
        table.add_column("IS Return")
        table.add_column("OOS Return")
        table.add_column("IS Sharpe")
        table.add_column("OOS Sharpe")
        table.add_column("Trades")
        table.add_column("Win Rate")
        for r in results:
            table.add_row(
                str(r.fold),
                f"{r.in_sample_return:+.2%}",
                f"{r.out_of_sample_return:+.2%}",
                f"{r.in_sample_sharpe:.2f}",
                f"{r.out_of_sample_sharpe:.2f}",
                str(r.trade_count),
                f"{r.win_rate:.0%}",
            )
        console.print(table)


def _run_monte_carlo(broker: Broker, symbols: list[str]):
    """Run backtest then Monte Carlo simulation on the resulting equity curve."""
    from core.monte_carlo import run_from_equity_curve, format_report

    for symbol in symbols:
        console.print(f"\nFetching historical data for {symbol}...")
        df = broker.get_historical_bars(symbol, days=Config.CANDLE_HISTORY_DAYS)
        console.print(f"Got {len(df)} bars")

        bt = Backtester(starting_cash=100.0)
        bt.run(df, symbol)

        if len(bt.equity_curve) < 10:
            console.print(f"[yellow]Not enough equity data for Monte Carlo on {symbol}[/yellow]")
            continue

        try:
            result = run_from_equity_curve(bt.equity_curve)
            console.print(f"\n[bold]Monte Carlo Analysis: {symbol}[/bold]")
            console.print(format_report(result))
        except Exception as e:
            console.print(f"[red]Error running Monte Carlo on {symbol}: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="Backtest trading strategies")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward analysis instead of standard backtest")
    parser.add_argument("--monte-carlo", action="store_true",
                        help="Run Monte Carlo simulation after backtest")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to backtest (default: all configured symbols)")
    args = parser.parse_args()

    Config.validate()
    broker = Broker()
    symbols = args.symbols or Config.SYMBOLS

    if args.walk_forward:
        _run_walk_forward(broker, symbols)
    elif args.monte_carlo:
        _run_monte_carlo(broker, symbols)
    else:
        for symbol in symbols:
            console.print(f"Fetching historical data for {symbol}...")
            df = broker.get_historical_bars(symbol, days=Config.CANDLE_HISTORY_DAYS)
            console.print(f"Got {len(df)} bars")

            bt = Backtester(starting_cash=100.0)
            bt.run(df, symbol)


if __name__ == "__main__":
    main()
