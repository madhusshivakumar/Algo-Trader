"""Backtester — test strategies against historical data before going live."""

import pandas as pd
from rich.console import Console
from rich.table import Table

from core.broker import Broker
from strategies.ensemble import compute_signals
from config import Config

console = Console()


class Backtester:
    def __init__(self, starting_cash: float = 100.0):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.position_qty = 0.0
        self.position_avg_price = 0.0
        self.trades: list[dict] = []
        self.equity_curve: list[float] = []

    def run(self, df: pd.DataFrame, symbol: str):
        """Run backtest on historical data."""
        console.print(f"\n[bold]Backtesting {symbol}[/bold] | {len(df)} bars | Starting cash: ${self.starting_cash:.2f}\n")

        window = 100  # bars needed for indicators

        for i in range(window, len(df)):
            slice_df = df.iloc[i - window:i + 1].copy()
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
                qty = size / current_price
                self.position_qty = qty
                self.position_avg_price = current_price
                self.cash -= size
                self.trades.append({
                    "bar": i, "side": "buy", "price": current_price,
                    "qty": qty, "amount": size, "reason": signal["reason"],
                })

            elif signal["action"] == "sell" and self.position_qty > 0:
                # Sell entire position
                proceeds = self.position_qty * current_price
                pnl = proceeds - (self.position_qty * self.position_avg_price)
                self.cash += proceeds
                self.trades.append({
                    "bar": i, "side": "sell", "price": current_price,
                    "qty": self.position_qty, "amount": proceeds,
                    "reason": signal["reason"], "pnl": pnl,
                })
                self.position_qty = 0
                self.position_avg_price = 0

            # Trailing stop check
            elif self.position_qty > 0:
                loss_pct = (current_price - self.position_avg_price) / self.position_avg_price
                if loss_pct <= -Config.STOP_LOSS_PCT:
                    proceeds = self.position_qty * current_price
                    pnl = proceeds - (self.position_qty * self.position_avg_price)
                    self.cash += proceeds
                    self.trades.append({
                        "bar": i, "side": "sell", "price": current_price,
                        "qty": self.position_qty, "amount": proceeds,
                        "reason": "stop-loss", "pnl": pnl,
                    })
                    self.position_qty = 0
                    self.position_avg_price = 0

        # Close any remaining position at last price
        if self.position_qty > 0:
            last_price = float(df["close"].iloc[-1])
            proceeds = self.position_qty * last_price
            pnl = proceeds - (self.position_qty * self.position_avg_price)
            self.cash += proceeds
            self.trades.append({
                "bar": len(df), "side": "sell", "price": last_price,
                "qty": self.position_qty, "amount": proceeds,
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


def main():
    Config.validate()
    broker = Broker()

    for symbol in Config.SYMBOLS:
        console.print(f"Fetching historical data for {symbol}...")
        df = broker.get_historical_bars(symbol, days=Config.CANDLE_HISTORY_DAYS)
        console.print(f"Got {len(df)} bars")

        bt = Backtester(starting_cash=100.0)
        bt.run(df, symbol)


if __name__ == "__main__":
    main()
