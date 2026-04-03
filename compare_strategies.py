"""Strategy Comparison — backtest all strategies and rank them."""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from rich.console import Console
from rich.table import Table

from core.broker import Broker
from config import Config

# Import all strategies
from strategies import (
    momentum,
    mean_reversion,
    ensemble,
    scalper,
    macd_crossover,
    triple_ema,
    rsi_divergence,
    volume_profile,
    mean_reversion_aggressive,
)

console = Console()

STRATEGIES = {
    "Momentum Breakout":     momentum.compute_signals,
    "Mean Reversion (RSI+BB)": mean_reversion.compute_signals,
    "Mean Rev Aggressive":   mean_reversion_aggressive.compute_signals,
    "Ensemble (Mom+MR)":     ensemble.compute_signals,
    "Scalper (EMA+VWAP)":    scalper.compute_signals,
    "MACD Crossover":        macd_crossover.compute_signals,
    "Triple EMA Trend":      triple_ema.compute_signals,
    "RSI Divergence":        rsi_divergence.compute_signals,
    "Volume Profile":        volume_profile.compute_signals,
}


def backtest_strategy(strategy_fn, df: pd.DataFrame, starting_cash: float = 100.0,
                      stop_loss_pct: float = 0.025, trailing_stop_pct: float = 0.02,
                      symbol: str = "UNKNOWN") -> dict:
    """Run a single strategy through the backtester. Returns metrics dict."""
    from core.transaction_costs import TransactionCostModel
    cost_model = TransactionCostModel() if Config.TC_ENABLED else None
    total_costs = 0.0
    cash = starting_cash
    position_qty = 0.0
    position_avg_price = 0.0
    highest_since_entry = 0.0
    trades = []
    equity_curve = []
    window = 100

    for i in range(window, len(df)):
        slice_df = df.iloc[i - window:i + 1].copy()
        current_price = float(slice_df["close"].iloc[-1])
        equity = cash + (position_qty * current_price)
        equity_curve.append(equity)

        # Check trailing stop first
        if position_qty > 0:
            highest_since_entry = max(highest_since_entry, current_price)
            trailing_stop_price = highest_since_entry * (1 - trailing_stop_pct)
            hard_stop_price = position_avg_price * (1 - stop_loss_pct)

            if current_price <= hard_stop_price:
                proceeds = position_qty * current_price
                if cost_model:
                    costs = cost_model.estimate(symbol, proceeds)
                    proceeds -= costs.total_cost
                    total_costs += costs.total_cost
                pnl = proceeds - (position_qty * position_avg_price)
                cash += proceeds
                trades.append({"side": "sell", "price": current_price, "pnl": pnl, "reason": "hard stop"})
                position_qty = 0
                continue

            if current_price <= trailing_stop_price:
                proceeds = position_qty * current_price
                if cost_model:
                    costs = cost_model.estimate(symbol, proceeds)
                    proceeds -= costs.total_cost
                    total_costs += costs.total_cost
                pnl = proceeds - (position_qty * position_avg_price)
                cash += proceeds
                trades.append({"side": "sell", "price": current_price, "pnl": pnl, "reason": "trailing stop"})
                position_qty = 0
                continue

        try:
            signal = strategy_fn(slice_df)
        except Exception:
            continue

        if signal["action"] == "buy" and position_qty == 0:
            size = cash * Config.MAX_POSITION_PCT * signal["strength"]
            size = min(size, cash - 0.01)
            if size < 1:
                continue
            if cost_model:
                costs = cost_model.estimate(symbol, size)
                net_size = size - costs.total_cost
                total_costs += costs.total_cost
                if net_size < 1:
                    continue
            else:
                net_size = size
            qty = net_size / current_price
            position_qty = qty
            position_avg_price = current_price
            highest_since_entry = current_price
            cash -= size
            trades.append({"side": "buy", "price": current_price, "amount": net_size})

        elif signal["action"] == "sell" and position_qty > 0:
            proceeds = position_qty * current_price
            if cost_model:
                costs = cost_model.estimate(symbol, proceeds)
                proceeds -= costs.total_cost
                total_costs += costs.total_cost
            pnl = proceeds - (position_qty * position_avg_price)
            cash += proceeds
            trades.append({"side": "sell", "price": current_price, "pnl": pnl, "reason": signal["reason"]})
            position_qty = 0

    # Close any open position
    if position_qty > 0:
        last_price = float(df["close"].iloc[-1])
        proceeds = position_qty * last_price
        if cost_model:
            costs = cost_model.estimate(symbol, proceeds)
            proceeds -= costs.total_cost
            total_costs += costs.total_cost
        pnl = proceeds - (position_qty * position_avg_price)
        cash += proceeds
        trades.append({"side": "sell", "price": last_price, "pnl": pnl, "reason": "end"})

    # Compute metrics
    final_equity = cash
    total_return = (final_equity - starting_cash) / starting_cash * 100
    buy_trades = [t for t in trades if t["side"] == "buy"]
    sell_trades = [t for t in trades if t["side"] == "sell"]
    wins = [t for t in sell_trades if t.get("pnl", 0) > 0]
    losses = [t for t in sell_trades if t.get("pnl", 0) < 0]
    num_trades = len(buy_trades)
    win_rate = (len(wins) / len(sell_trades) * 100) if sell_trades else 0
    total_pnl = sum(t.get("pnl", 0) for t in trades)

    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0

    # Max drawdown
    max_dd = 0
    peak = starting_cash
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    # Sharpe ratio (annualized, assuming 1-min bars)
    if len(equity_curve) > 1:
        returns = pd.Series(equity_curve).pct_change().dropna()
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * (525600 ** 0.5)  # annualized for 1-min
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Profit factor
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    # Buy and hold comparison
    first_price = float(df["close"].iloc[window])
    last_price = float(df["close"].iloc[-1])
    buy_hold_return = (last_price - first_price) / first_price * 100

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "wins": len(wins),
        "losses": len(losses),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "profit_factor": profit_factor,
        "buy_hold_return": buy_hold_return,
    }


def main():
    Config.validate()
    broker = Broker()

    all_symbols = Config.CRYPTO_SYMBOLS + Config.EQUITY_SYMBOLS
    for symbol in all_symbols:
        console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
        console.print(f"[bold cyan]  Fetching data for {symbol}...[/bold cyan]")
        console.print(f"[bold cyan]{'='*70}[/bold cyan]")

        df = broker.get_historical_bars(symbol, days=Config.CANDLE_HISTORY_DAYS)
        console.print(f"  Got {len(df)} bars ({Config.CANDLE_HISTORY_DAYS} days of 1-min data)\n")

        results = {}
        for name, strategy_fn in STRATEGIES.items():
            console.print(f"  Testing [bold]{name}[/bold]...", end=" ")
            metrics = backtest_strategy(strategy_fn, df)
            results[name] = metrics
            icon = "[green]+[/green]" if metrics["total_return"] > 0 else "[red]-[/red]"
            console.print(f"{icon} {metrics['total_return']:+.2f}% ({metrics['num_trades']} trades)")

        # Sort by total return
        ranked = sorted(results.items(), key=lambda x: x[1]["total_return"], reverse=True)

        # Main comparison table
        table = Table(title=f"\n Strategy Comparison — {symbol} (30-day backtest, $100 start)")
        table.add_column("#", style="dim", width=3)
        table.add_column("Strategy", width=24)
        table.add_column("Return", justify="right", width=9)
        table.add_column("PnL", justify="right", width=9)
        table.add_column("Trades", justify="right", width=7)
        table.add_column("Win Rate", justify="right", width=9)
        table.add_column("W/L", justify="right", width=7)
        table.add_column("Avg Win", justify="right", width=9)
        table.add_column("Avg Loss", justify="right", width=9)
        table.add_column("Max DD", justify="right", width=8)
        table.add_column("Sharpe", justify="right", width=8)
        table.add_column("PF", justify="right", width=6)

        for i, (name, m) in enumerate(ranked):
            ret_color = "green" if m["total_return"] > 0 else "red"
            pf_str = f"{m['profit_factor']:.1f}" if m["profit_factor"] < 100 else "inf"

            table.add_row(
                str(i + 1),
                name,
                f"[{ret_color}]{m['total_return']:+.2f}%[/{ret_color}]",
                f"[{ret_color}]${m['total_pnl']:+.2f}[/{ret_color}]",
                str(m["num_trades"]),
                f"{m['win_rate']:.0f}%",
                f"{m['wins']}/{m['losses']}",
                f"${m['avg_win']:.2f}",
                f"${m['avg_loss']:.2f}",
                f"{m['max_drawdown']:.1%}",
                f"{m['sharpe_ratio']:.2f}",
                pf_str,
            )

        console.print(table)

        # Buy & hold baseline
        bh = ranked[0][1]["buy_hold_return"]
        console.print(f"\n  [dim]Buy & Hold baseline: {bh:+.2f}%[/dim]")
        best_name = ranked[0][0]
        best_ret = ranked[0][1]["total_return"]
        beat_bh = best_ret > bh
        console.print(
            f"  [{'green' if beat_bh else 'yellow'}]Best strategy: {best_name} "
            f"({best_ret:+.2f}%) {'beats' if beat_bh else 'underperforms'} buy & hold[/{'green' if beat_bh else 'yellow'}]"
        )

        # Detailed stats for top 3
        console.print(f"\n[bold]  Top 3 Detailed:[/bold]")
        for i, (name, m) in enumerate(ranked[:3]):
            console.print(f"  {i+1}. [bold]{name}[/bold]")
            console.print(f"     Final equity: ${m['final_equity']:.2f} | Trades: {m['num_trades']} | "
                          f"Win rate: {m['win_rate']:.0f}% | Sharpe: {m['sharpe_ratio']:.2f}")
            console.print(f"     Profit factor: {m['profit_factor']:.2f} | Max drawdown: {m['max_drawdown']:.1%}")
            console.print()


if __name__ == "__main__":
    main()
