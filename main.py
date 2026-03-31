"""Algo Trader — Entry point.

Usage:
    python main.py              # Run the live/paper trading bot
    python main.py --backtest   # Run backtester on historical data
    python main.py --status     # Show account status and recent trades
"""

import sys
from config import Config
from utils.logger import log


def main():
    args = sys.argv[1:]

    Config.validate()

    if "--backtest" in args:
        from backtest import main as run_backtest
        run_backtest()

    elif "--status" in args:
        from core.broker import Broker
        broker = Broker()
        account = broker.get_account()
        positions = broker.get_positions()
        orders = broker.get_recent_orders(10)

        log.info(f"Equity: ${account['equity']:.2f} | Cash: ${account['cash']:.2f}")

        if positions:
            log.info("Open positions:")
            for p in positions:
                log.info(f"  {p['symbol']}: {p['qty']:.6f} units "
                         f"@ ${p['avg_entry_price']:.2f} → ${p['current_price']:.2f} "
                         f"(PnL: ${p['unrealized_pl']:.2f})")
        else:
            log.info("No open positions")

        log.print_summary()

    else:
        from core.engine import TradingEngine
        engine = TradingEngine()
        engine.run(interval_seconds=60)


if __name__ == "__main__":
    main()
