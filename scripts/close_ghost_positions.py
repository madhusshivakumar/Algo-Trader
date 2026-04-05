#!/usr/bin/env python3
"""One-time script to close orphaned positions with no trailing stops."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.broker import Broker
from config import Config
from utils.logger import log

GHOST_SYMBOLS = ["NVDA", "AAPL", "SPY", "META", "XOM", "XLE"]


def main():
    Config.validate()
    broker = Broker()
    positions = broker.get_positions()

    closed = 0
    for p in positions:
        sym = p["symbol"]
        if sym in GHOST_SYMBOLS:
            mkt_val = abs(float(p.get("market_value", 0)))
            pnl = float(p.get("unrealized_pl", 0))
            log.info(f"Closing ghost position: {sym} | value=${mkt_val:.2f} | PnL=${pnl:.2f}")
            try:
                result = broker.close_position(sym)
                if result:
                    log.info(f"  Closed {sym} successfully")
                    closed += 1
                else:
                    log.error(f"  FAILED to close {sym} — no position found")
            except Exception as e:
                log.error(f"  FAILED to close {sym}: {e}")

    log.info(f"Ghost position cleanup: {closed}/{len(GHOST_SYMBOLS)} closed")


if __name__ == "__main__":
    main()
