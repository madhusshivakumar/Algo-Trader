#!/usr/bin/env python3
"""Liquidate all open positions and cancel all open orders.

Sells every position at market price and cancels pending orders,
freeing the full equity pool for fresh algo trading deployment.

Usage:
    python scripts/liquidate_all.py [--dry-run]
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from core.broker import Broker
from utils.logger import log


def cancel_all_orders(broker: Broker, dry_run: bool) -> int:
    """Cancel all open/pending orders. Returns count cancelled."""
    try:
        orders = broker.trading_client.get_orders()
    except Exception as e:
        log.error(f"Failed to fetch open orders: {e}")
        return 0

    cancelled = 0
    for order in orders:
        sym = getattr(order, "symbol", "?")
        oid = str(order.id)
        if dry_run:
            log.info(f"  [DRY-RUN] Would cancel order {oid} for {sym}")
            cancelled += 1
            continue
        try:
            broker.trading_client.cancel_order_by_id(oid)
            log.info(f"  Cancelled order {oid} for {sym}")
            cancelled += 1
        except Exception as e:
            log.warning(f"  Could not cancel order {oid} for {sym}: {e}")

    return cancelled


def close_all_positions(broker: Broker, dry_run: bool) -> tuple[int, int, float]:
    """Close every open position at market. Returns (closed, failed, total_value)."""
    positions = broker.get_positions()

    if not positions:
        log.info("  No open positions found.")
        return 0, 0, 0.0

    closed = failed = 0
    total_value = 0.0

    for p in positions:
        sym = p["symbol"]
        qty = p["qty"]
        mkt_val = abs(float(p.get("market_value", 0)))
        pnl = float(p.get("unrealized_pl", 0))
        pnl_pct = float(p.get("unrealized_plpc", 0)) * 100
        total_value += mkt_val

        pnl_str = f"{'+'if pnl >= 0 else ''}{pnl:.2f} ({pnl_pct:+.2f}%)"
        log.info(f"  {sym:8s} | qty: {qty:12.6f} | value: ${mkt_val:>10,.2f} | PnL: {pnl_str}")

        if dry_run:
            log.info(f"  [DRY-RUN] Would close {sym}")
            closed += 1
            continue

        result = broker.close_position(sym)
        if result:
            log.info(f"    ✓ Market sell submitted — order {result['id']}")
            closed += 1
        else:
            log.error(f"    ✗ FAILED to close {sym}")
            failed += 1

        # Small pause to avoid hammering the API
        time.sleep(0.3)

    return closed, failed, total_value


def verify_final_state(broker: Broker) -> None:
    """Poll until all positions are confirmed closed, then print account summary."""
    log.info("\nWaiting for fills to confirm...")
    max_wait = 30  # seconds
    interval = 3
    elapsed = 0

    while elapsed < max_wait:
        time.sleep(interval)
        elapsed += interval
        remaining = broker.get_positions()
        if not remaining:
            log.info("  ✓ All positions confirmed closed")
            break
        syms = [p["symbol"] for p in remaining]
        log.info(f"  Still open ({elapsed}s): {', '.join(syms)}")
    else:
        remaining = broker.get_positions()
        if remaining:
            syms = [p["symbol"] for p in remaining]
            log.warning(f"  ⚠ Some positions still open after {max_wait}s: {', '.join(syms)}")
            log.warning("  They may still be filling — check the dashboard.")

    account = broker.get_account()
    log.info("\n── Final Account State ──────────────────────────────")
    log.info(f"  Equity:        ${account['equity']:>12,.2f}")
    log.info(f"  Cash:          ${account['cash']:>12,.2f}")
    log.info(f"  Buying Power:  ${account['buying_power']:>12,.2f}")
    log.info(f"  Day trades:    {account['daytrade_count']}")
    log.info("────────────────────────────────────────────────────")
    log.info("  Full equity pool is now free for fresh deployment.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Liquidate all positions and cancel all orders")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without submitting any orders")
    args = parser.parse_args()

    Config.validate()
    broker = Broker()

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    log.info(f"[Liquidate All — {mode}] Starting at {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Mode: {'paper' if Config.is_paper() else 'LIVE'} trading")

    # Step 1 — cancel all open orders first (prevents new fills during liquidation)
    log.info("\nStep 1: Cancelling all open orders...")
    n_cancelled = cancel_all_orders(broker, args.dry_run)
    log.info(f"  {n_cancelled} orders cancelled")

    # Step 2 — close all positions
    account_before = broker.get_account()
    log.info(f"\nStep 2: Closing all positions (equity before: ${account_before['equity']:,.2f})...")
    closed, failed, total_value = close_all_positions(broker, args.dry_run)
    log.info(f"\n  Submitted: {closed} | Failed: {failed} | Position value: ${total_value:,.2f}")

    if failed > 0:
        log.error(f"  ⚠ {failed} positions could NOT be closed — check broker dashboard")

    # Step 3 — confirm and print final state
    if not args.dry_run:
        verify_final_state(broker)
    else:
        log.info("\n[DRY-RUN] No orders were submitted. Run without --dry-run to execute.")


if __name__ == "__main__":
    main()
