#!/usr/bin/env python3
"""Sprint 5: smoke test for AlertManager configuration.

Run this after configuring `SLACK_WEBHOOK_URL` or `DISCORD_WEBHOOK_URL` in
`.env` to confirm alerts actually deliver. Fires one alert of each level so
you can see them in your channel.

Usage:
    python scripts/alert_test.py            # send INFO + WARNING + CRITICAL
    python scripts/alert_test.py --level=info
"""

from __future__ import annotations

import argparse
import sys
import time

# Allow running directly from project root
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.alerting import AlertLevel, AlertManager
from utils.logger import log


def main():
    parser = argparse.ArgumentParser(description="AlertManager smoke test")
    parser.add_argument("--level", choices=["info", "warning", "critical", "all"],
                       default="all", help="Which alert level to fire")
    args = parser.parse_args()

    am = AlertManager()
    if not am.channels:
        log.warning("No alert channels configured. Set SLACK_WEBHOOK_URL or "
                    "DISCORD_WEBHOOK_URL in .env.")
        return 1

    log.info(f"Configured channels: {[c.__class__.__name__ for c in am.channels]}")

    if args.level in ("info", "all"):
        log.info("Sending INFO alert...")
        am.alert("smoke_test", "Algo-trader alert smoke test (INFO)",
                AlertLevel.INFO, {"Test type": "info"})

    if args.level in ("warning", "all"):
        log.info("Sending WARNING alert...")
        am.alert("smoke_test", "Algo-trader alert smoke test (WARNING)",
                AlertLevel.WARNING, {"Test type": "warning"})

    if args.level in ("critical", "all"):
        log.info("Sending CRITICAL alert...")
        am.alert("smoke_test", "Algo-trader alert smoke test (CRITICAL)",
                AlertLevel.CRITICAL, {"Test type": "critical"})

    # Alerts run in daemon threads — give them a beat to flush
    time.sleep(2)
    log.info("Done. Check your Slack/Discord channels for the test alerts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
