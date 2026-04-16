"""Weekly modifier A/B report agent — Sprint 6D.

Reads the modifier A/B log (`data/modifier_ab/log.jsonl`), joins to the trades
database, computes each modifier's 30-day Sharpe contribution, and — if a
modifier's contribution is <= 0 over at least 5 matched trades — flips its
feature flag to False and alerts via AlertManager.

Runs weekly (invoked from a launchd plist / cron). Safe to run more often;
each run is idempotent because the auto-disable mutation is applied to
Config (in-process) and the next process start reads env defaults again.
To persist the disable beyond the process, the user must also flip the env
var in `.env` (the alert message calls this out explicitly).

The report is also written to `data/modifier_ab/report.json` so the
dashboard can surface it.

Usage:
    python -m agents.modifier_ab_report            # run and mutate Config
    python -m agents.modifier_ab_report --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime

from analytics.modifier_performance import (
    auto_disable_negative_modifiers,
    compute_all_reports,
    format_report_text,
)
from config import Config
from utils.logger import log

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "modifier_ab")
_REPORT_PATH = os.path.join(_DATA_DIR, "report.json")


def _persist_report(reports, disabled: list[str]) -> None:
    """Write report JSON for dashboard consumption. Best-effort."""
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        payload = {
            "generated_at": datetime.now().isoformat(),
            "reports": [asdict(r) for r in reports],
            "auto_disabled": disabled,
        }
        with open(_REPORT_PATH, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except OSError as e:
        log.warning(f"Could not write modifier A/B report JSON: {e}")


def _get_alert_manager():
    """Best-effort AlertManager load. Returns None if not available."""
    try:
        from core.alerting import AlertManager
        return AlertManager()
    except Exception as e:
        log.warning(f"AlertManager not available for A/B report: {e}")
        return None


def main(argv: list[str] | None = None) -> int:
    """Entrypoint. Returns process exit code."""
    parser = argparse.ArgumentParser(description="Weekly modifier A/B report.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute + persist report, but don't flip flags.")
    parser.add_argument(
        "--days", type=int, default=Config.MODIFIER_AB_REPORT_DAYS,
        help="Lookback window in days (default: Config.MODIFIER_AB_REPORT_DAYS).",
    )
    args = parser.parse_args(argv)

    log.info(f"Modifier A/B report running (days={args.days}, "
             f"dry_run={args.dry_run})")

    reports = compute_all_reports(days=args.days)
    log.info("\n" + format_report_text(reports))

    alert_mgr = None if args.dry_run else _get_alert_manager()
    disabled = auto_disable_negative_modifiers(
        reports, alert_manager=alert_mgr, dry_run=args.dry_run,
    )

    _persist_report(reports, disabled)

    if disabled and not args.dry_run:
        log.warning(
            f"Auto-disabled {len(disabled)} modifier(s): {', '.join(disabled)}. "
            "To persist across restarts, also flip the env var to 'false' in .env."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
