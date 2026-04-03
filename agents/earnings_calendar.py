"""Earnings Calendar Agent — fetches upcoming earnings dates for equity symbols.

Runs daily pre-market. Queries the Alpaca corporate actions API for upcoming
earnings announcements, writes results to data/earnings_calendar/output.json
for signal_modifiers to read.

Symbols with no known earnings date get in_blackout=false (safe default).
Crypto symbols are skipped (no earnings).
"""

import json
import os
import sys
from datetime import datetime, timedelta

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from config import Config
from utils.logger import log

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "earnings_calendar")
_OUTPUT_FILE = os.path.join(_DATA_DIR, "output.json")


def _ensure_data_dir():
    os.makedirs(_DATA_DIR, exist_ok=True)


def fetch_earnings(symbol: str, lookahead_days: int = 30) -> dict | None:
    """Fetch next earnings date for a symbol via Alpaca corporate actions API.

    Returns dict with next_earnings_date (str or None) and days_until (int or None),
    or None on error.
    """
    try:
        today = datetime.now(Config.MARKET_TZ).date()
        end_date = today + timedelta(days=lookahead_days)

        # Alpaca corporate actions announcements endpoint
        base_url = ("https://paper-api.alpaca.markets" if Config.is_paper()
                     else "https://api.alpaca.markets")
        url = f"{base_url}/v1/corporate-actions/announcements"
        headers = {
            "APCA-API-KEY-ID": Config.ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": Config.ALPACA_SECRET_KEY,
        }
        params = {
            "ca_types": "Earnings",
            "since": today.isoformat(),
            "until": end_date.isoformat(),
            "symbol": symbol,
        }

        resp = requests.get(url, headers=headers, params=params, timeout=10)

        if resp.status_code == 200:
            announcements = resp.json()
            if announcements:
                # Find the earliest upcoming earnings date
                earliest = None
                for ann in announcements:
                    date_str = ann.get("date") or ann.get("record_date")
                    if date_str:
                        try:
                            d = datetime.strptime(date_str, "%Y-%m-%d").date()
                            if d >= today and (earliest is None or d < earliest):
                                earliest = d
                        except ValueError:
                            continue

                if earliest:
                    days_until = (earliest - today).days
                    return {
                        "next_earnings_date": earliest.isoformat(),
                        "days_until": days_until,
                    }

            return {"next_earnings_date": None, "days_until": None}

        elif resp.status_code == 404:
            # No announcements endpoint — fall back gracefully
            return {"next_earnings_date": None, "days_until": None}
        else:
            log.warning(f"Earnings API returned {resp.status_code} for {symbol}")
            return None

    except requests.RequestException as e:
        log.error(f"Error fetching earnings for {symbol}: {e}")
        return None


def run_analysis(symbols: list[str] | None = None,
                 blackout_days: int | None = None) -> dict:
    """Run earnings calendar check for all equity symbols.

    Returns the full output dict to be written to disk.
    """
    if symbols is None:
        symbols = Config.EQUITY_SYMBOLS
    if blackout_days is None:
        blackout_days = Config.EARNINGS_BLACKOUT_DAYS

    earnings = {}
    for symbol in symbols:
        # Skip crypto — no earnings
        if Config.is_crypto(symbol):
            continue

        result = fetch_earnings(symbol)
        if result is not None:
            days_until = result.get("days_until")
            in_blackout = (
                days_until is not None
                and days_until <= blackout_days
            )
            earnings[symbol] = {
                "next_earnings_date": result.get("next_earnings_date"),
                "days_until": days_until,
                "in_blackout": in_blackout,
            }
        else:
            # API error — safe default: not in blackout
            earnings[symbol] = {
                "next_earnings_date": None,
                "days_until": None,
                "in_blackout": False,
            }

    return {
        "timestamp": datetime.now().isoformat(),
        "blackout_days": blackout_days,
        "earnings": earnings,
    }


def write_output(data: dict):
    """Write earnings data to JSON for signal_modifiers to read."""
    _ensure_data_dir()
    with open(_OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    log.success(f"Wrote earnings calendar for {len(data.get('earnings', {}))} symbols")


def main():
    """Main entry point — run full earnings calendar pipeline."""
    log.info("=" * 50)
    log.info("Earnings Calendar Agent starting")
    log.info("=" * 50)

    if not Config.EARNINGS_CALENDAR_ENABLED:
        log.warning("EARNINGS_CALENDAR_ENABLED=false, running anyway (agent invoked directly)")

    data = run_analysis()
    write_output(data)

    earnings = data.get("earnings", {})
    blackout_count = sum(1 for e in earnings.values() if e.get("in_blackout"))
    log.success(
        f"Earnings calendar complete: {len(earnings)} symbols checked, "
        f"{blackout_count} in blackout window"
    )
    return data


if __name__ == "__main__":
    main()
