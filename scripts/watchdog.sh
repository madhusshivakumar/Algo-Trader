#!/bin/bash
# Health watchdog — checks if algo-engine is running, restarts if not
# Schedule via launchd every 15 minutes

set -uo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$DIR/logs/watchdog.log"
DOCKER="${DOCKER:-docker}"

export PATH="/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin:$PATH"

timestamp() { date "+%Y-%m-%d %H:%M:%S PT"; }

mkdir -p "$DIR/logs"

# Check if Docker is running
if ! $DOCKER info >/dev/null 2>&1; then
    echo "[$(timestamp)] Docker is not running — cannot check engine" >> "$LOG"
    exit 1
fi

# Check if engine container is running.
#
# Sprint 8 (post-incident Apr 21): "container is running" is necessary
# but NOT sufficient. On Mon Apr 20 the container stayed up while the
# Python process inside stopped iterating for ~7 hours, and watchdog
# reported green the whole time. Now we ALSO check the heartbeat file
# during market hours and fire an alert if it's stale.
if $DOCKER ps --format '{{.Names}}' | grep -q "algo-engine"; then
    # Container is up. During market hours, additionally verify the
    # engine is actually iterating by inspecting the heartbeat.
    DOW=$(date +%u); HOUR=$(date +%H); MIN=$(date +%M)
    in_market_hours() {
        [ "$DOW" -gt 5 ] && return 1
        [ "$HOUR" -lt 6 ] && return 1
        [ "$HOUR" -gt 13 ] && return 1
        [ "$HOUR" -eq 13 ] && [ "$MIN" -ge 30 ] && return 1
        return 0
    }
    if in_market_hours && [ -f "$DIR/scripts/check_heartbeat.py" ]; then
        # Run the check; it will fire alerts on stale/missing.
        # Timeout so a hung script can't block the watchdog.
        HB_OUT=$(cd "$DIR" && python3 scripts/check_heartbeat.py 2>&1)
        HB_RC=$?
        if [ $HB_RC -eq 0 ]; then
            echo "[$(timestamp)] Engine alive (container up, heartbeat fresh): $HB_OUT" >> "$LOG"
        else
            echo "[$(timestamp)] WARNING: Engine container up but heartbeat check failed (rc=$HB_RC): $HB_OUT" >> "$LOG"
        fi
    else
        echo "[$(timestamp)] Engine is running (container up; heartbeat check skipped outside market hours)" >> "$LOG"
    fi
    exit 0
fi

# Engine is NOT running — decide whether to restart.
#
# Sprint 7 fix: don't revive the bot outside intended trading hours.
# The shutdown plist stops everything at 13:30 PT; without this guard
# the watchdog brings it back up at 13:45 and the schedule is meaningless.
#
# Allowed uptime window:
#   - Weekdays (Mon-Fri) 06:00 – 13:30 PT
#   - OR any time if sentinel `$DIR/.watchdog_always_on` exists
#     (touch that file for 24/7 operation, e.g. once crypto ATR stops
#     are validated per Sprint 0 plan)

DOW=$(date +%u)      # 1=Monday … 7=Sunday
HOUR=$(date +%H)     # 00–23
MIN=$(date +%M)      # 00–59

in_window() {
    # Returns 0 (true) if now is inside the allowed uptime window.
    [ -f "$DIR/.watchdog_always_on" ] && return 0
    # Weekend
    if [ "$DOW" -gt 5 ]; then return 1; fi
    # Before 06:00
    if [ "$HOUR" -lt 6 ]; then return 1; fi
    # After 13:30 (13:30–13:59 and 14:00+)
    if [ "$HOUR" -gt 13 ]; then return 1; fi
    if [ "$HOUR" -eq 13 ] && [ "$MIN" -ge 30 ]; then return 1; fi
    return 0
}

if ! in_window; then
    echo "[$(timestamp)] Engine down but outside trading window — not restarting (DOW=$DOW ${HOUR}:${MIN}). Touch .watchdog_always_on for 24/7." >> "$LOG"
    exit 0
fi

echo "[$(timestamp)] WARNING: algo-engine is DOWN in trading window — attempting restart" >> "$LOG"

if [ -f "$DIR/scripts/startup.sh" ]; then
    bash "$DIR/scripts/startup.sh" >> "$LOG" 2>&1

    sleep 10
    if $DOCKER ps --format '{{.Names}}' | grep -q "algo-engine"; then
        echo "[$(timestamp)] Engine restarted successfully" >> "$LOG"
    else
        echo "[$(timestamp)] CRITICAL: Engine restart FAILED" >> "$LOG"
        exit 1
    fi
else
    echo "[$(timestamp)] CRITICAL: startup.sh not found at $DIR/scripts/startup.sh" >> "$LOG"
    exit 1
fi
