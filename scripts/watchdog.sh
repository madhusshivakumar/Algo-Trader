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

# Check if engine container is running
if $DOCKER ps --format '{{.Names}}' | grep -q "algo-engine"; then
    echo "[$(timestamp)] Engine is running" >> "$LOG"
    exit 0
fi

# Engine is NOT running — attempt restart
echo "[$(timestamp)] WARNING: algo-engine is DOWN — attempting restart" >> "$LOG"

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
