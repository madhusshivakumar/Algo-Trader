#!/bin/bash
# ============================================================
#  Algo Trader — Graceful Shutdown Script
#
#  Stops the trading engine gracefully (saves state), then
#  runs post-market agents, and finally stops all containers.
#
#  Called by scheduler after market close (4:05 PM ET).
# ============================================================

set -uo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

# Ensure Docker tools (including docker-credential-desktop) are in PATH
export PATH="/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin:$PATH"

DOCKER="/Applications/Docker.app/Contents/Resources/bin/docker"
LOG="$DIR/logs/shutdown.log"
mkdir -p "$DIR/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

log "=========================================="
log "SHUTDOWN: Market close shutdown initiated"
log "=========================================="

# ── Step 1: Gracefully stop the engine (sends SIGTERM → saves state) ──
if $DOCKER inspect algo-engine &>/dev/null; then
    log "Stopping engine gracefully (SIGTERM)..."
    $DOCKER stop --timeout 30 algo-engine 2>&1 | tee -a "$LOG"
    log "Engine stopped"
else
    log "Engine container not found (already stopped?)"
fi

# ── Step 2: Run post-market agents in the agents container ──
if $DOCKER inspect algo-agents &>/dev/null; then
    log "Running post-market agents..."

    log "  Running trade_analyzer..."
    $DOCKER exec algo-agents python agents/trade_analyzer.py >> "$LOG" 2>&1 || \
        log "  WARNING: trade_analyzer failed"

    log "  Running conviction_scorer..."
    $DOCKER exec algo-agents python agents/conviction_scorer.py >> "$LOG" 2>&1 || \
        log "  WARNING: conviction_scorer failed"

    log "  Running health_check..."
    $DOCKER exec algo-agents python agents/health_check.py >> "$LOG" 2>&1 || \
        log "  WARNING: health_check failed"

    log "Post-market agents complete"
else
    log "WARNING: Agents container not found, skipping post-market agents"
fi

# ── Step 3: Stop all remaining containers ──
log "Stopping all containers..."
$DOCKER compose down --timeout 15 2>&1 | tee -a "$LOG"

log "=========================================="
log "SHUTDOWN COMPLETE: All services stopped"
log "=========================================="
