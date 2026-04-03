#!/bin/bash
# ============================================================
#  Algo Trader — Reliable Startup Script
#
#  Ensures Docker is running, starts all services, verifies
#  they're healthy, and logs everything. Designed to be called
#  by a scheduler (cron, launchd, Claude Code).
#
#  Exit codes:
#    0 = all services up and healthy
#    1 = partial failure (some services not healthy)
#    2 = Docker not available after retries
# ============================================================

set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

LOG="$DIR/logs/startup.log"
mkdir -p "$DIR/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

log "=========================================="
log "STARTUP: Algo Trader startup initiated"
log "=========================================="

# ── Step 1: Ensure Docker Desktop is running ──────────────────
if ! docker info &>/dev/null; then
    log "Docker not running. Starting Docker Desktop..."
    open -a Docker

    DOCKER_WAIT=0
    DOCKER_TIMEOUT=120
    while ! docker info &>/dev/null; do
        sleep 5
        DOCKER_WAIT=$((DOCKER_WAIT + 5))
        if [ $DOCKER_WAIT -ge $DOCKER_TIMEOUT ]; then
            log "FATAL: Docker Desktop failed to start after ${DOCKER_TIMEOUT}s"
            exit 2
        fi
        log "Waiting for Docker... (${DOCKER_WAIT}s)"
    done
    log "Docker Desktop ready (took ${DOCKER_WAIT}s)"
else
    log "Docker already running"
fi

# ── Step 2: Stop any stale containers ─────────────────────────
if docker compose ps --quiet 2>/dev/null | grep -q .; then
    log "Stopping stale containers from previous run..."
    docker compose down --timeout 30 2>&1 | tee -a "$LOG"
    sleep 2
fi

# ── Step 3: Start services with retry ─────────────────────────
MAX_RETRIES=3
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    log "Starting Docker Compose (attempt $((RETRY+1))/$MAX_RETRIES)..."
    if docker compose up -d --build 2>&1 | tee -a "$LOG"; then
        log "Docker Compose started successfully"
        break
    fi
    RETRY=$((RETRY+1))
    if [ $RETRY -lt $MAX_RETRIES ]; then
        log "WARNING: Startup failed, retrying in 10s..."
        sleep 10
    else
        log "FATAL: Docker Compose failed after $MAX_RETRIES attempts"
        exit 2
    fi
done

# ── Step 4: Wait for engine to be healthy ─────────────────────
log "Waiting for engine to become healthy..."
ENGINE_WAIT=0
ENGINE_TIMEOUT=300  # 5 min (pre-market agents take 2-3 min)
while [ $ENGINE_WAIT -lt $ENGINE_TIMEOUT ]; do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' algo-engine 2>/dev/null || echo "not_found")
    if [ "$STATUS" = "healthy" ]; then
        log "Engine is healthy (took ${ENGINE_WAIT}s)"
        break
    elif [ "$STATUS" = "not_found" ]; then
        log "FATAL: Engine container not found"
        exit 2
    fi
    sleep 10
    ENGINE_WAIT=$((ENGINE_WAIT + 10))
done

if [ $ENGINE_WAIT -ge $ENGINE_TIMEOUT ]; then
    log "WARNING: Engine health check timed out after ${ENGINE_TIMEOUT}s"
    log "Engine status: $(docker inspect --format='{{.State.Health.Status}}' algo-engine 2>/dev/null || echo 'unknown')"
fi

# ── Step 5: Verify all containers are running ─────────────────
FAILURES=0
for SERVICE in algo-engine algo-dashboard algo-agents; do
    STATE=$(docker inspect --format='{{.State.Status}}' "$SERVICE" 2>/dev/null || echo "missing")
    if [ "$STATE" = "running" ]; then
        log "  ✓ $SERVICE: running"
    else
        log "  ✗ $SERVICE: $STATE"
        FAILURES=$((FAILURES+1))
    fi
done

if [ $FAILURES -eq 0 ]; then
    log "=========================================="
    log "STARTUP COMPLETE: All 3 services running"
    log "Dashboard: http://localhost:5050"
    log "=========================================="
    exit 0
else
    log "=========================================="
    log "STARTUP WARNING: $FAILURES service(s) not running"
    log "=========================================="
    exit 1
fi
