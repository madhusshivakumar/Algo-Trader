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

# Ensure Docker tools (including docker-credential-desktop) are in PATH
export PATH="/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin:$PATH"

DOCKER="/Applications/Docker.app/Contents/Resources/bin/docker"
LOG="$DIR/logs/startup.log"
mkdir -p "$DIR/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

log "=========================================="
log "STARTUP: Algo Trader startup initiated"
log "=========================================="

# ── Step 0: Kill any local bot process (leftover from non-Docker runs) ──
if [ -f "$DIR/.bot.pid" ]; then
    OLD_PID=$(cat "$DIR/.bot.pid")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        log "Killing leftover local bot (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$DIR/.bot.pid"
fi

# ── Step 1: Ensure Docker Desktop is running ──────────────────
if ! $DOCKER info &>/dev/null; then
    log "Docker not running. Starting Docker Desktop..."
    open -a Docker

    DOCKER_WAIT=0
    DOCKER_TIMEOUT=120
    while ! $DOCKER info &>/dev/null; do
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
if $DOCKER compose ps --quiet 2>/dev/null | grep -q .; then
    log "Stopping stale containers from previous run..."
    $DOCKER compose down --timeout 30 2>&1 | tee -a "$LOG"
    sleep 2
fi

# ── Step 3: Start services with retry ─────────────────────────
MAX_RETRIES=3
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    log "Starting Docker Compose (attempt $((RETRY+1))/$MAX_RETRIES)..."
    # Force a no-cache build if any tracked source file changed since the
    # image was built. Prevents the Docker build-cache staleness bug that
    # repeatedly shipped stale broker.py / engine.py into production
    # (incidents Apr 15 and Apr 21). The guard only pays a cold-build
    # cost when there's actually new code — no-op on repeat launches.
    if [ -f "$DIR/.last_image_build_sha" ]; then
        LAST_SHA=$(cat "$DIR/.last_image_build_sha" 2>/dev/null)
    else
        LAST_SHA=""
    fi
    CUR_SHA=$(cd "$DIR" && find core strategies agents utils analytics config.py main.py requirements.txt Dockerfile 2>/dev/null -type f -exec shasum {} \; 2>/dev/null | shasum | awk '{print $1}')
    if [ "$CUR_SHA" != "$LAST_SHA" ]; then
        log "Source changed since last build — running --no-cache (old=$LAST_SHA, new=$CUR_SHA)"
        BUILD_FLAGS="--no-cache"
    else
        log "Source unchanged — reusing cached image"
        BUILD_FLAGS=""
    fi
    if [ -n "$BUILD_FLAGS" ]; then
        $DOCKER compose build $BUILD_FLAGS 2>&1 | tee -a "$LOG" || true
    fi
    if $DOCKER compose up -d --build 2>&1 | tee -a "$LOG"; then
        log "Docker Compose started successfully"
        echo "$CUR_SHA" > "$DIR/.last_image_build_sha"
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
    STATUS=$($DOCKER inspect --format='{{.State.Health.Status}}' algo-engine 2>/dev/null || echo "not_found")
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
    log "Engine status: $($DOCKER inspect --format='{{.State.Health.Status}}' algo-engine 2>/dev/null || echo 'unknown')"
fi

# ── Step 5: Verify all containers are running ─────────────────
FAILURES=0
for SERVICE in algo-engine algo-dashboard algo-agents; do
    STATE=$($DOCKER inspect --format='{{.State.Status}}' "$SERVICE" 2>/dev/null || echo "missing")
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
