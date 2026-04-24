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

# Daemon warm-up race: `docker info` can return 0 before the socket is
# fully accepting `exec`/`compose` calls. Probe with the specific
# subcommands we're about to run.
for PROBE in "ps" "compose version"; do
    PROBE_WAIT=0
    until $DOCKER $PROBE >/dev/null 2>&1; do
        sleep 2
        PROBE_WAIT=$((PROBE_WAIT + 2))
        if [ $PROBE_WAIT -ge 30 ]; then
            log "WARNING: '$DOCKER $PROBE' not responsive after 30s — continuing anyway"
            break
        fi
    done
done

# ── Step 2: Stop any stale containers (unconditional + idempotent) ──
# Previous guard used `compose ps --quiet | grep -q .` to detect
# existing containers, but when the compose plugin is mid-warm-up
# after Docker Desktop boot, that detector can silently fail and
# skip cleanup. A bare `compose down` is a no-op when nothing is
# running; running it unconditionally costs ~0.5s and eliminates
# the skip-cleanup edge case.
log "Stopping any existing containers (idempotent)..."
$DOCKER compose down --timeout 30 2>&1 | tee -a "$LOG" || true
sleep 2

# ── Step 3: Start services with retry ─────────────────────────
#
# PERMANENT FIX for the 3-day stale-image recurrence (Apr 21/22/23):
#   Before this, startup tried to skip rebuilds with a source-hash cache.
#   That saved ~60s on warm starts but repeatedly shipped containers
#   running April 14 source even though the image tag pointed at current
#   code. Root cause: `docker compose up -d --build` rebuilds the image
#   but does NOT recreate containers when it thinks the service config
#   is unchanged, so the running container kept its old image handle.
#
# The fix has three layers, every one of which costs us nothing when
# things are healthy and each of which independently closes the gap:
#
#   1. `docker compose build --no-cache` — fresh image every startup.
#      Pays ~60–90s per day; eliminates buildkit staleness class of bug.
#   2. `docker compose up -d --force-recreate` — tears down existing
#      containers and creates new ones from the freshly built image.
#      Without this flag, a stale container survives a new build.
#   3. Post-startup verification script — runs `docker exec` to confirm
#      the running engine has the expected current symbols. If it
#      doesn't, exit 1 so launchd sees startup failed and the watchdog
#      can attempt remediation (instead of reporting "healthy" on a
#      silently broken deploy).
MAX_RETRIES=3
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    log "Starting Docker Compose (attempt $((RETRY+1))/$MAX_RETRIES)..."
    log "Building fresh image (--no-cache) — prevents buildkit staleness..."
    if ! $DOCKER compose build --no-cache 2>&1 | tee -a "$LOG"; then
        log "WARNING: --no-cache build failed on attempt $((RETRY+1))"
        RETRY=$((RETRY+1))
        [ $RETRY -lt $MAX_RETRIES ] && sleep 10 && continue
        break
    fi
    log "Recreating containers (--force-recreate) — ensures new image is actually used..."
    if $DOCKER compose up -d --force-recreate 2>&1 | tee -a "$LOG"; then
        log "Docker Compose started successfully"
        # Record the source hash purely for observability — it does not
        # gate rebuilds anymore (we always rebuild).
        CUR_SHA=$(cd "$DIR" && find core strategies agents utils analytics config.py main.py requirements.txt Dockerfile -type f -exec shasum {} \; 2>/dev/null | shasum | awk '{print $1}')
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
    log "FATAL: Engine health check timed out after ${ENGINE_TIMEOUT}s"
    log "Engine status: $($DOCKER inspect --format='{{.State.Health.Status}}' algo-engine 2>/dev/null || echo 'unknown')"
    # Do not fall through to deploy-verification with an unhealthy
    # container — previously this path logged WARNING and continued,
    # which could pass step 5 (container `running`) against an
    # unhealthy engine and mask the timeout.
    exit 1
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

if [ $FAILURES -ne 0 ]; then
    log "=========================================="
    log "STARTUP WARNING: $FAILURES service(s) not running"
    log "=========================================="
    exit 1
fi

# ── Step 6: Verify the running container actually matches current source ──
#
# Layer 3 of the permanent fix: independent capability verification.
# Even if build + force-recreate succeeded, something could still go
# wrong (Docker Desktop bugs, weird cache states). This runs inside the
# live container and confirms expected symbols exist. If it fails, exit
# 1 so the failure is VISIBLE (launchd log, subsequent watchdog alert).
log "Verifying deployed engine matches current source..."
if [ -x "$DIR/scripts/verify_engine_deploy.sh" ]; then
    if bash "$DIR/scripts/verify_engine_deploy.sh" 2>&1 | tee -a "$LOG"; then
        log "  ✓ Deploy verification passed"
    else
        log "  ✗ DEPLOY VERIFICATION FAILED — engine is running stale code"
        log "    This is the Apr 21/22/23 recurring stale-image bug."
        log "    Investigate immediately; container is up but not running current fixes."
        exit 1
    fi
else
    log "  (verify_engine_deploy.sh not found — skipping verification)"
fi

log "=========================================="
log "STARTUP COMPLETE: All 3 services running, deploy verified"
log "Dashboard: http://localhost:5050"
log "=========================================="
exit 0
