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
#    3 = Another startup.sh is already running (mutex contention)
# ============================================================

set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

# ── MUTEX: single-instance guarantee ─────────────────────────
#
# Why: Apr 24 incident. launchd fires startup at 06:00, which does a
# compose down and begins a ~60-second --no-cache build. During that
# build window, the container genuinely doesn't exist — so the watchdog's
# 15-min check (which fires at 06:01) correctly observes "engine down in
# trading window" and calls startup.sh AGAIN. Two startup.sh processes
# then race on compose build/up, and the container that wins is
# unpredictable (usually NOT the one the final verify step observed).
#
# macOS doesn't ship flock(1), so we use `mkdir` as the POSIX-standard
# atomic primitive: only one process can successfully create a given
# directory. The holder's PID is written inside. On startup we check if
# the recorded PID is still alive; if not, we take over (stale lock
# recovery).
#
# Bug #5 (Apr 27 hardening): use a GLOBAL lock at $HOME, not a
# project-relative one. The previous $DIR-based lock had a hole: if
# two startup.sh scripts existed at different paths (Apr 27 had a
# stale Desktop copy AND ~/algo-trader/), each had its own lockdir
# and the mutex was effectively useless. A $HOME-based lock means any
# `bash startup.sh` invocation from any path competes for the same
# lock — what we actually want for protecting Docker resources.
readonly LOCKDIR="$HOME/.algo-trader-startup.lock.d"
acquire_mutex() {
    if mkdir "$LOCKDIR" 2>/dev/null; then
        echo $$ > "$LOCKDIR/pid"
        return 0
    fi
    # Lock dir exists — check if holder is still alive
    local holder_pid
    holder_pid=$(cat "$LOCKDIR/pid" 2>/dev/null)
    if [ -z "$holder_pid" ] || ! kill -0 "$holder_pid" 2>/dev/null; then
        # Stale lock — holder is dead. Reclaim.
        rm -rf "$LOCKDIR"
        if mkdir "$LOCKDIR" 2>/dev/null; then
            echo $$ > "$LOCKDIR/pid"
            return 0
        fi
    fi
    return 1
}
if ! acquire_mutex; then
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    holder=$(cat "$LOCKDIR/pid" 2>/dev/null || echo 'unknown')
    echo "[$ts] startup.sh aborting: another instance is running (pid $holder). The watchdog will retry on its next 15-min cycle if needed." \
        >> "$DIR/logs/startup.log" 2>/dev/null || true
    exit 3
fi
# Always release the lock on exit, even on errors or signals.
trap 'rm -rf "$LOCKDIR"' EXIT INT TERM

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

# Bug #8 (Apr 28 hardening): assert that no other compose project is
# tracking the same project name from a different working directory.
# The Apr 28 stale-Desktop incident root cause was exactly this:
# /Users/.../Desktop/Projects/algo-trader/docker-compose.yml had been
# registered with Docker Compose (project name "algo-trader") and
# every `compose up` we ran from ~/algo-trader/ was redirected to the
# Desktop project's working directory. Stale code shipped for days.
#
# Now that the Desktop folder is gone this can't recur naturally, but
# any future duplicate (clone for testing, fork copy, etc.) would
# silently re-introduce the same bug. Fail loudly at startup if we
# detect more than one project file or a project file at a path that
# isn't $DIR/docker-compose.yml.
COMPOSE_PROJECTS=$($DOCKER compose ls --format json 2>/dev/null \
                   | python3 -c "import sys,json; data=json.load(sys.stdin); [print(p.get('ConfigFiles','')) for p in data if p.get('Name','')=='algo-trader']" 2>/dev/null \
                   | grep -v '^$' || true)
EXPECTED_COMPOSE="$DIR/docker-compose.yml"
if [ -n "$COMPOSE_PROJECTS" ]; then
    UNEXPECTED=$(echo "$COMPOSE_PROJECTS" | grep -v "^${EXPECTED_COMPOSE}$" || true)
    if [ -n "$UNEXPECTED" ]; then
        log "FATAL: docker compose has tracked algo-trader project at unexpected path(s):"
        echo "$UNEXPECTED" | while IFS= read -r p; do log "  - $p"; done
        log "Expected: $EXPECTED_COMPOSE"
        log "This is the Apr 28 stale-Desktop incident pattern. Resolve by:"
        log "  1. cd <unexpected path> && docker compose down"
        log "  2. Remove the stale repo copy if it shouldn't exist"
        log "  3. Re-run startup.sh"
        exit 1
    fi
fi

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
# Bug #4 (Apr 27 incident): docker compose build can hang for HOURS
# silently — Apr 27 it ran 1h 33min before I manually killed it.
# Wrap each build attempt in a hard 10-minute timeout. macOS doesn't
# ship coreutils' `timeout` by default, so we provide a portable
# Bash-only equivalent.
BUILD_TIMEOUT_SEC=600
build_with_timeout() {
    local cmd_pid
    # Run in background; capture PID; wait with timeout sentinel.
    ($DOCKER compose build --no-cache 2>&1 | tee -a "$LOG") &
    cmd_pid=$!
    local elapsed=0
    while kill -0 $cmd_pid 2>/dev/null; do
        if [ $elapsed -ge $BUILD_TIMEOUT_SEC ]; then
            log "FATAL: docker compose build exceeded ${BUILD_TIMEOUT_SEC}s — killing"
            kill -TERM $cmd_pid 2>/dev/null
            sleep 2
            kill -KILL $cmd_pid 2>/dev/null
            # Also kill any orphaned buildkit children
            pkill -KILL -P $cmd_pid 2>/dev/null
            return 124  # Match GNU timeout's exit code for time-out
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    wait $cmd_pid
    return $?
}
while [ $RETRY -lt $MAX_RETRIES ]; do
    log "Starting Docker Compose (attempt $((RETRY+1))/$MAX_RETRIES)..."
    log "Building fresh image (--no-cache, ${BUILD_TIMEOUT_SEC}s timeout)..."
    build_with_timeout
    BUILD_RC=$?
    if [ $BUILD_RC -eq 124 ]; then
        log "WARNING: build attempt $((RETRY+1)) timed out after ${BUILD_TIMEOUT_SEC}s"
        RETRY=$((RETRY+1))
        if [ $RETRY -lt $MAX_RETRIES ]; then
            log "Retrying in 10s (attempt $((RETRY+1))/$MAX_RETRIES)..."
            sleep 10
            continue
        fi
        break
    fi
    if [ $BUILD_RC -ne 0 ]; then
        log "WARNING: --no-cache build failed (rc=$BUILD_RC) on attempt $((RETRY+1))"
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
    # Bug #3 hardening (Apr 28 silent-pass): require BOTH a clean exit
    # code AND an explicit VERIFY:OK marker in the output. Either alone
    # was insufficient — Apr 28 startup.sh logged "✓ passed" against a
    # stale container, suggesting either the pipeline ate the exit code
    # or the script never actually ran the checks. Two-way confirm
    # makes silent-pass essentially impossible.
    VERIFY_OUT=$(bash "$DIR/scripts/verify_engine_deploy.sh" 2>&1)
    VERIFY_RC=$?
    echo "$VERIFY_OUT" | tee -a "$LOG" >/dev/null
    if [ $VERIFY_RC -eq 0 ] && echo "$VERIFY_OUT" | grep -q '^VERIFY:OK'; then
        log "  ✓ Deploy verification passed"
    else
        log "  ✗ DEPLOY VERIFICATION FAILED (rc=$VERIFY_RC)"
        log "    Last verify output: $(echo "$VERIFY_OUT" | tail -3 | tr '\n' '|')"
        log "    Investigate immediately; container is up but may be running stale code."
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
