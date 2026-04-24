#!/bin/bash
# ============================================================
#  Deploy Verification — permanent fix for stale-image bug
#
#  Runs `docker exec algo-engine` checks to confirm the running
#  container actually has the expected current source. If any check
#  fails, exits 1 so the calling startup.sh can surface the failure
#  LOUDLY (launchd log + subsequent watchdog alert).
#
#  Why: from Apr 21–23 the container repeatedly started healthy per
#  docker ps and Alpaca heartbeats but was running April 14 source —
#  missing the broker.get_historical_bars timeframe kwarg, missing
#  core/heartbeat.py entirely. Silent failure meant days of degraded
#  trading with no alarm. This script turns silent failure into loud
#  exit 1.
#
#  Design choices from code review:
#   - Checks never import Python modules. Module import triggers
#     side effects (config load, network, agent scheduler) that can
#     mask the real symbol-presence signal with noise when things are
#     already broken. We only `cat`+`grep` or `test -f`.
#   - Explicit distinction between "infra failure" (docker/exec itself
#     errored) and "stale deploy" (symbol missing). Both exit 1 but
#     are logged separately so the cause is diagnosable.
#   - Each check represents a specific prior production incident.
# ============================================================

set -uo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin:$PATH"
DOCKER="${DOCKER:-docker}"
VERBOSE=false
[ "${1:-}" = "-v" ] && VERBOSE=true

FAIL=0
INFRA_FAIL=0

# dexec <description> <command-string>
#   Runs a command inside algo-engine via docker exec. Returns:
#     0 — command exited 0 inside container
#     1 — command exited non-zero inside container (symbol missing)
#     2 — docker exec itself failed (infra error, not a symbol issue)
dexec() {
    local out
    out=$($DOCKER exec algo-engine sh -c "$2" 2>&1)
    local rc=$?
    if [ $rc -eq 0 ]; then
        $VERBOSE && echo "  ✓ $1"
        return 0
    fi
    # Distinguish infra failure from symbol-missing failure. Docker exec
    # returns specific error strings on daemon/container issues.
    if echo "$out" | grep -qiE "no such container|is not running|cannot connect|daemon"; then
        echo "  ! INFRA $1 — $(echo "$out" | head -1)"
        INFRA_FAIL=$((INFRA_FAIL+1))
        return 2
    fi
    echo "  ✗ $1 — $(echo "$out" | head -1)"
    FAIL=$((FAIL+1))
    return 1
}

# ── Readiness gate ──────────────────────────────────────────
# Reviewer fix #3: don't run capability checks if the container's
# filesystem isn't mounted yet. We poll briefly for the engine's
# main module to exist as a quick "container is alive enough" gate.
WAIT=0
MAX_WAIT=60
while [ $WAIT -lt $MAX_WAIT ]; do
    if $DOCKER exec algo-engine test -f /app/main.py 2>/dev/null; then
        break
    fi
    sleep 2
    WAIT=$((WAIT+2))
done
if [ $WAIT -ge $MAX_WAIT ]; then
    echo "  ! INFRA container /app/main.py not accessible after ${MAX_WAIT}s"
    exit 1
fi

# ── Checks ──────────────────────────────────────────────────
# Each represents a specific prior production incident. Filesystem-
# only — no Python imports — so the signal is clean even when the
# engine's runtime code is half-broken.

# Apr 15 incident: broker.get_historical_bars gained timeframe kwarg
# so the regime detector could request daily bars.
dexec "broker.get_historical_bars has timeframe kwarg" \
      "grep -q 'timeframe: str' /app/core/broker.py"

# Apr 21 incident: core/heartbeat.py added to provide engine liveness
# signal.
dexec "core/heartbeat.py exists in container" \
      "test -f /app/core/heartbeat.py"

# Apr 21 incident: engine.py wired to call write_heartbeat() at the
# end of every run_cycle. Without this the file exists but never
# updates, leaving the same zombie-detection gap.
dexec "engine.py invokes write_heartbeat" \
      "grep -q 'write_heartbeat' /app/core/engine.py"

# Sanity: the check_heartbeat script the watchdog calls must be in
# place on the host (we run it from the host, but failing here is a
# good smoke test that the repo checkout is also current).
if [ ! -f "$DIR/scripts/check_heartbeat.py" ]; then
    echo "  ✗ host: scripts/check_heartbeat.py missing"
    FAIL=$((FAIL+1))
fi

# ── Result ──────────────────────────────────────────────────
if [ $INFRA_FAIL -gt 0 ]; then
    echo "Infra error: $INFRA_FAIL check(s) could not run. Container/daemon issue."
    exit 1
fi
if [ $FAIL -gt 0 ]; then
    echo "Stale deploy: $FAIL check(s) failed. Running container has OLD source."
    exit 1
fi
$VERBOSE && echo "All checks passed."
exit 0
