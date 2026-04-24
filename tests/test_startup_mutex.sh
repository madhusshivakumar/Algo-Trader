#!/bin/bash
# ============================================================
#  Integration test: concurrent startup.sh invocations
#
#  Reproduces the Apr 24 race where launchd fires startup.sh at
#  06:00 and the watchdog fires another startup.sh at 06:01 while
#  the first is still mid-build. Before the mkdir mutex, two
#  startup.sh's ran in parallel and raced on docker compose.
#
#  Pass criteria:
#    1. Second startup.sh aborts with exit 3 when first is running.
#    2. Stale lock (holder PID no longer alive) is reclaimed.
#    3. Watchdog refuses to invoke startup.sh while lock held.
# ============================================================

set -uo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
STARTUP="$DIR/scripts/startup.sh"
WATCHDOG="$DIR/scripts/watchdog.sh"
LOCKDIR="$DIR/.startup.lock.d"

FAIL=0
TMPDIR=$(mktemp -d)

cleanup() {
    rm -rf "$LOCKDIR" "$TMPDIR"
}
trap cleanup EXIT

# Fake docker that always succeeds. Used so startup.sh doesn't
# actually try to run compose commands during the mutex test.
cat > "$TMPDIR/docker" <<'EOF'
#!/bin/bash
# Stub: just succeed and log nothing
case "${1:-}" in
    info) exit 0 ;;
    ps) echo "" ;;
    compose) echo "stub-compose: $*" ;;
    inspect) echo "healthy" ;;
    *) : ;;
esac
exit 0
EOF
chmod +x "$TMPDIR/docker"
export DOCKER="$TMPDIR/docker"
export PATH="$TMPDIR:$PATH"

# ── Test 1: mkdir mutex is atomic ─────────────────────────
rm -rf "$LOCKDIR"
if mkdir "$LOCKDIR" 2>/dev/null; then
    echo "  ✓ first mkdir of lock succeeded"
else
    echo "  ✗ first mkdir failed unexpectedly"
    FAIL=1
fi
if mkdir "$LOCKDIR" 2>/dev/null; then
    echo "  ✗ second mkdir succeeded — mutex primitive broken"
    FAIL=1
else
    echo "  ✓ second mkdir rejected (EEXIST)"
fi
rm -rf "$LOCKDIR"

# ── Test 2: startup.sh exits 3 when lock dir exists with live PID ──
# Simulate a live holder: create lockdir with this shell's PID
mkdir "$LOCKDIR"
echo $$ > "$LOCKDIR/pid"

OUT=$(bash "$STARTUP" 2>&1)
RC=$?
rm -rf "$LOCKDIR"

if [ "$RC" -eq 3 ]; then
    echo "  ✓ startup.sh exits 3 when mutex held by live PID"
else
    echo "  ✗ startup.sh exit code was $RC (expected 3)"
    echo "    last output line: $(echo "$OUT" | tail -1)"
    FAIL=1
fi

# ── Test 3: startup.sh reclaims stale lock (dead PID) ──────
# Create lockdir with a fake PID that is guaranteed dead
mkdir "$LOCKDIR"
echo "9999999" > "$LOCKDIR/pid"  # PID that almost certainly doesn't exist

# Run startup.sh briefly — it should reclaim the lock, then fail
# later (because our fake docker is a stub), but NOT exit 3.
OUT=$(bash "$STARTUP" 2>&1 | head -20)
RC=$?

if [ "$RC" -ne 3 ]; then
    echo "  ✓ startup.sh reclaimed stale lock (exit $RC != 3)"
else
    echo "  ✗ startup.sh incorrectly treated stale lock as live"
    FAIL=1
fi
rm -rf "$LOCKDIR"

# ── Test 4: watchdog.sh skips startup when lock held ───────
# Give watchdog a fake "engine down" scenario by setting DOCKER
# to a stub that returns no containers from `ps`.
cat > "$TMPDIR/docker-nocontainer" <<'EOF'
#!/bin/bash
case "${1:-}" in
    info) exit 0 ;;
    ps) exit 0 ;;           # empty output = no containers
    inspect) echo "missing" ;;
    *) exit 0 ;;
esac
EOF
chmod +x "$TMPDIR/docker-nocontainer"

# Hold the lock with a live PID
mkdir "$LOCKDIR"
echo $$ > "$LOCKDIR/pid"

# Force watchdog into "in-window" by creating the override sentinel
touch "$DIR/.watchdog_always_on"

# Run watchdog — it should detect lock, log, and exit 0 WITHOUT
# calling startup.sh
BEFORE=$(wc -l < "$DIR/logs/watchdog.log" 2>/dev/null || echo 0)
DOCKER="$TMPDIR/docker-nocontainer" bash "$WATCHDOG" 2>&1
AFTER=$(wc -l < "$DIR/logs/watchdog.log" 2>/dev/null || echo 0)

# Lock dir should still be ours (watchdog didn't remove it)
if [ ! -d "$LOCKDIR" ]; then
    echo "  ✗ watchdog unexpectedly removed the lock"
    FAIL=1
fi

# Find the most recent watchdog log line
NEW_LINE=$(tail -1 "$DIR/logs/watchdog.log" 2>/dev/null)
if echo "$NEW_LINE" | grep -q "startup.sh is already running"; then
    echo "  ✓ watchdog logged 'already running' and deferred"
else
    echo "  ✗ watchdog did not log the expected defer message"
    echo "    last line: $NEW_LINE"
    FAIL=1
fi

rm -f "$DIR/.watchdog_always_on"
rm -rf "$LOCKDIR"

echo
if [ $FAIL -eq 0 ]; then
    echo "All 4 mutex integration tests passed."
    exit 0
else
    echo "Mutex integration tests FAILED ($FAIL failure(s))."
    exit 1
fi
