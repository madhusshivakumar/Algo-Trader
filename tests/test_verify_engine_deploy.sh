#!/bin/bash
# Tests for scripts/verify_engine_deploy.sh marker contract (Bug #3).
#
# Run with:  bash tests/test_verify_engine_deploy.sh
#
# These run on the host (not pytest) because the script is bash and
# pytest harness for shell is heavier than warranted. Each test stubs
# `docker` via PATH manipulation so we can simulate every container
# state without touching real Docker.

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO/scripts/verify_engine_deploy.sh"
PASS=0
FAIL=0

# Create a tmpdir with a fake `docker` shim. Each test rewrites the shim
# to simulate the desired container state.
TMP=$(mktemp -d)
# Force the script to use our shim by absolute path. The verify script
# prepends /Applications/Docker.app/... to PATH so a relative `docker`
# would still resolve to the real binary; absolute path bypasses that.
export DOCKER="$TMP/docker"

cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

assert_marker() {
    local label="$1"
    local out="$2"
    local expected="$3"
    if echo "$out" | grep -q "^${expected}"; then
        echo "  ✓ $label"
        PASS=$((PASS+1))
    else
        echo "  ✗ $label — expected '${expected}' marker"
        echo "    out: $(echo "$out" | tail -2 | tr '\n' '|')"
        FAIL=$((FAIL+1))
    fi
}

assert_exit() {
    local label="$1"
    local actual="$2"
    local expected="$3"
    if [ "$actual" = "$expected" ]; then
        echo "  ✓ $label (rc=$actual)"
        PASS=$((PASS+1))
    else
        echo "  ✗ $label — expected rc=$expected, got rc=$actual"
        FAIL=$((FAIL+1))
    fi
}

# ── Test 1: happy path — fresh container with all symbols ──────────
echo "Test 1: happy path"
cat > "$TMP/docker" <<'EOF'
#!/bin/bash
case "$*" in
    "exec algo-engine test -f /app/main.py") exit 0 ;;
    *"timeframe: str"*"/app/core/broker.py") exit 0 ;;
    *"test -f /app/core/heartbeat.py") exit 0 ;;
    *"write_heartbeat"*"/app/core/engine.py") exit 0 ;;
    *) exit 0 ;;
esac
EOF
chmod +x "$TMP/docker"
out=$(bash "$SCRIPT" 2>&1)
rc=$?
assert_marker "happy: VERIFY:OK marker present" "$out" "VERIFY:OK"
assert_exit "happy: exits 0" "$rc" "0"

# ── Test 2: stale code — broker.py missing timeframe ───────────────
# Mock matches the inner sh -c argument pattern. dexec passes the inner
# command verbatim as a single arg to `sh -c`, so it appears in $* with
# its single-quoted contents preserved literally.
echo "Test 2: stale broker.py"
cat > "$TMP/docker" <<'EOF'
#!/bin/bash
joined="$*"
case "$joined" in
    *"test -f /app/main.py"*)               exit 0 ;;
    *broker.py*timeframe*|*timeframe*broker.py*)  exit 1 ;;  # stale
    *heartbeat.py*test*-f*|*"test -f /app/core/heartbeat.py"*) exit 0 ;;
    *engine.py*write_heartbeat*|*write_heartbeat*engine.py*)   exit 0 ;;
    *) exit 0 ;;
esac
EOF
out=$(bash "$SCRIPT" 2>&1)
rc=$?
assert_marker "stale: VERIFY:FAIL:stale marker" "$out" "VERIFY:FAIL:stale"
assert_exit "stale: exits 1" "$rc" "1"

# ── Test 3: container down — readiness gate timeout ─────────────────
echo "Test 3: container missing"
cat > "$TMP/docker" <<'EOF'
#!/bin/bash
# main.py never accessible → readiness gate times out
exit 1
EOF
# Override MAX_WAIT to keep the test fast
sed 's/MAX_WAIT=60/MAX_WAIT=2/' "$SCRIPT" > "$TMP/verify_short.sh"
out=$(bash "$TMP/verify_short.sh" 2>&1)
rc=$?
assert_marker "no-container: VERIFY:FAIL:readiness marker" "$out" "VERIFY:FAIL:readiness"
assert_exit "no-container: exits 1" "$rc" "1"

# ── Test 4: infra error (daemon refusal) — distinguished from stale ─
echo "Test 4: infra error vs stale"
cat > "$TMP/docker" <<'EOF'
#!/bin/bash
joined="$*"
case "$joined" in
    *"test -f /app/main.py"*) exit 0 ;;
    *)
        # Print to BOTH stdout and stderr so dexec's 2>&1 catches it
        # regardless of which stream it's reading.
        echo "Cannot connect to the Docker daemon"
        exit 1
        ;;
esac
EOF
out=$(bash "$SCRIPT" 2>&1)
rc=$?
assert_marker "infra: VERIFY:FAIL:infra marker" "$out" "VERIFY:FAIL:infra"
assert_exit "infra: exits 1" "$rc" "1"

# ── Test 5: zero checks ran — script claims OK without running anything
# (Defensive against future refactor that drops dexec calls)
echo "Test 5: zero-checks defense"
# Intercept the script: replace all dexec calls with no-ops
sed 's/^dexec /# dexec /g' "$SCRIPT" > "$TMP/verify_nochecks.sh"
# But also remove host file check to truly run zero checks
# Actually: easier to just check the assertion fires. Patch: comment out dexec lines.
cat > "$TMP/docker" <<'EOF'
#!/bin/bash
[ "$*" = "exec algo-engine test -f /app/main.py" ] && exit 0
exit 0
EOF
out=$(bash "$TMP/verify_nochecks.sh" 2>&1)
rc=$?
assert_marker "zero-checks: VERIFY:FAIL:stale marker (count assertion)" "$out" "VERIFY:FAIL:stale"
assert_exit "zero-checks: exits 1" "$rc" "1"

# ── Summary ────────────────────────────────────────────────────────
echo
echo "── verify_engine_deploy.sh tests ──"
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
[ $FAIL -eq 0 ] && exit 0 || exit 1
