#!/bin/bash
# ============================================================
#  Engine Entrypoint
#
#  1. Runs pre-market agents (earnings, optimizer, sentiment, etc.)
#  2. Starts the trading engine
#
#  Pre-market agents run first so the engine has fresh signals
#  on its very first cycle. If any agent fails, the engine
#  still starts (graceful degradation).
# ============================================================

set -e
cd /app

# Run pre-market pipeline (failures are non-fatal)
echo "[$(date)] Running pre-market agents before engine start..."
bash scripts/run_premarket.sh >> /app/logs/agents.log 2>&1 || \
    echo "[$(date)] WARNING: Pre-market pipeline had errors (engine will start anyway)"

echo "[$(date)] Starting trading engine..."
exec python main.py
