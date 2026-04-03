#!/bin/bash
# ============================================================
#  Pre-Market Agent Pipeline (Parallelized)
#
#  Called on engine container startup before the engine begins
#  trading. Runs fast, time-sensitive agents only (~2-3 min).
#
#  Strategy optimizer runs weekly (Sunday) via crontab, not
#  here — backtest results are stable day-to-day.
#
#  All agents run in parallel since they're independent:
#    - Earnings calendar   (Alpaca API, ~5-30s)
#    - Sentiment agent     (Alpaca News + FinBERT, ~20-60s)
#    - Market scanner      (Alpaca bars, ~20-45s)
#    - LLM analyst         (Claude API, ~1-3 min)
# ============================================================

cd /app

echo "[$(date)] Starting pre-market agent pipeline..."

python agents/earnings_calendar.py >> /app/logs/agents.log 2>&1 &
PID_EARNINGS=$!

python agents/sentiment_agent.py >> /app/logs/agents.log 2>&1 &
PID_SENTIMENT=$!

python agents/market_scanner.py >> /app/logs/agents.log 2>&1 &
PID_SCANNER=$!

python agents/llm_analyst.py >> /app/logs/agents.log 2>&1 &
PID_LLM=$!

FAILED=0

wait $PID_EARNINGS || { echo "[$(date)] WARNING: earnings_calendar failed"; FAILED=$((FAILED+1)); }
echo "[$(date)] earnings_calendar done."

wait $PID_SCANNER || { echo "[$(date)] WARNING: market_scanner failed"; FAILED=$((FAILED+1)); }
echo "[$(date)] market_scanner done."

wait $PID_SENTIMENT || { echo "[$(date)] WARNING: sentiment_agent failed"; FAILED=$((FAILED+1)); }
echo "[$(date)] sentiment_agent done."

wait $PID_LLM || { echo "[$(date)] WARNING: llm_analyst failed"; FAILED=$((FAILED+1)); }
echo "[$(date)] llm_analyst done."

echo "[$(date)] Pre-market pipeline complete ($FAILED failures)."
