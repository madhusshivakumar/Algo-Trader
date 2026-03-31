#!/bin/bash
# ============================================================
#  Algo Trader — Quick Commands
#  Usage:  ./bot.sh [command]
# ============================================================

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

BOT_PID_FILE="$DIR/.bot.pid"
DASH_PID_FILE="$DIR/.dashboard.pid"

# --- Helpers ---
is_running() {
    local pidfile="$1"
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        else
            rm -f "$pidfile"
            return 1
        fi
    fi
    return 1
}

# --- Commands ---
start_bot() {
    if is_running "$BOT_PID_FILE"; then
        echo -e "${YELLOW}Bot is already running (PID: $(cat $BOT_PID_FILE))${NC}"
        return
    fi
    echo -e "${GREEN}${BOLD}Starting trading bot...${NC}"
    nohup python main.py > "$DIR/logs/bot.log" 2>&1 &
    echo $! > "$BOT_PID_FILE"
    echo -e "${GREEN}Bot started (PID: $!)${NC}"
    echo -e "${CYAN}Logs: tail -f $DIR/logs/bot.log${NC}"
}

stop_bot() {
    if is_running "$BOT_PID_FILE"; then
        local pid=$(cat "$BOT_PID_FILE")
        echo -e "${RED}Stopping bot (PID: $pid)...${NC}"
        kill "$pid" 2>/dev/null
        sleep 1
        kill -9 "$pid" 2>/dev/null
        rm -f "$BOT_PID_FILE"
        echo -e "${RED}Bot stopped.${NC}"
    else
        echo -e "${YELLOW}Bot is not running.${NC}"
    fi
}

start_dashboard() {
    if is_running "$DASH_PID_FILE"; then
        echo -e "${YELLOW}Dashboard is already running (PID: $(cat $DASH_PID_FILE))${NC}"
        echo -e "${BLUE}Open: http://localhost:5050${NC}"
        return
    fi
    echo -e "${GREEN}${BOLD}Starting dashboard...${NC}"
    nohup python dashboard.py > "$DIR/logs/dashboard.log" 2>&1 &
    echo $! > "$DASH_PID_FILE"
    sleep 1
    echo -e "${GREEN}Dashboard started (PID: $!)${NC}"
    echo -e "${BLUE}${BOLD}Open: http://localhost:5050${NC}"
}

stop_dashboard() {
    if is_running "$DASH_PID_FILE"; then
        local pid=$(cat "$DASH_PID_FILE")
        echo -e "${RED}Stopping dashboard (PID: $pid)...${NC}"
        kill "$pid" 2>/dev/null
        rm -f "$DASH_PID_FILE"
        echo -e "${RED}Dashboard stopped.${NC}"
    else
        echo -e "${YELLOW}Dashboard is not running.${NC}"
    fi
}

show_status() {
    echo -e "${BOLD}╔══════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║        Algo Trader Status            ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════╝${NC}"
    echo ""

    if is_running "$BOT_PID_FILE"; then
        echo -e "  Bot:       ${GREEN}● RUNNING${NC}  (PID: $(cat $BOT_PID_FILE))"
    else
        echo -e "  Bot:       ${RED}● STOPPED${NC}"
    fi

    if is_running "$DASH_PID_FILE"; then
        echo -e "  Dashboard: ${GREEN}● RUNNING${NC}  (PID: $(cat $DASH_PID_FILE))"
        echo -e "             ${BLUE}http://localhost:5050${NC}"
    else
        echo -e "  Dashboard: ${RED}● STOPPED${NC}"
    fi

    echo ""
    python main.py --status 2>/dev/null || echo -e "  ${YELLOW}(Set API keys in .env to see account status)${NC}"
}

show_logs() {
    echo -e "${CYAN}Following bot logs (Ctrl+C to stop)...${NC}"
    tail -f "$DIR/logs/bot.log" 2>/dev/null || echo -e "${YELLOW}No log file yet. Start the bot first.${NC}"
}

run_backtest() {
    echo -e "${BLUE}${BOLD}Running backtest...${NC}"
    python main.py --backtest
}

run_compare() {
    echo -e "${BLUE}${BOLD}Comparing all strategies...${NC}"
    python compare_strategies.py
}

run_optimizer() {
    echo -e "${BLUE}${BOLD}Running Strategy Optimizer agent...${NC}"
    python agents/strategy_optimizer.py
}

run_scanner() {
    echo -e "${BLUE}${BOLD}Running Market Scanner agent...${NC}"
    python agents/market_scanner.py
}

run_analyzer() {
    echo -e "${BLUE}${BOLD}Running Trade Analyzer agent...${NC}"
    python agents/trade_analyzer.py
}

run_health() {
    echo -e "${BLUE}${BOLD}Running Health Check...${NC}"
    python agents/health_check.py
}

run_tests() {
    echo -e "${BLUE}${BOLD}Running test suite...${NC}"
    python -m pytest tests/ -v --tb=short
}

run_sentiment() {
    echo -e "${BLUE}${BOLD}Running Sentiment Agent...${NC}"
    python agents/sentiment_agent.py
}

run_llm() {
    echo -e "${BLUE}${BOLD}Running LLM Analyst Agent...${NC}"
    python agents/llm_analyst.py
}

run_rl_train() {
    echo -e "${BLUE}${BOLD}Running RL Trainer Agent...${NC}"
    python agents/rl_trainer.py
}

run_agents() {
    echo -e "${BLUE}${BOLD}Running all agents in sequence...${NC}"
    echo ""
    run_optimizer
    echo ""
    run_scanner
    echo ""
    run_sentiment
    echo ""
    run_llm
    echo ""
    run_analyzer
}

show_help() {
    echo ""
    echo -e "${BOLD}Algo Trader Commands${NC}"
    echo -e "${BOLD}════════════════════${NC}"
    echo ""
    echo -e "  ${GREEN}./bot.sh start${NC}       Start the trading bot"
    echo -e "  ${RED}./bot.sh stop${NC}        Stop the trading bot"
    echo -e "  ${GREEN}./bot.sh dash${NC}        Start the web dashboard"
    echo -e "  ${RED}./bot.sh dash-stop${NC}   Stop the web dashboard"
    echo -e "  ${CYAN}./bot.sh status${NC}      Show bot status + account info"
    echo -e "  ${CYAN}./bot.sh logs${NC}        Follow live bot logs"
    echo -e "  ${BLUE}./bot.sh backtest${NC}    Run strategy backtest"
    echo -e "  ${BLUE}./bot.sh compare${NC}     Compare all strategies"
    echo ""
    echo -e "  ${GREEN}./bot.sh up${NC}          Start bot + dashboard together"
    echo -e "  ${RED}./bot.sh down${NC}        Stop everything"
    echo ""
    echo -e "  ${BOLD}Agents:${NC}"
    echo -e "  ${BLUE}./bot.sh optimizer${NC}   Run Strategy Optimizer"
    echo -e "  ${BLUE}./bot.sh scanner${NC}     Run Market Scanner"
    echo -e "  ${BLUE}./bot.sh analyzer${NC}    Run Trade Analyzer"
    echo -e "  ${BLUE}./bot.sh sentiment${NC}   Run Sentiment Agent (FinBERT)"
    echo -e "  ${BLUE}./bot.sh llm${NC}         Run LLM Analyst Agent (Claude)"
    echo -e "  ${BLUE}./bot.sh rl-train${NC}    Run RL Trainer Agent (DQN)"
    echo -e "  ${BLUE}./bot.sh agents${NC}      Run all agents in sequence"
    echo -e "  ${BLUE}./bot.sh health${NC}      Run system health check"
    echo -e "  ${BLUE}./bot.sh test${NC}        Run test suite"
    echo ""
}

# --- Main ---
mkdir -p "$DIR/logs"

case "${1:-help}" in
    start)      start_bot ;;
    stop)       stop_bot ;;
    dash)       start_dashboard ;;
    dash-stop)  stop_dashboard ;;
    status|st)  show_status ;;
    logs|log)   show_logs ;;
    backtest|bt) run_backtest ;;
    compare)    run_compare ;;
    optimizer)  run_optimizer ;;
    scanner)    run_scanner ;;
    analyzer)   run_analyzer ;;
    agents)     run_agents ;;
    sentiment)  run_sentiment ;;
    llm)        run_llm ;;
    rl-train)   run_rl_train ;;
    health)     run_health ;;
    test|tests) run_tests ;;
    up)
        start_bot
        start_dashboard
        echo ""
        echo -e "${GREEN}${BOLD}All systems go!${NC}"
        echo -e "${BLUE}Dashboard: http://localhost:5050${NC}"
        echo -e "${CYAN}Bot logs:  ./bot.sh logs${NC}"
        ;;
    down)
        stop_bot
        stop_dashboard
        echo -e "${RED}${BOLD}All systems stopped.${NC}"
        ;;
    *)          show_help ;;
esac
