# Algo Trader

A fully automated algorithmic trading system for crypto and US equities, powered by [Alpaca](https://alpaca.markets/) with AI-enhanced signal processing.

The bot runs 9 technical strategies, selects the best one per symbol using backtesting or reinforcement learning, and optionally enriches signals with NLP sentiment analysis (FinBERT) and LLM-powered conviction scoring (Claude). Seven autonomous agents handle optimization, scanning, analysis, health checks, and AI model management on a daily schedule.

---

## Table of Contents

- [Architecture](#architecture)
- [High-Level Design](#high-level-design)
- [Strategies](#strategies)
- [AI Signal Modifiers (v2)](#ai-signal-modifiers-v2)
- [Autonomous Agents](#autonomous-agents)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Architecture

```
                                    Alpaca API
                                   /    |    \
                            Crypto  Equities  News
                               \      |      /
                        ┌───────────────────────────┐
                        │      Trading Engine        │
                        │   (60-second loop cycle)   │
                        └─────────┬─────────────────┘
                                  │
                    ┌─────────────┼─────────────────┐
                    │             │                  │
              ┌─────┴──────┐  ┌──┴───────┐  ┌──────┴──────┐
              │  Strategy   │  │   Risk   │  │   Signal    │
              │   Router    │  │ Manager  │  │  Modifiers  │
              └─────┬──────┘  └──────────┘  └──────┬──────┘
                    │                              │
         ┌─────────┼──────────┐         ┌──────────┼──────────┐
         │         │          │         │          │          │
    ┌────┴──┐ ┌───┴───┐ ┌───┴──┐  ┌───┴───┐  ┌──┴───┐  ┌──┴──┐
    │9 Tech │ │  RL   │ │STRAT │  │Senti- │  │ LLM  │  │Score│
    │Strats │ │Select │ │ MAP  │  │ment   │  │Convic│  │Clamp│
    └───────┘ └───────┘ └──────┘  └───────┘  └──────┘  └─────┘
```

---

## High-Level Design

### Core Loop

The trading engine runs a **60-second cycle** for each configured symbol:

1. **Fetch Data** — Pull latest 1-minute OHLCV bars from Alpaca
2. **Check Stops** — Evaluate trailing stops on existing positions
3. **Select Strategy** — RL model picks the best strategy (or fallback to optimizer assignments)
4. **Compute Signal** — Run the selected strategy's technical analysis on the DataFrame
5. **Enrich Signal** — Apply sentiment modifier (FinBERT) and LLM conviction modifier (Claude)
6. **Risk Check** — Validate exposure limits, cooldowns, drawdown, PDT protection
7. **Size & Execute** — Calculate position size (strength-weighted) and submit order
8. **Log** — Record trade details to SQLite with all metadata

### Signal Flow

```
router.compute_signals(symbol, df)
  │
  ├─ 1. Strategy Selection
  │     RL Model (DQN)  ──or──  STRATEGY_MAP (optimizer-assigned)
  │
  ├─ 2. Technical Signal
  │     strategy.compute_signals(df) → {action, strength, reason}
  │
  ├─ 3. Sentiment Modifier (if enabled)
  │     FinBERT score from data/sentiment/scores.json
  │     Buy + positive sentiment → boost strength by score × 0.15
  │     Buy + negative sentiment → dampen strength by 30%
  │
  ├─ 4. LLM Conviction Modifier (if enabled)
  │     Claude score from data/llm_analyst/convictions.json
  │     Buy + bullish conviction → boost strength by score × 0.20
  │     Buy + bearish conviction → dampen strength by 30%
  │
  └─ 5. Clamp strength to [0.3, 1.0]
       → Final signal: {action, strength, reason, sentiment_score, llm_conviction, rl_selected}
```

### Risk Management

| Control | Value | Description |
|---------|-------|-------------|
| **Trailing Stop** | 2% | Follows highest price, triggers exit on pullback |
| **Max Position** | 50% of equity | No single position exceeds this |
| **Exposure Cap** | 90% of equity | Total portfolio exposure limit |
| **Daily Drawdown** | 10% | Halts all trading if daily loss exceeds this |
| **Cooldown (Crypto)** | 15 min | Minimum time between trades on same symbol |
| **Cooldown (Equity)** | 5 min | Minimum time between trades on same symbol |
| **PDT Protection** | Enabled | Prevents same-day equity sells under $25k |
| **LLM Budget** | $1.00/day | Hard cap on Claude API spend |

### Data Flow Between Components

```
                     ┌─────────────────────┐
                     │    Agent Scheduler   │
                     │      (crontab)       │
                     └──┬──┬──┬──┬──┬──┬──┘
                        │  │  │  │  │  │
    5:00 AM ────────────┤  │  │  │  │  │── health_check.py
    5:30 AM ───────────────┤  │  │  │  │── strategy_optimizer.py
    5:45 AM ──────────────────┤  │  │  │── sentiment_agent.py
    6:00 AM ─────────────────────┤  │  │── market_scanner.py
    6:15 AM ────────────────────────┤  │── llm_analyst.py
    5:00 PM ───────────────────────────┤── trade_analyzer.py
    Sun 2 AM ──────────────────────────┤── rl_trainer.py
                                       │
                     ┌─────────────────────┐
                     │     data/ (JSON)     │
                     ├─────────────────────┤
                     │ agent_state.json     │◄── All agents write status
                     │ learnings.json       │◄── Analyzer writes, Optimizer/Scanner read
                     │ fallback_config.json │◄── Analyzer writes on profitable days
                     │ optimizer/           │◄── Optimizer writes backtest results
                     │   strategy_assign... │◄── Scanner reads, Router reads
                     │ sentiment/scores.json│◄── Sentiment agent writes, Router reads
                     │ llm_analyst/         │◄── LLM agent writes, Router reads
                     │   convictions.json   │
                     │   spend_log.json     │
                     │ rl_models/           │◄── RL trainer writes, Selector reads
                     │   dqn_latest.zip     │
                     └─────────────────────┘
```

---

## Strategies

The system ships with **9 technical strategies**, each producing a signal of `{action: buy|sell|hold, strength: 0-1, reason: string}`:

| # | Strategy | Indicators | Style |
|---|----------|-----------|-------|
| 1 | **Mean Rev Aggressive** | RSI(10) + BB(15, 1.5σ) | High-frequency mean reversion |
| 2 | **Mean Reversion** | RSI(14) + BB(20, 2σ) | Classic mean reversion |
| 3 | **Volume Profile** | OBV + MFI + Volume spikes | Volume-driven entries |
| 4 | **Momentum Breakout** | EMA(50) + MACD + breakouts | Trend-following |
| 5 | **MACD Crossover** | MACD/signal line crossovers | Momentum |
| 6 | **Triple EMA Trend** | EMA(8/21/55) + ADX trend filter | Multi-timeframe trend |
| 7 | **RSI Divergence** | Price-RSI divergence detection | Reversal detection |
| 8 | **Scalper** | EMA(5/13) + VWAP + StochRSI | High-frequency scalping |
| 9 | **Ensemble** | Momentum + Mean Reversion vote | Multi-strategy consensus |

The **Strategy Optimizer** agent backtests all 9 strategies against each symbol weekly and assigns the best-performing one via a composite score (40% Sharpe + 30% Return + 20% Win Rate + 10% (1-MaxDD)).

---

## AI Signal Modifiers (v2)

All AI features are **disabled by default** and controlled via environment variables. When disabled, signals pass through unchanged.

### 1. NLP Sentiment Analysis (FinBERT)

```
Alpaca News API → Headlines → FinBERT (local) → Sentiment Score [-1, +1]
```

- Fetches last 24h of news headlines per symbol via Alpaca's free News API
- Scores each headline using [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) (runs locally, no API cost)
- Aggregates into a per-symbol sentiment score written to `data/sentiment/scores.json`
- The signal modifier adjusts strength by `sentiment_score × weight` (default weight: 0.15)
- If sentiment strongly opposes signal direction (e.g., bearish news on a buy signal), strength is dampened by 30%

### 2. LLM Multi-Agent Analyst (Claude)

```
For each symbol:
  ├─ News Analyst (Haiku) ──────► News summary + sentiment
  ├─ Technical Analyst (Haiku) ─► RSI/MACD/BB analysis in natural language
  └─ Debate Synthesizer (Sonnet) ► conviction_score [-1, +1], bias, key_factors, risk_flags
```

- Three-step analysis pipeline using Claude Haiku (cheap sub-analysts) and Sonnet (synthesis)
- Cost: ~$0.17/day for 15 symbols, with a configurable daily budget cap (default $1.00)
- Spend tracking in `data/llm_analyst/spend_log.json` — stops analysis when budget is hit
- Conviction scores written to `data/llm_analyst/convictions.json`
- The signal modifier adjusts strength by `conviction × weight` (default weight: 0.20)

### 3. RL Strategy Selector (DQN)

```
Market Features (10-dim) → Trained DQN → Best Strategy Index (0-8)
```

**State vector (10 features):**
RSI (normalized), Bollinger %B, MACD histogram, Volume ratio, ATR %, 5-day momentum, 20-day momentum, Hour (sin), Hour (cos), Regime proxy (vol ratio)

**Training:**
- Gymnasium environment where action = select 1 of 9 strategies
- Reward = rolling Sharpe ratio + switching penalty
- Walk-forward validation: 70/30 split, only deployed if validation Sharpe > 0.5
- Trains weekly on 90 days of historical data using DQN (stable-baselines3)

**Inference:**
- Extracts features from current DataFrame
- DQN predicts best strategy index → maps to strategy key
- Falls back to optimizer-assigned strategy if model is missing or prediction fails

---

## Autonomous Agents

| Agent | Schedule | Purpose | Inputs | Outputs |
|-------|----------|---------|--------|---------|
| **Health Check** | 5:00 AM M-F | System validation | pytest, files, DB, imports | `health_report.json` |
| **Strategy Optimizer** | 5:30 AM M-F | Backtest all strategies | Historical bars, learnings | `strategy_assignments.json` |
| **Sentiment Agent** | 5:45 AM M-F | Score news sentiment | Alpaca News API | `sentiment/scores.json` |
| **Market Scanner** | 6:00 AM M-F | Screen & select equities | Market data, assignments | Updates `.env` EQUITY_SYMBOLS |
| **LLM Analyst** | 6:15 AM M-F | Claude multi-agent analysis | News + technicals | `llm_analyst/convictions.json` |
| **Trade Analyzer** | 5:00 PM M-F | Review daily trades | `trades.db` | Reports, learnings, risk adjustments |
| **RL Trainer** | Sun 2:00 AM | Retrain DQN model | 90-day bars | `rl_models/dqn_latest.zip` |

All agents communicate through **JSON files** in the `data/` directory. No agent directly modifies another agent's code or the strategy registry — they only write data files that the router reads at runtime.

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Alpaca](https://alpaca.markets/) account (free paper trading account works)
- (Optional) [Anthropic API key](https://console.anthropic.com/) for LLM analyst

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/algo-trader.git
cd algo-trader

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Quick Setup

1. **Get Alpaca API keys** from [alpaca.markets](https://alpaca.markets/) (use Paper Trading keys to start)

2. **Edit `.env`** with your keys:
```env
ALPACA_API_KEY=your_paper_key_here
ALPACA_SECRET_KEY=your_paper_secret_here
TRADING_MODE=paper
```

3. **Verify setup:**
```bash
./bot.sh health    # Run system health check
./bot.sh test      # Run 562 tests
python main.py --status  # Check Alpaca connection
```

4. **Start trading:**
```bash
./bot.sh start     # Start the bot (backgrounded)
./bot.sh dash      # Start web dashboard at http://localhost:5050
./bot.sh logs      # Follow live logs
```

---

## Configuration

### Environment Variables (`.env`)

#### Required
| Variable | Description | Example |
|----------|-------------|---------|
| `ALPACA_API_KEY` | Alpaca API key | `PK...` |
| `ALPACA_SECRET_KEY` | Alpaca secret key | `FC...` |
| `TRADING_MODE` | `paper` or `live` | `paper` |

#### Symbols
| Variable | Default | Description |
|----------|---------|-------------|
| `CRYPTO_SYMBOLS` | `BTC/USD,ETH/USD` | Crypto pairs (trade 24/7) |
| `EQUITY_SYMBOLS` | `TSLA,NVDA,AMD,...` | Stocks (market hours only) |

#### Risk Management
| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_POSITION_PCT` | `0.50` | Max single position as % of equity |
| `STOP_LOSS_PCT` | `0.025` | Trailing stop percentage |
| `DAILY_DRAWDOWN_LIMIT` | `0.10` | Halt trading if daily loss exceeds this |

#### AI Features (v2) — All Disabled by Default
| Variable | Default | Description |
|----------|---------|-------------|
| `SENTIMENT_ENABLED` | `false` | Enable FinBERT sentiment analysis |
| `SENTIMENT_WEIGHT` | `0.15` | Sentiment modifier weight |
| `LLM_ANALYST_ENABLED` | `false` | Enable Claude LLM analyst |
| `LLM_CONVICTION_WEIGHT` | `0.2` | LLM conviction modifier weight |
| `LLM_BUDGET_DAILY` | `1.00` | Max daily LLM API spend (USD) |
| `ANTHROPIC_API_KEY` | `` | Required for LLM analyst |
| `RL_STRATEGY_ENABLED` | `false` | Enable RL strategy selection |
| `RL_MIN_SHARPE_THRESHOLD` | `0.5` | Min validation Sharpe to deploy model |

---

## Usage

### Bot Commands (`bot.sh`)

```bash
# ── Lifecycle ─────────────────────────
./bot.sh start       # Start the trading bot (background)
./bot.sh stop        # Stop the trading bot
./bot.sh dash        # Start web dashboard (http://localhost:5050)
./bot.sh dash-stop   # Stop the dashboard
./bot.sh up          # Start bot + dashboard together
./bot.sh down        # Stop everything
./bot.sh status      # Show bot status + account info
./bot.sh logs        # Follow live bot logs

# ── Analysis ──────────────────────────
./bot.sh backtest    # Run strategy backtest
./bot.sh compare     # Compare all 9 strategies

# ── Agents ────────────────────────────
./bot.sh health      # Run system health check
./bot.sh optimizer   # Run Strategy Optimizer
./bot.sh scanner     # Run Market Scanner
./bot.sh sentiment   # Run Sentiment Agent (FinBERT)
./bot.sh llm         # Run LLM Analyst Agent (Claude)
./bot.sh rl-train    # Run RL Trainer Agent (DQN)
./bot.sh analyzer    # Run Trade Analyzer
./bot.sh agents      # Run all agents in sequence
./bot.sh test        # Run full test suite
```

### Python CLI

```bash
python main.py              # Run the live/paper trading bot (foreground)
python main.py --backtest   # Run backtester
python main.py --status     # Show account status and recent trades
```

### Enabling AI Features

**Sentiment Analysis** (no API key required):
```bash
# In .env:
SENTIMENT_ENABLED=true

# Run once manually to populate scores:
./bot.sh sentiment
```

**LLM Analyst** (requires Anthropic API key):
```bash
# In .env:
LLM_ANALYST_ENABLED=true
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Run once manually:
./bot.sh llm
```

**RL Strategy Selector** (requires training first):
```bash
# Train the initial model (needs Alpaca historical data):
./bot.sh rl-train

# If training succeeds (val Sharpe > 0.5), enable in .env:
RL_STRATEGY_ENABLED=true
```

### Agent Scheduling

All agents are scheduled via crontab. The cron jobs are installed automatically. To verify or modify:

```bash
crontab -l    # View scheduled agents
crontab -e    # Edit schedule
```

Default schedule (all times US Eastern):
```
5:00 AM  M-F  health_check.py
5:30 AM  M-F  strategy_optimizer.py
5:45 AM  M-F  sentiment_agent.py
6:00 AM  M-F  market_scanner.py
6:15 AM  M-F  llm_analyst.py
5:00 PM  M-F  trade_analyzer.py
2:00 AM  Sun  rl_trainer.py
```

---

## Testing

```bash
# Run full suite (562 tests)
python -m pytest

# Run with coverage report
python -m pytest --cov --cov-report=term-missing

# Run specific module tests
python -m pytest tests/test_signal_modifiers.py -v
python -m pytest tests/test_rl_features.py -v
python -m pytest tests/test_router_enhanced.py -v

# Run only v2 tests
python -m pytest tests/test_news_client.py tests/test_sentiment_analyzer.py \
                 tests/test_sentiment_agent.py tests/test_llm_client.py \
                 tests/test_llm_analyst.py tests/test_rl_features.py \
                 tests/test_rl_environment.py tests/test_rl_selector.py \
                 tests/test_rl_trainer.py tests/test_router_enhanced.py -v
```

Current coverage: **562 tests, 95% line coverage**.

---

## Project Structure

```
algo-trader/
├── agents/                              # 7 autonomous agents
│   ├── health_check.py                  # System validation (pytest, files, DB, imports)
│   ├── strategy_optimizer.py            # Backtest 9 strategies, assign best per symbol
│   ├── market_scanner.py                # Screen 80 stocks, select 10-20 for trading
│   ├── sentiment_agent.py               # FinBERT news sentiment scoring
│   ├── llm_analyst.py                   # Claude 3-step analysis (news→tech→synthesis)
│   ├── rl_trainer.py                    # Weekly DQN model retraining
│   └── trade_analyzer.py               # Daily trade review, learnings, risk adjustments
│
├── core/                                # Trading engine & infrastructure
│   ├── engine.py                        # Main 60s trading loop
│   ├── broker.py                        # Alpaca API (crypto + equities, orders, bars)
│   ├── risk_manager.py                  # Trailing stops, position sizing, drawdown
│   ├── signal_modifiers.py              # Sentiment + LLM conviction blending
│   ├── news_client.py                   # Alpaca News API client
│   ├── sentiment_analyzer.py            # FinBERT pipeline wrapper
│   ├── llm_client.py                    # Anthropic SDK (retries, budget tracking)
│   ├── rl_features.py                   # 10-dim feature extraction for RL
│   ├── rl_environment.py                # Gymnasium env for strategy selection
│   └── rl_strategy_selector.py          # DQN inference + fallback
│
├── strategies/                          # 9 trading strategies
│   ├── router.py                        # Dispatches symbol → strategy + modifiers
│   ├── mean_reversion_aggressive.py     # RSI(10) + BB(15, 1.5σ)
│   ├── mean_reversion.py               # RSI(14) + BB(20, 2σ)
│   ├── volume_profile.py               # OBV + MFI + volume spikes
│   ├── momentum.py                      # EMA(50) + MACD + breakouts
│   ├── macd_crossover.py               # MACD/signal crossovers
│   ├── triple_ema.py                    # EMA(8/21/55) + ADX
│   ├── rsi_divergence.py               # Price-RSI divergence
│   ├── scalper.py                       # EMA(5/13) + VWAP + StochRSI
│   └── ensemble.py                      # Momentum + Mean Reversion vote
│
├── utils/
│   └── logger.py                        # Rich console + SQLite trade logging
│
├── data/                                # Agent communication (JSON files)
│   ├── sentiment/scores.json            # FinBERT scores per symbol
│   ├── llm_analyst/convictions.json     # Claude conviction scores
│   ├── llm_analyst/spend_log.json       # Daily LLM API spend tracking
│   ├── rl_models/dqn_latest.zip         # Trained DQN model
│   ├── optimizer/                       # Backtest results & assignments
│   ├── scanner/                         # Selected symbols & candidates
│   ├── analyzer/reports/                # Daily YYYY-MM-DD.json reports
│   └── history/                         # Daily archives
│
├── tests/                               # 562 tests, 95% coverage
│   ├── conftest.py                      # Shared fixtures
│   └── test_*.py                        # 29 test files
│
├── config.py                            # Environment-based configuration
├── main.py                              # Entry point (bot, backtest, status)
├── compare_strategies.py                # Strategy benchmarking tool
├── dashboard.py                         # Flask web dashboard
├── backtest.py                          # Historical backtester
├── bot.sh                               # CLI wrapper for all commands
├── trades.db                            # SQLite database (trades + equity snapshots)
├── requirements.txt                     # Python dependencies
├── .coveragerc                          # Coverage configuration
└── .env                                 # API keys, symbols, feature flags
```

---

## Graceful Degradation

The system is designed to **never fail due to a missing AI component**:

| Scenario | Behavior |
|----------|----------|
| FinBERT model not downloaded | Sentiment agent logs error, signals pass through unchanged |
| Alpaca News API unreachable | `fetch_news()` returns empty list, sentiment skipped |
| `ANTHROPIC_API_KEY` not set | LLM agent exits early, convictions file not created |
| LLM daily budget exceeded | Remaining symbols skipped, existing scores still used |
| RL model file missing | Selector returns `None`, router uses STRATEGY_MAP fallback |
| RL prediction fails | Falls back to optimizer-assigned strategy |
| `scores.json` missing/corrupt | `apply_sentiment()` returns signal unchanged |
| `convictions.json` missing/corrupt | `apply_llm_conviction()` returns signal unchanged |
| Any feature flag set to `false` | Modifier is never called, zero overhead |

