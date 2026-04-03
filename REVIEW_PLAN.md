# Algo Trader — Full System Review Plan

**Target start:** After market close, April 3 2026 (~1:00 PM PT)
**Scope:** Every module reviewed for bugs, edge cases, security, reliability, and test coverage
**Codebase:** 26,521 lines across 53 source files + 45 test files + infra

---

## Module 1: Configuration & Entry Points
**Files:** `config.py` (192), `main.py` (50)
**Review focus:**
- All env vars have sane defaults and validation
- No secrets hardcoded, no `os.getenv()` outside config.py
- Feature flag interactions — any conflicting combinations?
- `Config.validate()` — does it catch all invalid states?
- Entry point error handling — what happens if broker is unreachable at startup?

---

## Module 2: Core Engine
**Files:** `core/engine.py` (828)
**Review focus:**
- Main trading loop — can it hang, crash, or silently stop?
- Symbol iteration — does one symbol's failure isolate from others?
- Feature flag gating — are all v2/v3 features properly gated?
- SIGHUP reload — does it actually work without race conditions?
- Memory leaks — anything accumulating over a full trading day?
- Signal flow: strategy → signal modifiers → risk checks → order → execution
- Integration points with all subsystems (broker, risk, order manager, execution algo, portfolio optimizer)
- Logging — enough to diagnose issues, not so much it floods

---

## Module 3: Broker & Order Management
**Files:** `core/broker.py` (271), `core/order_manager.py` (407)
**Review focus:**
- API error handling — retries, timeouts, rate limits
- Symbol format handling (BTC/USD vs BTCUSD)
- Position tracking accuracy — does it match Alpaca's view?
- Order lifecycle — submit, fill, partial fill, reject, cancel
- Concurrent order safety — can two cycles submit conflicting orders?
- PDT protection — in-memory only, what happens on restart?
- Fractional shares and crypto quantity precision

---

## Module 4: Risk Management
**Files:** `core/risk_manager.py` (227), `core/transaction_costs.py` (154)
**Review focus:**
- Daily loss limits — can they be bypassed by timing?
- Position size limits — enforced before or after signal modifiers?
- Max drawdown protection — does it actually stop trading?
- Correlation risk — multiple correlated positions?
- Transaction cost model accuracy — slippage, spread, commission
- Edge cases: zero equity, negative P&L, market gaps

---

## Module 5: Execution & Order Algo
**Files:** `core/execution_algo.py` (332)
**Review focus:**
- VWAP/TWAP child order splitting logic
- Partial fill handling — filled_notional vs total_notional (previous bug)
- Timeout handling for stale execution plans
- What happens if broker rejects a child order mid-execution?
- Feature flag disabled path — clean passthrough?

---

## Module 6: Signal Modifiers & Strategy Router
**Files:** `core/signal_modifiers.py` (201), `strategies/router.py` (170)
**Review focus:**
- Signal contract compliance — do all modifiers preserve the contract?
- Modifier ordering — does sequence matter? Is it deterministic?
- Stale data handling — what if sentiment/LLM data is hours old?
- Strategy registry vs strategy map — any desync possible?
- Default map fallback — correct for all symbols?
- Ensemble strategy — how does it combine signals?

---

## Module 7: Trading Strategies (9 strategies)
**Files:** `strategies/` — `momentum.py`, `mean_reversion.py`, `mean_reversion_aggressive.py`, `macd_crossover.py`, `rsi_divergence.py`, `triple_ema.py`, `scalper.py`, `volume_profile.py`, `ensemble.py`
**Review focus:**
- Signal contract compliance — action, reason, strength all correct?
- Edge cases: empty DataFrame, single row, NaN values, zero volume
- Indicator calculation correctness — periods, smoothing
- Strength values — actually bounded 0.3–1.0?
- Are any strategies redundant or contradictory?

---

## Module 8: AI/ML Signal Pipeline
**Files:** `core/sentiment_analyzer.py` (97), `core/llm_client.py` (204), `core/rl_strategy_selector.py` (90), `core/rl_features.py` (148), `core/rl_environment.py` (126)
**Review focus:**
- FinBERT — model loading, memory usage, error handling
- LLM client — prompt injection risk, response parsing, timeout, cost
- RL selector — model loading, stale model fallback, action space
- Feature extraction — NaN handling, normalization
- All gracefully degrade when disabled or data missing?

---

## Module 9: Portfolio Analytics
**Files:** `core/portfolio_metrics.py` (375), `core/portfolio_optimizer.py` (363), `core/monte_carlo.py` (338)
**Review focus:**
- Return calculations — divide by zero (already has warnings)
- Kelly criterion — edge cases (0 trades, 100% win rate, negative payoff)
- Mean-variance optimizer — convergence, constraint violations
- Monte Carlo — statistical validity, bootstrap assumptions
- Cache invalidation in PortfolioOptimizer — stale allocations?

---

## Module 10: State & Data Management
**Files:** `core/state_store.py` (292), `core/data_fetcher.py` (82), `utils/logger.py` (187), `utils/db_rotation.py` (109)
**Review focus:**
- SQLite thread safety — each call opens/closes connection?
- DB file locking — engine + dashboard + agents all accessing same DB
- State persistence — what survives restart vs what's lost?
- Data fetcher — cache behavior, stale data, API errors
- Log rotation — disk space management
- DB rotation — does it lose data? Corruption risk?

---

## Module 11: Monitoring & Resilience
**Files:** `core/alerting.py` (238), `core/drift_detector.py` (180), `core/position_reconciler.py` (176), `core/walk_forward.py` (179), `core/config_reloader.py` (125), `core/multi_timeframe.py` (257), `core/influencer_registry.py` (359)
**Review focus:**
- Alerting — does it actually notify anyone? Channels configured?
- Drift detection — false positives? Threshold tuning?
- Position reconciliation — broker vs local state divergence handling
- Walk-forward validation — is it used? Does it feed back?
- Config reloader — race conditions with engine loop?
- Multi-timeframe — data alignment across timeframes
- Influencer registry — what is this? Is it used?

---

## Module 12: Agents (Pre-Market)
**Files:** `agents/earnings_calendar.py` (169), `agents/sentiment_agent.py` (97), `agents/market_scanner.py` (471), `agents/llm_analyst.py` (275), `agents/llm_analyst_v2.py` (831)
**Review focus:**
- Output format — all write valid JSON to data/<name>/output.json?
- Error handling — API failures, empty results, malformed responses
- Staleness — do consumers check data age?
- LLM analyst v1 vs v2 — which is used? Is v1 dead code?
- Market scanner — symbol selection logic, is it sound?
- Resource usage — memory, API calls, rate limits

---

## Module 13: Agents (Post-Market & Weekly)
**Files:** `agents/trade_analyzer.py` (620), `agents/health_check.py` (578), `agents/strategy_optimizer.py` (272), `agents/rl_trainer.py` (263), `agents/conviction_scorer.py` (432), `agents/pattern_discoverer.py` (265)
**Review focus:**
- Trade analyzer — P&L calculation accuracy, report completeness
- Health check — does it catch real issues? False alarms?
- Strategy optimizer — backtest methodology, overfitting risk
- RL trainer — training stability, model versioning, rollback
- Conviction scorer — scoring methodology, does it feed back to LLM weights?
- Pattern discoverer — what patterns? How are they used?

---

## Module 14: Dashboard
**Files:** `dashboard.py` (351)
**Review focus:**
- Security — any unauthenticated endpoints that modify state?
- SQL injection via query parameters?
- XSS in rendered data?
- Performance — expensive queries on every page load?
- Does it work with empty DB (fresh start)?
- Docker binding (0.0.0.0) — is this exposed beyond localhost?

---

## Module 15: Backtesting & Comparison
**Files:** `backtest.py` (298), `compare_strategies.py` (281)
**Review focus:**
- Look-ahead bias — any future data leaking into signals?
- Survivorship bias — symbol selection
- Transaction costs included?
- Monte Carlo integration — working correctly?
- Results accuracy — do they match live trading behavior?

---

## Module 16: Infrastructure & DevOps
**Files:** `Dockerfile`, `docker-compose.yml`, `crontab`, `bot.sh`, `scripts/startup.sh`, `scripts/entrypoint.sh`, `scripts/run_premarket.sh`, `.dockerignore`, `.env.example`, `requirements.txt`
**Review focus:**
- Dockerfile — security (running as root?), image size, layer caching
- Compose — volume permissions, restart behavior, health checks
- Crontab — timing correctness (UTC vs ET vs PT)
- Startup script — all failure paths handled?
- .env.example — complete? Matches actual config.py?
- requirements.txt — pinned versions? Any vulnerable packages?
- bot.sh — all commands work for both local and Docker?

---

## Module 17: Test Suite
**Files:** 45 test files, 1324 tests
**Review focus:**
- Coverage gaps — any untested modules or paths?
- Mock quality — are mocks realistic? Do they mask real bugs?
- Flaky tests — any timing-dependent or order-dependent tests?
- Integration tests — do they cover the full signal→trade flow?
- Missing test scenarios from modules reviewed above

---

## Review Process (per module)
1. Read every line of source code
2. Check error handling and edge cases
3. Verify test coverage (run with --cov for that module)
4. Note bugs, risks, and improvements
5. Fix critical bugs immediately
6. Log non-critical issues for follow-up
7. Re-run tests after any fixes
