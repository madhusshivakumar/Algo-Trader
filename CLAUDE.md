# Algo-Trader — Claude Code Guidelines

## Project Overview
Python algorithmic trading bot using Alpaca API. Trades crypto (24/7) and equities (market hours). 9 strategies with AI signal modifiers (FinBERT sentiment, Claude LLM analyst, RL strategy selector). All AI features are behind feature flags.

## Architecture Rules

### Signal Contract
Every strategy's `compute_signals(df)` must return:
```python
{"action": "buy"|"sell"|"hold", "reason": str, "strength": float(0.3-1.0)}
```
Signal modifiers may add `sentiment_score`, `llm_conviction`, `rl_selected` keys. Never break this contract.

### Agent Communication
Agents communicate via JSON files in `data/`, never by importing each other. The router reads these files. Pattern: agent writes `data/<module>/output.json`, consumer reads it with graceful fallback if missing/stale.

### Feature Flags
All AI/ML upgrades are behind `_ENABLED` env vars (default `false`). If a feature is disabled or its data is missing, the system must behave identically to the base version. Never make a feature flag's disabled path produce errors.

### Strategy Registry
`strategies/router.py` has two layers:
- `STRATEGY_REGISTRY`: hardcoded name-to-function map (never modified at runtime)
- `STRATEGY_MAP`: symbol-to-name map (loaded from agent JSON files)
Agents only write string-based mappings. The router resolves strings to functions.

## Coding Standards

### Python
- Python 3.11+, type hints on all public functions
- Use `dataclass` for data containers, not plain dicts (except signal contract)
- Imports: stdlib, third-party, local — separated by blank lines
- No wildcard imports. No circular imports between `core/`, `strategies/`, `agents/`
- Use `Config` class for all configuration; never read `os.getenv()` outside `config.py`

### Error Handling
- Broker/API calls: always wrap in try/except, log the error, return None or fallback
- Never let a single symbol's failure crash the engine loop
- Use `log.error()` for failures, `log.warning()` for degraded behavior, `log.info()` for normal ops
- Validate at system boundaries (API responses, file reads), trust internal code

### Testing
- Every new module must have a corresponding `tests/test_<module>.py`
- Target >90% line coverage on new code
- Mock external dependencies (Alpaca API, Anthropic API, filesystem) — never make real API calls in tests
- Use `patch.dict("sys.modules", ...)` when mocking imports that happen inside functions (lazy imports)
- Test the happy path, error/fallback paths, and edge cases (empty data, missing files, disabled flags)
- Run `python -m pytest tests/ -x -q` before considering any change complete

### File Organization
```
config.py          — all env vars and constants
core/              — broker, engine, risk manager, order manager, signal modifiers
strategies/        — one file per strategy, plus router.py
agents/            — scheduled agents (daily/weekly), write to data/
utils/             — logger, helpers
tests/             — mirrors source structure with test_ prefix
data/              — runtime state (gitignored), agent outputs
```

## Key Patterns

### Adding a New Strategy
1. Create `strategies/<name>.py` with `compute_signals(df) -> dict`
2. Add to `STRATEGY_REGISTRY` and `STRATEGY_DISPLAY_NAMES` in `router.py`
3. Add to `_DEFAULT_MAP` for relevant symbols
4. Write tests in `tests/test_strategies.py` or a dedicated file

### Adding a New Signal Modifier
1. Add function to `core/signal_modifiers.py`
2. Gate behind a feature flag in `config.py`
3. Call from `router.py:compute_signals()` after technical signal
4. Modifier must return signal unchanged if data is missing

### Adding a New Agent
1. Create `agents/<name>.py` with `main()` entrypoint
2. Agent writes output to `data/<name>/output.json`
3. Add command to `bot.sh`
4. Add staleness check to `agents/health_check.py`
5. Write tests mocking all external dependencies

## Common Pitfalls
- Alpaca crypto symbols use `/` (e.g., `BTC/USD`) but orders need it stripped (`BTCUSD`)
- `broker.get_position()` uses stripped symbol internally
- PDT protection is in-memory only — lost on restart (known gap being fixed)
- SQLite operations are not thread-safe; each call opens/closes its own connection
- FinBERT and RL models are lazy-loaded on first use; don't import at module level
- `STRATEGY_MAP` is loaded once at import time; restart bot to pick up new assignments

## Commands
```bash
./bot.sh start          # start trading bot
./bot.sh stop           # stop trading bot
./bot.sh status         # check bot status
./bot.sh report         # run trade analyzer
./bot.sh health         # run health check
./bot.sh optimize       # run strategy optimizer
./bot.sh scan           # run market scanner
./bot.sh sentiment      # run sentiment analysis
./bot.sh llm            # run LLM analyst
./bot.sh rl-train       # run RL trainer
python -m pytest tests/ -x -q           # run all tests
python -m pytest tests/ --cov --cov-report=term-missing  # with coverage
```

## Do NOT
- Commit `.env`, `trades.db`, or anything in `data/` (check `.gitignore`)
- Add dependencies without updating `requirements.txt`
- Change the signal contract without updating all strategies and tests
- Use `time.sleep()` in tests
- Skip writing tests for new code
- Add `Co-Authored-By:` trailers to commits unless the user explicitly asks for them
