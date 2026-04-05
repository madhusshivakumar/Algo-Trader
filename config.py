import os
from datetime import datetime, time as dtime
import pytz
from dotenv import load_dotenv

load_dotenv(override=True)


class Config:
    # Alpaca
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
    TRADING_MODE = os.getenv("TRADING_MODE", "paper")

    # Crypto symbols (24/7)
    CRYPTO_SYMBOLS = [
        s.strip()
        for s in os.getenv("CRYPTO_SYMBOLS", "BTC/USD,ETH/USD").split(",")
        if s.strip()
    ]

    # Equity symbols (market hours only)
    EQUITY_SYMBOLS = [
        s.strip()
        for s in os.getenv("EQUITY_SYMBOLS", "TSLA,NVDA,AMD,AAPL,META,SPY").split(",")
        if s.strip()
    ]

    # Combined for backward compatibility
    SYMBOLS = CRYPTO_SYMBOLS + EQUITY_SYMBOLS

    # Account
    STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "100000"))

    # Risk management
    MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.50"))
    STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.025"))
    DAILY_DRAWDOWN_LIMIT = float(os.getenv("DAILY_DRAWDOWN_LIMIT", "0.10"))

    # Strategy parameters
    MOMENTUM_LOOKBACK = 20
    MOMENTUM_VOLUME_MULT = 1.5
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2.0

    # Timeframe
    CANDLE_TIMEFRAME = "1Min"
    CANDLE_HISTORY_DAYS = 30

    # Trailing stop
    TRAILING_STOP_PCT = 0.02

    # Market hours (US Eastern)
    MARKET_TZ = pytz.timezone("US/Eastern")
    MARKET_OPEN = dtime(9, 30)
    MARKET_CLOSE = dtime(16, 0)

    # PDT protection — avoid day trading equities under $25k
    PDT_PROTECTION = True

    # ── v2 Upgrades ────────────────────────────────────────────────────
    # Sentiment analysis (FinBERT + Alpaca News API)
    SENTIMENT_ENABLED = os.getenv("SENTIMENT_ENABLED", "true").lower() == "true"
    SENTIMENT_WEIGHT = float(os.getenv("SENTIMENT_WEIGHT", "0.15"))

    # LLM multi-agent analyst (Claude API)
    LLM_ANALYST_ENABLED = os.getenv("LLM_ANALYST_ENABLED", "true").lower() == "true"
    LLM_CONVICTION_WEIGHT = float(os.getenv("LLM_CONVICTION_WEIGHT", "0.2"))
    LLM_QUICK_MODEL = os.getenv("LLM_QUICK_MODEL", "claude-haiku-4-5-20251001")
    LLM_DEEP_MODEL = os.getenv("LLM_DEEP_MODEL", "claude-sonnet-4-20250514")
    LLM_BUDGET_DAILY = float(os.getenv("LLM_BUDGET_DAILY", "1.00"))
    LLM_ANALYST_V2_ENABLED = os.getenv("LLM_ANALYST_V2_ENABLED", "true").lower() == "true"
    LLM_FEEDBACK_LOOP_ENABLED = os.getenv("LLM_FEEDBACK_LOOP_ENABLED", "true").lower() == "true"
    LLM_INFLUENCER_TRACKING_ENABLED = os.getenv("LLM_INFLUENCER_TRACKING_ENABLED", "true").lower() == "true"
    LLM_PATTERN_DISCOVERY_ENABLED = os.getenv("LLM_PATTERN_DISCOVERY_ENABLED", "true").lower() == "true"
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    # RL-based strategy selection (DQN)
    RL_STRATEGY_ENABLED = os.getenv("RL_STRATEGY_ENABLED", "true").lower() == "true"
    RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", "data/rl_models/dqn_latest.zip")
    RL_MIN_SHARPE_THRESHOLD = float(os.getenv("RL_MIN_SHARPE_THRESHOLD", "0.5"))

    # ── v3 Production Hardening ─────────────────────────────────────
    # Sprint 1: Order management & state persistence
    ORDER_MANAGEMENT_ENABLED = os.getenv("ORDER_MANAGEMENT_ENABLED", "true").lower() == "true"
    STATE_PERSISTENCE_ENABLED = os.getenv("STATE_PERSISTENCE_ENABLED", "true").lower() == "true"
    ORDER_STALE_TIMEOUT_SECONDS = int(os.getenv("ORDER_STALE_TIMEOUT_SECONDS", "300"))

    # Sprint 2: Risk management upgrades
    ATR_STOPS_ENABLED = os.getenv("ATR_STOPS_ENABLED", "true").lower() == "true"
    ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER", "2.0"))
    ATR_STOP_MULTIPLIER_CRYPTO = float(os.getenv("ATR_STOP_MULTIPLIER_CRYPTO", "3.0"))
    MAX_HOLD_HOURS = float(os.getenv("MAX_HOLD_HOURS", "48"))
    MAX_HOLD_HOURS_CRYPTO = float(os.getenv("MAX_HOLD_HOURS_CRYPTO", "24"))
    VOLATILITY_SIZING_ENABLED = os.getenv("VOLATILITY_SIZING_ENABLED", "true").lower() == "true"
    MAX_TRADES_PER_DAY_CRYPTO = int(os.getenv("MAX_TRADES_PER_DAY_CRYPTO", "8"))
    MAX_TRADES_PER_DAY_EQUITY = int(os.getenv("MAX_TRADES_PER_DAY_EQUITY", "4"))
    MIN_SIGNAL_STRENGTH = float(os.getenv("MIN_SIGNAL_STRENGTH", "0.55"))
    CORRELATION_CHECK_ENABLED = os.getenv("CORRELATION_CHECK_ENABLED", "true").lower() == "true"
    CORRELATION_THRESHOLD = float(os.getenv("CORRELATION_THRESHOLD", "0.7"))
    SENTIMENT_FRESHNESS_CHECK = os.getenv("SENTIMENT_FRESHNESS_CHECK", "true").lower() == "true"
    SENTIMENT_MAX_AGE_HOURS = float(os.getenv("SENTIMENT_MAX_AGE_HOURS", "24.0"))
    LLM_FRESHNESS_CHECK = os.getenv("LLM_FRESHNESS_CHECK", "true").lower() == "true"
    LLM_MAX_AGE_HOURS = float(os.getenv("LLM_MAX_AGE_HOURS", "24.0"))

    # Sprint 3: Execution & performance
    PARALLEL_FETCH_ENABLED = os.getenv("PARALLEL_FETCH_ENABLED", "true").lower() == "true"
    FETCH_WORKERS = int(os.getenv("FETCH_WORKERS", "5"))
    DB_ROTATION_ENABLED = os.getenv("DB_ROTATION_ENABLED", "true").lower() == "true"
    DB_ROTATION_MAX_ROWS = int(os.getenv("DB_ROTATION_MAX_ROWS", "50000"))
    DB_ROTATION_KEEP_DAYS = int(os.getenv("DB_ROTATION_KEEP_DAYS", "30"))
    DB_ROTATION_MAX_AGE_DAYS = int(os.getenv("DB_ROTATION_MAX_AGE_DAYS", "90"))

    # Sprint 4: Backtesting & monitoring
    HOT_RELOAD_ENABLED = os.getenv("HOT_RELOAD_ENABLED", "true").lower() == "true"
    DRIFT_DETECTION_ENABLED = os.getenv("DRIFT_DETECTION_ENABLED", "true").lower() == "true"
    DRIFT_LOOKBACK_DAYS = int(os.getenv("DRIFT_LOOKBACK_DAYS", "7"))
    DRIFT_MIN_TRADES = int(os.getenv("DRIFT_MIN_TRADES", "5"))

    # ── Transaction Cost Modeling ──────────────────────────────────────
    TC_ENABLED = os.getenv("TC_ENABLED", "true").lower() == "true"
    TC_COMMISSION_PCT = float(os.getenv("TC_COMMISSION_PCT", "0.0"))
    TC_SPREAD_BPS_EQUITY = float(os.getenv("TC_SPREAD_BPS_EQUITY", "5.0"))
    TC_SPREAD_BPS_CRYPTO = float(os.getenv("TC_SPREAD_BPS_CRYPTO", "15.0"))
    TC_SLIPPAGE_BPS = float(os.getenv("TC_SLIPPAGE_BPS", "3.0"))

    # ── Multi-Timeframe Analysis ──────────────────────────────────────
    MTF_ENABLED = os.getenv("MTF_ENABLED", "true").lower() == "true"
    MTF_WEIGHT = max(0.0, min(float(os.getenv("MTF_WEIGHT", "0.15")), 0.5))

    # ── Position Reconciliation ────────────────────────────────────────
    POSITION_RECONCILIATION_ENABLED = os.getenv("POSITION_RECONCILIATION_ENABLED", "true").lower() == "true"
    RECONCILIATION_INTERVAL_CYCLES = max(1, int(os.getenv("RECONCILIATION_INTERVAL_CYCLES", "10")))
    RECONCILIATION_AUTO_FIX = os.getenv("RECONCILIATION_AUTO_FIX", "true").lower() == "true"
    RECONCILIATION_ENTRY_TOLERANCE = float(os.getenv("RECONCILIATION_ENTRY_TOLERANCE", "0.02"))

    # ── Earnings Calendar Awareness ─────────────────────────────────
    EARNINGS_CALENDAR_ENABLED = os.getenv("EARNINGS_CALENDAR_ENABLED", "true").lower() == "true"
    EARNINGS_BLACKOUT_DAYS = int(os.getenv("EARNINGS_BLACKOUT_DAYS", "2"))
    EARNINGS_CLOSE_POSITIONS = os.getenv("EARNINGS_CLOSE_POSITIONS", "true").lower() == "true"
    EARNINGS_SIZE_REDUCTION = float(os.getenv("EARNINGS_SIZE_REDUCTION", "0.5"))

    # ── VWAP/TWAP Execution ─────────────────────────────────────────
    VWAP_TWAP_ENABLED = os.getenv("VWAP_TWAP_ENABLED", "true").lower() == "true"
    VWAP_TWAP_ALGO = os.getenv("VWAP_TWAP_ALGO", "twap")
    VWAP_TWAP_NUM_SLICES = int(os.getenv("VWAP_TWAP_NUM_SLICES", "5"))
    VWAP_TWAP_INTERVAL_SECONDS = int(os.getenv("VWAP_TWAP_INTERVAL_SECONDS", "60"))
    VWAP_TWAP_MIN_NOTIONAL = float(os.getenv("VWAP_TWAP_MIN_NOTIONAL", "100.0"))
    VWAP_VOLUME_LOOKBACK_DAYS = int(os.getenv("VWAP_VOLUME_LOOKBACK_DAYS", "5"))

    # ── Monte Carlo Simulation ────────────────────────────────────────
    MONTE_CARLO_ENABLED = os.getenv("MONTE_CARLO_ENABLED", "true").lower() == "true"
    MONTE_CARLO_NUM_SIMULATIONS = int(os.getenv("MONTE_CARLO_NUM_SIMULATIONS", "1000"))
    MONTE_CARLO_CONFIDENCE_LEVELS = [0.95, 0.99]  # VaR confidence levels
    MONTE_CARLO_HORIZON_DAYS = int(os.getenv("MONTE_CARLO_HORIZON_DAYS", "252"))

    # ── Portfolio Optimization ────────────────────────────────────────
    PORTFOLIO_OPTIMIZATION_ENABLED = os.getenv("PORTFOLIO_OPTIMIZATION_ENABLED", "true").lower() == "true"
    KELLY_SIZING_ENABLED = os.getenv("KELLY_SIZING_ENABLED", "true").lower() == "true"
    KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))  # fractional Kelly (conservative)
    KELLY_MIN_TRADES = int(os.getenv("KELLY_MIN_TRADES", "20"))  # min trades for reliable estimate
    MEAN_VARIANCE_ENABLED = os.getenv("MEAN_VARIANCE_ENABLED", "true").lower() == "true"
    MEAN_VARIANCE_LOOKBACK_DAYS = int(os.getenv("MEAN_VARIANCE_LOOKBACK_DAYS", "60"))
    MEAN_VARIANCE_RISK_FREE_RATE = float(os.getenv("MEAN_VARIANCE_RISK_FREE_RATE", "0.05"))
    MAX_SINGLE_POSITION_PCT = float(os.getenv("MAX_SINGLE_POSITION_PCT", "0.25"))

    # ── Alerting ──────────────────────────────────────────────────────
    ALERTING_ENABLED = os.getenv("ALERTING_ENABLED", "false").lower() == "true"
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
    ALERT_ON_TRADE = os.getenv("ALERT_ON_TRADE", "true").lower() == "true"
    ALERT_ON_ERROR = os.getenv("ALERT_ON_ERROR", "true").lower() == "true"

    @classmethod
    def is_paper(cls) -> bool:
        return cls.TRADING_MODE == "paper"

    @classmethod
    def is_crypto(cls, symbol: str) -> bool:
        """Check if a symbol is crypto (contains '/')."""
        return "/" in symbol

    @classmethod
    def is_market_open(cls) -> bool:
        """Check if US stock market is currently open."""
        now = datetime.now(cls.MARKET_TZ)
        # Weekends
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        return cls.MARKET_OPEN <= current_time < cls.MARKET_CLOSE

    @classmethod
    def validate(cls):
        if not cls.ALPACA_API_KEY or cls.ALPACA_API_KEY == "your_api_key_here":
            raise ValueError("Set ALPACA_API_KEY in .env file")
        if not cls.ALPACA_SECRET_KEY or cls.ALPACA_SECRET_KEY == "your_secret_key_here":
            raise ValueError("Set ALPACA_SECRET_KEY in .env file")
        if cls.TRADING_MODE not in ("paper", "live"):
            raise ValueError(f"TRADING_MODE must be 'paper' or 'live', got '{cls.TRADING_MODE}'")
        if not cls.SYMBOLS:
            raise ValueError("No symbols configured — set CRYPTO_SYMBOLS or EQUITY_SYMBOLS")
        if cls.MAX_POSITION_PCT <= 0 or cls.MAX_POSITION_PCT > 1.0:
            raise ValueError(f"MAX_POSITION_PCT must be (0, 1.0], got {cls.MAX_POSITION_PCT}")
        if cls.STOP_LOSS_PCT <= 0 or cls.STOP_LOSS_PCT > 0.5:
            raise ValueError(f"STOP_LOSS_PCT must be (0, 0.5], got {cls.STOP_LOSS_PCT}")
        if cls.DAILY_DRAWDOWN_LIMIT <= 0 or cls.DAILY_DRAWDOWN_LIMIT > 1.0:
            raise ValueError(f"DAILY_DRAWDOWN_LIMIT must be (0, 1.0], got {cls.DAILY_DRAWDOWN_LIMIT}")
        if cls.LLM_ANALYST_ENABLED and not cls.ANTHROPIC_API_KEY:
            raise ValueError("LLM_ANALYST_ENABLED requires ANTHROPIC_API_KEY")
