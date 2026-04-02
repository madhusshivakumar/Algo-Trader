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

    # ── v2 Upgrades (all disabled by default) ────────────────────────
    # Sentiment analysis (FinBERT + Alpaca News API)
    SENTIMENT_ENABLED = os.getenv("SENTIMENT_ENABLED", "false").lower() == "true"
    SENTIMENT_WEIGHT = float(os.getenv("SENTIMENT_WEIGHT", "0.15"))

    # LLM multi-agent analyst (Claude API)
    LLM_ANALYST_ENABLED = os.getenv("LLM_ANALYST_ENABLED", "false").lower() == "true"
    LLM_CONVICTION_WEIGHT = float(os.getenv("LLM_CONVICTION_WEIGHT", "0.2"))
    LLM_QUICK_MODEL = os.getenv("LLM_QUICK_MODEL", "claude-haiku-4-5-20251001")
    LLM_DEEP_MODEL = os.getenv("LLM_DEEP_MODEL", "claude-sonnet-4-20250514")
    LLM_BUDGET_DAILY = float(os.getenv("LLM_BUDGET_DAILY", "1.00"))
    LLM_ANALYST_V2_ENABLED = os.getenv("LLM_ANALYST_V2_ENABLED", "false").lower() == "true"
    LLM_FEEDBACK_LOOP_ENABLED = os.getenv("LLM_FEEDBACK_LOOP_ENABLED", "false").lower() == "true"
    LLM_INFLUENCER_TRACKING_ENABLED = os.getenv("LLM_INFLUENCER_TRACKING_ENABLED", "false").lower() == "true"
    LLM_PATTERN_DISCOVERY_ENABLED = os.getenv("LLM_PATTERN_DISCOVERY_ENABLED", "false").lower() == "true"
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    # RL-based strategy selection (DQN)
    RL_STRATEGY_ENABLED = os.getenv("RL_STRATEGY_ENABLED", "false").lower() == "true"
    RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", "data/rl_models/dqn_latest.zip")
    RL_MIN_SHARPE_THRESHOLD = float(os.getenv("RL_MIN_SHARPE_THRESHOLD", "0.5"))

    # ── v3 Production Hardening ─────────────────────────────────────
    # Sprint 1: Order management & state persistence
    ORDER_MANAGEMENT_ENABLED = os.getenv("ORDER_MANAGEMENT_ENABLED", "false").lower() == "true"
    STATE_PERSISTENCE_ENABLED = os.getenv("STATE_PERSISTENCE_ENABLED", "false").lower() == "true"
    ORDER_STALE_TIMEOUT_SECONDS = int(os.getenv("ORDER_STALE_TIMEOUT_SECONDS", "300"))

    # Sprint 2: Risk management upgrades
    ATR_STOPS_ENABLED = os.getenv("ATR_STOPS_ENABLED", "false").lower() == "true"
    ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER", "2.0"))
    VOLATILITY_SIZING_ENABLED = os.getenv("VOLATILITY_SIZING_ENABLED", "false").lower() == "true"
    CORRELATION_CHECK_ENABLED = os.getenv("CORRELATION_CHECK_ENABLED", "false").lower() == "true"
    CORRELATION_THRESHOLD = float(os.getenv("CORRELATION_THRESHOLD", "0.7"))
    SENTIMENT_FRESHNESS_CHECK = os.getenv("SENTIMENT_FRESHNESS_CHECK", "false").lower() == "true"
    SENTIMENT_MAX_AGE_HOURS = float(os.getenv("SENTIMENT_MAX_AGE_HOURS", "24.0"))

    # Sprint 3: Execution & performance
    PARALLEL_FETCH_ENABLED = os.getenv("PARALLEL_FETCH_ENABLED", "false").lower() == "true"
    FETCH_WORKERS = int(os.getenv("FETCH_WORKERS", "5"))
    DB_ROTATION_ENABLED = os.getenv("DB_ROTATION_ENABLED", "false").lower() == "true"
    DB_ROTATION_MAX_ROWS = int(os.getenv("DB_ROTATION_MAX_ROWS", "50000"))
    DB_ROTATION_KEEP_DAYS = int(os.getenv("DB_ROTATION_KEEP_DAYS", "30"))

    # Sprint 4: Backtesting & monitoring
    HOT_RELOAD_ENABLED = os.getenv("HOT_RELOAD_ENABLED", "false").lower() == "true"
    DRIFT_DETECTION_ENABLED = os.getenv("DRIFT_DETECTION_ENABLED", "false").lower() == "true"
    DRIFT_LOOKBACK_DAYS = int(os.getenv("DRIFT_LOOKBACK_DAYS", "7"))
    DRIFT_MIN_TRADES = int(os.getenv("DRIFT_MIN_TRADES", "5"))

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
        return cls.MARKET_OPEN <= current_time <= cls.MARKET_CLOSE

    @classmethod
    def validate(cls):
        if not cls.ALPACA_API_KEY or cls.ALPACA_API_KEY == "your_api_key_here":
            raise ValueError("Set ALPACA_API_KEY in .env file")
        if not cls.ALPACA_SECRET_KEY or cls.ALPACA_SECRET_KEY == "your_secret_key_here":
            raise ValueError("Set ALPACA_SECRET_KEY in .env file")
