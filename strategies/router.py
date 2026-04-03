"""Strategy Router — assigns the best-performing strategy to each symbol.

Architecture:
  - STRATEGY_REGISTRY: hardcoded map of strategy names → compute_signals functions (never modified by agents)
  - STRATEGY_MAP: symbol → strategy name (updated by Market Scanner agent via data/optimizer/strategy_assignments.json)
  - Agents only write string-based mappings; this module resolves them to functions
"""

import json
import os

from config import Config
from core.signal_modifiers import apply_sentiment, apply_llm_conviction, apply_earnings_blackout
from core.multi_timeframe import apply_mtf_filter

from strategies import (
    mean_reversion_aggressive,
    mean_reversion,
    volume_profile,
    momentum,
    macd_crossover,
    triple_ema,
    rsi_divergence,
    scalper,
    ensemble,
)

# ── Registry: name → function (never modified by agents) ──────────────
STRATEGY_REGISTRY = {
    "mean_reversion_aggressive": mean_reversion_aggressive.compute_signals,
    "mean_reversion":            mean_reversion.compute_signals,
    "volume_profile":            volume_profile.compute_signals,
    "momentum":                  momentum.compute_signals,
    "macd_crossover":            macd_crossover.compute_signals,
    "triple_ema":                triple_ema.compute_signals,
    "rsi_divergence":            rsi_divergence.compute_signals,
    "scalper":                   scalper.compute_signals,
    "ensemble":                  ensemble.compute_signals,
}

# Human-readable display names
STRATEGY_DISPLAY_NAMES = {
    "mean_reversion_aggressive": "Mean Rev Aggressive",
    "mean_reversion":            "Mean Reversion (RSI+BB)",
    "volume_profile":            "Volume Profile",
    "momentum":                  "Momentum Breakout",
    "macd_crossover":            "MACD Crossover",
    "triple_ema":                "Triple EMA Trend",
    "rsi_divergence":            "RSI Divergence",
    "scalper":                   "Scalper (EMA+VWAP)",
    "ensemble":                  "Ensemble (Mom+MR)",
}

DEFAULT_STRATEGY = "mean_reversion_aggressive"

# ── Strategy Map: symbol → strategy name (loaded from file or defaults) ──

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_ASSIGNMENTS_FILE = os.path.join(_DATA_DIR, "optimizer", "strategy_assignments.json")
_FALLBACK_FILE = os.path.join(_DATA_DIR, "fallback_config.json")

# Default assignments (used if no agent files exist)
_DEFAULT_MAP = {
    "BTC/USD": "mean_reversion_aggressive",
    "ETH/USD": "volume_profile",
    "TSLA":    "mean_reversion_aggressive",
    "NVDA":    "mean_reversion_aggressive",
    "AMD":     "mean_reversion_aggressive",
    "AAPL":    "mean_reversion_aggressive",
    "META":    "mean_reversion_aggressive",
    "SPY":     "mean_reversion_aggressive",
}


def _load_strategy_map() -> dict[str, str]:
    """Load strategy assignments from agent output files, falling back gracefully."""
    # Try optimizer assignments first
    if os.path.exists(_ASSIGNMENTS_FILE):
        try:
            with open(_ASSIGNMENTS_FILE) as f:
                data = json.load(f)
            assignments = data.get("assignments", {})
            # Validate that all strategy names exist in the registry
            result = {}
            for symbol, info in assignments.items():
                name = info if isinstance(info, str) else info.get("strategy", DEFAULT_STRATEGY)
                if name in STRATEGY_REGISTRY:
                    result[symbol] = name
                else:
                    result[symbol] = DEFAULT_STRATEGY
            if result:
                return result
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Try fallback config
    if os.path.exists(_FALLBACK_FILE):
        try:
            with open(_FALLBACK_FILE) as f:
                data = json.load(f)
            smap = data.get("strategy_map", {})
            if smap and all(v in STRATEGY_REGISTRY for v in smap.values()):
                return smap
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Use hardcoded defaults
    return _DEFAULT_MAP.copy()


# Load at import time (bot reads this once at startup)
STRATEGY_MAP = _load_strategy_map()


def get_strategy(symbol: str) -> tuple[str, callable]:
    """Return (display_name, compute_signals_fn) for a given symbol."""
    strategy_key = STRATEGY_MAP.get(symbol, DEFAULT_STRATEGY)
    display_name = STRATEGY_DISPLAY_NAMES.get(strategy_key, strategy_key)
    fn = STRATEGY_REGISTRY[strategy_key]
    return display_name, fn


def get_strategy_key(symbol: str) -> str:
    """Return the raw strategy key for a symbol (used by logger)."""
    return STRATEGY_MAP.get(symbol, DEFAULT_STRATEGY)


def compute_signals(symbol: str, df):
    """Route to the correct strategy, compute signals, and apply modifiers."""
    rl_selected = ""

    # RL strategy selection (if enabled and model available)
    if Config.RL_STRATEGY_ENABLED and df is not None:
        try:
            from core.rl_strategy_selector import select_strategy as rl_select
            rl_key = rl_select(df)
            if rl_key and rl_key in STRATEGY_REGISTRY:
                rl_selected = rl_key
                name = STRATEGY_DISPLAY_NAMES.get(rl_key, rl_key)
                fn = STRATEGY_REGISTRY[rl_key]
            else:
                name, fn = get_strategy(symbol)
        except Exception:
            name, fn = get_strategy(symbol)
    else:
        name, fn = get_strategy(symbol)

    if df is None:
        return {"action": "hold", "reason": "no data", "strength": 0, "strategy": name}

    signal = fn(df)
    signal["strategy"] = name
    if rl_selected:
        signal["rl_selected"] = rl_selected

    # Apply v2 signal modifiers (each is a no-op if disabled or data missing)
    if Config.SENTIMENT_ENABLED:
        signal = apply_sentiment(signal, symbol, weight=Config.SENTIMENT_WEIGHT)
    if Config.LLM_ANALYST_ENABLED:
        signal = apply_llm_conviction(signal, symbol, weight=Config.LLM_CONVICTION_WEIGHT)

    # Apply multi-timeframe filter (resamples existing bars, no extra API calls)
    if Config.MTF_ENABLED:
        signal = apply_mtf_filter(signal, df, weight=Config.MTF_WEIGHT)

    # Apply earnings blackout (equity only — reduces sizing near earnings)
    if Config.EARNINGS_CALENDAR_ENABLED:
        signal = apply_earnings_blackout(signal, symbol)

    return signal
