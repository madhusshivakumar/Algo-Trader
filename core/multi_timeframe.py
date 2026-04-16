"""Multi-timeframe analysis — confirms or dampens signals using higher timeframes.

Resamples 1-minute bars into higher timeframes (5m, 15m, 1h) and computes
trend alignment. Acts as a signal modifier: buy signals are boosted when
higher timeframes agree on uptrend, dampened when they disagree.

All analysis is done by resampling existing data — no additional API calls.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.logger import log


# Timeframe definitions: label -> minutes per bar
TIMEFRAMES = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
}


@dataclass
class TimeframeTrend:
    """Trend assessment for a single timeframe."""
    timeframe: str
    trend: str          # "bullish", "bearish", or "neutral"
    ema_fast: float     # short EMA value
    ema_slow: float     # long EMA value
    strength: float     # 0.0 to 1.0 — how decisive the trend is


@dataclass
class MTFAnalysis:
    """Result of multi-timeframe analysis."""
    trends: list[TimeframeTrend]
    alignment: str      # "bullish", "bearish", "mixed", or "neutral"
    confidence: float   # 0.0 to 1.0 — how aligned the timeframes are

    @property
    def supports_buy(self) -> bool:
        return self.alignment == "bullish"

    @property
    def supports_sell(self) -> bool:
        return self.alignment == "bearish"


def resample_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample 1-minute OHLCV bars into a higher timeframe.

    Args:
        df: DataFrame with open, high, low, close, volume columns.
        minutes: Target bar size in minutes.

    Returns:
        Resampled DataFrame with the same columns, or empty DataFrame on failure.
    """
    if df is None or df.empty or minutes <= 1:
        return pd.DataFrame()

    try:
        # Use the time column as index for resampling
        work = df.copy()
        if "time" in work.columns:
            work = work.set_index("time")
        elif not isinstance(work.index, pd.DatetimeIndex):
            return pd.DataFrame()

        rule = f"{minutes}min"
        resampled = work.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        if len(resampled) < 2:
            return pd.DataFrame()

        result = resampled.reset_index()
        # Ensure the time column is consistently named "time"
        if "time" not in result.columns:
            # Index may have been unnamed or named differently
            result = result.rename(columns={result.columns[0]: "time"})
        return result
    except Exception:
        return pd.DataFrame()


def compute_trend(df: pd.DataFrame, fast_period: int = 8,
                  slow_period: int = 21) -> TimeframeTrend | None:
    """Compute trend direction from a resampled DataFrame.

    Uses dual EMA crossover: fast EMA above slow = bullish, below = bearish.
    Strength is based on the percentage separation between EMAs.

    Args:
        df: Resampled OHLCV DataFrame.
        fast_period: Fast EMA window.
        slow_period: Slow EMA window.

    Returns:
        TimeframeTrend or None if insufficient data.
    """
    if df is None or len(df) < slow_period + 1:
        return None

    close = df["close"]
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()

    fast_val = float(ema_fast.iloc[-1])
    slow_val = float(ema_slow.iloc[-1])

    # Guard against NaN from gaps in data or zero prices
    if slow_val == 0 or np.isnan(fast_val) or np.isnan(slow_val):
        return None

    # Percentage separation
    separation = (fast_val - slow_val) / slow_val

    # Determine trend
    if separation > 0.001:      # >0.1% above
        trend = "bullish"
    elif separation < -0.001:   # >0.1% below
        trend = "bearish"
    else:
        trend = "neutral"

    # Strength: clamp |separation| to [0, 0.02] and normalize to [0, 1]
    strength = min(abs(separation) / 0.02, 1.0)

    return TimeframeTrend(
        timeframe="",  # will be set by caller
        trend=trend,
        ema_fast=fast_val,
        ema_slow=slow_val,
        strength=strength,
    )


def analyze_timeframes(df: pd.DataFrame,
                       timeframes: dict[str, int] | None = None) -> MTFAnalysis:
    """Run multi-timeframe trend analysis on 1-minute bar data.

    Args:
        df: 1-minute OHLCV DataFrame.
        timeframes: Dict of {label: minutes}. Defaults to 5m/15m/1h.

    Returns:
        MTFAnalysis with trend assessments and overall alignment.
    """
    if timeframes is None:
        timeframes = TIMEFRAMES

    trends = []
    for label, minutes in timeframes.items():
        resampled = resample_bars(df, minutes)
        if resampled.empty:
            continue
        trend = compute_trend(resampled)
        if trend is None:
            continue
        trend.timeframe = label
        trends.append(trend)

    if not trends:
        return MTFAnalysis(trends=[], alignment="neutral", confidence=0.0)

    # Count directional votes
    bullish = sum(1 for t in trends if t.trend == "bullish")
    bearish = sum(1 for t in trends if t.trend == "bearish")
    total = len(trends)

    if bullish == total:
        alignment = "bullish"
        confidence = sum(t.strength for t in trends) / total
    elif bearish == total:
        alignment = "bearish"
        confidence = sum(t.strength for t in trends) / total
    elif bullish > bearish:
        alignment = "mixed"
        confidence = (bullish - bearish) / total * 0.5
    elif bearish > bullish:
        alignment = "mixed"
        confidence = (bearish - bullish) / total * 0.5
    else:
        alignment = "neutral"
        confidence = 0.0

    return MTFAnalysis(trends=trends, alignment=alignment, confidence=confidence)


def _ab_log(symbol: str | None, before: dict, after: dict) -> None:
    """Best-effort A/B logger — mirrors the helper in core.signal_modifiers.

    Lazy import keeps this module importable even if analytics is missing,
    and swallows any error so instrumentation never breaks signal flow.
    """
    if symbol is None:
        return
    try:
        from analytics.modifier_ab import log_delta
        log_delta(symbol, "mtf", before, after)
    except Exception:
        pass


def apply_mtf_filter(signal: dict, df: pd.DataFrame,
                     weight: float = 0.15,
                     symbol: str | None = None) -> dict:
    """Apply multi-timeframe analysis as a signal modifier.

    - Buy signals are boosted when higher timeframes are bullish,
      dampened when bearish.
    - Sell signals are boosted when higher timeframes are bearish,
      dampened when bullish.
    - Hold signals are unchanged.

    Args:
        signal: Signal dict with action/reason/strength.
        df: 1-minute OHLCV DataFrame.
        weight: How much MTF analysis affects strength (0.0-0.5).
        symbol: Trading symbol — used only for Sprint 6D A/B instrumentation.
            Optional to preserve backward-compatible test call sites.

    Returns:
        Modified signal dict with mtf_alignment and mtf_confidence keys added.
    """
    # Sprint 6D: capture pre-modifier state so we can emit an A/B record
    # regardless of which branch we exit through.
    before = {"action": signal.get("action", "hold"),
              "strength": signal.get("strength", 0.0)}

    if signal["action"] == "hold":
        _ab_log(symbol, before, signal)
        return signal

    if df is None or len(df) < 60:
        # Need at least 60 1-min bars for meaningful higher-TF analysis
        _ab_log(symbol, before, signal)
        return signal

    analysis = analyze_timeframes(df)

    # Add MTF metadata to signal
    signal["mtf_alignment"] = analysis.alignment
    signal["mtf_confidence"] = round(analysis.confidence, 3)

    action = signal["action"]
    strength = signal["strength"]

    if action == "buy":
        if analysis.supports_buy:
            # Boost: higher TFs confirm bullish trend
            boost = weight * analysis.confidence
            signal["strength"] = min(1.0, strength + boost)
            signal["reason"] += f" [MTF: bullish +{boost:.0%}]"
        elif analysis.supports_sell:
            # Dampen: buying against higher-TF bearish trend
            penalty = weight * analysis.confidence
            signal["strength"] = max(0.3, strength - penalty)
            signal["reason"] += f" [MTF: bearish -{penalty:.0%}]"

    elif action == "sell":
        if analysis.supports_sell:
            # Boost: higher TFs confirm bearish trend
            boost = weight * analysis.confidence
            signal["strength"] = min(1.0, strength + boost)
            signal["reason"] += f" [MTF: bearish +{boost:.0%}]"
        elif analysis.supports_buy:
            # Dampen: selling against higher-TF bullish trend
            penalty = weight * analysis.confidence
            signal["strength"] = max(0.3, strength - penalty)
            signal["reason"] += f" [MTF: bullish -{penalty:.0%}]"

    _ab_log(symbol, before, signal)
    return signal
