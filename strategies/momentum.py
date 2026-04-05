"""Momentum breakout strategy.

Signal: Price breaks above the highest high of the last N candles
with above-average volume. Exit via trailing stop.
"""

import pandas as pd
import ta

from config import Config


def compute_signals(df: pd.DataFrame) -> dict:
    """Analyze recent bars and return trading signal.

    Returns:
        {"action": "buy"|"sell"|"hold", "reason": str, "strength": float 0-1}
    """
    if len(df) < Config.MOMENTUM_LOOKBACK + 5:
        return {"action": "hold", "reason": "insufficient data", "strength": 0}

    close = df["close"]
    high = df["high"]
    volume = df["volume"]

    # Highest high over lookback (excluding current bar)
    lookback_high = high.iloc[-(Config.MOMENTUM_LOOKBACK + 1):-1].max()
    current_close = close.iloc[-1]
    prev_close = close.iloc[-2]

    # Volume confirmation
    avg_volume = volume.iloc[-(Config.MOMENTUM_LOOKBACK + 1):-1].mean()
    current_volume = volume.iloc[-1]
    volume_confirmed = current_volume > avg_volume * Config.MOMENTUM_VOLUME_MULT

    # EMA trend filter — only buy if price is above 50-period EMA
    ema_50 = ta.trend.ema_indicator(close, window=50)
    above_ema = current_close > ema_50.iloc[-1] if len(df) > 50 else True

    # MACD for momentum confirmation
    macd = ta.trend.macd_diff(close)
    macd_positive = macd.iloc[-1] > 0 if len(df) > 26 else True

    # Guard against NaN indicators
    if pd.isna(lookback_high) or pd.isna(avg_volume) or pd.isna(ema_50.iloc[-1]) or pd.isna(macd.iloc[-1]):
        return {"action": "hold", "reason": "insufficient data", "strength": 0.0}

    # Breakout condition
    breakout = current_close > lookback_high and prev_close <= lookback_high

    if breakout and volume_confirmed and above_ema and macd_positive:
        strength = min((current_close - lookback_high) / lookback_high * 100, 1.0)
        return {
            "action": "buy",
            "reason": f"Breakout above ${lookback_high:.2f} with {current_volume/avg_volume:.1f}x volume",
            "strength": max(strength, 0.5),
        }

    # Check for breakdown (for exiting)
    lookback_low = df["low"].iloc[-(Config.MOMENTUM_LOOKBACK + 1):-1].min()
    breakdown = current_close < lookback_low

    if breakdown:
        return {
            "action": "sell",
            "reason": f"Breakdown below ${lookback_low:.2f}",
            "strength": 0.7,
        }

    return {"action": "hold", "reason": "no signal", "strength": 0}
