"""Scalping strategy — frequent small trades on short-term price moves.

Uses VWAP + short EMAs to catch micro-trends. Quick entries and exits.
"""

import pandas as pd
import ta

from config import Config


def compute_signals(df: pd.DataFrame) -> dict:
    if len(df) < 30:
        return {"action": "hold", "reason": "insufficient data", "strength": 0}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    current_price = close.iloc[-1]

    # Short EMAs for quick signals
    ema_5 = ta.trend.ema_indicator(close, window=5)
    ema_13 = ta.trend.ema_indicator(close, window=13)

    # VWAP approximation (cumulative typical price * volume / cumulative volume)
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    if cum_vol.iloc[-1] == 0:
        return {"action": "hold", "reason": "no volume data for VWAP", "strength": 0}
    vwap = cum_tp_vol / cum_vol

    current_vwap = vwap.iloc[-1]
    ema5_val = ema_5.iloc[-1]
    ema13_val = ema_13.iloc[-1]

    # Stochastic RSI for timing
    stoch_rsi = ta.momentum.stochrsi(close, window=14)
    stoch_val = stoch_rsi.iloc[-1] if len(stoch_rsi) > 14 else 0.5

    # BUY: EMA5 crosses above EMA13 + price above VWAP + stoch RSI not overbought
    ema5_prev = ema_5.iloc[-2]
    ema13_prev = ema_13.iloc[-2]
    bullish_cross = ema5_prev <= ema13_prev and ema5_val > ema13_val
    above_vwap = current_price > current_vwap

    if bullish_cross and above_vwap and stoch_val < 0.8:
        strength = min(abs(ema5_val - ema13_val) / ema13_val * 1000, 1.0)
        return {
            "action": "buy",
            "reason": f"EMA5/13 bullish cross above VWAP, StochRSI={stoch_val:.2f}",
            "strength": max(strength, 0.5),
        }

    # SELL: EMA5 crosses below EMA13 OR price drops below VWAP with bearish momentum
    bearish_cross = ema5_prev >= ema13_prev and ema5_val < ema13_val
    below_vwap = current_price < current_vwap

    if bearish_cross or (below_vwap and stoch_val > 0.7):
        reason = "EMA5/13 bearish cross" if bearish_cross else "Below VWAP + high StochRSI"
        return {
            "action": "sell",
            "reason": reason,
            "strength": 0.7,
        }

    return {"action": "hold", "reason": "no scalp signal", "strength": 0}
