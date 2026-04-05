"""Triple EMA (TEMA) Trend-Following strategy.

Uses 3 EMAs (8, 21, 55) to identify trend direction and entry points.
Buy when short > medium > long (uptrend alignment).
Sell when short < medium (trend weakening).
"""

import pandas as pd
import ta


def compute_signals(df: pd.DataFrame) -> dict:
    if len(df) < 60:
        return {"action": "hold", "reason": "insufficient data", "strength": 0}

    close = df["close"]
    current_price = close.iloc[-1]

    ema_8 = ta.trend.ema_indicator(close, window=8)
    ema_21 = ta.trend.ema_indicator(close, window=21)
    ema_55 = ta.trend.ema_indicator(close, window=55)

    e8 = ema_8.iloc[-1]
    e21 = ema_21.iloc[-1]
    e55 = ema_55.iloc[-1]

    e8_prev = ema_8.iloc[-2]
    e21_prev = ema_21.iloc[-2]

    # Guard against NaN indicators
    if pd.isna(e8) or pd.isna(e21) or pd.isna(e55) or pd.isna(e8_prev) or pd.isna(e21_prev):
        return {"action": "hold", "reason": "insufficient data", "strength": 0.0}

    # ADX for trend strength
    adx = ta.trend.adx(df["high"], df["low"], close, window=14)
    adx_val = adx.iloc[-1] if len(adx) > 14 else 20

    # Full uptrend alignment: EMA8 > EMA21 > EMA55
    uptrend = e8 > e21 > e55
    # EMA8 just crossed above EMA21
    bullish_cross = e8_prev <= e21_prev and e8 > e21

    # Full downtrend: EMA8 < EMA21 < EMA55
    downtrend = e8 < e21 < e55
    bearish_cross = e8_prev >= e21_prev and e8 < e21

    if bullish_cross and adx_val > 20:
        strength = min(adx_val / 50, 1.0)
        trend_note = " in uptrend" if uptrend else ""
        return {
            "action": "buy",
            "reason": f"EMA8/21 bullish cross{trend_note}, ADX={adx_val:.0f}",
            "strength": max(strength, 0.5),
        }

    if uptrend and adx_val > 25 and current_price > e8:
        # Pullback buy in strong uptrend — price bouncing off EMA8
        prev_price = close.iloc[-2]
        if prev_price < e8_prev and current_price > e8:
            return {
                "action": "buy",
                "reason": f"Pullback bounce in uptrend, ADX={adx_val:.0f}",
                "strength": 0.6,
            }

    if bearish_cross:
        strength = 0.7 if downtrend else 0.5
        return {
            "action": "sell",
            "reason": f"EMA8/21 bearish cross, ADX={adx_val:.0f}",
            "strength": strength,
        }

    if downtrend and adx_val > 25:
        return {
            "action": "sell",
            "reason": f"Strong downtrend, ADX={adx_val:.0f}",
            "strength": 0.6,
        }

    return {"action": "hold", "reason": "no trend signal", "strength": 0}
