"""Volume-Weighted Momentum strategy.

Combines OBV (On-Balance Volume), volume spikes, and price momentum
to detect institutional-level moves. Buys on volume-confirmed breakouts.
"""

import pandas as pd
import ta
import numpy as np


def compute_signals(df: pd.DataFrame) -> dict:
    if len(df) < 40:
        return {"action": "hold", "reason": "insufficient data", "strength": 0}

    close = df["close"]
    volume = df["volume"]
    high = df["high"]
    low = df["low"]

    current_price = close.iloc[-1]
    prev_price = close.iloc[-2]

    # On-Balance Volume and its EMA
    obv = ta.volume.on_balance_volume(close, volume)
    obv_ema = ta.trend.ema_indicator(obv, window=20)

    # Guard against NaN indicators
    if pd.isna(obv.iloc[-1]) or pd.isna(obv_ema.iloc[-1]) or pd.isna(obv.iloc[-2]) or pd.isna(obv_ema.iloc[-2]):
        return {"action": "hold", "reason": "insufficient data", "strength": 0.0}

    obv_above = obv.iloc[-1] > obv_ema.iloc[-1]
    obv_prev_above = obv.iloc[-2] > obv_ema.iloc[-2]

    # Volume spike detection
    avg_vol = volume.iloc[-21:-1].mean()
    current_vol = volume.iloc[-1]
    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1

    # MFI (Money Flow Index) — RSI but weighted by volume
    mfi = ta.volume.money_flow_index(high, low, close, volume, window=14)
    current_mfi = mfi.iloc[-1] if len(mfi) > 14 else 50

    # Price momentum (rate of change)
    roc = ta.momentum.roc(close, window=10)
    current_roc = roc.iloc[-1] if len(roc) > 10 else 0

    # BUY: OBV crosses above its EMA + volume spike + positive momentum
    obv_bullish_cross = not obv_prev_above and obv_above
    price_up = current_price > prev_price

    if obv_bullish_cross and vol_ratio > 1.3 and price_up and current_mfi < 75:
        strength = min(vol_ratio / 3, 1.0)
        return {
            "action": "buy",
            "reason": f"OBV breakout + {vol_ratio:.1f}x volume, MFI={current_mfi:.0f}",
            "strength": max(strength, 0.5),
        }

    # Also buy on extreme volume with positive momentum
    if vol_ratio > 2.0 and current_roc > 0.1 and current_mfi < 70 and obv_above:
        return {
            "action": "buy",
            "reason": f"Volume spike {vol_ratio:.1f}x + ROC={current_roc:.2f}%",
            "strength": 0.6,
        }

    # SELL: OBV crosses below EMA + high MFI (distribution)
    obv_bearish_cross = obv_prev_above and not obv_above
    if obv_bearish_cross and current_mfi > 60:
        return {
            "action": "sell",
            "reason": f"OBV breakdown, MFI={current_mfi:.0f} (distribution)",
            "strength": 0.7,
        }

    # Sell on volume spike with negative momentum
    if vol_ratio > 2.0 and current_roc < -0.1 and not obv_above:
        return {
            "action": "sell",
            "reason": f"Sell-off volume {vol_ratio:.1f}x, ROC={current_roc:.2f}%",
            "strength": 0.6,
        }

    return {"action": "hold", "reason": "no volume signal", "strength": 0}
