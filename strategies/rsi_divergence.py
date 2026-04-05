"""RSI Divergence strategy.

Detects bullish/bearish divergence between price and RSI.
Bullish divergence: price makes lower low but RSI makes higher low → buy.
Bearish divergence: price makes higher high but RSI makes lower high → sell.
"""

import pandas as pd
import ta


def compute_signals(df: pd.DataFrame) -> dict:
    if len(df) < 50:
        return {"action": "hold", "reason": "insufficient data", "strength": 0}

    close = df["close"]

    rsi = ta.momentum.rsi(close, window=14)
    current_rsi = rsi.iloc[-1]

    # Guard against NaN indicators
    if pd.isna(current_rsi):
        return {"action": "hold", "reason": "insufficient data", "strength": 0.0}

    # Look for divergence over last 20 bars
    lookback = 20
    price_window = close.iloc[-lookback:]
    rsi_window = rsi.iloc[-lookback:]

    # Find local lows (for bullish divergence)
    price_lows = []
    rsi_lows = []
    for i in range(2, len(price_window) - 2):
        if (price_window.iloc[i] < price_window.iloc[i-1] and
            price_window.iloc[i] < price_window.iloc[i-2] and
            price_window.iloc[i] < price_window.iloc[i+1] and
            price_window.iloc[i] < price_window.iloc[i+2]):
            price_lows.append((i, price_window.iloc[i]))
            rsi_lows.append((i, rsi_window.iloc[i]))

    # Find local highs (for bearish divergence)
    price_highs = []
    rsi_highs = []
    for i in range(2, len(price_window) - 2):
        if (price_window.iloc[i] > price_window.iloc[i-1] and
            price_window.iloc[i] > price_window.iloc[i-2] and
            price_window.iloc[i] > price_window.iloc[i+1] and
            price_window.iloc[i] > price_window.iloc[i+2]):
            price_highs.append((i, price_window.iloc[i]))
            rsi_highs.append((i, rsi_window.iloc[i]))

    # Bullish divergence: price lower low, RSI higher low
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        if (price_lows[-1][1] < price_lows[-2][1] and
            rsi_lows[-1][1] > rsi_lows[-2][1] and
            current_rsi < 45):
            strength = min((rsi_lows[-1][1] - rsi_lows[-2][1]) / 10, 1.0)
            return {
                "action": "buy",
                "reason": f"Bullish RSI divergence (RSI={current_rsi:.0f})",
                "strength": max(strength, 0.5),
            }

    # Bearish divergence: price higher high, RSI lower high
    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        if (price_highs[-1][1] > price_highs[-2][1] and
            rsi_highs[-1][1] < rsi_highs[-2][1] and
            current_rsi > 55):
            strength = min((rsi_highs[-2][1] - rsi_highs[-1][1]) / 10, 1.0)
            return {
                "action": "sell",
                "reason": f"Bearish RSI divergence (RSI={current_rsi:.0f})",
                "strength": max(strength, 0.6),
            }

    # Simple RSI extremes as fallback
    if current_rsi < 25:
        return {
            "action": "buy",
            "reason": f"RSI extremely oversold ({current_rsi:.0f})",
            "strength": 0.5,
        }
    if current_rsi > 80:
        return {
            "action": "sell",
            "reason": f"RSI extremely overbought ({current_rsi:.0f})",
            "strength": 0.5,
        }

    return {"action": "hold", "reason": f"RSI={current_rsi:.0f}, no divergence", "strength": 0}
