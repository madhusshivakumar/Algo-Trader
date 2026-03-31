"""MACD Crossover strategy.

Classic signal: buy when MACD line crosses above signal line,
sell when it crosses below. Enhanced with histogram momentum.
"""

import pandas as pd
import ta


def compute_signals(df: pd.DataFrame) -> dict:
    if len(df) < 35:
        return {"action": "hold", "reason": "insufficient data", "strength": 0}

    close = df["close"]

    # MACD components
    macd_line = ta.trend.macd(close, window_slow=26, window_fast=12)
    signal_line = ta.trend.macd_signal(close, window_slow=26, window_fast=12)
    histogram = ta.trend.macd_diff(close, window_slow=26, window_fast=12)

    curr_macd = macd_line.iloc[-1]
    prev_macd = macd_line.iloc[-2]
    curr_signal = signal_line.iloc[-1]
    prev_signal = signal_line.iloc[-2]
    curr_hist = histogram.iloc[-1]
    prev_hist = histogram.iloc[-2]

    # Bullish crossover: MACD crosses above signal line
    bullish_cross = prev_macd <= prev_signal and curr_macd > curr_signal

    # Bearish crossover: MACD crosses below signal line
    bearish_cross = prev_macd >= prev_signal and curr_macd < curr_signal

    # Histogram acceleration (momentum increasing)
    hist_accelerating = abs(curr_hist) > abs(prev_hist)

    if bullish_cross:
        # Stronger signal if histogram is accelerating upward
        strength = 0.6 if hist_accelerating else 0.5
        # Even stronger if crossing from well below zero
        if curr_macd < 0:
            strength += 0.2  # buying the dip recovery
        return {
            "action": "buy",
            "reason": f"MACD bullish cross (MACD={curr_macd:.2f}, Signal={curr_signal:.2f})",
            "strength": min(strength, 1.0),
        }

    if bearish_cross:
        strength = 0.7 if hist_accelerating else 0.5
        return {
            "action": "sell",
            "reason": f"MACD bearish cross (MACD={curr_macd:.2f}, Signal={curr_signal:.2f})",
            "strength": min(strength, 1.0),
        }

    # Strong sell if MACD divergence is widening bearish
    if curr_macd < curr_signal and curr_hist < prev_hist and curr_hist < 0:
        if abs(curr_hist) > abs(prev_hist) * 1.5:
            return {
                "action": "sell",
                "reason": f"MACD bearish acceleration",
                "strength": 0.6,
            }

    return {"action": "hold", "reason": f"MACD={curr_macd:.2f}, no cross", "strength": 0}
