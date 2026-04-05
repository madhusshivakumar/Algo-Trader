"""Aggressive Mean Reversion — looser thresholds, more trades.

Trades RSI + Bollinger with relaxed entry criteria for higher frequency.
"""

import pandas as pd
import ta


def compute_signals(df: pd.DataFrame) -> dict:
    if len(df) < 25:
        return {"action": "hold", "reason": "insufficient data", "strength": 0}

    close = df["close"]
    current_price = close.iloc[-1]

    # RSI with shorter period for more signals
    rsi = ta.momentum.rsi(close, window=10)
    current_rsi = rsi.iloc[-1]

    # Bollinger Bands — tighter (1.5 std)
    bb = ta.volatility.BollingerBands(close, window=15, window_dev=1.5)
    upper = bb.bollinger_hband().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]
    middle = bb.bollinger_mavg().iloc[-1]
    bb_width = (upper - lower) / middle if middle > 0 else 0

    # %B — where price is relative to bands (0 = lower, 1 = upper)
    pct_b = bb.bollinger_pband().iloc[-1]

    # Guard against NaN indicators
    if pd.isna(current_rsi) or pd.isna(upper) or pd.isna(lower) or pd.isna(pct_b):
        return {"action": "hold", "reason": "insufficient data", "strength": 0.0}

    # Extreme buy: RSI < 30 or %B < 0.05 (check extreme first)
    if current_rsi < 30 or pct_b < 0.05:
        return {
            "action": "buy",
            "reason": f"Extreme oversold: RSI={current_rsi:.0f}, %B={pct_b:.2f}",
            "strength": 0.7,
        }

    # Buy: RSI < 40 and price in lower 20% of BB
    if current_rsi < 40 and pct_b < 0.2:
        strength = min((40 - current_rsi) / 30 + (0.2 - pct_b), 1.0)
        return {
            "action": "buy",
            "reason": f"Mean rev buy: RSI={current_rsi:.0f}, %B={pct_b:.2f}",
            "strength": max(strength, 0.5),
        }

    # ── Sell logic (v2: data-driven thresholds) ──
    # Winners exit avg RSI 62.8, losers avg 59.9 — old RSI 60 was in noise zone
    if current_rsi > 80 and pct_b > 0.95:
        return {
            "action": "sell",
            "reason": f"Extreme overbought: RSI={current_rsi:.0f}, %B={pct_b:.2f}",
            "strength": 0.8,
        }
    if current_rsi > 80:
        return {
            "action": "sell",
            "reason": f"RSI extreme: RSI={current_rsi:.0f}, %B={pct_b:.2f}",
            "strength": 0.7,
        }
    if current_rsi > 70 and pct_b > 0.85:
        strength = min((current_rsi - 70) / 20 + (pct_b - 0.85), 1.0)
        return {
            "action": "sell",
            "reason": f"Mean rev sell: RSI={current_rsi:.0f}, %B={pct_b:.2f}",
            "strength": max(strength, 0.5),
        }

    return {"action": "hold", "reason": f"RSI={current_rsi:.0f}, neutral zone", "strength": 0}
