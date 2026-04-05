"""Mean reversion strategy using RSI + Bollinger Bands.

Signal: Buy when RSI is oversold AND price touches lower Bollinger Band.
Sell when RSI is overbought OR price touches upper Bollinger Band.
"""

import pandas as pd
import ta

from config import Config


def compute_signals(df: pd.DataFrame) -> dict:
    """Analyze recent bars and return trading signal.

    Returns:
        {"action": "buy"|"sell"|"hold", "reason": str, "strength": float 0-1}
    """
    if len(df) < max(Config.RSI_PERIOD, Config.BOLLINGER_PERIOD) + 5:
        return {"action": "hold", "reason": "insufficient data", "strength": 0}

    close = df["close"]

    # RSI
    rsi = ta.momentum.rsi(close, window=Config.RSI_PERIOD)
    current_rsi = rsi.iloc[-1]

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(
        close, window=Config.BOLLINGER_PERIOD, window_dev=Config.BOLLINGER_STD
    )
    upper_band = bb.bollinger_hband().iloc[-1]
    lower_band = bb.bollinger_lband().iloc[-1]
    middle_band = bb.bollinger_mavg().iloc[-1]
    current_price = close.iloc[-1]

    # Guard against NaN indicators
    if pd.isna(current_rsi) or pd.isna(upper_band) or pd.isna(lower_band) or pd.isna(current_price):
        return {"action": "hold", "reason": "insufficient data", "strength": 0.0}

    # Buy signal: RSI oversold + price near/below lower band
    if current_rsi < Config.RSI_OVERSOLD and current_price <= lower_band * 1.005:
        strength = (Config.RSI_OVERSOLD - current_rsi) / Config.RSI_OVERSOLD
        return {
            "action": "buy",
            "reason": f"RSI={current_rsi:.0f} oversold + price at lower BB ${lower_band:.2f}",
            "strength": max(0.3, min(strength, 1.0)),
        }

    # Sell signal: RSI overbought OR price at upper band
    if current_rsi > Config.RSI_OVERBOUGHT and current_price >= upper_band * 0.995:
        strength = (current_rsi - Config.RSI_OVERBOUGHT) / (100 - Config.RSI_OVERBOUGHT)
        return {
            "action": "sell",
            "reason": f"RSI={current_rsi:.0f} overbought + price at upper BB ${upper_band:.2f}",
            "strength": max(0.3, min(strength, 1.0)),
        }

    # Mild sell if just RSI overbought
    if current_rsi > Config.RSI_OVERBOUGHT + 5:
        return {
            "action": "sell",
            "reason": f"RSI={current_rsi:.0f} strongly overbought",
            "strength": 0.5,
        }

    return {"action": "hold", "reason": f"RSI={current_rsi:.0f}, neutral", "strength": 0}
