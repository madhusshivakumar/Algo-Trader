"""Ensemble strategy — combines momentum and mean reversion signals."""

import pandas as pd

from strategies import momentum, mean_reversion
from utils.logger import log


def compute_signals(df: pd.DataFrame) -> dict:
    """Combine multiple strategy signals into one decision.

    Requires agreement between strategies for buys (conservative).
    Either strategy can trigger a sell (protective).
    """
    mom_signal = momentum.compute_signals(df)
    mr_signal = mean_reversion.compute_signals(df)

    # For SELL: either strategy saying sell is enough
    if mom_signal["action"] == "sell" or mr_signal["action"] == "sell":
        sell_signal = mom_signal if mom_signal["action"] == "sell" else mr_signal
        return {
            "action": "sell",
            "reason": f"[ensemble] {sell_signal['reason']}",
            "strength": sell_signal["strength"],
        }

    # For BUY: at least one strong buy signal
    buy_signals = [s for s in [mom_signal, mr_signal] if s["action"] == "buy"]

    if len(buy_signals) == 2:
        # Both agree — strong signal
        avg_strength = (buy_signals[0]["strength"] + buy_signals[1]["strength"]) / 2
        return {
            "action": "buy",
            "reason": f"[ensemble] Both strategies agree: {buy_signals[0]['reason']} + {buy_signals[1]['reason']}",
            "strength": min(avg_strength * 1.2, 1.0),
        }
    elif len(buy_signals) == 1:
        # One buy signal — only take it if strong enough
        signal = buy_signals[0]
        if signal["strength"] >= 0.6:
            return {
                "action": "buy",
                "reason": f"[ensemble] {signal['reason']}",
                "strength": max(signal["strength"] * 0.8, 0.3),  # reduce confidence
            }

    return {"action": "hold", "reason": "[ensemble] no consensus", "strength": 0}
