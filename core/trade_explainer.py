"""Plain-language trade explainer — Sprint 6G.

Converts a strategy's machine-readable signal into one sentence a beginner can
understand. The goal is to teach the user what the bot is actually doing over
time — not replace the technical reason, but sit alongside it.

Example:
    signal = {"action": "buy", "reason": "Mean rev buy: RSI=32, %B=0.18",
              "strength": 0.6, "strategy": "Mean Rev Aggressive"}
    explain(signal, df, symbol="AAPL")
    # → "Bought $15 of AAPL because the price dropped ~2% below its recent
    #    average — often a setup for a short-term bounce."

The explainer is deliberately conservative about the claims it makes. We say
"often precedes a bounce" rather than "will bounce" — beginners anchor on
certainty language and we don't want to mislead.

Called from engine.py at trade time; the result is stored in the new
`trades.explanation` column and surfaced on the dashboard.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# ── Strategy key → template function ────────────────────────────────────────
#
# Each template takes the same args and returns a plain-English sentence.
# Templates try to extract whatever numbers are useful (RSI level, %B, price
# distance from MA) from the df, and fall back to a generic sentence if the
# data is missing.
#
# We always hedge with "often" / "typically" — never "will".


def _safe_tail_stat(df: pd.DataFrame, col: str, window: int, stat: str = "mean") -> Optional[float]:
    """Compute a tail statistic over `window` bars, or None if data is missing."""
    if df is None or col not in df.columns or len(df) < window:
        return None
    try:
        if stat == "mean":
            return float(df[col].tail(window).mean())
        if stat == "max":
            return float(df[col].tail(window).max())
        if stat == "min":
            return float(df[col].tail(window).min())
    except Exception:
        return None
    return None


def _current_price(df: pd.DataFrame) -> Optional[float]:
    if df is None or "close" not in df.columns or len(df) == 0:
        return None
    try:
        return float(df["close"].iloc[-1])
    except Exception:
        return None


def _pct_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """(a - b) / b, safe for None/zero."""
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b


def _explain_mean_reversion(
    signal: dict, df: pd.DataFrame, symbol: str, notional: float
) -> str:
    price = _current_price(df)
    ma20 = _safe_tail_stat(df, "close", 20, "mean")
    drop = _pct_diff(price, ma20)

    action = signal.get("action", "hold")
    if action == "buy":
        if drop is not None and drop < -0.005:
            return (
                f"Bought ${notional:.0f} of {symbol} because the price is ~{abs(drop):.1%} "
                f"below its 20-bar average — often precedes a short-term bounce."
            )
        return (
            f"Bought ${notional:.0f} of {symbol} because indicators flagged it as oversold — "
            f"a common setup for a short-term rebound."
        )
    if action == "sell":
        if drop is not None and drop > 0.005:
            return (
                f"Sold {symbol} because the price is ~{drop:.1%} above its 20-bar average — "
                f"mean reversion strategies typically take profits here."
            )
        return (
            f"Sold {symbol} because indicators flagged it as overbought — "
            f"a common exit for mean reversion."
        )
    return f"Holding {symbol} — no clear mean-reversion setup right now."


def _explain_momentum(
    signal: dict, df: pd.DataFrame, symbol: str, notional: float
) -> str:
    price = _current_price(df)
    high20 = _safe_tail_stat(df, "close", 20, "max")
    breakout = _pct_diff(price, high20)

    action = signal.get("action", "hold")
    if action == "buy":
        if breakout is not None and breakout > -0.002:
            return (
                f"Bought ${notional:.0f} of {symbol} because it's breaking above its recent "
                f"20-bar high — momentum strategies try to ride that continuation."
            )
        return (
            f"Bought ${notional:.0f} of {symbol} because momentum indicators turned bullish."
        )
    if action == "sell":
        return (
            f"Sold {symbol} because momentum faded — taking profit before a reversal."
        )
    return f"Holding {symbol} — no fresh momentum signal."


def _explain_macd(
    signal: dict, df: pd.DataFrame, symbol: str, notional: float
) -> str:
    action = signal.get("action", "hold")
    if action == "buy":
        return (
            f"Bought ${notional:.0f} of {symbol} after a bullish MACD crossover — "
            f"short-term momentum is turning up relative to longer-term."
        )
    if action == "sell":
        return (
            f"Sold {symbol} after a bearish MACD crossover — "
            f"short-term momentum rolled over."
        )
    return f"Holding {symbol} — no MACD crossover yet."


def _explain_volume_profile(
    signal: dict, df: pd.DataFrame, symbol: str, notional: float
) -> str:
    action = signal.get("action", "hold")
    if action == "buy":
        return (
            f"Bought ${notional:.0f} of {symbol} because price returned to a high-volume "
            f"support zone — these areas often act as floors in the short term."
        )
    if action == "sell":
        return (
            f"Sold {symbol} because price hit a high-volume resistance zone — "
            f"often a short-term ceiling."
        )
    return f"Holding {symbol} — price isn't near a meaningful volume zone."


def _explain_rsi_divergence(
    signal: dict, df: pd.DataFrame, symbol: str, notional: float
) -> str:
    action = signal.get("action", "hold")
    if action == "buy":
        return (
            f"Bought ${notional:.0f} of {symbol} because RSI showed bullish divergence — "
            f"price made a new low but momentum didn't, often a reversal cue."
        )
    if action == "sell":
        return (
            f"Sold {symbol} because RSI showed bearish divergence — "
            f"price made a new high but momentum didn't confirm."
        )
    return f"Holding {symbol} — no clear RSI divergence."


def _explain_triple_ema(
    signal: dict, df: pd.DataFrame, symbol: str, notional: float
) -> str:
    action = signal.get("action", "hold")
    if action == "buy":
        return (
            f"Bought ${notional:.0f} of {symbol} because three moving averages lined up "
            f"bullish — a classic trend-following setup."
        )
    if action == "sell":
        return (
            f"Sold {symbol} because the moving average stack turned bearish."
        )
    return f"Holding {symbol} — trend is unclear."


def _explain_scalper(
    signal: dict, df: pd.DataFrame, symbol: str, notional: float
) -> str:
    action = signal.get("action", "hold")
    if action == "buy":
        return (
            f"Bought ${notional:.0f} of {symbol} on a short-term bullish EMA+VWAP alignment — "
            f"scalping the next few bars of expected upside."
        )
    if action == "sell":
        return (
            f"Sold {symbol} — scalper exit, short-term edge has decayed."
        )
    return f"Holding {symbol} — no short-term edge right now."


def _explain_ensemble(
    signal: dict, df: pd.DataFrame, symbol: str, notional: float
) -> str:
    action = signal.get("action", "hold")
    if action == "buy":
        return (
            f"Bought ${notional:.0f} of {symbol} — multiple strategies (momentum + mean "
            f"reversion) agreed on a bullish setup."
        )
    if action == "sell":
        return (
            f"Sold {symbol} — multiple strategies agreed the setup has run its course."
        )
    return f"Holding {symbol} — strategies are mixed."


# Canonical strategy-key → explainer map.
_EXPLAINERS = {
    "mean_reversion": _explain_mean_reversion,
    "mean_reversion_aggressive": _explain_mean_reversion,
    "momentum": _explain_momentum,
    "macd_crossover": _explain_macd,
    "volume_profile": _explain_volume_profile,
    "rsi_divergence": _explain_rsi_divergence,
    "triple_ema": _explain_triple_ema,
    "scalper": _explain_scalper,
    "ensemble": _explain_ensemble,
}

# Display-name → strategy-key (for when callers pass the human label).
_DISPLAY_TO_KEY = {
    "Mean Reversion (RSI+BB)": "mean_reversion",
    "Mean Rev Aggressive": "mean_reversion_aggressive",
    "Momentum Breakout": "momentum",
    "MACD Crossover": "macd_crossover",
    "Volume Profile": "volume_profile",
    "RSI Divergence": "rsi_divergence",
    "Triple EMA Trend": "triple_ema",
    "Scalper (EMA+VWAP)": "scalper",
    "Ensemble (Mom+MR)": "ensemble",
}


def _resolve_strategy_key(signal: dict) -> str:
    """Extract a canonical strategy key from the signal dict.

    Accepts either `strategy_key` (preferred) or `strategy` (display name,
    what the router currently sets). Returns "unknown" if nothing maps.
    """
    key = signal.get("strategy_key")
    if key and key in _EXPLAINERS:
        return key
    display = signal.get("strategy") or ""
    if display in _DISPLAY_TO_KEY:
        return _DISPLAY_TO_KEY[display]
    # Last-resort: lowercase the display name and replace spaces.
    guess = display.lower().replace(" ", "_")
    if guess in _EXPLAINERS:
        return guess
    return "unknown"


def _rl_suffix(signal: dict) -> str:
    """If the RL selector picked this strategy, tack on a short note."""
    rl = signal.get("rl_selected") or ""
    if not rl:
        return ""
    # Keep this brief and non-promotional.
    return " (RL selector picked this strategy based on recent conditions.)"


def explain(
    signal: dict,
    df: Optional[pd.DataFrame] = None,
    symbol: str = "",
    notional: float = 0.0,
) -> str:
    """Return a one-sentence plain-language explanation of a trade signal.

    Args:
        signal: Output of `strategies.router.compute_signals()`. Must have
            at minimum `action`; `strategy` and `strategy_key` are used to
            pick the template.
        df: Recent OHLCV bars. Optional — templates degrade gracefully without.
        symbol: The ticker/pair being traded (e.g., "AAPL", "BTC/USD").
        notional: Dollar size of the trade. If 0, "some" is used.

    Returns:
        A single sentence, no trailing newline. Always non-empty.
    """
    key = _resolve_strategy_key(signal)
    action = signal.get("action", "hold")

    # Normalize notional for template use — we don't want "$0 of AAPL".
    display_notional = notional if notional and notional > 0 else 0.0

    fn = _EXPLAINERS.get(key)
    if fn is None:
        # Unknown strategy — fall back to a generic template that mentions the
        # action and symbol but makes no strategy-specific claims.
        if action == "buy":
            base = (
                f"Bought ${display_notional:.0f} of {symbol} — signals aligned for entry."
                if display_notional
                else f"Bought {symbol} — signals aligned for entry."
            )
        elif action == "sell":
            base = f"Sold {symbol} — exit conditions were met."
        else:
            base = f"Holding {symbol} — no trade signal."
        return base + _rl_suffix(signal)

    try:
        base = fn(signal, df if df is not None else pd.DataFrame(), symbol, display_notional)
    except Exception:
        # Any template error → generic fallback. Never crash the engine here.
        if action == "buy":
            base = f"Bought {symbol} — entry signal triggered."
        elif action == "sell":
            base = f"Sold {symbol} — exit signal triggered."
        else:
            base = f"Holding {symbol}."
    return base + _rl_suffix(signal)
