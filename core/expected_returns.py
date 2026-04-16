"""Honest expected-return framing — Sprint 6I.

The bot previously printed naked metrics — "Val Sharpe 2.6542", "+12.5%
annualized" — that a beginner retail user will read as a return promise.
That's a known trap for every algo-trading platform: highlight-reel
numbers beget unrealistic expectations and poor capital preservation.

This module wraps every historical performance number with a short
reality-anchor sentence before it's shown to the user (dashboard, logs,
README excerpt). Three helpers:

    * ``frame_sharpe(v, context)``  — one line for a Sharpe value.
    * ``frame_return(v, context)``  — one line for a return percentage.
    * ``frame_backtest(metrics)``  — multi-line block for the dashboard.

The text is deliberately conservative — emphasis on **"past test result"**
and **"retail algo traders typically see materially lower"**. Numbers are
shown; they're just never shown alone.

Rule of thumb (industry heuristic, not a guarantee):
    retail algo Sharpe       ≈  0.5 – 1.5  (post-cost, real money, small account)
    retail annualized return ≈  -20% — +15%  (annual, before tax)

These are the anchors the framing references.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


# Anchor ranges — industry-rough, quoted so the user can see where we draw
# the expectation band from.
_REALISTIC_SHARPE_LOW = 0.5
_REALISTIC_SHARPE_HIGH = 1.5
_REALISTIC_RETURN_LOW_PCT = -20.0
_REALISTIC_RETURN_HIGH_PCT = 15.0


# ── Public helpers ─────────────────────────────────────────────────────────

def frame_sharpe(value: float, context: str = "backtest") -> str:
    """Wrap a Sharpe value with a reality check.

    Args:
        value: The raw Sharpe value (already computed).
        context: Short label for where the value came from — e.g.
            ``"backtest"``, ``"RL validation"``, ``"paper trading (30d)"``.

    Returns:
        A single-line string safe to log or display.
    """
    if value is None:
        return f"No Sharpe available ({context})."
    v = float(value)
    anchor = (f"Real-world retail algo traders typically see Sharpe "
              f"{_REALISTIC_SHARPE_LOW:.1f}–{_REALISTIC_SHARPE_HIGH:.1f}, "
              f"or losses.")
    if v >= _REALISTIC_SHARPE_HIGH:
        return (f"{context.capitalize()} Sharpe {v:.2f} — this is a past "
                f"test result. {anchor} Expect materially lower going forward.")
    if v >= _REALISTIC_SHARPE_LOW:
        return (f"{context.capitalize()} Sharpe {v:.2f} — in the realistic "
                f"retail range ({_REALISTIC_SHARPE_LOW:.1f}–"
                f"{_REALISTIC_SHARPE_HIGH:.1f}), but past performance does "
                f"not predict future returns.")
    if v >= 0:
        return (f"{context.capitalize()} Sharpe {v:.2f} — below the realistic "
                f"retail range. The strategy may be underperforming costs.")
    return (f"{context.capitalize()} Sharpe {v:.2f} (NEGATIVE) — the "
            f"strategy is losing money on a risk-adjusted basis in this "
            f"sample. Don't deploy without investigating.")


def frame_return(value_pct: float, context: str = "backtest") -> str:
    """Wrap an annualized return pct with a reality check."""
    if value_pct is None:
        return f"No return available ({context})."
    v = float(value_pct)
    anchor = (f"Retail algo traders commonly see "
              f"{_REALISTIC_RETURN_LOW_PCT:.0f}% to "
              f"{_REALISTIC_RETURN_HIGH_PCT:.0f}% annually, with most losing "
              f"money.")
    if v >= _REALISTIC_RETURN_HIGH_PCT:
        return (f"{context.capitalize()} return {v:+.1f}% — this is a past "
                f"test result. {anchor} Do not expect this to repeat live.")
    if v >= 0:
        return (f"{context.capitalize()} return {v:+.1f}% — within the "
                f"realistic retail range. {anchor}")
    return (f"{context.capitalize()} return {v:+.1f}% (NEGATIVE). {anchor}")


@dataclass(frozen=True)
class BacktestFrame:
    """Structured framing block for the dashboard (multi-line)."""
    sharpe_line: str
    return_line: str
    pct_band_line: str
    disclaimer_line: str

    def as_text(self) -> str:
        return "\n".join([self.sharpe_line, self.return_line,
                          self.pct_band_line, self.disclaimer_line])


def frame_backtest(sharpe: float, total_return_pct: float,
                   mc_p5: float | None = None,
                   mc_p50: float | None = None,
                   mc_p95: float | None = None,
                   context: str = "backtest") -> BacktestFrame:
    """Build a multi-line framing block for a backtest result.

    Monte Carlo percentiles are optional — when present they communicate
    sequencing uncertainty, which is exactly the thing naked Sharpe hides.
    """
    sh = frame_sharpe(sharpe, context)
    rt = frame_return(total_return_pct, context)

    if all(x is not None for x in (mc_p5, mc_p50, mc_p95)):
        band = (f"Monte-Carlo sequencing band (1k shuffles): "
                f"p5={mc_p5:+.1%}  p50={mc_p50:+.1%}  p95={mc_p95:+.1%}.")
    else:
        band = "Monte-Carlo band not available for this run."

    disclaimer = ("IMPORTANT: Past backtest results are NOT a prediction. "
                  "Retail algo traders most commonly lose money. "
                  "Do not allocate capital you cannot afford to lose.")
    return BacktestFrame(sharpe_line=sh, return_line=rt,
                         pct_band_line=band,
                         disclaimer_line=disclaimer)


# ── Industry context blurbs (used by dashboard + README) ──────────────────

INDUSTRY_RANGE_SHARPE = (
    f"Industry heuristic: retail algo-traders commonly see Sharpe "
    f"{_REALISTIC_SHARPE_LOW:.1f}–{_REALISTIC_SHARPE_HIGH:.1f} "
    f"(post-cost, real money). Higher numbers in backtests are normal; "
    f"they rarely survive live trading intact."
)


INDUSTRY_RANGE_RETURN = (
    f"Industry heuristic: retail algo-traders commonly see "
    f"{_REALISTIC_RETURN_LOW_PCT:.0f}% to {_REALISTIC_RETURN_HIGH_PCT:.0f}% "
    f"annualized, with the majority losing money after costs."
)


_CANNOT_DO = [
    "Predict market crashes or black-swan events.",
    "Guarantee positive returns — in any week, month, or year.",
    "Replicate hedge-fund-scale edge with retail data feeds.",
    "Beat buy-and-hold S&P on a small account after transaction costs.",
    "Trade options, futures, forex, or leveraged instruments.",
    "Act on inside information, rumors, or social-media sentiment directly.",
    "Recover from a broker outage faster than a human (we watchdog, we don't guarantee).",
    "Interpret your personal risk tolerance — you must still size positions you can afford to lose.",
]


def what_this_bot_cannot_do() -> Sequence[str]:
    """Return the canonical list of disclaimers for the dashboard/README."""
    return tuple(_CANNOT_DO)
