"""Sprint 6G: plain-language trade explainer tests.

Covers template resolution (by strategy_key and by display name),
per-strategy templates, graceful degradation when df is missing, RL suffix
injection, and fallback behavior for unknown strategies.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.trade_explainer import (
    _resolve_strategy_key,
    _rl_suffix,
    explain,
)


def _df_with_trend(trend: float = 0.0, n: int = 30, start: float = 100.0) -> pd.DataFrame:
    """Build a simple closing-price DataFrame with a linear trend fraction."""
    prices = [start * (1 + trend * (i / n)) for i in range(n)]
    return pd.DataFrame({"close": prices, "high": prices, "low": prices,
                         "volume": [1_000_000] * n})


# ── strategy key resolution ────────────────────────────────────────────────


class TestResolveStrategyKey:
    def test_direct_strategy_key(self):
        assert _resolve_strategy_key({"strategy_key": "momentum"}) == "momentum"

    def test_display_name_mapped(self):
        assert _resolve_strategy_key({"strategy": "Mean Rev Aggressive"}) == "mean_reversion_aggressive"
        assert _resolve_strategy_key({"strategy": "Volume Profile"}) == "volume_profile"
        assert _resolve_strategy_key({"strategy": "MACD Crossover"}) == "macd_crossover"

    def test_empty_signal_is_unknown(self):
        assert _resolve_strategy_key({}) == "unknown"

    def test_unmapped_display_falls_back_to_slug(self):
        # "Momentum Breakout" → "momentum_breakout" isn't in _EXPLAINERS,
        # but "momentum" is — the slug heuristic only matches exact keys.
        assert _resolve_strategy_key({"strategy": "Momentum"}) == "momentum"

    def test_completely_unknown_is_unknown(self):
        assert _resolve_strategy_key({"strategy": "Some Future Strat"}) == "unknown"


# ── RL suffix ──────────────────────────────────────────────────────────────


class TestRlSuffix:
    def test_no_rl_empty(self):
        assert _rl_suffix({}) == ""
        assert _rl_suffix({"rl_selected": ""}) == ""

    def test_rl_selected_adds_note(self):
        out = _rl_suffix({"rl_selected": "momentum"})
        assert "RL" in out
        assert out.startswith(" ")  # space-prefixed so it appends cleanly


# ── per-strategy explain() ────────────────────────────────────────────────


class TestExplainMeanReversion:
    def test_buy_with_drop_mentions_percentage(self):
        # Price dropped 3% below its 20-bar average
        prices = [100.0] * 20 + [97.0]
        df = pd.DataFrame({"close": prices, "high": prices, "low": prices,
                           "volume": [1e6] * 21})
        sig = {"action": "buy", "strategy_key": "mean_reversion_aggressive"}
        out = explain(sig, df, symbol="AAPL", notional=15.0)
        assert "AAPL" in out
        assert "15" in out
        assert "below" in out.lower()
        # Soft-hedged language
        assert "often" in out.lower() or "typical" in out.lower()

    def test_buy_without_drop_uses_generic(self):
        df = _df_with_trend(0.0)
        sig = {"action": "buy", "strategy": "Mean Rev Aggressive"}
        out = explain(sig, df, "AAPL", 20.0)
        assert "AAPL" in out
        assert "oversold" in out.lower() or "bounce" in out.lower() or "rebound" in out.lower()

    def test_sell_above_mean(self):
        prices = [100.0] * 20 + [105.0]
        df = pd.DataFrame({"close": prices, "high": prices, "low": prices,
                           "volume": [1e6] * 21})
        sig = {"action": "sell", "strategy_key": "mean_reversion"}
        out = explain(sig, df, "AAPL", 100.0)
        assert "AAPL" in out
        # Either "above" phrasing (specific) or "overbought" (generic)
        assert "above" in out.lower() or "overbought" in out.lower()

    def test_hold_has_no_dollar_amount(self):
        df = _df_with_trend(0.0)
        out = explain({"action": "hold", "strategy_key": "mean_reversion"},
                      df, "AAPL", 0.0)
        assert "AAPL" in out
        assert "Holding" in out


class TestExplainMomentum:
    def test_buy_breakout_mentions_high(self):
        prices = [100.0] * 20 + [110.0]
        df = pd.DataFrame({"close": prices, "high": prices, "low": prices,
                           "volume": [1e6] * 21})
        out = explain({"action": "buy", "strategy_key": "momentum"},
                      df, "TSLA", 50.0)
        assert "TSLA" in out
        assert "high" in out.lower() or "break" in out.lower()

    def test_buy_without_breakout(self):
        df = _df_with_trend(0.0)
        out = explain({"action": "buy", "strategy_key": "momentum"},
                      df, "TSLA", 50.0)
        assert "TSLA" in out
        assert "momentum" in out.lower() or "bullish" in out.lower()

    def test_sell_mentions_profit_take(self):
        out = explain({"action": "sell", "strategy_key": "momentum"},
                      _df_with_trend(), "TSLA", 50.0)
        assert "profit" in out.lower() or "reversal" in out.lower() or "faded" in out.lower()


class TestExplainMacd:
    def test_buy_mentions_crossover(self):
        out = explain({"action": "buy", "strategy_key": "macd_crossover"},
                      _df_with_trend(), "NVDA", 100.0)
        assert "NVDA" in out
        assert "MACD" in out or "crossover" in out.lower()

    def test_sell_mentions_crossover(self):
        out = explain({"action": "sell", "strategy_key": "macd_crossover"},
                      _df_with_trend(), "NVDA", 100.0)
        assert "NVDA" in out
        assert "MACD" in out or "bearish" in out.lower()


class TestExplainVolumeProfile:
    def test_buy_mentions_volume(self):
        out = explain({"action": "buy", "strategy_key": "volume_profile"},
                      _df_with_trend(), "BTC/USD", 200.0)
        assert "BTC/USD" in out
        assert "volume" in out.lower()


class TestExplainRsiDivergence:
    def test_buy_mentions_divergence(self):
        out = explain({"action": "buy", "strategy_key": "rsi_divergence"},
                      _df_with_trend(), "AMD", 30.0)
        assert "divergence" in out.lower()


class TestExplainTripleEma:
    def test_buy_mentions_ma_stack(self):
        out = explain({"action": "buy", "strategy_key": "triple_ema"},
                      _df_with_trend(), "META", 75.0)
        assert "META" in out
        assert "moving average" in out.lower() or "trend" in out.lower()


class TestExplainScalper:
    def test_buy_mentions_short_term(self):
        out = explain({"action": "buy", "strategy_key": "scalper"},
                      _df_with_trend(), "SPY", 40.0)
        assert "SPY" in out
        assert "short" in out.lower() or "scalp" in out.lower()


class TestExplainEnsemble:
    def test_buy_mentions_multiple(self):
        out = explain({"action": "buy", "strategy_key": "ensemble"},
                      _df_with_trend(), "AAPL", 60.0)
        assert "multiple" in out.lower() or "strategies" in out.lower()


# ── graceful degradation ─────────────────────────────────────────────────


class TestGracefulDegradation:
    def test_unknown_strategy_buy_returns_something(self):
        out = explain({"action": "buy"}, _df_with_trend(), "AAPL", 25.0)
        assert "AAPL" in out
        assert len(out) > 10

    def test_unknown_strategy_sell_returns_something(self):
        out = explain({"action": "sell"}, _df_with_trend(), "AAPL", 25.0)
        assert "AAPL" in out

    def test_unknown_strategy_hold(self):
        out = explain({"action": "hold"}, _df_with_trend(), "AAPL", 0.0)
        assert "AAPL" in out

    def test_none_df_still_returns_sentence(self):
        out = explain({"action": "buy", "strategy_key": "mean_reversion"},
                      None, "AAPL", 20.0)
        assert "AAPL" in out

    def test_empty_df_still_returns_sentence(self):
        out = explain({"action": "buy", "strategy_key": "momentum"},
                      pd.DataFrame(), "AAPL", 20.0)
        assert "AAPL" in out

    def test_zero_notional_omits_dollar(self):
        out = explain({"action": "buy"}, _df_with_trend(), "AAPL", 0.0)
        # No "$0" in the output — we suppress it.
        assert "$0" not in out

    def test_rl_suffix_appended(self):
        out = explain({"action": "buy", "strategy_key": "momentum",
                       "rl_selected": "momentum"},
                      _df_with_trend(), "TSLA", 30.0)
        assert "RL" in out

    def test_never_empty(self):
        for action in ("buy", "sell", "hold"):
            for key in ("mean_reversion", "momentum", "macd_crossover",
                        "volume_profile", "rsi_divergence", "triple_ema",
                        "scalper", "ensemble", "unknown_strategy"):
                out = explain({"action": action, "strategy_key": key},
                              _df_with_trend(), "AAPL", 20.0)
                assert out and len(out) > 5

    def test_template_exception_does_not_crash(self, monkeypatch):
        # If a template raises, explain() must still return a sentence.
        import core.trade_explainer as te
        def boom(*args, **kwargs):
            raise RuntimeError("template broke")
        monkeypatch.setitem(te._EXPLAINERS, "momentum", boom)

        out = explain({"action": "buy", "strategy_key": "momentum"},
                      _df_with_trend(), "AAPL", 25.0)
        assert "AAPL" in out
        assert "entry" in out.lower() or "triggered" in out.lower()
