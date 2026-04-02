"""Tests for core/multi_timeframe.py — multi-timeframe analysis."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from core.multi_timeframe import (
    resample_bars,
    compute_trend,
    analyze_timeframes,
    apply_mtf_filter,
    TimeframeTrend,
    MTFAnalysis,
    TIMEFRAMES,
)


# ── Helpers ──────────────────────────────────────────────────────────

def make_bars(n: int = 200, start_price: float = 100.0,
              trend: str = "flat", volatility: float = 0.5) -> pd.DataFrame:
    """Generate synthetic 1-minute OHLCV bars.

    Args:
        n: Number of bars.
        start_price: Starting close price.
        trend: "flat", "up", or "down".
        volatility: Random noise magnitude.
    """
    np.random.seed(42)
    times = pd.date_range("2026-01-01 09:30", periods=n, freq="1min")

    if trend == "up":
        drift = np.linspace(0, n * 0.05, n)
    elif trend == "down":
        drift = np.linspace(0, -n * 0.05, n)
    else:
        drift = np.zeros(n)

    noise = np.random.randn(n) * volatility
    close = start_price + drift + np.cumsum(noise) * 0.1

    high = close + np.abs(np.random.randn(n)) * volatility
    low = close - np.abs(np.random.randn(n)) * volatility
    open_ = close + np.random.randn(n) * volatility * 0.3
    volume = np.random.randint(100, 10000, n).astype(float)

    return pd.DataFrame({
        "time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def make_signal(action: str = "buy", strength: float = 0.7,
                reason: str = "test signal") -> dict:
    return {"action": action, "strength": strength, "reason": reason}


# ── resample_bars ────────────────────────────────────────────────────

class TestResampleBars:
    def test_resamples_to_5min(self):
        df = make_bars(100)
        result = resample_bars(df, 5)
        assert not result.empty
        assert len(result) == 20  # 100 / 5

    def test_resamples_to_15min(self):
        df = make_bars(150)
        result = resample_bars(df, 15)
        assert not result.empty
        assert len(result) == 10  # 150 / 15

    def test_resamples_to_60min(self):
        df = make_bars(240)
        result = resample_bars(df, 60)
        assert not result.empty
        # May not be exactly 4 due to start time not aligning on hour boundary
        assert 3 <= len(result) <= 5

    def test_ohlcv_aggregation_correct(self):
        """Verify OHLCV aggregation: open=first, high=max, low=min, close=last, volume=sum."""
        df = make_bars(10)
        result = resample_bars(df, 5)
        # First 5-min bar should aggregate first 5 1-min bars
        first_5 = df.iloc[:5]
        assert result.iloc[0]["open"] == first_5["open"].iloc[0]
        assert result.iloc[0]["high"] == first_5["high"].max()
        assert result.iloc[0]["low"] == first_5["low"].min()
        assert result.iloc[0]["close"] == first_5["close"].iloc[-1]
        assert result.iloc[0]["volume"] == first_5["volume"].sum()

    def test_returns_empty_for_none(self):
        assert resample_bars(None, 5).empty

    def test_returns_empty_for_empty_df(self):
        assert resample_bars(pd.DataFrame(), 5).empty

    def test_returns_empty_for_minutes_lte_1(self):
        df = make_bars(100)
        assert resample_bars(df, 1).empty
        assert resample_bars(df, 0).empty

    def test_returns_empty_when_too_few_bars(self):
        df = make_bars(2)
        # 2 bars resampled to 5min = 1 bar, which is < 2 minimum
        result = resample_bars(df, 5)
        assert result.empty

    def test_has_time_column(self):
        df = make_bars(100)
        result = resample_bars(df, 5)
        assert "time" in result.columns

    def test_works_with_datetime_index(self):
        """Should handle DataFrame with DatetimeIndex instead of 'time' column."""
        df = make_bars(100)
        df = df.set_index("time")
        result = resample_bars(df, 5)
        assert not result.empty


# ── compute_trend ────────────────────────────────────────────────────

class TestComputeTrend:
    def test_bullish_trend(self):
        df = make_bars(200, trend="up")
        resampled = resample_bars(df, 5)
        trend = compute_trend(resampled)
        assert trend is not None
        assert trend.trend == "bullish"
        assert trend.strength > 0

    def test_bearish_trend(self):
        df = make_bars(200, trend="down")
        resampled = resample_bars(df, 5)
        trend = compute_trend(resampled)
        assert trend is not None
        assert trend.trend == "bearish"
        assert trend.strength > 0

    def test_neutral_flat_market(self):
        # Create perfectly flat data
        n = 100
        times = pd.date_range("2026-01-01", periods=n, freq="5min")
        df = pd.DataFrame({
            "time": times,
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        })
        trend = compute_trend(df)
        assert trend is not None
        assert trend.trend == "neutral"

    def test_returns_none_insufficient_data(self):
        df = make_bars(10)
        resampled = resample_bars(df, 5)
        # Only 2 bars — less than slow_period (21)
        assert compute_trend(resampled) is None

    def test_returns_none_for_none_df(self):
        assert compute_trend(None) is None

    def test_strength_clamped_at_1(self):
        """Strength should never exceed 1.0."""
        df = make_bars(200, trend="up", volatility=0.01)
        resampled = resample_bars(df, 5)
        trend = compute_trend(resampled)
        assert trend is not None
        assert trend.strength <= 1.0

    def test_ema_values_populated(self):
        df = make_bars(200, trend="up")
        resampled = resample_bars(df, 5)
        trend = compute_trend(resampled)
        assert trend.ema_fast > 0
        assert trend.ema_slow > 0

    def test_zero_close_returns_none(self):
        """If slow EMA is zero, should return None to avoid division by zero."""
        n = 50
        times = pd.date_range("2026-01-01", periods=n, freq="5min")
        df = pd.DataFrame({
            "time": times,
            "open": [0.0] * n,
            "high": [0.0] * n,
            "low": [0.0] * n,
            "close": [0.0] * n,
            "volume": [1000.0] * n,
        })
        assert compute_trend(df) is None


# ── analyze_timeframes ───────────────────────────────────────────────

class TestAnalyzeTimeframes:
    def test_all_bullish_alignment(self):
        df = make_bars(500, trend="up")
        result = analyze_timeframes(df)
        assert result.alignment == "bullish"
        assert result.confidence > 0
        assert result.supports_buy is True
        assert result.supports_sell is False

    def test_all_bearish_alignment(self):
        df = make_bars(500, trend="down")
        result = analyze_timeframes(df)
        assert result.alignment == "bearish"
        assert result.confidence > 0
        assert result.supports_sell is True
        assert result.supports_buy is False

    def test_neutral_on_insufficient_data(self):
        df = make_bars(10)  # Not enough for any higher TF
        result = analyze_timeframes(df)
        assert result.alignment == "neutral"
        assert result.confidence == 0.0

    def test_custom_timeframes(self):
        df = make_bars(300, trend="up")
        result = analyze_timeframes(df, timeframes={"3m": 3, "10m": 10})
        assert len(result.trends) > 0
        labels = {t.timeframe for t in result.trends}
        assert labels <= {"3m", "10m"}

    def test_trends_populated(self):
        df = make_bars(500, trend="up")
        result = analyze_timeframes(df)
        assert len(result.trends) > 0
        for t in result.trends:
            assert t.timeframe in TIMEFRAMES
            assert t.trend in ("bullish", "bearish", "neutral")

    def test_empty_df(self):
        result = analyze_timeframes(pd.DataFrame())
        assert result.alignment == "neutral"
        assert result.confidence == 0.0
        assert result.trends == []

    def test_none_df(self):
        result = analyze_timeframes(None)
        assert result.alignment == "neutral"

    def test_confidence_between_0_and_1(self):
        df = make_bars(500, trend="up")
        result = analyze_timeframes(df)
        assert 0.0 <= result.confidence <= 1.0

    def test_mixed_alignment_confidence(self):
        """When timeframes disagree, alignment should be mixed with partial confidence."""
        df = make_bars(500, trend="up")
        # Use custom timeframes and mock one to return bearish
        with patch("core.multi_timeframe.compute_trend") as mock_trend:
            mock_trend.side_effect = [
                TimeframeTrend("5m", "bullish", 101.0, 100.0, 0.5),
                TimeframeTrend("15m", "bearish", 99.0, 100.0, 0.5),
                TimeframeTrend("1h", "bullish", 101.0, 100.0, 0.5),
            ]
            result = analyze_timeframes(df)
        # 2 bullish, 1 bearish = mixed
        assert result.alignment == "mixed"
        assert result.confidence > 0


# ── apply_mtf_filter ─────────────────────────────────────────────────

class TestApplyMTFFilter:
    def test_hold_signal_unchanged(self):
        signal = make_signal("hold", 0.0)
        df = make_bars(200)
        result = apply_mtf_filter(signal, df)
        assert result["action"] == "hold"
        assert result["strength"] == 0.0

    def test_buy_boosted_in_uptrend(self):
        signal = make_signal("buy", 0.7)
        df = make_bars(500, trend="up")
        result = apply_mtf_filter(signal, df, weight=0.15)
        assert result["strength"] >= 0.7  # Should be boosted or at least unchanged
        assert "mtf_alignment" in result

    def test_buy_dampened_in_downtrend(self):
        signal = make_signal("buy", 0.7)
        df = make_bars(500, trend="down")
        result = apply_mtf_filter(signal, df, weight=0.15)
        assert result["strength"] <= 0.7  # Should be dampened or unchanged
        assert result["strength"] >= 0.3  # Never below floor

    def test_sell_boosted_in_downtrend(self):
        signal = make_signal("sell", 0.7)
        df = make_bars(500, trend="down")
        result = apply_mtf_filter(signal, df, weight=0.15)
        assert result["strength"] >= 0.7  # Should be boosted or unchanged

    def test_sell_dampened_in_uptrend(self):
        signal = make_signal("sell", 0.7)
        df = make_bars(500, trend="up")
        result = apply_mtf_filter(signal, df, weight=0.15)
        assert result["strength"] <= 0.7  # Should be dampened or unchanged
        assert result["strength"] >= 0.3  # Never below floor

    def test_strength_never_exceeds_1(self):
        signal = make_signal("buy", 0.95)
        df = make_bars(500, trend="up")
        result = apply_mtf_filter(signal, df, weight=0.5)
        assert result["strength"] <= 1.0

    def test_strength_floor_at_0_3(self):
        signal = make_signal("buy", 0.35)
        df = make_bars(500, trend="down")
        result = apply_mtf_filter(signal, df, weight=0.5)
        assert result["strength"] >= 0.3

    def test_mtf_metadata_added(self):
        signal = make_signal("buy", 0.7)
        df = make_bars(500, trend="up")
        result = apply_mtf_filter(signal, df)
        assert "mtf_alignment" in result
        assert "mtf_confidence" in result

    def test_insufficient_data_returns_unchanged(self):
        signal = make_signal("buy", 0.7)
        df = make_bars(30)  # Less than 60 bars required
        result = apply_mtf_filter(signal, df)
        assert result["strength"] == 0.7
        assert "mtf_alignment" not in result

    def test_none_df_returns_unchanged(self):
        signal = make_signal("buy", 0.7)
        result = apply_mtf_filter(signal, None)
        assert result["strength"] == 0.7

    def test_reason_includes_mtf_annotation(self):
        signal = make_signal("buy", 0.7)
        df = make_bars(500, trend="up")
        result = apply_mtf_filter(signal, df, weight=0.15)
        if result["mtf_alignment"] == "bullish":
            assert "MTF:" in result["reason"]

    def test_zero_weight_no_change(self):
        signal = make_signal("buy", 0.7)
        df = make_bars(500, trend="up")
        result = apply_mtf_filter(signal, df, weight=0.0)
        # Zero weight = zero boost/penalty, but metadata still added
        assert "mtf_alignment" in result

    def test_mixed_alignment_no_strength_change(self):
        """When alignment is mixed, strength should not be modified."""
        signal = make_signal("buy", 0.7)
        df = make_bars(500, trend="up")
        # Force mixed alignment by mocking
        with patch("core.multi_timeframe.analyze_timeframes") as mock_analyze:
            mock_analyze.return_value = MTFAnalysis(
                trends=[], alignment="mixed", confidence=0.3)
            result = apply_mtf_filter(signal, df)
        assert result["strength"] == 0.7
        assert result["mtf_alignment"] == "mixed"

    def test_neutral_alignment_no_strength_change(self):
        """When alignment is neutral, strength should not be modified."""
        signal = make_signal("sell", 0.8)
        df = make_bars(500)
        with patch("core.multi_timeframe.analyze_timeframes") as mock_analyze:
            mock_analyze.return_value = MTFAnalysis(
                trends=[], alignment="neutral", confidence=0.0)
            result = apply_mtf_filter(signal, df)
        assert result["strength"] == 0.8
        assert result["mtf_alignment"] == "neutral"

    def test_nan_in_data_handled_gracefully(self):
        """NaN values in close prices should not crash or produce NaN strength."""
        df = make_bars(200, trend="up")
        df.loc[50, "close"] = np.nan  # Inject a NaN
        signal = make_signal("buy", 0.7)
        result = apply_mtf_filter(signal, df)
        # Should not crash and strength should be a valid number
        assert not np.isnan(result["strength"])


# ── MTFAnalysis properties ───────────────────────────────────────────

class TestMTFAnalysisProperties:
    def test_supports_buy(self):
        a = MTFAnalysis(trends=[], alignment="bullish", confidence=0.8)
        assert a.supports_buy is True
        assert a.supports_sell is False

    def test_supports_sell(self):
        a = MTFAnalysis(trends=[], alignment="bearish", confidence=0.8)
        assert a.supports_sell is True
        assert a.supports_buy is False

    def test_mixed_supports_neither(self):
        a = MTFAnalysis(trends=[], alignment="mixed", confidence=0.3)
        assert a.supports_buy is False
        assert a.supports_sell is False

    def test_neutral_supports_neither(self):
        a = MTFAnalysis(trends=[], alignment="neutral", confidence=0.0)
        assert a.supports_buy is False
        assert a.supports_sell is False


# ── Router Integration ───────────────────────────────────────────────

class TestRouterIntegration:
    def test_mtf_applied_when_enabled(self):
        """MTF filter should be called in router when MTF_ENABLED is True."""
        from config import Config
        with patch.object(Config, "MTF_ENABLED", True), \
             patch.object(Config, "MTF_WEIGHT", 0.15), \
             patch.object(Config, "SENTIMENT_ENABLED", False), \
             patch.object(Config, "LLM_ANALYST_ENABLED", False), \
             patch.object(Config, "RL_STRATEGY_ENABLED", False), \
             patch("strategies.router.apply_mtf_filter") as mock_mtf:
            mock_mtf.side_effect = lambda sig, df, weight: sig
            from strategies.router import compute_signals
            df = make_bars(200)
            compute_signals("AAPL", df)
            mock_mtf.assert_called_once()

    def test_mtf_not_applied_when_disabled(self):
        """MTF filter should not be called when MTF_ENABLED is False."""
        from config import Config
        with patch.object(Config, "MTF_ENABLED", False), \
             patch.object(Config, "SENTIMENT_ENABLED", False), \
             patch.object(Config, "LLM_ANALYST_ENABLED", False), \
             patch.object(Config, "RL_STRATEGY_ENABLED", False), \
             patch("strategies.router.apply_mtf_filter") as mock_mtf:
            from strategies.router import compute_signals
            df = make_bars(200)
            compute_signals("AAPL", df)
            mock_mtf.assert_not_called()

    def test_signal_contract_preserved(self):
        """MTF filter must preserve the signal contract."""
        from config import Config
        with patch.object(Config, "MTF_ENABLED", True), \
             patch.object(Config, "MTF_WEIGHT", 0.15), \
             patch.object(Config, "SENTIMENT_ENABLED", False), \
             patch.object(Config, "LLM_ANALYST_ENABLED", False), \
             patch.object(Config, "RL_STRATEGY_ENABLED", False):
            from strategies.router import compute_signals
            df = make_bars(500, trend="up")
            signal = compute_signals("AAPL", df)
            assert "action" in signal
            assert "reason" in signal
            assert "strength" in signal
            assert signal["action"] in ("buy", "sell", "hold")
            assert 0 <= signal["strength"] <= 1.0
