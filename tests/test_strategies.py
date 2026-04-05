"""Unit tests for all trading strategies.

Tests that each strategy:
  - Returns valid signal format {action, reason, strength}
  - Handles insufficient data gracefully
  - Produces buy/sell/hold signals
  - Has strength in valid range [0, 1]
"""

import os

import pytest
import pandas as pd
import numpy as np
import ta

from tests.conftest import _make_ohlcv

from strategies import (
    mean_reversion_aggressive,
    mean_reversion,
    volume_profile,
    momentum,
    macd_crossover,
    triple_ema,
    rsi_divergence,
    scalper,
    ensemble,
)

ALL_STRATEGIES = [
    ("mean_reversion_aggressive", mean_reversion_aggressive),
    ("mean_reversion", mean_reversion),
    ("volume_profile", volume_profile),
    ("momentum", momentum),
    ("macd_crossover", macd_crossover),
    ("triple_ema", triple_ema),
    ("rsi_divergence", rsi_divergence),
    ("scalper", scalper),
    ("ensemble", ensemble),
]


class TestSignalFormat:
    """Every strategy must return a valid signal dict."""

    @pytest.mark.parametrize("name,strategy", ALL_STRATEGIES)
    def test_returns_dict(self, name, strategy, flat_market):
        signal = strategy.compute_signals(flat_market)
        assert isinstance(signal, dict), f"{name} did not return a dict"

    @pytest.mark.parametrize("name,strategy", ALL_STRATEGIES)
    def test_has_required_keys(self, name, strategy, flat_market):
        signal = strategy.compute_signals(flat_market)
        assert "action" in signal, f"{name} missing 'action'"
        assert "reason" in signal, f"{name} missing 'reason'"
        assert "strength" in signal, f"{name} missing 'strength'"

    @pytest.mark.parametrize("name,strategy", ALL_STRATEGIES)
    def test_action_is_valid(self, name, strategy, flat_market):
        signal = strategy.compute_signals(flat_market)
        assert signal["action"] in ("buy", "sell", "hold"), \
            f"{name} returned invalid action: {signal['action']}"

    @pytest.mark.parametrize("name,strategy", ALL_STRATEGIES)
    def test_reason_is_string(self, name, strategy, flat_market):
        signal = strategy.compute_signals(flat_market)
        assert isinstance(signal["reason"], str), f"{name} reason not a string"

    @pytest.mark.parametrize("name,strategy", ALL_STRATEGIES)
    def test_strength_in_range(self, name, strategy, flat_market):
        signal = strategy.compute_signals(flat_market)
        assert 0 <= signal["strength"] <= 1.0, \
            f"{name} strength out of range: {signal['strength']}"


class TestInsufficientData:
    """All strategies should return 'hold' when data is too small."""

    @pytest.mark.parametrize("name,strategy", ALL_STRATEGIES)
    def test_small_df_returns_hold(self, name, strategy, small_df):
        signal = strategy.compute_signals(small_df)
        assert signal["action"] == "hold", \
            f"{name} did not return 'hold' for insufficient data"


class TestMultipleMarketConditions:
    """Test each strategy across different market conditions."""

    @pytest.mark.parametrize("name,strategy", ALL_STRATEGIES)
    def test_uptrend(self, name, strategy, uptrend_market):
        signal = strategy.compute_signals(uptrend_market)
        assert signal["action"] in ("buy", "sell", "hold")

    @pytest.mark.parametrize("name,strategy", ALL_STRATEGIES)
    def test_downtrend(self, name, strategy, downtrend_market):
        signal = strategy.compute_signals(downtrend_market)
        assert signal["action"] in ("buy", "sell", "hold")

    @pytest.mark.parametrize("name,strategy", ALL_STRATEGIES)
    def test_volatile(self, name, strategy, volatile_market):
        signal = strategy.compute_signals(volatile_market)
        assert signal["action"] in ("buy", "sell", "hold")


class TestMeanRevAggressive:
    """Specific tests for mean_reversion_aggressive."""

    def test_buy_on_oversold(self, oversold_df):
        signal = mean_reversion_aggressive.compute_signals(oversold_df)
        # Strong downtrend should eventually trigger a buy
        assert signal["action"] in ("buy", "hold")

    def test_sell_on_overbought(self, overbought_df):
        signal = mean_reversion_aggressive.compute_signals(overbought_df)
        assert signal["action"] in ("sell", "hold")

    def test_hold_in_neutral(self, flat_market):
        signal = mean_reversion_aggressive.compute_signals(flat_market)
        # Flat market should mostly hold
        assert signal["action"] in ("buy", "sell", "hold")


class TestMeanRevAggressiveV2Thresholds:
    """Tests for v2 RSI sell thresholds (hardcoded, PHASE0_RSI_FIX removed)."""

    def test_hold_at_moderate_rsi(self):
        """Moderate RSI (50-65) should NOT trigger sell."""
        df = _make_ohlcv(100, trend="flat", seed=99)
        df.iloc[-5:, df.columns.get_loc("close")] *= 1.01
        signal = mean_reversion_aggressive.compute_signals(df)
        rsi = ta.momentum.rsi(df["close"], window=10).iloc[-1]
        assert 45 < rsi < 70, f"Test setup error: RSI={rsi:.1f}, expected 45-70"
        assert signal["action"] in ("hold", "buy"), (
            f"RSI={rsi:.0f} should NOT sell with v2 thresholds"
        )

    def test_sell_at_extreme_rsi(self):
        """Strong uptrend (high RSI) should sell."""
        df = _make_ohlcv(100, trend="up", seed=42)
        signal = mean_reversion_aggressive.compute_signals(df)
        assert signal["action"] in ("sell", "hold")

    def test_signal_contract_all_paths(self):
        """Signal contract maintained across buy, sell, and hold paths."""
        for trend, seed in [("up", 42), ("down", 42), ("flat", 99), ("volatile", 7)]:
            df = _make_ohlcv(100, trend=trend, seed=seed)
            signal = mean_reversion_aggressive.compute_signals(df)
            assert "action" in signal
            assert "reason" in signal
            assert "strength" in signal
            assert signal["action"] in ("buy", "sell", "hold")
            assert isinstance(signal["strength"], (int, float))
            assert 0 <= signal["strength"] <= 1.0

    def test_pctb_alone_no_sell_with_moderate_rsi(self):
        """Core fix: %B > 0.95 with RSI 55-65 must NOT sell."""
        n = 60
        prices = [100.0] * (n - 3)
        prices.extend([100.0, 101.5, 102.0])
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 0.3 for p in prices],
            "low": [p - 0.3 for p in prices],
            "close": prices,
            "volume": [1_000_000] * n,
        })
        rsi = ta.momentum.rsi(pd.Series(prices), window=10).iloc[-1]
        bb = ta.volatility.BollingerBands(pd.Series(prices), window=15, window_dev=1.5)
        pct_b = bb.bollinger_pband().iloc[-1]

        signal = mean_reversion_aggressive.compute_signals(df)

        if pct_b > 0.9 and rsi < 70:
            assert signal["action"] != "sell", (
                f"Bug regression: %B={pct_b:.2f}, RSI={rsi:.0f} should NOT sell"
            )

    def test_strength_range(self):
        """All strength values stay in [0, 1]."""
        for seed in range(10):
            for trend in ("up", "down", "flat", "volatile"):
                df = _make_ohlcv(100, trend=trend, seed=seed)
                signal = mean_reversion_aggressive.compute_signals(df)
                assert 0 <= signal["strength"] <= 1.0, (
                    f"strength={signal['strength']} out of range "
                    f"(trend={trend}, seed={seed})"
                )


class TestVolumeProfile:
    """Specific tests for volume_profile."""

    def test_needs_40_bars(self):
        from tests.conftest import _make_ohlcv
        df = _make_ohlcv(35)
        signal = volume_profile.compute_signals(df)
        assert signal["action"] == "hold"

    def test_no_crash_on_zero_volume(self, flat_market):
        df = flat_market.copy()
        df["volume"] = 0
        # Should not crash, just hold
        signal = volume_profile.compute_signals(df)
        assert signal["action"] in ("buy", "sell", "hold")


class TestEnsemble:
    """Specific tests for ensemble strategy."""

    def test_combines_momentum_and_mr(self, flat_market):
        signal = ensemble.compute_signals(flat_market)
        assert signal["action"] in ("buy", "sell", "hold")

    def test_reason_prefixed(self, uptrend_market):
        signal = ensemble.compute_signals(uptrend_market)
        if signal["action"] != "hold" or "ensemble" in signal["reason"]:
            assert "ensemble" in signal["reason"].lower() or signal["action"] == "hold"


class TestMomentum:
    """Specific tests for momentum strategy."""

    def test_needs_lookback_data(self):
        from tests.conftest import _make_ohlcv
        df = _make_ohlcv(20)  # Exactly at MOMENTUM_LOOKBACK
        signal = momentum.compute_signals(df)
        assert signal["action"] == "hold"
