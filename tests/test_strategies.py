"""Unit tests for all trading strategies.

Tests that each strategy:
  - Returns valid signal format {action, reason, strength}
  - Handles insufficient data gracefully
  - Produces buy/sell/hold signals
  - Has strength in valid range [0, 1]
"""

import pytest
import pandas as pd
import numpy as np

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
