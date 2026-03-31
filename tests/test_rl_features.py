"""Tests for core/rl_features.py — RL feature extraction."""

import numpy as np
import pandas as pd
import pytest


def _make_df(n=60, with_volume=True, with_highlow=True):
    """Create a sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    data = {"close": close}
    if with_highlow:
        data["high"] = close + np.abs(np.random.randn(n) * 0.3)
        data["low"] = close - np.abs(np.random.randn(n) * 0.3)
    if with_volume:
        data["volume"] = np.random.randint(1000, 5000, n)
    return pd.DataFrame(data)


class TestComputeRSI:
    def test_returns_series(self):
        from core.rl_features import compute_rsi
        df = _make_df()
        rsi = compute_rsi(df["close"])
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(df)

    def test_values_in_range(self):
        from core.rl_features import compute_rsi
        df = _make_df(100)
        rsi = compute_rsi(df["close"])
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestComputeBollingerPctB:
    def test_returns_series(self):
        from core.rl_features import compute_bollinger_pct_b
        df = _make_df()
        pct_b = compute_bollinger_pct_b(df["close"])
        assert isinstance(pct_b, pd.Series)

    def test_midpoint_near_half(self):
        from core.rl_features import compute_bollinger_pct_b
        # Constant price -> %B = 0.5 (NaN due to zero std, but concept)
        close = pd.Series([100.0] * 30)
        pct_b = compute_bollinger_pct_b(close)
        # Will be NaN because std=0, which is expected
        assert pd.isna(pct_b.iloc[-1])


class TestComputeMACDHist:
    def test_returns_series(self):
        from core.rl_features import compute_macd_hist
        df = _make_df()
        hist = compute_macd_hist(df["close"])
        assert isinstance(hist, pd.Series)
        assert len(hist) == len(df)


class TestComputeATR:
    def test_returns_series(self):
        from core.rl_features import compute_atr
        df = _make_df()
        atr = compute_atr(df)
        assert isinstance(atr, pd.Series)

    def test_positive_values(self):
        from core.rl_features import compute_atr
        df = _make_df(100)
        atr = compute_atr(df)
        valid = atr.dropna()
        assert (valid >= 0).all()


class TestExtractFeatures:
    def test_returns_correct_shape(self):
        from core.rl_features import extract_features, NUM_FEATURES
        df = _make_df()
        features = extract_features(df)
        assert features is not None
        assert features.shape == (NUM_FEATURES,)

    def test_returns_float32(self):
        from core.rl_features import extract_features
        df = _make_df()
        features = extract_features(df)
        assert features.dtype == np.float32

    def test_no_nans(self):
        from core.rl_features import extract_features
        df = _make_df()
        features = extract_features(df)
        assert not np.isnan(features).any()

    def test_insufficient_data_returns_none(self):
        from core.rl_features import extract_features
        df = _make_df(10)
        assert extract_features(df) is None

    def test_none_df_returns_none(self):
        from core.rl_features import extract_features
        assert extract_features(None) is None

    def test_without_volume(self):
        from core.rl_features import extract_features, NUM_FEATURES
        df = _make_df(with_volume=False)
        features = extract_features(df)
        assert features is not None
        assert features.shape == (NUM_FEATURES,)

    def test_without_highlow(self):
        from core.rl_features import extract_features, NUM_FEATURES
        df = _make_df(with_highlow=False)
        features = extract_features(df)
        assert features is not None
        assert features.shape == (NUM_FEATURES,)

    def test_with_datetime_index(self):
        from core.rl_features import extract_features, NUM_FEATURES
        df = _make_df()
        df.index = pd.date_range("2024-01-01 09:30", periods=len(df), freq="1min")
        features = extract_features(df)
        assert features is not None
        assert features.shape == (NUM_FEATURES,)

    def test_features_change_with_data(self):
        from core.rl_features import extract_features
        df1 = _make_df()
        # Create a different df with different seed
        np.random.seed(99)
        close2 = 200 + np.cumsum(np.random.randn(60) * 2.0)
        df2 = pd.DataFrame({
            "close": close2,
            "high": close2 + np.abs(np.random.randn(60) * 0.3),
            "low": close2 - np.abs(np.random.randn(60) * 0.3),
            "volume": np.random.randint(1000, 5000, 60),
        })
        f1 = extract_features(df1)
        f2 = extract_features(df2)
        # Features should differ for different data
        assert not np.allclose(f1, f2)


class TestConstants:
    def test_strategy_keys_count(self):
        from core.rl_features import STRATEGY_KEYS, NUM_STRATEGIES
        assert len(STRATEGY_KEYS) == NUM_STRATEGIES
        assert NUM_STRATEGIES == 9

    def test_num_features(self):
        from core.rl_features import NUM_FEATURES
        assert NUM_FEATURES == 10
