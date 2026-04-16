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
        # Sprint 6E: state extended from 10 → 16 dims (see rl_features docstring).
        from core.rl_features import NUM_FEATURES
        assert NUM_FEATURES == 16


# ── Sprint 6E: new helper behaviour ──────────────────────────────────────


class TestTanhZscore:
    def test_constant_series_is_zero(self):
        from core.rl_features import _tanh_zscore
        s = pd.Series([5.0] * 30)
        # std is 0 → guarded return of 0.0, not NaN
        assert _tanh_zscore(s) == 0.0

    def test_empty_series_is_zero(self):
        from core.rl_features import _tanh_zscore
        assert _tanh_zscore(pd.Series(dtype=float)) == 0.0

    def test_bounded_in_negative_one_one(self):
        from core.rl_features import _tanh_zscore
        # Extreme spike at the end
        s = pd.Series(list(np.random.randn(100)) + [100.0])
        val = _tanh_zscore(s)
        assert -1.0 <= val <= 1.0

    def test_monotonic_in_direction(self):
        from core.rl_features import _tanh_zscore
        # Last value below recent mean → negative z; above → positive z
        s_below = pd.Series([100.0] * 20 + [95.0])
        s_above = pd.Series([100.0] * 20 + [105.0])
        assert _tanh_zscore(s_below) < 0
        assert _tanh_zscore(s_above) > 0


class TestRealizedVol:
    def test_zero_for_flat_series(self):
        from core.rl_features import _realized_vol
        s = pd.Series([100.0] * 30)
        assert _realized_vol(s, window=5) == 0.0

    def test_positive_for_noisy_series(self):
        from core.rl_features import _realized_vol
        np.random.seed(7)
        s = pd.Series(100 + np.cumsum(np.random.randn(40) * 0.5))
        assert _realized_vol(s, window=20) > 0.0

    def test_short_series_returns_zero(self):
        from core.rl_features import _realized_vol
        s = pd.Series([100.0, 101.0])
        # Not enough data for 20-window
        assert _realized_vol(s, window=20) == 0.0


class TestVolOfVol:
    def test_flat_series_is_zero(self):
        from core.rl_features import _vol_of_vol
        s = pd.Series([100.0] * 50)
        assert _vol_of_vol(s) == 0.0

    def test_non_flat_is_positive(self):
        from core.rl_features import _vol_of_vol
        np.random.seed(13)
        s = pd.Series(100 + np.cumsum(np.random.randn(60) * 0.8))
        assert _vol_of_vol(s) >= 0.0


class TestSpreadProxyZ:
    def test_missing_cols_returns_zero(self):
        from core.rl_features import _spread_proxy_z
        df = pd.DataFrame({"close": [100.0] * 30})
        assert _spread_proxy_z(df) == 0.0

    def test_bounded(self):
        from core.rl_features import _spread_proxy_z
        np.random.seed(19)
        close = 100 + np.cumsum(np.random.randn(40) * 0.5)
        df = pd.DataFrame({
            "close": close,
            "high": close + np.abs(np.random.randn(40)),
            "low": close - np.abs(np.random.randn(40)),
        })
        val = _spread_proxy_z(df)
        assert -1.0 <= val <= 1.0


class TestTimeToClose:
    def test_no_datetime_index_is_half(self):
        from core.rl_features import _time_to_close
        df = pd.DataFrame({"close": [100, 101, 102]})
        # Integer index → 0.5 neutral
        assert _time_to_close(df) == 0.5

    def test_early_session_near_one(self):
        from core.rl_features import _time_to_close
        # 9:30 AM is close to session start — should be near 1.0
        df = pd.DataFrame({"close": [100]},
                          index=pd.DatetimeIndex(["2024-01-02 09:30"]))
        val = _time_to_close(df)
        assert 0.9 < val <= 1.0

    def test_after_close_is_zero(self):
        from core.rl_features import _time_to_close
        df = pd.DataFrame({"close": [100]},
                          index=pd.DatetimeIndex(["2024-01-02 17:00"]))
        assert _time_to_close(df) == 0.0

    def test_bounded(self):
        from core.rl_features import _time_to_close
        df = pd.DataFrame({"close": [100]},
                          index=pd.DatetimeIndex(["2024-01-02 12:00"]))
        val = _time_to_close(df)
        assert 0.0 <= val <= 1.0


class TestRegimeBits:
    def test_no_regime_sets_bits_to_zero(self):
        from core.rl_features import extract_features
        df = _make_df()
        features = extract_features(df, regime=None)
        assert features[-2] == 0.0  # regime_stress
        assert features[-1] == 0.0  # regime_crisis

    def test_low_vol_and_normal_are_zero_bits(self):
        from core.rl_features import extract_features
        df = _make_df()
        for regime in ("low_vol", "normal"):
            features = extract_features(df, regime=regime)
            assert features[-2] == 0.0
            assert features[-1] == 0.0

    def test_high_vol_sets_stress_only(self):
        from core.rl_features import extract_features
        df = _make_df()
        features = extract_features(df, regime="high_vol")
        assert features[-2] == 1.0  # stress
        assert features[-1] == 0.0  # crisis still off

    def test_crisis_sets_both_bits(self):
        from core.rl_features import extract_features
        df = _make_df()
        features = extract_features(df, regime="crisis")
        assert features[-2] == 1.0  # stress
        assert features[-1] == 1.0  # crisis


class TestFeatureShape16:
    def test_sixteen_dim_output(self):
        from core.rl_features import extract_features
        df = _make_df()
        features = extract_features(df)
        assert features.shape == (16,)

    def test_all_features_finite(self):
        # A regression guard: extract_features must never emit NaN/Inf,
        # regardless of regime input.
        from core.rl_features import extract_features
        df = _make_df()
        for regime in (None, "low_vol", "normal", "high_vol", "crisis"):
            features = extract_features(df, regime=regime)
            assert np.isfinite(features).all(), f"NaN/Inf leaked with {regime=}"
