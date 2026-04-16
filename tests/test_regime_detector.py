"""Sprint 6C: regime detector tests.

Covers the pure math (annualize_vol, classify_vol, compute_realized_vol),
the stateful classifier, and the strategy-compatibility policy.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from core import regime_detector
from core.regime_detector import (
    MIN_BARS_REQUIRED,
    THRESHOLD_CRISIS,
    THRESHOLD_HIGH,
    THRESHOLD_LOW,
    RegimeDetector,
    RegimeSnapshot,
    annualize_vol,
    classify_vol,
    compute_realized_vol,
    is_strategy_allowed,
    regime_one_hot,
    reload_overrides,
)


# ── helpers ───────────────────────────────────────────────────────────────


def _make_price_df(returns: list[float], start_price: float = 100.0) -> pd.DataFrame:
    """Build an OHLCV DataFrame from a sequence of daily returns."""
    prices = [start_price]
    for r in returns:
        prices.append(prices[-1] * (1.0 + r))
    return pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1_000_000] * len(prices),
    })


def _daily_ret_for_annualized_vol(target: float, n: int = 25) -> list[float]:
    """Construct an alternating-return series whose annualized std ≈ target."""
    # std of alternating ±r is r; annualized is r * sqrt(252). So r = target / sqrt(252).
    r = target / math.sqrt(252)
    # Use +r, -r alternation so the variance is exact.
    return [r if i % 2 == 0 else -r for i in range(n)]


# ── annualize_vol ──────────────────────────────────────────────────────────


class TestAnnualizeVol:
    def test_empty_returns_zero(self):
        assert annualize_vol([]) == 0.0
        assert annualize_vol(np.array([])) == 0.0

    def test_none_returns_zero(self):
        assert annualize_vol(None) == 0.0

    def test_single_value_returns_zero(self):
        # std needs at least 2 samples for ddof=1
        assert annualize_vol([0.01]) == 0.0

    def test_all_nan_returns_zero(self):
        assert annualize_vol([np.nan, np.nan]) == 0.0

    def test_constant_zero_returns_zero(self):
        assert annualize_vol([0.0, 0.0, 0.0, 0.0]) == 0.0

    def test_known_value(self):
        # 1% daily std → 1% * sqrt(252) ≈ 0.1587
        rets = _daily_ret_for_annualized_vol(0.20, n=30)
        vol = annualize_vol(rets)
        assert abs(vol - 0.20) < 0.01

    def test_mixed_nan_dropped(self):
        # NaNs shouldn't blow up the computation
        vol = annualize_vol([0.01, np.nan, -0.01, 0.02, np.nan])
        assert vol > 0 and math.isfinite(vol)


# ── classify_vol ───────────────────────────────────────────────────────────


class TestClassifyVol:
    def test_zero_is_normal(self):
        # Degenerate — don't spuriously halt trading.
        assert classify_vol(0.0) == "normal"

    def test_negative_is_normal(self):
        assert classify_vol(-0.1) == "normal"

    def test_nan_is_normal(self):
        assert classify_vol(float("nan")) == "normal"

    def test_inf_is_normal(self):
        assert classify_vol(float("inf")) == "normal"

    def test_low_vol_boundary_below(self):
        assert classify_vol(THRESHOLD_LOW - 0.001) == "low_vol"

    def test_low_vol_boundary_at_threshold(self):
        # Exactly at 15% is normal (right-exclusive low-end)
        assert classify_vol(THRESHOLD_LOW) == "normal"

    def test_normal_midrange(self):
        assert classify_vol(0.20) == "normal"

    def test_normal_to_high_boundary(self):
        assert classify_vol(THRESHOLD_HIGH - 0.001) == "normal"
        assert classify_vol(THRESHOLD_HIGH) == "high_vol"

    def test_high_to_crisis_boundary(self):
        assert classify_vol(THRESHOLD_CRISIS - 0.001) == "high_vol"
        assert classify_vol(THRESHOLD_CRISIS) == "crisis"

    def test_extreme_vol_is_crisis(self):
        assert classify_vol(1.0) == "crisis"
        assert classify_vol(5.0) == "crisis"


# ── compute_realized_vol ──────────────────────────────────────────────────


class TestComputeRealizedVol:
    def test_none_df_zero(self):
        assert compute_realized_vol(None) == 0.0

    def test_empty_df_zero(self):
        assert compute_realized_vol(pd.DataFrame()) == 0.0

    def test_missing_price_col_zero(self):
        df = pd.DataFrame({"open": [1, 2, 3]})
        assert compute_realized_vol(df) == 0.0

    def test_single_bar_zero(self):
        assert compute_realized_vol(pd.DataFrame({"close": [100]})) == 0.0

    def test_stable_prices_near_zero(self):
        df = _make_price_df([0.0] * 25)
        assert compute_realized_vol(df) < 0.001

    def test_low_vol_recognized(self):
        # ~8% annualized target
        rets = _daily_ret_for_annualized_vol(0.08, n=30)
        df = _make_price_df(rets)
        vol = compute_realized_vol(df, window=20)
        assert 0.04 < vol < 0.12

    def test_high_vol_recognized(self):
        # ~35% annualized target
        rets = _daily_ret_for_annualized_vol(0.35, n=30)
        df = _make_price_df(rets)
        vol = compute_realized_vol(df, window=20)
        assert 0.30 < vol < 0.40

    def test_custom_price_col(self):
        df = pd.DataFrame({"px": [100, 101, 99, 102, 98, 103, 97, 104, 96, 105]
                                  + [100] * 12})
        vol = compute_realized_vol(df, window=15, price_col="px")
        assert vol > 0


# ── RegimeDetector ─────────────────────────────────────────────────────────


class TestRegimeDetector:
    def test_initial_state(self):
        rd = RegimeDetector()
        assert rd.get_current_regime() == "normal"
        assert rd.get_current_vol() == 0.0
        assert rd.get_regime_history() == []

    def test_update_with_insufficient_data_returns_none(self):
        rd = RegimeDetector()
        df = _make_price_df([0.01] * (MIN_BARS_REQUIRED - 2))
        assert rd.update(df) is None
        assert rd.get_current_regime() == "normal"  # unchanged

    def test_update_with_none_df_safe(self):
        rd = RegimeDetector()
        assert rd.update(None) is None

    def test_update_low_vol_regime(self):
        # 10% annualized target → low_vol (< 15%)
        rets = _daily_ret_for_annualized_vol(0.10, n=40)
        df = _make_price_df(rets)
        rd = RegimeDetector()
        snap = rd.update(df)
        assert snap is not None
        assert snap.regime == "low_vol"
        assert rd.get_current_regime() == "low_vol"

    def test_update_normal_regime(self):
        rets = _daily_ret_for_annualized_vol(0.20, n=40)
        df = _make_price_df(rets)
        rd = RegimeDetector()
        snap = rd.update(df)
        assert snap.regime == "normal"

    def test_update_high_vol_regime(self):
        rets = _daily_ret_for_annualized_vol(0.30, n=40)
        df = _make_price_df(rets)
        rd = RegimeDetector()
        snap = rd.update(df)
        assert snap.regime == "high_vol"

    def test_update_crisis_regime(self):
        rets = _daily_ret_for_annualized_vol(0.60, n=40)
        df = _make_price_df(rets)
        rd = RegimeDetector()
        snap = rd.update(df)
        assert snap.regime == "crisis"

    def test_history_bounded_by_max_history(self):
        rd = RegimeDetector(max_history=3)
        df = _make_price_df(_daily_ret_for_annualized_vol(0.20, n=40))
        for _ in range(10):
            rd.update(df)
        assert len(rd.get_regime_history()) == 3

    def test_reset_clears_history(self):
        rd = RegimeDetector()
        df = _make_price_df(_daily_ret_for_annualized_vol(0.20, n=40))
        rd.update(df)
        assert rd.get_regime_history()
        rd.reset()
        assert rd.get_regime_history() == []
        assert rd.get_current_regime() == "normal"

    def test_snapshot_has_timestamp(self):
        rd = RegimeDetector()
        df = _make_price_df(_daily_ret_for_annualized_vol(0.20, n=40))
        fixed = datetime(2025, 3, 15, 12, 0, 0)
        snap = rd.update(df, as_of=fixed)
        assert snap.as_of == fixed

    def test_snapshot_bars_used_bounded(self):
        rd = RegimeDetector(window=20)
        df = _make_price_df(_daily_ret_for_annualized_vol(0.20, n=50))
        snap = rd.update(df)
        assert snap.bars_used == 21  # window + 1

    def test_snapshot_str_is_readable(self):
        snap = RegimeSnapshot(
            regime="high_vol", annualized_vol=0.28,
            as_of=datetime.now(), bars_used=20,
        )
        s = str(snap)
        assert "high_vol" in s and "28" in s


# ── is_strategy_allowed ────────────────────────────────────────────────────


class TestStrategyCompatibility:
    def test_ensemble_always_allowed(self):
        for regime in ("low_vol", "normal", "high_vol", "crisis"):
            assert is_strategy_allowed("ensemble", regime) is True

    def test_mean_reversion_blocked_in_high_vol(self):
        assert is_strategy_allowed("mean_reversion", "high_vol") is False
        assert is_strategy_allowed("mean_reversion_aggressive", "high_vol") is False
        assert is_strategy_allowed("rsi_divergence", "high_vol") is False
        assert is_strategy_allowed("volume_profile", "high_vol") is False

    def test_mean_reversion_blocked_in_crisis(self):
        assert is_strategy_allowed("mean_reversion", "crisis") is False
        assert is_strategy_allowed("mean_reversion_aggressive", "crisis") is False

    def test_mean_reversion_allowed_in_normal_and_low(self):
        assert is_strategy_allowed("mean_reversion", "normal") is True
        assert is_strategy_allowed("mean_reversion", "low_vol") is True

    def test_momentum_blocked_in_low_vol(self):
        assert is_strategy_allowed("momentum", "low_vol") is False
        assert is_strategy_allowed("macd_crossover", "low_vol") is False
        assert is_strategy_allowed("triple_ema", "low_vol") is False
        assert is_strategy_allowed("scalper", "low_vol") is False

    def test_momentum_allowed_in_normal_and_higher(self):
        for regime in ("normal", "high_vol", "crisis"):
            assert is_strategy_allowed("momentum", regime) is True

    def test_unknown_strategy_allowed_by_default(self):
        # Policy should be permissive for strategies not in the compatibility lists.
        assert is_strategy_allowed("some_future_strategy", "high_vol") is True


# ── regime_one_hot ─────────────────────────────────────────────────────────


class TestRegimeOneHot:
    def test_low_vol(self):
        assert regime_one_hot("low_vol") == (1.0, 0.0, 0.0, 0.0)

    def test_normal(self):
        assert regime_one_hot("normal") == (0.0, 1.0, 0.0, 0.0)

    def test_high_vol(self):
        assert regime_one_hot("high_vol") == (0.0, 0.0, 1.0, 0.0)

    def test_crisis(self):
        assert regime_one_hot("crisis") == (0.0, 0.0, 0.0, 1.0)

    def test_sum_is_one(self):
        for r in ("low_vol", "normal", "high_vol", "crisis"):
            assert sum(regime_one_hot(r)) == 1.0


# ── Sprint 6F: data-driven override path ──────────────────────────────────

class TestHardSkipOverrides:
    def _reset(self):
        """Clear overrides and reload — always run between cases."""
        regime_detector._HARD_SKIP_OVERRIDES = set()

    def test_override_blocks_otherwise_allowed_pair(self, monkeypatch):
        self._reset()
        # mean_reversion in 'normal' is normally allowed.
        assert is_strategy_allowed("mean_reversion", "normal") is True

        monkeypatch.setattr(regime_detector, "_HARD_SKIP_OVERRIDES",
                            {("mean_reversion", "normal")}, raising=False)
        assert is_strategy_allowed("mean_reversion", "normal") is False

    def test_override_does_not_affect_ensemble(self, monkeypatch):
        """ensemble short-circuits before overrides — always allowed."""
        self._reset()
        monkeypatch.setattr(regime_detector, "_HARD_SKIP_OVERRIDES",
                            {("ensemble", "crisis")}, raising=False)
        assert is_strategy_allowed("ensemble", "crisis") is True

    def test_hardcoded_rules_still_apply_without_overrides(self, monkeypatch):
        self._reset()
        monkeypatch.setattr(regime_detector, "_HARD_SKIP_OVERRIDES",
                            set(), raising=False)
        # mean_reversion in crisis is still blocked via hardcoded policy.
        assert is_strategy_allowed("mean_reversion", "crisis") is False

    def test_reload_overrides_reads_fresh_file(self, tmp_path, monkeypatch):
        """reload_overrides() reflects changes to disk without a restart."""
        from analytics import strategy_regime_matrix as srm

        hs_path = tmp_path / "hs.json"
        monkeypatch.setattr(srm, "_HARD_SKIP_JSON", str(hs_path))

        # Empty state
        regime_detector._HARD_SKIP_OVERRIDES = set()
        assert reload_overrides() == set()

        # Add overrides
        srm.write_hard_skip([("momentum", "high_vol")], path=str(hs_path))
        out = reload_overrides()
        assert ("momentum", "high_vol") in out

        # Behaviour reflects the reload
        assert is_strategy_allowed("momentum", "high_vol") is False

        # Cleanup for other tests
        regime_detector._HARD_SKIP_OVERRIDES = set()

    def test_load_overrides_from_disk_swallows_errors(self, monkeypatch):
        """The inner try/except means a raising loader ⇒ empty set."""
        from analytics import strategy_regime_matrix as srm

        def _bad_load(path=None):
            raise RuntimeError("malformed JSON")

        monkeypatch.setattr(srm, "load_hard_skip", _bad_load)
        assert regime_detector._load_overrides_from_disk() == set()
