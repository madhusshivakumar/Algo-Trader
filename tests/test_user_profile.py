"""Tests for core/user_profile.py — capital-tier defaults."""

import pytest

from core.user_profile import (
    BEGINNER,
    HOBBYIST,
    LEARNER,
    PROFILES,
    UserProfile,
    detect_profile,
    resolve_profile,
)


class TestProfileDefaults:
    def test_beginner_is_most_conservative(self):
        # Beginner should never be laxer than Hobbyist on safety parameters.
        assert BEGINNER.base_position_pct_equity <= HOBBYIST.base_position_pct_equity
        assert BEGINNER.base_position_pct_crypto <= HOBBYIST.base_position_pct_crypto
        assert BEGINNER.kelly_fraction <= HOBBYIST.kelly_fraction
        assert BEGINNER.max_single_position_pct <= HOBBYIST.max_single_position_pct
        assert BEGINNER.max_daily_loss_pct <= HOBBYIST.max_daily_loss_pct
        assert BEGINNER.max_trades_per_day_equity <= HOBBYIST.max_trades_per_day_equity
        assert BEGINNER.max_var_contribution_pct <= HOBBYIST.max_var_contribution_pct

    def test_hobbyist_no_laxer_than_learner(self):
        assert HOBBYIST.base_position_pct_equity <= LEARNER.base_position_pct_equity
        assert HOBBYIST.kelly_fraction <= LEARNER.kelly_fraction
        assert HOBBYIST.max_daily_loss_pct <= LEARNER.max_daily_loss_pct

    def test_beginner_tiny_position_sizes(self):
        # Beginner must ship 1-2% per position — the whole point of this tier
        assert 0 < BEGINNER.base_position_pct_equity <= 0.02
        assert 0 < BEGINNER.base_position_pct_crypto <= 0.02

    def test_learner_matches_legacy_defaults(self):
        # Learner should mirror pre-profile defaults (KELLY 0.25, MAX_SINGLE_POSITION 0.25)
        assert LEARNER.kelly_fraction == 0.25
        assert LEARNER.max_single_position_pct == 0.25

    def test_all_profiles_registered(self):
        assert set(PROFILES.keys()) == {"beginner", "hobbyist", "learner"}
        assert PROFILES["beginner"] is BEGINNER


class TestMaxDailyLossUsd:
    def test_tiny_account_hits_dollar_cap(self):
        # $800 × 3% = $24, but cap is $50 — so halt at $24 (pct is tighter here)
        loss = BEGINNER.max_daily_loss_usd(equity=800)
        assert loss == pytest.approx(24.0)

    def test_beginner_cap_activates_at_break_even(self):
        # At $1,667 × 3% = $50 exactly — both equal
        loss = BEGINNER.max_daily_loss_usd(equity=1_667)
        assert loss == pytest.approx(50.0, abs=0.5)

    def test_beginner_cap_activates_above_break_even(self):
        # Beyond $1,667 the $50 cap starts dominating (tighter than pct)
        loss = BEGINNER.max_daily_loss_usd(equity=10_000)
        assert loss == pytest.approx(50.0)  # cap, not 300

    def test_hobbyist_cap_activates_for_big_accounts(self):
        # $8,000 × 4% = $320, cap $200 — cap dominates
        loss = HOBBYIST.max_daily_loss_usd(equity=8_000)
        assert loss == pytest.approx(200.0)

    def test_learner_effectively_uses_pct(self):
        # Learner cap is $1M — pct always dominates for real accounts
        loss = LEARNER.max_daily_loss_usd(equity=50_000)
        assert loss == pytest.approx(2_500.0)  # 5% of 50K

    def test_zero_equity_yields_zero(self):
        assert BEGINNER.max_daily_loss_usd(equity=0) == 0.0

    def test_negative_equity_clamped(self):
        # Edge case: should never produce negative halt thresholds
        assert BEGINNER.max_daily_loss_usd(equity=-100) >= 0


class TestDetectProfile:
    def test_beginner_boundary_low(self):
        assert detect_profile(500).name == "beginner"

    def test_beginner_boundary_high(self):
        assert detect_profile(1_999).name == "beginner"

    def test_hobbyist_boundary_low(self):
        assert detect_profile(2_000).name == "hobbyist"

    def test_hobbyist_middle(self):
        assert detect_profile(5_000).name == "hobbyist"

    def test_hobbyist_boundary_high(self):
        assert detect_profile(9_999).name == "hobbyist"

    def test_learner_boundary_low(self):
        assert detect_profile(10_000).name == "learner"

    def test_learner_large_account(self):
        assert detect_profile(500_000).name == "learner"


class TestResolveProfile:
    def test_override_beginner(self):
        # Even with large equity, explicit override wins
        assert resolve_profile(equity=100_000, override="beginner").name == "beginner"

    def test_override_learner(self):
        assert resolve_profile(equity=500, override="learner").name == "learner"

    def test_override_case_insensitive(self):
        assert resolve_profile(equity=5_000, override="BEGINNER").name == "beginner"
        assert resolve_profile(equity=5_000, override="  Hobbyist  ").name == "hobbyist"

    def test_unknown_override_falls_back(self, capsys):
        result = resolve_profile(equity=5_000, override="expert")
        # Should fall back to auto-detect (hobbyist for $5K)
        assert result.name == "hobbyist"
        captured = capsys.readouterr()
        assert "Unknown override" in captured.out

    def test_no_override_auto_detects(self):
        assert resolve_profile(equity=500).name == "beginner"
        assert resolve_profile(equity=5_000).name == "hobbyist"
        assert resolve_profile(equity=50_000).name == "learner"

    def test_empty_override_falls_through(self):
        assert resolve_profile(equity=5_000, override="").name == "hobbyist"
        assert resolve_profile(equity=5_000, override=None).name == "hobbyist"


class TestDashboardView:
    def test_progressive_disclosure(self):
        assert BEGINNER.dashboard_default_view == "simple"
        assert HOBBYIST.dashboard_default_view == "standard"
        assert LEARNER.dashboard_default_view == "advanced"


class TestFrozen:
    def test_profile_is_immutable(self):
        # Safety: profiles should not be mutable at runtime
        with pytest.raises((AttributeError, TypeError)):
            BEGINNER.kelly_fraction = 0.99  # type: ignore
