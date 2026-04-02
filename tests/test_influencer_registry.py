"""Tests for the influencer registry — matching, context building."""

import pytest
from core.influencer_registry import (
    INFLUENCER_REGISTRY,
    match_influencers,
    get_influencer_context_for_sector,
    get_influencer_context_for_symbol,
    get_influencer_summary,
    InfluencerMatch,
)


class TestInfluencerRegistry:
    def test_registry_has_key_figures(self):
        assert "Jerome Powell" in INFLUENCER_REGISTRY
        assert "Elon Musk" in INFLUENCER_REGISTRY
        assert "Warren Buffett" in INFLUENCER_REGISTRY
        assert "Jensen Huang" in INFLUENCER_REGISTRY
        assert "OPEC" in INFLUENCER_REGISTRY

    def test_all_entries_have_required_fields(self):
        for name, info in INFLUENCER_REGISTRY.items():
            assert "role" in info, f"{name} missing role"
            assert "sectors" in info, f"{name} missing sectors"
            assert "keywords" in info, f"{name} missing keywords"
            assert "historical_patterns" in info, f"{name} missing patterns"
            assert len(info["keywords"]) >= 1, f"{name} has no keywords"

    def test_keywords_are_lowercase(self):
        for name, info in INFLUENCER_REGISTRY.items():
            for kw in info["keywords"]:
                assert kw == kw.lower(), f"{name} keyword not lowercase: {kw}"


class TestMatchInfluencers:
    def test_matches_powell(self):
        headlines = ["Jerome Powell signals rate pause at FOMC meeting"]
        matches = match_influencers(headlines)
        assert len(matches) == 1
        assert matches[0].name == "Jerome Powell"
        assert "Finance" in matches[0].sectors

    def test_matches_musk(self):
        headlines = ["Elon Musk announces new Tesla factory in Texas"]
        matches = match_influencers(headlines)
        assert len(matches) == 1
        assert matches[0].name == "Elon Musk"
        assert "TSLA" in matches[0].symbols

    def test_case_insensitive(self):
        headlines = ["JEROME POWELL SPEAKS ON INFLATION"]
        matches = match_influencers(headlines)
        assert len(matches) == 1
        assert matches[0].name == "Jerome Powell"

    def test_no_match(self):
        headlines = ["S&P 500 closes higher on strong earnings"]
        matches = match_influencers(headlines)
        assert len(matches) == 0

    def test_multiple_influencers_in_different_headlines(self):
        headlines = [
            "Elon Musk tweets about Bitcoin",
            "Warren Buffett increases Apple stake",
        ]
        matches = match_influencers(headlines)
        names = {m.name for m in matches}
        assert "Elon Musk" in names
        assert "Warren Buffett" in names

    def test_same_influencer_different_headlines_both_matched(self):
        headlines = [
            "Elon Musk announces Cybertruck update",
            "Elon Musk responds to SEC inquiry",
        ]
        matches = match_influencers(headlines)
        assert len(matches) == 2

    def test_deduplication_same_headline(self):
        headlines = ["Jerome Powell signals rate pause"]  # only one
        matches = match_influencers(headlines)
        assert len(matches) == 1  # not duplicated by multiple keywords

    def test_opec_match(self):
        headlines = ["OPEC+ announces production cut of 1M barrels/day"]
        matches = match_influencers(headlines)
        assert len(matches) == 1
        assert matches[0].name == "OPEC"
        assert "Energy" in matches[0].sectors

    def test_empty_headlines(self):
        assert match_influencers([]) == []

    def test_match_returns_patterns(self):
        headlines = ["Jensen Huang unveils next-gen GPU at keynote"]
        matches = match_influencers(headlines)
        assert len(matches) == 1
        assert len(matches[0].patterns) > 0


class TestGetInfluencerContextForSector:
    def test_returns_context_for_matching_sector(self):
        headlines = ["Jerome Powell signals hawkish stance on rates"]
        ctx = get_influencer_context_for_sector("Finance", headlines)
        assert "Jerome Powell" in ctx
        assert "Influencer Activity" in ctx

    def test_returns_empty_for_unrelated_sector(self):
        headlines = ["Jensen Huang unveils new GPU"]
        ctx = get_influencer_context_for_sector("Energy", headlines)
        assert ctx == ""

    def test_includes_historical_pattern(self):
        headlines = ["Jerome Powell signals hawkish stance"]
        ctx = get_influencer_context_for_sector("Finance", headlines)
        assert "Historical" in ctx or "hawkish" in ctx.lower()

    def test_caps_at_five_entries(self):
        headlines = [f"Jerome Powell statement {i}" for i in range(10)]
        ctx = get_influencer_context_for_sector("Finance", headlines)
        # Should have at most 5 influencer entries (deduped by name = 1 here)
        assert ctx.count("Jerome Powell") <= 5

    def test_no_headlines(self):
        ctx = get_influencer_context_for_sector("Tech", [])
        assert ctx == ""


class TestGetInfluencerContextForSymbol:
    def test_tsla_with_musk_headline(self):
        headlines = ["Elon Musk announces record Tesla deliveries"]
        ctx = get_influencer_context_for_symbol("TSLA", headlines)
        assert "Elon Musk" in ctx
        assert "TSLA" in ctx or "Influencer" in ctx

    def test_no_match_for_unrelated_symbol(self):
        headlines = ["Elon Musk announces record Tesla deliveries"]
        ctx = get_influencer_context_for_symbol("XOM", headlines)
        assert ctx == ""

    def test_nvda_with_huang_headline(self):
        headlines = ["Jensen Huang says AI demand is insatiable"]
        ctx = get_influencer_context_for_symbol("NVDA", headlines)
        assert "Jensen Huang" in ctx

    def test_empty_headlines(self):
        ctx = get_influencer_context_for_symbol("TSLA", [])
        assert ctx == ""


class TestGetInfluencerSummary:
    def test_returns_summary(self):
        headlines = [
            "Jerome Powell hints at rate cut",
            "Elon Musk announces new factory",
        ]
        summary = get_influencer_summary(headlines)
        assert "Key Figure Activity" in summary
        assert "Jerome Powell" in summary
        assert "Elon Musk" in summary

    def test_no_matches(self):
        headlines = ["Market closes flat"]
        summary = get_influencer_summary(headlines)
        assert summary == ""

    def test_deduplicates_by_name(self):
        headlines = [
            "Elon Musk tweets about Bitcoin",
            "Elon Musk responds to SEC",
        ]
        summary = get_influencer_summary(headlines)
        # Only one "- Elon Musk" entry line (name may appear in headline text too)
        assert summary.count("- Elon Musk") == 1
