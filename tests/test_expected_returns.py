"""Tests for core/expected_returns.py — Sprint 6I."""

from __future__ import annotations

import pytest

from core.expected_returns import (
    INDUSTRY_RANGE_RETURN,
    INDUSTRY_RANGE_SHARPE,
    BacktestFrame,
    frame_backtest,
    frame_return,
    frame_sharpe,
    what_this_bot_cannot_do,
)


class TestFrameSharpe:
    def test_none_returns_no_sharpe_message(self):
        out = frame_sharpe(None, context="test")
        assert "No Sharpe" in out
        assert "test" in out

    def test_very_high_sharpe_flagged_as_unlikely(self):
        out = frame_sharpe(2.65, context="backtest")
        assert "2.65" in out
        assert "past" in out.lower()
        assert "lower" in out.lower()

    def test_realistic_sharpe_gets_in_range_message(self):
        out = frame_sharpe(1.0, context="validation")
        assert "1.00" in out
        assert "realistic retail range" in out.lower()

    def test_sub_realistic_sharpe_warns(self):
        out = frame_sharpe(0.2, context="oos")
        assert "0.20" in out
        assert "below" in out.lower()

    def test_negative_sharpe_calls_it_out(self):
        out = frame_sharpe(-0.5, context="test")
        assert "NEGATIVE" in out
        assert "losing" in out.lower()

    def test_zero_sharpe_treated_as_sub_realistic(self):
        out = frame_sharpe(0.0)
        assert "0.00" in out
        assert "below" in out.lower()

    def test_context_capitalized(self):
        """First letter of context should be capitalized in output."""
        out = frame_sharpe(1.0, context="backtest")
        assert out.startswith("Backtest")


class TestFrameReturn:
    def test_none_returns_no_return_message(self):
        out = frame_return(None, context="paper")
        assert "No return" in out
        assert "paper" in out

    def test_very_high_return_flagged(self):
        out = frame_return(35.0, context="backtest")
        assert "+35.0%" in out
        assert "past" in out.lower()

    def test_realistic_return_shows_anchor(self):
        out = frame_return(5.0, context="backtest")
        assert "+5.0%" in out
        assert "retail" in out.lower()

    def test_negative_return_flagged(self):
        out = frame_return(-15.0, context="test")
        assert "-15.0%" in out
        assert "NEGATIVE" in out

    def test_context_capitalized(self):
        out = frame_return(5.0, context="live paper")
        assert out.startswith("Live paper")


class TestFrameBacktest:
    def test_returns_all_four_lines(self):
        fb = frame_backtest(sharpe=1.0, total_return_pct=5.0)
        assert isinstance(fb, BacktestFrame)
        assert fb.sharpe_line
        assert fb.return_line
        assert fb.pct_band_line
        assert fb.disclaimer_line

    def test_mc_band_included_when_percentiles_passed(self):
        fb = frame_backtest(
            sharpe=1.0, total_return_pct=5.0,
            mc_p5=-0.1, mc_p50=0.05, mc_p95=0.20,
        )
        assert "Monte-Carlo" in fb.pct_band_line
        assert "-10.0%" in fb.pct_band_line
        assert "+5.0%" in fb.pct_band_line
        assert "+20.0%" in fb.pct_band_line

    def test_mc_band_absent_when_percentiles_missing(self):
        fb = frame_backtest(sharpe=1.0, total_return_pct=5.0)
        assert "not available" in fb.pct_band_line.lower()

    def test_disclaimer_always_present(self):
        fb = frame_backtest(sharpe=1.0, total_return_pct=5.0)
        assert "IMPORTANT" in fb.disclaimer_line
        assert "not" in fb.disclaimer_line.lower()

    def test_as_text_joins_with_newlines(self):
        fb = frame_backtest(sharpe=1.0, total_return_pct=5.0)
        text = fb.as_text()
        assert text.count("\n") == 3
        assert fb.sharpe_line in text
        assert fb.disclaimer_line in text

    def test_custom_context_propagates(self):
        fb = frame_backtest(sharpe=1.2, total_return_pct=8.0,
                            context="6-month paper")
        assert "6-month paper" in fb.sharpe_line.lower()


class TestIndustryConstants:
    def test_sharpe_range_has_numbers(self):
        assert "0.5" in INDUSTRY_RANGE_SHARPE
        assert "1.5" in INDUSTRY_RANGE_SHARPE
        assert "retail" in INDUSTRY_RANGE_SHARPE.lower()

    def test_return_range_has_numbers(self):
        assert "-20" in INDUSTRY_RANGE_RETURN
        assert "15" in INDUSTRY_RANGE_RETURN


class TestCannotDoList:
    def test_returns_non_empty_sequence(self):
        items = what_this_bot_cannot_do()
        assert len(items) > 3

    def test_each_item_is_sentence(self):
        for item in what_this_bot_cannot_do():
            assert isinstance(item, str)
            assert len(item) > 10
            assert item.endswith(".")

    def test_mentions_crash_prediction_disclaimer(self):
        items = what_this_bot_cannot_do()
        joined = " ".join(items).lower()
        assert "crash" in joined or "black-swan" in joined

    def test_mentions_risk_tolerance(self):
        items = what_this_bot_cannot_do()
        joined = " ".join(items).lower()
        assert "risk" in joined
