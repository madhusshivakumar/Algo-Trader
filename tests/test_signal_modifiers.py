"""Tests for core/signal_modifiers.py — signal blending logic."""

import json
import os
import pytest
from unittest.mock import patch


class TestClamp:
    def test_within_range(self):
        from core.signal_modifiers import _clamp
        assert _clamp(0.5) == 0.5

    def test_below_min(self):
        from core.signal_modifiers import _clamp
        assert _clamp(0.1) == 0.3

    def test_above_max(self):
        from core.signal_modifiers import _clamp
        assert _clamp(1.5) == 1.0

    def test_custom_bounds(self):
        from core.signal_modifiers import _clamp
        assert _clamp(0.2, lo=0.0, hi=0.5) == 0.2
        assert _clamp(-1.0, lo=0.0, hi=0.5) == 0.0
        assert _clamp(1.0, lo=0.0, hi=0.5) == 0.5


class TestLoadJson:
    def test_valid_file(self, tmp_path):
        from core.signal_modifiers import _load_json
        f = tmp_path / "test.json"
        f.write_text(json.dumps({"key": "value"}))
        assert _load_json(str(f)) == {"key": "value"}

    def test_missing_file(self, tmp_path):
        from core.signal_modifiers import _load_json
        assert _load_json(str(tmp_path / "nope.json")) == {}

    def test_corrupt_file(self, tmp_path):
        from core.signal_modifiers import _load_json
        f = tmp_path / "bad.json"
        f.write_text("{bad json")
        assert _load_json(str(f)) == {}


class TestApplySentiment:
    def _make_signal(self, action="buy", strength=0.5, reason="test"):
        return {"action": action, "strength": strength, "reason": reason}

    def test_no_data_file_returns_unchanged(self, tmp_path):
        from core.signal_modifiers import apply_sentiment
        signal = self._make_signal()
        with patch("core.signal_modifiers._SENTIMENT_FILE", str(tmp_path / "nope.json")):
            result = apply_sentiment(signal, "AAPL")
        assert result["strength"] == 0.5
        assert "sentiment_score" not in result

    def test_symbol_not_in_scores(self, tmp_path):
        from core.signal_modifiers import apply_sentiment
        f = tmp_path / "scores.json"
        f.write_text(json.dumps({"scores": {"MSFT": {"sentiment_score": 0.8}}}))
        signal = self._make_signal()
        with patch("core.signal_modifiers._SENTIMENT_FILE", str(f)):
            result = apply_sentiment(signal, "AAPL")
        assert "sentiment_score" not in result

    def test_buy_positive_sentiment_boosts(self, tmp_path):
        from core.signal_modifiers import apply_sentiment
        f = tmp_path / "scores.json"
        f.write_text(json.dumps({"scores": {"AAPL": {"sentiment_score": 0.8}}}))
        signal = self._make_signal(action="buy", strength=0.6)
        with patch("core.signal_modifiers._SENTIMENT_FILE", str(f)):
            result = apply_sentiment(signal, "AAPL", weight=0.15)
        assert result["strength"] > 0.6
        assert result["sentiment_score"] == 0.8

    def test_buy_negative_sentiment_dampens(self, tmp_path):
        from core.signal_modifiers import apply_sentiment
        f = tmp_path / "scores.json"
        f.write_text(json.dumps({"scores": {"AAPL": {"sentiment_score": -0.8}}}))
        signal = self._make_signal(action="buy", strength=0.7)
        with patch("core.signal_modifiers._SENTIMENT_FILE", str(f)):
            result = apply_sentiment(signal, "AAPL", weight=0.15)
        # Negative sentiment on buy: strength reduced + 0.7x dampen
        assert result["strength"] < 0.7

    def test_sell_negative_sentiment_boosts(self, tmp_path):
        from core.signal_modifiers import apply_sentiment
        f = tmp_path / "scores.json"
        f.write_text(json.dumps({"scores": {"AAPL": {"sentiment_score": -0.8}}}))
        signal = self._make_signal(action="sell", strength=0.6)
        with patch("core.signal_modifiers._SENTIMENT_FILE", str(f)):
            result = apply_sentiment(signal, "AAPL", weight=0.15)
        # Negative sentiment on sell: strength increased (subtracting negative)
        assert result["strength"] > 0.6

    def test_sell_positive_sentiment_dampens(self, tmp_path):
        from core.signal_modifiers import apply_sentiment
        f = tmp_path / "scores.json"
        f.write_text(json.dumps({"scores": {"AAPL": {"sentiment_score": 0.8}}}))
        signal = self._make_signal(action="sell", strength=0.7)
        with patch("core.signal_modifiers._SENTIMENT_FILE", str(f)):
            result = apply_sentiment(signal, "AAPL", weight=0.15)
        # Positive sentiment opposes sell: dampened by 0.7x
        assert result["strength"] < 0.7

    def test_hold_signal_unchanged(self, tmp_path):
        from core.signal_modifiers import apply_sentiment
        f = tmp_path / "scores.json"
        f.write_text(json.dumps({"scores": {"AAPL": {"sentiment_score": 0.9}}}))
        signal = self._make_signal(action="hold", strength=0)
        with patch("core.signal_modifiers._SENTIMENT_FILE", str(f)):
            result = apply_sentiment(signal, "AAPL")
        # Hold signals keep strength at 0 -> clamped to 0.3
        assert result["strength"] == 0.3

    def test_strength_clamped_high(self, tmp_path):
        from core.signal_modifiers import apply_sentiment
        f = tmp_path / "scores.json"
        f.write_text(json.dumps({"scores": {"AAPL": {"sentiment_score": 1.0}}}))
        signal = self._make_signal(action="buy", strength=0.95)
        with patch("core.signal_modifiers._SENTIMENT_FILE", str(f)):
            result = apply_sentiment(signal, "AAPL", weight=0.15)
        assert result["strength"] <= 1.0

    def test_strength_clamped_low(self, tmp_path):
        from core.signal_modifiers import apply_sentiment
        f = tmp_path / "scores.json"
        f.write_text(json.dumps({"scores": {"AAPL": {"sentiment_score": -1.0}}}))
        signal = self._make_signal(action="buy", strength=0.3)
        with patch("core.signal_modifiers._SENTIMENT_FILE", str(f)):
            result = apply_sentiment(signal, "AAPL", weight=0.15)
        assert result["strength"] >= 0.3

    def test_reason_appended(self, tmp_path):
        from core.signal_modifiers import apply_sentiment
        f = tmp_path / "scores.json"
        f.write_text(json.dumps({"scores": {"AAPL": {"sentiment_score": 0.5}}}))
        signal = self._make_signal()
        with patch("core.signal_modifiers._SENTIMENT_FILE", str(f)):
            result = apply_sentiment(signal, "AAPL")
        assert "sentiment=+0.50" in result["reason"]

    def test_invalid_score_type_ignored(self, tmp_path):
        from core.signal_modifiers import apply_sentiment
        f = tmp_path / "scores.json"
        f.write_text(json.dumps({"scores": {"AAPL": {"sentiment_score": "not_a_number"}}}))
        signal = self._make_signal()
        with patch("core.signal_modifiers._SENTIMENT_FILE", str(f)):
            result = apply_sentiment(signal, "AAPL")
        assert "sentiment_score" not in result

    def test_zero_weight_no_change(self, tmp_path):
        from core.signal_modifiers import apply_sentiment
        f = tmp_path / "scores.json"
        f.write_text(json.dumps({"scores": {"AAPL": {"sentiment_score": 1.0}}}))
        signal = self._make_signal(action="buy", strength=0.6)
        with patch("core.signal_modifiers._SENTIMENT_FILE", str(f)):
            result = apply_sentiment(signal, "AAPL", weight=0.0)
        assert result["strength"] == 0.6


class TestApplyLLMConviction:
    def _make_signal(self, action="buy", strength=0.5, reason="test"):
        return {"action": action, "strength": strength, "reason": reason}

    def test_no_data_file_returns_unchanged(self, tmp_path):
        from core.signal_modifiers import apply_llm_conviction
        signal = self._make_signal()
        with patch("core.signal_modifiers._LLM_FILE", str(tmp_path / "nope.json")):
            result = apply_llm_conviction(signal, "AAPL")
        assert result["strength"] == 0.5
        assert "llm_conviction" not in result

    def test_symbol_not_in_convictions(self, tmp_path):
        from core.signal_modifiers import apply_llm_conviction
        f = tmp_path / "convictions.json"
        f.write_text(json.dumps({"convictions": {"MSFT": {"score": 0.7}}}))
        signal = self._make_signal()
        with patch("core.signal_modifiers._LLM_FILE", str(f)):
            result = apply_llm_conviction(signal, "AAPL")
        assert "llm_conviction" not in result

    def test_buy_bullish_conviction_boosts(self, tmp_path):
        from core.signal_modifiers import apply_llm_conviction
        f = tmp_path / "convictions.json"
        f.write_text(json.dumps({"convictions": {"AAPL": {"score": 0.8}}}))
        signal = self._make_signal(action="buy", strength=0.6)
        with patch("core.signal_modifiers._LLM_FILE", str(f)):
            result = apply_llm_conviction(signal, "AAPL", weight=0.2)
        assert result["strength"] > 0.6
        assert result["llm_conviction"] == 0.8

    def test_buy_bearish_conviction_dampens(self, tmp_path):
        from core.signal_modifiers import apply_llm_conviction
        f = tmp_path / "convictions.json"
        f.write_text(json.dumps({"convictions": {"AAPL": {"score": -0.8}}}))
        signal = self._make_signal(action="buy", strength=0.7)
        with patch("core.signal_modifiers._LLM_FILE", str(f)):
            result = apply_llm_conviction(signal, "AAPL", weight=0.2)
        assert result["strength"] < 0.7

    def test_sell_bearish_conviction_boosts(self, tmp_path):
        from core.signal_modifiers import apply_llm_conviction
        f = tmp_path / "convictions.json"
        f.write_text(json.dumps({"convictions": {"AAPL": {"score": -0.8}}}))
        signal = self._make_signal(action="sell", strength=0.6)
        with patch("core.signal_modifiers._LLM_FILE", str(f)):
            result = apply_llm_conviction(signal, "AAPL", weight=0.2)
        assert result["strength"] > 0.6

    def test_sell_bullish_conviction_dampens(self, tmp_path):
        from core.signal_modifiers import apply_llm_conviction
        f = tmp_path / "convictions.json"
        f.write_text(json.dumps({"convictions": {"AAPL": {"score": 0.8}}}))
        signal = self._make_signal(action="sell", strength=0.7)
        with patch("core.signal_modifiers._LLM_FILE", str(f)):
            result = apply_llm_conviction(signal, "AAPL", weight=0.2)
        assert result["strength"] < 0.7

    def test_reason_appended(self, tmp_path):
        from core.signal_modifiers import apply_llm_conviction
        f = tmp_path / "convictions.json"
        f.write_text(json.dumps({"convictions": {"AAPL": {"score": -0.3}}}))
        signal = self._make_signal()
        with patch("core.signal_modifiers._LLM_FILE", str(f)):
            result = apply_llm_conviction(signal, "AAPL")
        assert "llm=-0.30" in result["reason"]

    def test_invalid_score_type_ignored(self, tmp_path):
        from core.signal_modifiers import apply_llm_conviction
        f = tmp_path / "convictions.json"
        f.write_text(json.dumps({"convictions": {"AAPL": {"score": None}}}))
        signal = self._make_signal()
        with patch("core.signal_modifiers._LLM_FILE", str(f)):
            result = apply_llm_conviction(signal, "AAPL")
        assert "llm_conviction" not in result

    def test_strength_stays_in_bounds(self, tmp_path):
        from core.signal_modifiers import apply_llm_conviction
        f = tmp_path / "convictions.json"
        f.write_text(json.dumps({"convictions": {"AAPL": {"score": 1.0}}}))
        signal = self._make_signal(action="buy", strength=0.95)
        with patch("core.signal_modifiers._LLM_FILE", str(f)):
            result = apply_llm_conviction(signal, "AAPL", weight=0.2)
        assert 0.3 <= result["strength"] <= 1.0


class TestModifierChaining:
    """Test that sentiment and LLM modifiers can be applied sequentially."""

    def test_both_modifiers_stack(self, tmp_path):
        from core.signal_modifiers import apply_sentiment, apply_llm_conviction

        sf = tmp_path / "scores.json"
        sf.write_text(json.dumps({"scores": {"AAPL": {"sentiment_score": 0.5}}}))
        lf = tmp_path / "convictions.json"
        lf.write_text(json.dumps({"convictions": {"AAPL": {"score": 0.6}}}))

        signal = {"action": "buy", "strength": 0.5, "reason": "test"}

        with patch("core.signal_modifiers._SENTIMENT_FILE", str(sf)):
            signal = apply_sentiment(signal, "AAPL", weight=0.15)
        with patch("core.signal_modifiers._LLM_FILE", str(lf)):
            signal = apply_llm_conviction(signal, "AAPL", weight=0.2)

        assert "sentiment_score" in signal
        assert "llm_conviction" in signal
        assert "sentiment=" in signal["reason"]
        assert "llm=" in signal["reason"]
        # Both bullish on a buy → strength should be higher than original
        assert signal["strength"] > 0.5

    def test_opposing_modifiers_cancel(self, tmp_path):
        from core.signal_modifiers import apply_sentiment, apply_llm_conviction

        sf = tmp_path / "scores.json"
        sf.write_text(json.dumps({"scores": {"AAPL": {"sentiment_score": 0.8}}}))
        lf = tmp_path / "convictions.json"
        lf.write_text(json.dumps({"convictions": {"AAPL": {"score": -0.8}}}))

        signal = {"action": "buy", "strength": 0.6, "reason": "test"}

        with patch("core.signal_modifiers._SENTIMENT_FILE", str(sf)):
            signal = apply_sentiment(signal, "AAPL", weight=0.15)
        after_sentiment = signal["strength"]

        with patch("core.signal_modifiers._LLM_FILE", str(lf)):
            signal = apply_llm_conviction(signal, "AAPL", weight=0.2)

        # LLM bearish should pull strength back down from sentiment boost
        assert signal["strength"] < after_sentiment
