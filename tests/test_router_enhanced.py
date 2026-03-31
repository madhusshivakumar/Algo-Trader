"""Integration tests for the enhanced signal chain with all v2 modifiers."""

import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _make_df(n=60):
    """Create sample OHLCV DataFrame."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "close": close,
        "open": close - np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "volume": np.random.randint(1000, 5000, n),
    })


class TestFullSignalChain:
    """Test the complete signal chain: strategy → sentiment → LLM → output."""

    @patch("strategies.router.Config")
    def test_no_modifiers_enabled(self, mock_config):
        """With all modifiers disabled, signal passes through unchanged."""
        from strategies.router import compute_signals, STRATEGY_REGISTRY

        mock_config.SENTIMENT_ENABLED = False
        mock_config.LLM_ANALYST_ENABLED = False
        mock_config.RL_STRATEGY_ENABLED = False

        df = _make_df()
        # Mock the strategy function
        with patch.dict(STRATEGY_REGISTRY, {"mean_reversion_aggressive": lambda df: {
            "action": "buy", "strength": 0.6, "reason": "test"
        }}):
            with patch("strategies.router.STRATEGY_MAP", {"AAPL": "mean_reversion_aggressive"}):
                signal = compute_signals("AAPL", df)

        assert signal["action"] == "buy"
        assert signal["strength"] == 0.6
        assert "sentiment_score" not in signal
        assert "llm_conviction" not in signal

    @patch("strategies.router.Config")
    def test_sentiment_only(self, mock_config, tmp_path):
        """With only sentiment enabled, only sentiment modifier is applied."""
        from strategies.router import compute_signals, STRATEGY_REGISTRY

        mock_config.SENTIMENT_ENABLED = True
        mock_config.SENTIMENT_WEIGHT = 0.15
        mock_config.LLM_ANALYST_ENABLED = False
        mock_config.RL_STRATEGY_ENABLED = False

        # Write sentiment data
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps({
            "scores": {"AAPL": {"sentiment_score": 0.8}}
        }))

        df = _make_df()
        with patch.dict(STRATEGY_REGISTRY, {"mean_reversion_aggressive": lambda df: {
            "action": "buy", "strength": 0.6, "reason": "test"
        }}):
            with patch("strategies.router.STRATEGY_MAP", {"AAPL": "mean_reversion_aggressive"}):
                with patch("core.signal_modifiers._SENTIMENT_FILE", str(scores_file)):
                    signal = compute_signals("AAPL", df)

        assert signal["action"] == "buy"
        assert signal["strength"] > 0.6  # Boosted by positive sentiment
        assert signal["sentiment_score"] == 0.8
        assert "llm_conviction" not in signal

    @patch("strategies.router.Config")
    def test_llm_only(self, mock_config, tmp_path):
        """With only LLM enabled, only LLM modifier is applied."""
        from strategies.router import compute_signals, STRATEGY_REGISTRY

        mock_config.SENTIMENT_ENABLED = False
        mock_config.LLM_ANALYST_ENABLED = True
        mock_config.LLM_CONVICTION_WEIGHT = 0.2
        mock_config.RL_STRATEGY_ENABLED = False

        conv_file = tmp_path / "convictions.json"
        conv_file.write_text(json.dumps({
            "convictions": {"AAPL": {"score": 0.7}}
        }))

        df = _make_df()
        with patch.dict(STRATEGY_REGISTRY, {"mean_reversion_aggressive": lambda df: {
            "action": "buy", "strength": 0.6, "reason": "test"
        }}):
            with patch("strategies.router.STRATEGY_MAP", {"AAPL": "mean_reversion_aggressive"}):
                with patch("core.signal_modifiers._LLM_FILE", str(conv_file)):
                    signal = compute_signals("AAPL", df)

        assert signal["strength"] > 0.6
        assert signal["llm_conviction"] == 0.7
        assert "sentiment_score" not in signal

    @patch("strategies.router.Config")
    def test_all_modifiers(self, mock_config, tmp_path):
        """With all modifiers enabled, signal gets both adjustments."""
        from strategies.router import compute_signals, STRATEGY_REGISTRY

        mock_config.SENTIMENT_ENABLED = True
        mock_config.SENTIMENT_WEIGHT = 0.15
        mock_config.LLM_ANALYST_ENABLED = True
        mock_config.LLM_CONVICTION_WEIGHT = 0.2
        mock_config.RL_STRATEGY_ENABLED = False

        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps({
            "scores": {"AAPL": {"sentiment_score": 0.5}}
        }))
        conv_file = tmp_path / "convictions.json"
        conv_file.write_text(json.dumps({
            "convictions": {"AAPL": {"score": 0.6}}
        }))

        df = _make_df()
        with patch.dict(STRATEGY_REGISTRY, {"mean_reversion_aggressive": lambda df: {
            "action": "buy", "strength": 0.5, "reason": "test"
        }}):
            with patch("strategies.router.STRATEGY_MAP", {"AAPL": "mean_reversion_aggressive"}):
                with patch("core.signal_modifiers._SENTIMENT_FILE", str(scores_file)):
                    with patch("core.signal_modifiers._LLM_FILE", str(conv_file)):
                        signal = compute_signals("AAPL", df)

        assert "sentiment_score" in signal
        assert "llm_conviction" in signal
        assert "sentiment=" in signal["reason"]
        assert "llm=" in signal["reason"]
        assert signal["strength"] > 0.5  # Both bullish

    @patch("strategies.router.Config")
    def test_rl_strategy_selection(self, mock_config):
        """With RL enabled, strategy should come from RL selector."""
        from strategies.router import compute_signals, STRATEGY_REGISTRY

        mock_config.SENTIMENT_ENABLED = False
        mock_config.LLM_ANALYST_ENABLED = False
        mock_config.RL_STRATEGY_ENABLED = True

        df = _make_df()
        mock_rl = MagicMock(return_value="momentum")

        with patch.dict(STRATEGY_REGISTRY, {
            "mean_reversion_aggressive": lambda df: {"action": "hold", "strength": 0, "reason": "mr"},
            "momentum": lambda df: {"action": "buy", "strength": 0.7, "reason": "breakout"},
        }):
            with patch("strategies.router.STRATEGY_MAP", {"AAPL": "mean_reversion_aggressive"}):
                with patch("strategies.router.rl_select", mock_rl, create=True):
                    with patch("core.rl_strategy_selector.select_strategy", mock_rl):
                        signal = compute_signals("AAPL", df)

        assert signal["action"] == "buy"
        assert signal["rl_selected"] == "momentum"

    @patch("strategies.router.Config")
    def test_rl_fallback_on_invalid(self, mock_config):
        """RL returns invalid strategy key → falls back to STRATEGY_MAP."""
        from strategies.router import compute_signals, STRATEGY_REGISTRY

        mock_config.SENTIMENT_ENABLED = False
        mock_config.LLM_ANALYST_ENABLED = False
        mock_config.RL_STRATEGY_ENABLED = True

        df = _make_df()
        mock_rl = MagicMock(return_value="nonexistent_strategy")

        with patch.dict(STRATEGY_REGISTRY, {
            "mean_reversion_aggressive": lambda df: {"action": "hold", "strength": 0, "reason": "mr"},
        }):
            with patch("strategies.router.STRATEGY_MAP", {"AAPL": "mean_reversion_aggressive"}):
                with patch("core.rl_strategy_selector.select_strategy", mock_rl):
                    signal = compute_signals("AAPL", df)

        assert "rl_selected" not in signal or signal.get("rl_selected") == ""

    @patch("strategies.router.Config")
    def test_signal_strength_always_bounded(self, mock_config, tmp_path):
        """No matter what modifiers do, strength stays in [0.3, 1.0]."""
        from strategies.router import compute_signals, STRATEGY_REGISTRY

        mock_config.SENTIMENT_ENABLED = True
        mock_config.SENTIMENT_WEIGHT = 0.15
        mock_config.LLM_ANALYST_ENABLED = True
        mock_config.LLM_CONVICTION_WEIGHT = 0.2
        mock_config.RL_STRATEGY_ENABLED = False

        # Both extremely positive
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps({
            "scores": {"AAPL": {"sentiment_score": 1.0}}
        }))
        conv_file = tmp_path / "convictions.json"
        conv_file.write_text(json.dumps({
            "convictions": {"AAPL": {"score": 1.0}}
        }))

        df = _make_df()
        with patch.dict(STRATEGY_REGISTRY, {"mean_reversion_aggressive": lambda df: {
            "action": "buy", "strength": 0.95, "reason": "test"
        }}):
            with patch("strategies.router.STRATEGY_MAP", {"AAPL": "mean_reversion_aggressive"}):
                with patch("core.signal_modifiers._SENTIMENT_FILE", str(scores_file)):
                    with patch("core.signal_modifiers._LLM_FILE", str(conv_file)):
                        signal = compute_signals("AAPL", df)

        assert 0.3 <= signal["strength"] <= 1.0


class TestSentimentCorrelation:
    """Test the new trade analyzer v2 analytics."""

    def test_analyze_sentiment_correlation(self):
        from agents.trade_analyzer import analyze_sentiment_correlation

        trades = [
            {"side": "sell", "pnl": 10, "sentiment_score": 0.5},   # aligned (profit + positive)
            {"side": "sell", "pnl": -5, "sentiment_score": -0.3},  # aligned (loss + negative)
            {"side": "sell", "pnl": 10, "sentiment_score": -0.8},  # opposed (profit + negative)
            {"side": "buy", "pnl": 0, "sentiment_score": 0.5},     # buy, not sell
        ]

        result = analyze_sentiment_correlation(trades)
        assert result is not None
        assert result["trades_with_sentiment"] == 3
        assert result["aligned"] == 2
        assert result["opposed"] == 1

    def test_no_sentiment_data(self):
        from agents.trade_analyzer import analyze_sentiment_correlation
        trades = [
            {"side": "sell", "pnl": 10},
            {"side": "sell", "pnl": -5},
        ]
        result = analyze_sentiment_correlation(trades)
        assert result is None


class TestRLPerformance:
    def test_analyze_rl_performance(self):
        from agents.trade_analyzer import analyze_rl_performance

        trades = [
            {"side": "sell", "pnl": 10, "rl_selected": "momentum"},
            {"side": "sell", "pnl": -5, "rl_selected": ""},
            {"side": "sell", "pnl": 20, "rl_selected": "scalper"},
            {"side": "buy", "pnl": 0, "rl_selected": ""},
        ]

        result = analyze_rl_performance(trades)
        assert result is not None
        assert result["rl_selected"]["trades"] == 2
        assert result["optimizer_selected"]["trades"] == 2

    def test_no_trades(self):
        from agents.trade_analyzer import analyze_rl_performance
        result = analyze_rl_performance([])
        assert result is None
