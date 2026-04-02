"""Tests for agents/llm_analyst.py — LLM multi-agent analyst."""

import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


class TestAnalyzeNews:
    @patch("agents.llm_analyst.call_llm_json")
    def test_analyzes_headlines(self, mock_call):
        from agents.llm_analyst import analyze_news
        mock_call.return_value = {
            "parsed": {
                "summary": "Positive earnings",
                "sentiment": "bullish",
                "key_events": ["earnings beat"],
            }
        }

        result = analyze_news("AAPL", ["AAPL beats earnings", "Apple revenue up"])
        assert result["sentiment"] == "bullish"
        mock_call.assert_called_once()

    def test_empty_headlines(self):
        from agents.llm_analyst import analyze_news
        result = analyze_news("AAPL", [])
        assert result["sentiment"] == "neutral"

    @patch("agents.llm_analyst.call_llm_json")
    def test_handles_api_error(self, mock_call):
        from agents.llm_analyst import analyze_news
        mock_call.side_effect = Exception("API error")

        result = analyze_news("AAPL", ["headline"])
        assert result["sentiment"] == "neutral"


class TestAnalyzeTechnicals:
    @patch("agents.llm_analyst.call_llm_json")
    def test_analyzes_indicators(self, mock_call):
        from agents.llm_analyst import analyze_technicals
        mock_call.return_value = {
            "parsed": {
                "summary": "Uptrend",
                "trend": "up",
                "key_signals": ["RSI oversold bounce"],
            }
        }

        result = analyze_technicals("AAPL", {"RSI": 35, "MACD": 0.5})
        assert result["trend"] == "up"

    def test_empty_indicators(self):
        from agents.llm_analyst import analyze_technicals
        result = analyze_technicals("AAPL", {})
        assert result["trend"] == "sideways"

    @patch("agents.llm_analyst.call_llm_json")
    def test_handles_api_error(self, mock_call):
        from agents.llm_analyst import analyze_technicals
        mock_call.side_effect = Exception("API error")

        result = analyze_technicals("AAPL", {"RSI": 50})
        assert result["trend"] == "sideways"


class TestSynthesize:
    @patch("agents.llm_analyst.call_llm_json")
    def test_produces_conviction(self, mock_call):
        from agents.llm_analyst import synthesize
        mock_call.return_value = {
            "parsed": {
                "conviction_score": 0.72,
                "bias": "bullish",
                "confidence": 0.8,
                "key_factors": ["strong earnings"],
                "risk_flags": [],
                "reasoning": "test",
            }
        }

        result = synthesize("AAPL", {"sentiment": "bullish"}, {"trend": "up"})
        assert result["conviction_score"] == 0.72
        assert result["bias"] == "bullish"

    @patch("agents.llm_analyst.call_llm_json")
    def test_clamps_score(self, mock_call):
        from agents.llm_analyst import synthesize
        mock_call.return_value = {
            "parsed": {
                "conviction_score": 2.5,
                "bias": "bullish",
            }
        }

        result = synthesize("AAPL", {}, {})
        assert result["conviction_score"] == 1.0

    @patch("agents.llm_analyst.call_llm_json")
    def test_clamps_negative_score(self, mock_call):
        from agents.llm_analyst import synthesize
        mock_call.return_value = {
            "parsed": {
                "conviction_score": -3.0,
                "bias": "bearish",
            }
        }

        result = synthesize("AAPL", {}, {})
        assert result["conviction_score"] == -1.0

    @patch("agents.llm_analyst.call_llm_json")
    def test_handles_api_error(self, mock_call):
        from agents.llm_analyst import synthesize
        mock_call.side_effect = Exception("API error")

        result = synthesize("AAPL", {}, {})
        assert result["conviction_score"] == 0
        assert result["bias"] == "neutral"
        assert "analysis_failed" in result["risk_flags"]


class TestExtractIndicators:
    def test_extracts_rsi(self):
        from agents.llm_analyst import extract_indicators
        df = pd.DataFrame({"close": [100], "rsi": [35.5]})
        result = extract_indicators(df)
        assert result["RSI"] == 35.5

    def test_extracts_macd(self):
        from agents.llm_analyst import extract_indicators
        df = pd.DataFrame({"close": [100], "macd": [0.0025], "macd_signal": [0.001]})
        result = extract_indicators(df)
        assert result["MACD"] == 0.0025
        assert result["MACD_Signal"] == 0.001

    def test_extracts_momentum(self):
        from agents.llm_analyst import extract_indicators
        prices = list(range(100, 125))  # 25 data points, ascending
        df = pd.DataFrame({"close": prices})
        result = extract_indicators(df)
        assert "5d_change" in result
        assert "20d_change" in result
        assert result["5d_change"] > 0

    def test_extracts_volume(self):
        from agents.llm_analyst import extract_indicators
        df = pd.DataFrame({
            "close": list(range(100, 125)),
            "volume": [1000] * 24 + [2000],  # last day 2x volume
        })
        result = extract_indicators(df)
        assert result["Volume_vs_avg"] > 1.0

    def test_empty_dataframe(self):
        from agents.llm_analyst import extract_indicators
        result = extract_indicators(pd.DataFrame())
        assert result == {}

    def test_none_dataframe(self):
        from agents.llm_analyst import extract_indicators
        result = extract_indicators(None)
        assert result == {}


class TestAnalyzeSymbol:
    @patch("agents.llm_analyst.synthesize")
    @patch("agents.llm_analyst.analyze_technicals")
    @patch("agents.llm_analyst.analyze_news")
    @patch("agents.llm_analyst.get_daily_spend", return_value=0.0)
    def test_full_analysis(self, mock_spend, mock_news, mock_tech, mock_synth):
        from agents.llm_analyst import analyze_symbol
        mock_news.return_value = {"sentiment": "bullish"}
        mock_tech.return_value = {"trend": "up"}
        mock_synth.return_value = {
            "conviction_score": 0.7,
            "bias": "bullish",
            "confidence": 0.8,
            "key_factors": [],
            "risk_flags": [],
            "reasoning": "test",
        }

        with patch("agents.llm_analyst.Config") as mock_config:
            mock_config.LLM_BUDGET_DAILY = 1.0
            result = analyze_symbol("AAPL", headlines=["headline"])

        assert result["score"] == 0.7
        assert result["bias"] == "bullish"

    @patch("agents.llm_analyst.get_daily_spend", return_value=5.0)
    def test_budget_exceeded_returns_none(self, mock_spend):
        from agents.llm_analyst import analyze_symbol
        with patch("agents.llm_analyst.Config") as mock_config:
            mock_config.LLM_BUDGET_DAILY = 1.0
            result = analyze_symbol("AAPL")
        assert result is None


class TestRunAnalysis:
    @patch("agents.llm_analyst.analyze_symbol")
    @patch("agents.llm_analyst.get_daily_spend", return_value=0.0)
    def test_analyzes_all_symbols(self, mock_spend, mock_analyze):
        from agents.llm_analyst import run_analysis
        mock_analyze.side_effect = [
            {"score": 0.5, "bias": "bullish", "confidence": 0.8,
             "key_factors": [], "risk_flags": [], "reasoning": ""},
            None,  # Budget exceeded
        ]

        with patch("agents.llm_analyst.Config") as mock_config:
            mock_config.LLM_BUDGET_DAILY = 1.0
            mock_config.SYMBOLS = ["AAPL", "MSFT"]
            result = run_analysis(symbols=["AAPL", "MSFT"])

        assert "AAPL" in result["convictions"]
        assert "MSFT" not in result["convictions"]

    @patch("agents.llm_analyst.analyze_symbol")
    @patch("agents.llm_analyst.get_daily_spend", return_value=0.0)
    def test_uses_config_symbols(self, mock_spend, mock_analyze):
        from agents.llm_analyst import run_analysis
        mock_analyze.return_value = None

        with patch("agents.llm_analyst.Config") as mock_config:
            mock_config.SYMBOLS = ["A", "B"]
            mock_config.LLM_BUDGET_DAILY = 1.0
            run_analysis()
        assert mock_analyze.call_count == 2


class TestWriteConvictions:
    def test_writes_json(self, tmp_path):
        from agents.llm_analyst import write_convictions
        conv_file = tmp_path / "llm_analyst" / "convictions.json"

        with patch("agents.llm_analyst._DATA_DIR", str(tmp_path / "llm_analyst")), \
             patch("agents.llm_analyst._CONVICTIONS_FILE", str(conv_file)):
            data = {
                "timestamp": "2024-01-01",
                "convictions": {"AAPL": {"score": 0.7}},
            }
            write_convictions(data)

        assert conv_file.exists()
        written = json.loads(conv_file.read_text())
        assert written["convictions"]["AAPL"]["score"] == 0.7


class TestMain:
    @patch("agents.llm_analyst.write_convictions")
    @patch("agents.llm_analyst.run_analysis")
    def test_main_success(self, mock_run, mock_write):
        from agents.llm_analyst import main
        mock_run.return_value = {
            "timestamp": "2024-01-01",
            "convictions": {"AAPL": {"score": 0.5}},
            "daily_spend": 0.05,
        }

        with patch("agents.llm_analyst.Config") as mock_config:
            mock_config.LLM_ANALYST_V2_ENABLED = False
            mock_config.LLM_ANALYST_ENABLED = True
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.SYMBOLS = ["AAPL"]
            result = main()

        mock_run.assert_called_once()
        mock_write.assert_called_once()
        assert result is not None

    @patch("agents.llm_analyst.write_convictions")
    @patch("agents.llm_analyst.run_analysis")
    def test_main_no_api_key(self, mock_run, mock_write):
        from agents.llm_analyst import main
        with patch("agents.llm_analyst.Config") as mock_config:
            mock_config.LLM_ANALYST_V2_ENABLED = False
            mock_config.LLM_ANALYST_ENABLED = True
            mock_config.ANTHROPIC_API_KEY = ""
            result = main()

        mock_run.assert_not_called()
        assert result is None
