"""Tests for agents/sentiment_agent.py — pre-market sentiment agent."""

import json
import pytest
from unittest.mock import patch, MagicMock


class TestAnalyzeSymbol:
    @patch("agents.sentiment_agent.score_headlines")
    @patch("agents.sentiment_agent.fetch_headlines")
    def test_returns_scores_when_headlines_found(self, mock_fetch, mock_score):
        from agents.sentiment_agent import analyze_symbol
        mock_fetch.return_value = ["AAPL up 5%", "Apple launches product"]
        mock_score.return_value = {
            "sentiment_score": 0.65,
            "num_articles": 2,
            "positive_pct": 1.0,
            "negative_pct": 0.0,
        }

        result = analyze_symbol("AAPL")
        assert result is not None
        assert result["sentiment_score"] == 0.65
        mock_fetch.assert_called_once_with("AAPL", lookback_hours=24)

    @patch("agents.sentiment_agent.fetch_headlines")
    def test_returns_none_when_no_headlines(self, mock_fetch):
        from agents.sentiment_agent import analyze_symbol
        mock_fetch.return_value = []

        result = analyze_symbol("AAPL")
        assert result is None

    @patch("agents.sentiment_agent.score_headlines")
    @patch("agents.sentiment_agent.fetch_headlines")
    def test_custom_lookback(self, mock_fetch, mock_score):
        from agents.sentiment_agent import analyze_symbol
        mock_fetch.return_value = ["headline"]
        mock_score.return_value = {"sentiment_score": 0.5, "num_articles": 1,
                                   "positive_pct": 1.0, "negative_pct": 0.0}

        analyze_symbol("AAPL", lookback_hours=48)
        mock_fetch.assert_called_once_with("AAPL", lookback_hours=48)


class TestRunAnalysis:
    @patch("agents.sentiment_agent.analyze_symbol")
    def test_analyzes_all_symbols(self, mock_analyze):
        from agents.sentiment_agent import run_analysis
        mock_analyze.side_effect = [
            {"sentiment_score": 0.5, "num_articles": 3, "positive_pct": 0.7, "negative_pct": 0.1},
            None,  # No headlines for second symbol
            {"sentiment_score": -0.3, "num_articles": 2, "positive_pct": 0.0, "negative_pct": 0.8},
        ]

        result = run_analysis(symbols=["AAPL", "MSFT", "TSLA"])
        assert "timestamp" in result
        assert "scores" in result
        assert len(result["scores"]) == 2  # MSFT had None
        assert "AAPL" in result["scores"]
        assert "TSLA" in result["scores"]
        assert "MSFT" not in result["scores"]

    @patch("agents.sentiment_agent.analyze_symbol")
    def test_uses_config_symbols_by_default(self, mock_analyze):
        from agents.sentiment_agent import run_analysis
        mock_analyze.return_value = None

        with patch("agents.sentiment_agent.Config") as mock_config:
            mock_config.SYMBOLS = ["SYM1", "SYM2"]
            run_analysis()
            assert mock_analyze.call_count == 2

    @patch("agents.sentiment_agent.analyze_symbol")
    def test_empty_symbols_list(self, mock_analyze):
        from agents.sentiment_agent import run_analysis
        result = run_analysis(symbols=[])
        assert result["scores"] == {}
        mock_analyze.assert_not_called()


class TestWriteScores:
    def test_writes_json_file(self, tmp_path):
        from agents.sentiment_agent import write_scores
        scores_file = tmp_path / "sentiment" / "scores.json"

        with patch("agents.sentiment_agent._DATA_DIR", str(tmp_path / "sentiment")), \
             patch("agents.sentiment_agent._SCORES_FILE", str(scores_file)):
            data = {
                "timestamp": "2024-01-01T00:00:00",
                "scores": {"AAPL": {"sentiment_score": 0.5}},
            }
            write_scores(data)

        assert scores_file.exists()
        written = json.loads(scores_file.read_text())
        assert written["scores"]["AAPL"]["sentiment_score"] == 0.5

    def test_creates_directory(self, tmp_path):
        from agents.sentiment_agent import write_scores
        nested = tmp_path / "new" / "dir"

        with patch("agents.sentiment_agent._DATA_DIR", str(nested)), \
             patch("agents.sentiment_agent._SCORES_FILE", str(nested / "scores.json")):
            write_scores({"timestamp": "now", "scores": {}})

        assert nested.exists()


class TestMain:
    @patch("agents.sentiment_agent.write_scores")
    @patch("agents.sentiment_agent.run_analysis")
    def test_main_success(self, mock_run, mock_write):
        from agents.sentiment_agent import main
        mock_run.return_value = {
            "timestamp": "2024-01-01",
            "scores": {"AAPL": {"sentiment_score": 0.5}},
        }

        with patch("agents.sentiment_agent.Config") as mock_config:
            mock_config.SENTIMENT_ENABLED = True
            mock_config.SYMBOLS = ["AAPL"]
            result = main()

        mock_run.assert_called_once()
        mock_write.assert_called_once()
        assert result["scores"]["AAPL"]["sentiment_score"] == 0.5

    @patch("agents.sentiment_agent.write_scores")
    @patch("agents.sentiment_agent.run_analysis")
    def test_main_runs_even_when_disabled(self, mock_run, mock_write):
        from agents.sentiment_agent import main
        mock_run.return_value = {"timestamp": "now", "scores": {}}

        with patch("agents.sentiment_agent.Config") as mock_config:
            mock_config.SENTIMENT_ENABLED = False
            mock_config.SYMBOLS = []
            main()

        mock_run.assert_called_once()
