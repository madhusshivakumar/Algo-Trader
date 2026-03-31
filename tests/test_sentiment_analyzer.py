"""Tests for core/sentiment_analyzer.py — FinBERT sentiment scoring."""

import pytest
from unittest.mock import patch, MagicMock


class TestScoreText:
    @patch("core.sentiment_analyzer._get_pipeline")
    def test_positive_sentiment(self, mock_get_pipe):
        from core.sentiment_analyzer import score_text
        mock_pipe = MagicMock()
        mock_pipe.return_value = [[
            {"label": "positive", "score": 0.9},
            {"label": "negative", "score": 0.05},
            {"label": "neutral", "score": 0.05},
        ]]
        mock_get_pipe.return_value = mock_pipe

        result = score_text("Apple beats earnings expectations")
        assert result > 0
        assert result == round(0.9 - 0.05, 4)

    @patch("core.sentiment_analyzer._get_pipeline")
    def test_negative_sentiment(self, mock_get_pipe):
        from core.sentiment_analyzer import score_text
        mock_pipe = MagicMock()
        mock_pipe.return_value = [[
            {"label": "positive", "score": 0.05},
            {"label": "negative", "score": 0.85},
            {"label": "neutral", "score": 0.10},
        ]]
        mock_get_pipe.return_value = mock_pipe

        result = score_text("Company faces major lawsuit")
        assert result < 0
        assert result == round(0.05 - 0.85, 4)

    @patch("core.sentiment_analyzer._get_pipeline")
    def test_neutral_sentiment(self, mock_get_pipe):
        from core.sentiment_analyzer import score_text
        mock_pipe = MagicMock()
        mock_pipe.return_value = [[
            {"label": "positive", "score": 0.4},
            {"label": "negative", "score": 0.4},
            {"label": "neutral", "score": 0.2},
        ]]
        mock_get_pipe.return_value = mock_pipe

        result = score_text("Company reports quarterly results")
        assert result == 0.0

    def test_empty_text_returns_zero(self):
        from core.sentiment_analyzer import score_text
        assert score_text("") == 0.0
        assert score_text("   ") == 0.0

    @patch("core.sentiment_analyzer._get_pipeline")
    def test_truncates_long_text(self, mock_get_pipe):
        from core.sentiment_analyzer import score_text
        mock_pipe = MagicMock()
        mock_pipe.return_value = [[
            {"label": "positive", "score": 0.5},
            {"label": "negative", "score": 0.3},
            {"label": "neutral", "score": 0.2},
        ]]
        mock_get_pipe.return_value = mock_pipe

        long_text = "x" * 1000
        score_text(long_text)
        # Should truncate to 512 chars
        called_text = mock_pipe.call_args[0][0]
        assert len(called_text) == 512

    @patch("core.sentiment_analyzer._get_pipeline")
    def test_missing_label_treated_as_zero(self, mock_get_pipe):
        from core.sentiment_analyzer import score_text
        mock_pipe = MagicMock()
        mock_pipe.return_value = [[
            {"label": "positive", "score": 0.7},
            {"label": "neutral", "score": 0.3},
        ]]
        mock_get_pipe.return_value = mock_pipe

        result = score_text("Some news")
        assert result == 0.7  # positive - 0 (no negative)


class TestScoreHeadlines:
    @patch("core.sentiment_analyzer.score_text")
    def test_aggregate_positive(self, mock_score):
        from core.sentiment_analyzer import score_headlines
        mock_score.side_effect = [0.5, 0.3, 0.8]

        result = score_headlines(["h1", "h2", "h3"])
        assert result["sentiment_score"] == round((0.5 + 0.3 + 0.8) / 3, 4)
        assert result["num_articles"] == 3
        assert result["positive_pct"] == 1.0  # all > 0.1

    @patch("core.sentiment_analyzer.score_text")
    def test_aggregate_negative(self, mock_score):
        from core.sentiment_analyzer import score_headlines
        mock_score.side_effect = [-0.5, -0.3, -0.8]

        result = score_headlines(["h1", "h2", "h3"])
        assert result["sentiment_score"] < 0
        assert result["negative_pct"] == 1.0

    @patch("core.sentiment_analyzer.score_text")
    def test_mixed_sentiment(self, mock_score):
        from core.sentiment_analyzer import score_headlines
        mock_score.side_effect = [0.5, -0.5, 0.0]

        result = score_headlines(["h1", "h2", "h3"])
        assert result["sentiment_score"] == 0.0
        assert result["positive_pct"] == round(1 / 3, 3)
        assert result["negative_pct"] == round(1 / 3, 3)

    def test_empty_headlines(self):
        from core.sentiment_analyzer import score_headlines
        result = score_headlines([])
        assert result["sentiment_score"] == 0.0
        assert result["num_articles"] == 0

    @patch("core.sentiment_analyzer.score_text")
    def test_error_in_scoring_skips(self, mock_score):
        from core.sentiment_analyzer import score_headlines
        mock_score.side_effect = [0.5, Exception("model error"), 0.3]

        result = score_headlines(["h1", "h2", "h3"])
        assert result["num_articles"] == 2
        assert result["sentiment_score"] == round((0.5 + 0.3) / 2, 4)

    @patch("core.sentiment_analyzer.score_text")
    def test_all_scoring_errors(self, mock_score):
        from core.sentiment_analyzer import score_headlines
        mock_score.side_effect = Exception("model error")

        result = score_headlines(["h1", "h2"])
        assert result["sentiment_score"] == 0.0
        assert result["num_articles"] == 2

    @patch("core.sentiment_analyzer.score_text")
    def test_result_keys(self, mock_score):
        from core.sentiment_analyzer import score_headlines
        mock_score.return_value = 0.5

        result = score_headlines(["headline"])
        assert "sentiment_score" in result
        assert "num_articles" in result
        assert "positive_pct" in result
        assert "negative_pct" in result


class TestGetPipeline:
    def test_loads_model_on_first_call(self):
        import core.sentiment_analyzer as sa
        sa._pipeline = None

        mock_pipeline_fn = MagicMock(return_value=MagicMock())
        mock_transformers = MagicMock()
        mock_transformers.pipeline = mock_pipeline_fn

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            result = sa._get_pipeline()

        mock_pipeline_fn.assert_called_once_with(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
        )
        assert result is not None
        # Clean up
        sa._pipeline = None

    def test_caches_pipeline(self):
        import core.sentiment_analyzer as sa
        fake_pipe = MagicMock()
        sa._pipeline = fake_pipe

        result = sa._get_pipeline()
        assert result is fake_pipe
        # Clean up
        sa._pipeline = None

    def test_raises_on_load_failure(self):
        import core.sentiment_analyzer as sa
        sa._pipeline = None

        mock_transformers = MagicMock()
        mock_transformers.pipeline.side_effect = Exception("model not found")

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            with pytest.raises(Exception, match="model not found"):
                sa._get_pipeline()
        sa._pipeline = None
