"""Tests for core/news_client.py — Alpaca News API client."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestFetchNews:
    def _mock_article(self, headline="Test headline", summary="Test summary",
                      source="reuters", created_at="2024-01-01T12:00:00Z"):
        article = MagicMock()
        article.headline = headline
        article.summary = summary
        article.source = source
        article.created_at = created_at
        return article

    @patch("core.news_client.NewsClient")
    def test_returns_articles(self, mock_cls):
        from core.news_client import fetch_news
        mock_response = MagicMock()
        mock_response.news = [
            self._mock_article("AAPL beats earnings"),
            self._mock_article("Apple launches new product"),
        ]
        mock_cls.return_value.get_news.return_value = mock_response

        result = fetch_news("AAPL")
        assert len(result) == 2
        assert result[0]["headline"] == "AAPL beats earnings"
        assert result[1]["headline"] == "Apple launches new product"

    @patch("core.news_client.NewsClient")
    def test_article_dict_keys(self, mock_cls):
        from core.news_client import fetch_news
        mock_response = MagicMock()
        mock_response.news = [self._mock_article()]
        mock_cls.return_value.get_news.return_value = mock_response

        result = fetch_news("AAPL")
        assert "headline" in result[0]
        assert "summary" in result[0]
        assert "source" in result[0]
        assert "created_at" in result[0]

    @patch("core.news_client.NewsClient")
    def test_empty_response(self, mock_cls):
        from core.news_client import fetch_news
        mock_response = MagicMock()
        mock_response.news = []
        mock_cls.return_value.get_news.return_value = mock_response

        result = fetch_news("AAPL")
        assert result == []

    @patch("core.news_client.NewsClient")
    def test_api_error_returns_empty(self, mock_cls):
        from core.news_client import fetch_news
        mock_cls.return_value.get_news.side_effect = Exception("API error")

        result = fetch_news("AAPL")
        assert result == []

    @patch("core.news_client.NewsClient")
    def test_crypto_symbol_stripped(self, mock_cls):
        from core.news_client import fetch_news
        mock_response = MagicMock()
        mock_response.news = []
        mock_cls.return_value.get_news.return_value = mock_response

        fetch_news("BTC/USD")
        call_args = mock_cls.return_value.get_news.call_args[0][0]
        # The symbol should have "/" removed
        assert call_args.symbols == "BTCUSD"

    @patch("core.news_client.NewsClient")
    def test_custom_lookback(self, mock_cls):
        from core.news_client import fetch_news
        mock_response = MagicMock()
        mock_response.news = []
        mock_cls.return_value.get_news.return_value = mock_response

        fetch_news("AAPL", lookback_hours=48)
        call_args = mock_cls.return_value.get_news.call_args[0][0]
        assert call_args.limit == 50  # default MAX_ARTICLES

    @patch("core.news_client.NewsClient")
    def test_custom_limit(self, mock_cls):
        from core.news_client import fetch_news
        mock_response = MagicMock()
        mock_response.news = []
        mock_cls.return_value.get_news.return_value = mock_response

        fetch_news("AAPL", limit=10)
        call_args = mock_cls.return_value.get_news.call_args[0][0]
        assert call_args.limit == 10

    @patch("core.news_client.NewsClient")
    def test_missing_article_attributes(self, mock_cls):
        from core.news_client import fetch_news
        article = MagicMock(spec=[])  # no attributes
        mock_response = MagicMock()
        mock_response.news = [article]
        mock_cls.return_value.get_news.return_value = mock_response

        result = fetch_news("AAPL")
        assert len(result) == 1
        assert result[0]["headline"] == ""


class TestFetchHeadlines:
    @patch("core.news_client.fetch_news")
    def test_returns_headline_strings(self, mock_fetch):
        from core.news_client import fetch_headlines
        mock_fetch.return_value = [
            {"headline": "AAPL up 5%", "summary": "..."},
            {"headline": "Apple earnings beat", "summary": "..."},
        ]

        result = fetch_headlines("AAPL")
        assert result == ["AAPL up 5%", "Apple earnings beat"]

    @patch("core.news_client.fetch_news")
    def test_filters_empty_headlines(self, mock_fetch):
        from core.news_client import fetch_headlines
        mock_fetch.return_value = [
            {"headline": "Good news", "summary": "..."},
            {"headline": "", "summary": "..."},
            {"headline": "More news", "summary": "..."},
        ]

        result = fetch_headlines("AAPL")
        assert result == ["Good news", "More news"]

    @patch("core.news_client.fetch_news")
    def test_empty_articles(self, mock_fetch):
        from core.news_client import fetch_headlines
        mock_fetch.return_value = []

        result = fetch_headlines("AAPL")
        assert result == []
