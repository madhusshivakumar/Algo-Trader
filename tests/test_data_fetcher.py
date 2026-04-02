"""Tests for core/data_fetcher.py — parallel bar fetching."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from core.data_fetcher import DataFetcher


def _make_df(n=50, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "open": 100 + rng.randn(n),
        "high": 101 + rng.randn(n),
        "low": 99 + rng.randn(n),
        "close": 100 + rng.randn(n),
        "volume": rng.randint(100000, 1000000, n),
    })


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.get_recent_bars.return_value = _make_df()
    broker.get_historical_bars.return_value = _make_df()
    return broker


@pytest.fixture
def fetcher(mock_broker):
    return DataFetcher(mock_broker, max_workers=3)


class TestFetchRecentBarsBatch:
    def test_all_success(self, fetcher, mock_broker):
        result = fetcher.fetch_recent_bars_batch(["AAPL", "TSLA", "NVDA"])
        assert len(result) == 3
        assert all(not df.empty for df in result.values())
        assert mock_broker.get_recent_bars.call_count == 3

    def test_one_failure(self, fetcher, mock_broker):
        def side_effect(symbol, limit=100):
            if symbol == "BAD":
                raise Exception("API error")
            return _make_df()

        mock_broker.get_recent_bars.side_effect = side_effect
        result = fetcher.fetch_recent_bars_batch(["AAPL", "BAD", "TSLA"])
        assert len(result) == 3
        assert not result["AAPL"].empty
        assert result["BAD"].empty
        assert not result["TSLA"].empty

    def test_all_fail(self, fetcher, mock_broker):
        mock_broker.get_recent_bars.side_effect = Exception("API down")
        result = fetcher.fetch_recent_bars_batch(["AAPL", "TSLA"])
        assert all(df.empty for df in result.values())

    def test_empty_symbols(self, fetcher):
        result = fetcher.fetch_recent_bars_batch([])
        assert result == {}

    def test_single_symbol(self, fetcher, mock_broker):
        result = fetcher.fetch_recent_bars_batch(["AAPL"])
        assert len(result) == 1
        assert "AAPL" in result

    def test_results_have_all_symbols(self, fetcher):
        symbols = ["AAPL", "TSLA", "NVDA", "AMD", "META"]
        result = fetcher.fetch_recent_bars_batch(symbols)
        assert set(result.keys()) == set(symbols)

    def test_passes_limit_parameter(self, fetcher, mock_broker):
        fetcher.fetch_recent_bars_batch(["AAPL"], limit=50)
        mock_broker.get_recent_bars.assert_called_once_with("AAPL", limit=50)


class TestFetchHistoricalBarsBatch:
    def test_all_success(self, fetcher, mock_broker):
        result = fetcher.fetch_historical_bars_batch(["AAPL", "TSLA"])
        assert len(result) == 2
        assert mock_broker.get_historical_bars.call_count == 2

    def test_one_failure(self, fetcher, mock_broker):
        def side_effect(symbol, days=30):
            if symbol == "BAD":
                raise Exception("fail")
            return _make_df()

        mock_broker.get_historical_bars.side_effect = side_effect
        result = fetcher.fetch_historical_bars_batch(["AAPL", "BAD"])
        assert not result["AAPL"].empty
        assert result["BAD"].empty

    def test_empty_symbols(self, fetcher):
        result = fetcher.fetch_historical_bars_batch([])
        assert result == {}

    def test_passes_days_parameter(self, fetcher, mock_broker):
        fetcher.fetch_historical_bars_batch(["AAPL"], days=60)
        mock_broker.get_historical_bars.assert_called_once_with("AAPL", days=60)


class TestMaxWorkers:
    def test_respects_max_workers(self, mock_broker):
        fetcher = DataFetcher(mock_broker, max_workers=2)
        assert fetcher.max_workers == 2

    def test_default_workers(self, mock_broker):
        fetcher = DataFetcher(mock_broker)
        assert fetcher.max_workers == 5
