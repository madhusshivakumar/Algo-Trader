"""Unit tests for config module."""

import pytest
from config import Config


class TestConfig:
    """Test configuration loading and helpers."""

    def test_crypto_symbols_not_empty(self):
        assert len(Config.CRYPTO_SYMBOLS) > 0

    def test_equity_symbols_not_empty(self):
        assert len(Config.EQUITY_SYMBOLS) > 0

    def test_combined_symbols(self):
        assert Config.SYMBOLS == Config.CRYPTO_SYMBOLS + Config.EQUITY_SYMBOLS

    def test_is_crypto_with_slash(self):
        assert Config.is_crypto("BTC/USD") is True
        assert Config.is_crypto("ETH/USD") is True

    def test_is_crypto_without_slash(self):
        assert Config.is_crypto("AAPL") is False
        assert Config.is_crypto("TSLA") is False

    def test_is_paper_returns_bool(self):
        assert isinstance(Config.is_paper(), bool)

    def test_is_market_open_returns_bool(self):
        assert isinstance(Config.is_market_open(), bool)

    def test_risk_params_are_positive(self):
        assert Config.MAX_POSITION_PCT > 0
        assert Config.STOP_LOSS_PCT > 0
        assert Config.DAILY_DRAWDOWN_LIMIT > 0
        assert Config.TRAILING_STOP_PCT > 0

    def test_risk_params_reasonable(self):
        assert Config.MAX_POSITION_PCT <= 1.0
        assert Config.STOP_LOSS_PCT <= 0.10
        assert Config.DAILY_DRAWDOWN_LIMIT <= 0.50

    def test_rsi_thresholds(self):
        assert Config.RSI_OVERSOLD < Config.RSI_OVERBOUGHT
        assert 0 < Config.RSI_OVERSOLD < 50
        assert 50 < Config.RSI_OVERBOUGHT < 100

    def test_bollinger_params(self):
        assert Config.BOLLINGER_PERIOD > 0
        assert Config.BOLLINGER_STD > 0

    def test_market_timezone(self):
        assert Config.MARKET_TZ is not None
        assert str(Config.MARKET_TZ) == "US/Eastern"

    def test_market_hours_order(self):
        assert Config.MARKET_OPEN < Config.MARKET_CLOSE
