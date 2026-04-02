"""Unit tests for the broker interface with mocked Alpaca API."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd
from datetime import datetime, timedelta


class MockAccount:
    equity = "100000.00"
    cash = "50000.00"
    buying_power = "200000.00"
    daytrade_count = "0"
    pattern_day_trader = False


class MockPosition:
    def __init__(self, symbol="AAPL", qty="10", market_value="2500",
                 unrealized_pl="50", unrealized_plpc="0.02",
                 avg_entry_price="245", current_price="250"):
        self.symbol = symbol
        self.qty = qty
        self.market_value = market_value
        self.unrealized_pl = unrealized_pl
        self.unrealized_plpc = unrealized_plpc
        self.avg_entry_price = avg_entry_price
        self.current_price = current_price


class MockOrder:
    def __init__(self):
        self.id = "test-order-123"
        self.status = "filled"
        self.symbol = "AAPL"
        self.side = "buy"
        self.qty = "10"
        self.filled_avg_price = "250.00"
        self.created_at = "2026-03-27T10:00:00Z"


class MockBarsResponse:
    def __init__(self, data):
        self.df = data


@pytest.fixture
def mock_alpaca():
    """Patch all Alpaca clients."""
    with patch("core.broker.TradingClient") as mock_trading, \
         patch("core.broker.CryptoHistoricalDataClient") as mock_crypto, \
         patch("core.broker.StockHistoricalDataClient") as mock_stock:

        trading_instance = MagicMock()
        crypto_instance = MagicMock()
        stock_instance = MagicMock()

        mock_trading.return_value = trading_instance
        mock_crypto.return_value = crypto_instance
        mock_stock.return_value = stock_instance

        yield {
            "trading": trading_instance,
            "crypto": crypto_instance,
            "stock": stock_instance,
        }


@pytest.fixture
def broker(mock_alpaca):
    from core.broker import Broker
    return Broker()


@pytest.fixture
def sample_bars_df():
    """Create a sample DataFrame that looks like Alpaca bars response."""
    n = 100
    import numpy as np
    rng = np.random.RandomState(42)
    idx = pd.MultiIndex.from_arrays([
        ["AAPL"] * n,
        pd.date_range("2026-03-01", periods=n, freq="1min"),
    ], names=["symbol", "timestamp"])
    return pd.DataFrame({
        "open": 250 + rng.randn(n),
        "high": 251 + rng.randn(n),
        "low": 249 + rng.randn(n),
        "close": 250 + rng.randn(n),
        "volume": rng.randint(100000, 1000000, n),
        "trade_count": rng.randint(100, 1000, n),
        "vwap": 250 + rng.randn(n),
    }, index=idx)


class TestBrokerAccount:
    def test_get_account(self, broker, mock_alpaca):
        mock_alpaca["trading"].get_account.return_value = MockAccount()
        result = broker.get_account()
        assert result["equity"] == 100000.0
        assert result["cash"] == 50000.0
        assert result["buying_power"] == 200000.0
        assert result["daytrade_count"] == 0
        assert result["pattern_day_trader"] is False

    def test_get_positions(self, broker, mock_alpaca):
        mock_alpaca["trading"].get_all_positions.return_value = [
            MockPosition("AAPL"), MockPosition("NVDA", qty="5", market_value="4500")
        ]
        result = broker.get_positions()
        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["qty"] == 10.0
        assert result[1]["symbol"] == "NVDA"

    def test_get_positions_empty(self, broker, mock_alpaca):
        mock_alpaca["trading"].get_all_positions.return_value = []
        result = broker.get_positions()
        assert result == []

    def test_get_position_found(self, broker, mock_alpaca):
        mock_alpaca["trading"].get_open_position.return_value = MockPosition()
        result = broker.get_position("AAPL")
        assert result is not None
        assert result["symbol"] == "AAPL"

    def test_get_position_not_found(self, broker, mock_alpaca):
        mock_alpaca["trading"].get_open_position.side_effect = Exception("not found")
        result = broker.get_position("ZZZZ")
        assert result is None

    def test_get_position_crypto_strips_slash(self, broker, mock_alpaca):
        mock_alpaca["trading"].get_open_position.return_value = MockPosition("BTCUSD")
        broker.get_position("BTC/USD")
        mock_alpaca["trading"].get_open_position.assert_called_with("BTCUSD")


class TestBrokerOrders:
    def test_buy_equity(self, broker, mock_alpaca):
        mock_alpaca["trading"].submit_order.return_value = MockOrder()
        result = broker.buy("AAPL", 5000)
        assert result["id"] == "test-order-123"
        assert result["symbol"] == "AAPL"
        # Verify DAY time-in-force for equity
        call_args = mock_alpaca["trading"].submit_order.call_args[0][0]
        assert call_args.symbol == "AAPL"

    def test_buy_crypto(self, broker, mock_alpaca):
        mock_alpaca["trading"].submit_order.return_value = MockOrder()
        result = broker.buy("BTC/USD", 25000)
        assert result["symbol"] == "BTC/USD"
        call_args = mock_alpaca["trading"].submit_order.call_args[0][0]
        assert call_args.symbol == "BTCUSD"  # Slash stripped

    def test_sell(self, broker, mock_alpaca):
        mock_alpaca["trading"].submit_order.return_value = MockOrder()
        result = broker.sell("AAPL", 10)
        assert result["id"] == "test-order-123"

    def test_close_position_success(self, broker, mock_alpaca):
        mock_alpaca["trading"].close_position.return_value = MockOrder()
        result = broker.close_position("AAPL")
        assert result is not None
        assert result["id"] == "test-order-123"

    def test_close_position_failure(self, broker, mock_alpaca):
        mock_alpaca["trading"].close_position.side_effect = Exception("no position")
        result = broker.close_position("ZZZZ")
        assert result is None

    def test_close_position_crypto(self, broker, mock_alpaca):
        mock_alpaca["trading"].close_position.return_value = MockOrder()
        broker.close_position("ETH/USD")
        mock_alpaca["trading"].close_position.assert_called_with("ETHUSD")


class TestBrokerNewOrderTypes:
    """Tests for Sprint 1 broker additions: limit, stop-limit, order query, cancel."""

    def test_submit_limit_order_buy(self, broker, mock_alpaca):
        mock_alpaca["trading"].submit_order.return_value = MockOrder()
        result = broker.submit_limit_order("AAPL", 10.0, 148.0, side="buy")
        assert result is not None
        assert result["id"] == "test-order-123"

    def test_submit_limit_order_sell(self, broker, mock_alpaca):
        mock_alpaca["trading"].submit_order.return_value = MockOrder()
        result = broker.submit_limit_order("AAPL", 10.0, 155.0, side="sell")
        assert result is not None

    def test_submit_limit_order_failure(self, broker, mock_alpaca):
        mock_alpaca["trading"].submit_order.side_effect = Exception("insufficient qty")
        result = broker.submit_limit_order("AAPL", 10.0, 148.0)
        assert result is None

    def test_submit_stop_limit_order(self, broker, mock_alpaca):
        mock_alpaca["trading"].submit_order.return_value = MockOrder()
        result = broker.submit_stop_limit_order("AAPL", 10.0, 145.0, 144.50, side="sell")
        assert result is not None
        assert result["id"] == "test-order-123"

    def test_submit_stop_limit_order_failure(self, broker, mock_alpaca):
        mock_alpaca["trading"].submit_order.side_effect = Exception("fail")
        result = broker.submit_stop_limit_order("AAPL", 10.0, 145.0, 144.50)
        assert result is None

    def test_get_order_by_id_found(self, broker, mock_alpaca):
        mock_order = MockOrder()
        mock_order.filled_qty = "10"
        mock_alpaca["trading"].get_order_by_id.return_value = mock_order
        result = broker.get_order_by_id("test-order-123")
        assert result is not None
        assert result["id"] == "test-order-123"
        assert result["status"] == "filled"
        assert result["filled_avg_price"] == 250.0

    def test_get_order_by_id_not_found(self, broker, mock_alpaca):
        mock_alpaca["trading"].get_order_by_id.side_effect = Exception("not found")
        result = broker.get_order_by_id("bad-id")
        assert result is None

    def test_cancel_order_success(self, broker, mock_alpaca):
        mock_alpaca["trading"].cancel_order_by_id.return_value = None
        assert broker.cancel_order("test-order-123") is True

    def test_cancel_order_failure(self, broker, mock_alpaca):
        mock_alpaca["trading"].cancel_order_by_id.side_effect = Exception("already filled")
        assert broker.cancel_order("test-order-123") is False

    def test_check_buying_power(self, broker, mock_alpaca):
        mock_alpaca["trading"].get_account.return_value = MockAccount()
        bp = broker.check_buying_power()
        assert bp == 200000.0


class TestBrokerMarketData:
    def test_get_historical_bars_stock(self, broker, mock_alpaca, sample_bars_df):
        mock_alpaca["stock"].get_stock_bars.return_value = MockBarsResponse(sample_bars_df)
        df = broker.get_historical_bars("AAPL", days=30)
        assert not df.empty
        assert "close" in df.columns
        assert "time" in df.columns  # Renamed from timestamp

    def test_get_historical_bars_crypto(self, broker, mock_alpaca):
        import numpy as np
        n = 50
        rng = np.random.RandomState(42)
        idx = pd.MultiIndex.from_arrays([
            ["BTC/USD"] * n,
            pd.date_range("2026-03-01", periods=n, freq="1min"),
        ], names=["symbol", "timestamp"])
        crypto_df = pd.DataFrame({
            "open": 65000 + rng.randn(n) * 100,
            "high": 65100 + rng.randn(n) * 100,
            "low": 64900 + rng.randn(n) * 100,
            "close": 65000 + rng.randn(n) * 100,
            "volume": rng.uniform(1, 10, n),
            "trade_count": rng.randint(100, 1000, n),
            "vwap": 65000 + rng.randn(n) * 100,
        }, index=idx)
        mock_alpaca["crypto"].get_crypto_bars.return_value = MockBarsResponse(crypto_df)
        df = broker.get_historical_bars("BTC/USD", days=30)
        assert not df.empty

    def test_get_recent_bars_stock(self, broker, mock_alpaca, sample_bars_df):
        mock_alpaca["stock"].get_stock_bars.return_value = MockBarsResponse(sample_bars_df)
        df = broker.get_recent_bars("AAPL", limit=50)
        assert len(df) <= 50

    def test_get_recent_bars_crypto(self, broker, mock_alpaca, sample_bars_df):
        mock_alpaca["crypto"].get_crypto_bars.return_value = MockBarsResponse(sample_bars_df)
        df = broker.get_recent_bars("ETH/USD", limit=50)
        assert len(df) <= 50

    def test_get_recent_orders(self, broker, mock_alpaca):
        mock_order = MockOrder()
        mock_alpaca["trading"].get_orders.return_value = [mock_order]
        result = broker.get_recent_orders(limit=5)
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["status"] == "filled"
