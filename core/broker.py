"""Alpaca broker interface — handles crypto AND equity trading."""

from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd

from config import Config


class Broker:
    def __init__(self):
        self.trading_client = TradingClient(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
            paper=Config.is_paper(),
        )
        self.crypto_data = CryptoHistoricalDataClient(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
        )
        self.stock_data = StockHistoricalDataClient(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
        )

    # ── Account & Positions ──────────────────────────────────────────

    def get_account(self) -> dict:
        account = self.trading_client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "daytrade_count": int(account.daytrade_count),
            "pattern_day_trader": bool(account.pattern_day_trader),
        }

    def get_positions(self) -> list[dict]:
        positions = self.trading_client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "asset_class": "crypto" if Config.is_crypto(p.symbol) or "USD" in p.symbol else "equity",
            }
            for p in positions
        ]

    def get_position(self, symbol: str) -> dict | None:
        alpaca_symbol = symbol.replace("/", "")
        try:
            p = self.trading_client.get_open_position(alpaca_symbol)
            return {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
            }
        except Exception:
            return None

    # ── Orders ───────────────────────────────────────────────────────

    def buy(self, symbol: str, notional: float) -> dict:
        """Buy by dollar amount — works for both crypto and stocks."""
        alpaca_symbol = symbol.replace("/", "")
        is_crypto = Config.is_crypto(symbol)

        req = MarketOrderRequest(
            symbol=alpaca_symbol,
            notional=round(notional, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC if is_crypto else TimeInForce.DAY,
        )
        order = self.trading_client.submit_order(req)
        return {"id": str(order.id), "status": str(order.status), "symbol": symbol}

    def sell(self, symbol: str, qty: float) -> dict:
        """Sell by quantity."""
        alpaca_symbol = symbol.replace("/", "")
        is_crypto = Config.is_crypto(symbol)

        req = MarketOrderRequest(
            symbol=alpaca_symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC if is_crypto else TimeInForce.DAY,
        )
        order = self.trading_client.submit_order(req)
        return {"id": str(order.id), "status": str(order.status), "symbol": symbol}

    def submit_limit_order(self, symbol: str, qty: float, limit_price: float,
                           side: str = "buy") -> dict | None:
        """Submit a limit order. side = 'buy' or 'sell'."""
        alpaca_symbol = symbol.replace("/", "")
        is_crypto = Config.is_crypto(symbol)
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        try:
            req = LimitOrderRequest(
                symbol=alpaca_symbol,
                qty=qty,
                limit_price=limit_price,
                side=order_side,
                time_in_force=TimeInForce.GTC if is_crypto else TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(req)
            return {"id": str(order.id), "status": str(order.status), "symbol": symbol}
        except Exception:
            return None

    def submit_stop_limit_order(self, symbol: str, qty: float, stop_price: float,
                                limit_price: float, side: str = "sell") -> dict | None:
        """Submit a stop-limit order. Typically used for protective stops."""
        alpaca_symbol = symbol.replace("/", "")
        is_crypto = Config.is_crypto(symbol)
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        try:
            req = StopLimitOrderRequest(
                symbol=alpaca_symbol,
                qty=qty,
                stop_price=stop_price,
                limit_price=limit_price,
                side=order_side,
                time_in_force=TimeInForce.GTC if is_crypto else TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(req)
            return {"id": str(order.id), "status": str(order.status), "symbol": symbol}
        except Exception:
            return None

    def close_position(self, symbol: str) -> dict | None:
        alpaca_symbol = symbol.replace("/", "")
        try:
            order = self.trading_client.close_position(alpaca_symbol)
            return {"id": str(order.id), "status": str(order.status)}
        except Exception:
            return None

    def get_order_by_id(self, order_id: str) -> dict | None:
        """Fetch current state of an order by its ID."""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return {
                "id": str(order.id),
                "symbol": order.symbol,
                "side": str(order.side),
                "status": str(order.status),
                "filled_qty": float(order.filled_qty or 0),
                "filled_avg_price": float(order.filled_avg_price or 0),
                "created_at": str(order.created_at),
            }
        except Exception:
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if successful."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def check_buying_power(self) -> float:
        """Return current buying power."""
        return self.get_account()["buying_power"]

    # ── Market Data ──────────────────────────────────────────────────

    def get_historical_bars(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical 1-min bars — auto-detects crypto vs stock."""
        if Config.is_crypto(symbol):
            return self._get_crypto_bars(symbol, days=days)
        else:
            return self._get_stock_bars(symbol, days=days)

    def get_recent_bars(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get the most recent N bars."""
        if Config.is_crypto(symbol):
            return self._get_recent_crypto_bars(symbol, limit)
        else:
            return self._get_recent_stock_bars(symbol, limit)

    # ── Crypto Data ──────────────────────────────────────────────────

    def _get_crypto_bars(self, symbol: str, days: int = 30) -> pd.DataFrame:
        req = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=datetime.now() - timedelta(days=days),
        )
        bars = self.crypto_data.get_crypto_bars(req)
        df = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol]
        df = df.rename(columns={"timestamp": "time"})
        return df

    def _get_recent_crypto_bars(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        req = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=datetime.now() - timedelta(hours=3),
        )
        bars = self.crypto_data.get_crypto_bars(req)
        df = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol]
        df = df.rename(columns={"timestamp": "time"})
        return df.tail(limit)

    # ── Stock Data ───────────────────────────────────────────────────

    def _get_stock_bars(self, symbol: str, days: int = 30) -> pd.DataFrame:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=datetime.now() - timedelta(days=days),
        )
        bars = self.stock_data.get_stock_bars(req)
        df = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol]
        df = df.rename(columns={"timestamp": "time"})
        return df

    def _get_recent_stock_bars(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=datetime.now() - timedelta(hours=3),
        )
        bars = self.stock_data.get_stock_bars(req)
        df = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol]
        df = df.rename(columns={"timestamp": "time"})
        return df.tail(limit)

    # ── Orders History ───────────────────────────────────────────────

    def get_recent_orders(self, limit: int = 20) -> list[dict]:
        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=limit,
        )
        orders = self.trading_client.get_orders(req)
        return [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "side": str(o.side),
                "qty": str(o.qty),
                "filled_avg_price": str(o.filled_avg_price),
                "status": str(o.status),
                "created_at": str(o.created_at),
            }
            for o in orders
        ]
