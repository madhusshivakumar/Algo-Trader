"""GET /api/account — live account + positions from Alpaca."""

from http.server import BaseHTTPRequestHandler
import json
import os

from alpaca.trading.client import TradingClient


def _get_client():
    return TradingClient(
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_SECRET_KEY"],
        paper=os.environ.get("TRADING_MODE", "paper") == "paper",
    )


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            client = _get_client()

            acct = client.get_account()
            account = {
                "equity": float(acct.equity),
                "cash": float(acct.cash),
                "buying_power": float(acct.buying_power),
                "daytrade_count": int(acct.daytrade_count),
            }

            positions = []
            for p in client.get_all_positions():
                positions.append({
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                })

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "account": account,
                "positions": positions,
            }).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
