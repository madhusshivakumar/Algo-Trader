"""GET /api/stats — trading statistics from Alpaca portfolio history + order matching."""

from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            api_key = os.environ["ALPACA_API_KEY"]
            secret_key = os.environ["ALPACA_SECRET_KEY"]
            is_paper = os.environ.get("TRADING_MODE", "paper") == "paper"

            # Get authoritative total P&L from portfolio history
            base_url = "https://paper-api.alpaca.markets" if is_paper else "https://api.alpaca.markets"
            url = f"{base_url}/v2/account"
            req = urllib.request.Request(url)
            req.add_header("APCA-API-KEY-ID", api_key)
            req.add_header("APCA-API-SECRET-KEY", secret_key)
            with urllib.request.urlopen(req, timeout=10) as resp:
                acct = json.loads(resp.read().decode())

            url = f"{base_url}/v2/account/portfolio/history?period=all&timeframe=1D"
            req = urllib.request.Request(url)
            req.add_header("APCA-API-KEY-ID", api_key)
            req.add_header("APCA-API-SECRET-KEY", secret_key)
            with urllib.request.urlopen(req, timeout=10) as resp:
                hist = json.loads(resp.read().decode())

            base_value = float(hist.get("base_value", 0))
            equity = float(acct.get("equity", 0))
            total_pnl = equity - base_value if base_value > 0 else 0

            # Get filled orders for win/loss breakdown
            client = TradingClient(api_key, secret_key, paper=is_paper)
            order_req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500)
            orders = client.get_orders(order_req)

            filled = [o for o in orders if "filled" in str(o.status).lower()]
            buys = [o for o in filled if "buy" in str(o.side).lower()]

            # Reconstruct per-trade PnL using weighted avg cost basis
            cost_basis = {}
            wins = 0
            losses = 0
            win_pnl = 0.0
            loss_pnl = 0.0

            for o in reversed(filled):
                symbol = o.symbol
                price = float(o.filled_avg_price) if o.filled_avg_price else 0
                qty = float(o.filled_qty) if o.filled_qty else 0
                if price == 0 or qty == 0:
                    continue

                if "buy" in str(o.side).lower():
                    prev = cost_basis.get(symbol, (0.0, 0.0))
                    cost_basis[symbol] = (prev[0] + price * qty, prev[1] + qty)
                elif "sell" in str(o.side).lower():
                    basis = cost_basis.get(symbol)
                    if basis and basis[1] > 0:
                        avg_cost = basis[0] / basis[1]
                        pnl = (price - avg_cost) * qty
                        if pnl > 0:
                            wins += 1
                            win_pnl += pnl
                        elif pnl < 0:
                            losses += 1
                            loss_pnl += pnl
                        remaining_qty = max(0, basis[1] - qty)
                        cost_basis[symbol] = (avg_cost * remaining_qty, remaining_qty)

            total_sells = wins + losses
            win_rate = (wins / total_sells * 100) if total_sells > 0 else 0
            avg_win = (win_pnl / wins) if wins > 0 else 0
            avg_loss = (loss_pnl / losses) if losses > 0 else 0

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "total_trades": len(buys),
                "wins": wins,
                "losses": losses,
                "win_rate": round(win_rate, 1),
                "total_pnl": round(total_pnl, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
            }).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
