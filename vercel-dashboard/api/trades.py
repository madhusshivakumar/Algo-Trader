"""GET /api/trades — filled orders from Alpaca with reconstructed PnL."""

from http.server import BaseHTTPRequestHandler
import json
import os

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


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

            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=500,
            )
            orders = client.get_orders(req)

            filled = []
            for o in orders:
                status_str = str(o.status).lower().replace("orderstatus.", "")
                if status_str not in ("filled", "partially_filled"):
                    continue
                filled.append(o)

            # Build cost basis to compute PnL on sells (oldest first)
            cost_basis = {}
            pnl_map = {}  # order_id -> pnl
            for o in reversed(filled):
                symbol = o.symbol
                price = float(o.filled_avg_price) if o.filled_avg_price else 0
                qty = float(o.filled_qty) if o.filled_qty else 0
                side = str(o.side).lower().replace("orderside.", "")
                if price == 0 or qty == 0:
                    continue

                if side == "buy":
                    prev = cost_basis.get(symbol, (0.0, 0.0))
                    cost_basis[symbol] = (prev[0] + price * qty, prev[1] + qty)
                elif side == "sell":
                    basis = cost_basis.get(symbol)
                    if basis and basis[1] > 0:
                        avg_cost = basis[0] / basis[1]
                        pnl = (price - avg_cost) * qty
                        pnl_map[str(o.id)] = round(pnl, 2)
                        remaining_qty = max(0, basis[1] - qty)
                        cost_basis[symbol] = (avg_cost * remaining_qty, remaining_qty)

            # Group TWAP/VWAP slices into single logical trades.
            # Slices are same symbol+side within a 5-minute window.
            from datetime import datetime as dt, timedelta
            MERGE_WINDOW = timedelta(minutes=5)

            # Sort oldest-first for grouping, then reverse for display
            filled_sorted = sorted(filled, key=lambda o: o.filled_at or o.created_at)

            groups = []  # list of lists
            for o in filled_sorted:
                price = float(o.filled_avg_price) if o.filled_avg_price else 0
                qty = float(o.filled_qty) if o.filled_qty else 0
                if price == 0 or qty == 0:
                    continue
                side = str(o.side).lower().replace("orderside.", "")
                ts = o.filled_at or o.created_at

                merged = False
                for g in groups:
                    g_first = g[0]
                    g_ts = g_first["_ts"]
                    if (g_first["symbol"] == o.symbol
                            and g_first["side"] == side
                            and abs((ts - g_ts).total_seconds()) < MERGE_WINDOW.total_seconds()):
                        g.append({"price": price, "qty": qty, "id": str(o.id),
                                  "symbol": o.symbol, "side": side, "_ts": ts})
                        merged = True
                        break
                if not merged:
                    groups.append([{"price": price, "qty": qty, "id": str(o.id),
                                    "symbol": o.symbol, "side": side, "_ts": ts}])

            # Build consolidated response
            trades = []
            for g in reversed(groups):  # newest first
                total_qty = sum(e["qty"] for e in g)
                total_notional = sum(e["price"] * e["qty"] for e in g)
                avg_price = total_notional / total_qty if total_qty else 0
                slices = len(g)
                label = f"TWAP ({slices} slices)" if slices > 1 else ""

                # Sum PnL for all orders in group
                group_pnl = None
                for e in g:
                    p = pnl_map.get(e["id"])
                    if p is not None:
                        group_pnl = (group_pnl or 0) + p

                trades.append({
                    "id": g[0]["id"],
                    "timestamp": str(g[0]["_ts"]),
                    "symbol": g[0]["symbol"],
                    "side": g[0]["side"],
                    "amount": round(total_notional, 2),
                    "price": round(avg_price, 2),
                    "reason": label,
                    "pnl": round(group_pnl, 2) if group_pnl is not None else None,
                    "strategy": "",
                })

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(trades).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
