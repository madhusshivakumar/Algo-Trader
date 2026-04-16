"""GET /api/equity — portfolio equity curve from Alpaca REST API."""

from http.server import BaseHTTPRequestHandler
import json
import os
from datetime import datetime
import urllib.request
import base64


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            api_key = os.environ["ALPACA_API_KEY"]
            secret_key = os.environ["ALPACA_SECRET_KEY"]
            is_paper = os.environ.get("TRADING_MODE", "paper") == "paper"

            base_url = "https://paper-api.alpaca.markets" if is_paper else "https://api.alpaca.markets"
            url = f"{base_url}/v2/account/portfolio/history?period=1M&timeframe=1D"

            req = urllib.request.Request(url)
            req.add_header("APCA-API-KEY-ID", api_key)
            req.add_header("APCA-API-SECRET-KEY", secret_key)

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            snapshots = []
            timestamps = data.get("timestamp", [])
            equities = data.get("equity", [])

            for ts, eq in zip(timestamps, equities):
                if eq is not None:
                    snapshots.append({
                        "timestamp": datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                        "equity": float(eq),
                        "cash": 0,
                    })

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(snapshots).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
