"""GET /api/view_data — progressive-disclosure payload for dashboard v2.

Accepts ``?view=simple|standard|advanced`` (default: simple).

Aggregates account, stats, trades, equity, regime, framing, and
protection-status into one response shaped for the requested view mode.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import urllib.parse
import urllib.request

# Add the repo root so we can import core modules.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Parse view mode from query string.
            qs = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(qs)
            view_mode = params.get("view", ["simple"])[0]

            # Lazy imports so cold-start is tolerable.
            from core.dashboard_views import build_view_payload

            # Delegate to the aggregation helper.
            payload = build_view_payload(
                view_mode,
                account=_fetch_account(),
                stats=_fetch_stats(),
                trades=_fetch_trades(),
                equity_curve=_fetch_equity(),
                regime=_read_regime(),
                framing=_read_framing(),
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(payload, default=str).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


# ── Helpers ────────────────────────────────────────────────────────────────

def _alpaca_base():
    is_paper = os.environ.get("TRADING_MODE", "paper") == "paper"
    return "https://paper-api.alpaca.markets" if is_paper else "https://api.alpaca.markets"


def _alpaca_req(path: str) -> dict:
    url = _alpaca_base() + path
    req = urllib.request.Request(url)
    req.add_header("APCA-API-KEY-ID", os.environ["ALPACA_API_KEY"])
    req.add_header("APCA-API-SECRET-KEY", os.environ["ALPACA_SECRET_KEY"])
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def _fetch_account() -> dict:
    try:
        return _alpaca_req("/v2/account")
    except Exception:
        return {}


def _fetch_stats() -> dict:
    """Minimal stats (reuses /api/stats logic inline to avoid circular)."""
    try:
        acct = _fetch_account()
        return {"total_pnl": float(acct.get("equity", 0)) -
                             float(acct.get("last_equity", 0))}
    except Exception:
        return {}


def _fetch_trades() -> list[dict]:
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        client = TradingClient(
            os.environ["ALPACA_API_KEY"],
            os.environ["ALPACA_SECRET_KEY"],
            paper=os.environ.get("TRADING_MODE", "paper") == "paper",
        )
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=50)
        orders = client.get_orders(req)
        return [{"symbol": o.symbol, "side": str(o.side),
                 "timestamp": str(o.filled_at or o.created_at)}
                for o in orders if "filled" in str(o.status).lower()]
    except Exception:
        return []


def _fetch_equity() -> list[dict]:
    try:
        data = _alpaca_req("/v2/account/portfolio/history?period=1M&timeframe=1D")
        out = []
        for ts, eq in zip(data.get("timestamp", []),
                          data.get("equity", [])):
            if eq is not None:
                out.append({"timestamp": ts, "equity": eq})
        return out
    except Exception:
        return []


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _read_regime() -> str | None:
    """Read the latest regime classification from the runtime state."""
    state = _read_json(os.path.join(_ROOT, "data", "runtime_state.json"))
    if state:
        return state.get("regime")
    return None


def _read_framing() -> dict | None:
    """Read the latest backtest framing from the most recent OOS run."""
    oos_dir = os.path.join(_ROOT, "data", "backtest_oos")
    if not os.path.isdir(oos_dir):
        return None
    # Find the newest timestamped directory.
    try:
        dirs = sorted(os.listdir(oos_dir), reverse=True)
    except OSError:
        return None
    for d in dirs:
        summary = _read_json(os.path.join(oos_dir, d, "summary.json"))
        if summary and "framing" in summary:
            return summary["framing"]
    return None
