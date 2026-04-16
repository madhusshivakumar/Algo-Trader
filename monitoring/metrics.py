"""Sprint 7B: Prometheus metrics endpoint.

Exposes ``/metrics`` on port 9100 (configurable via ``METRICS_PORT``).

Metrics:
    Gauges:
        algo_equity, algo_cash, algo_open_positions, algo_unrealized_pnl,
        algo_daily_pnl
    Counters:
        algo_trades_total{side, strategy, symbol}
        algo_orders_rejected_total{reason}
        algo_alerts_total{level}
    Histograms:
        algo_cycle_duration_seconds
        algo_broker_latency_seconds{method}

Usage:
    from monitoring.metrics import start_metrics_server, EQUITY, TRADES

    start_metrics_server()         # starts background HTTP /metrics handler
    EQUITY.set(10_500.0)           # update in engine loop
    TRADES.labels("buy", "mr", "TSLA").inc()

When ``METRICS_ENABLED=false`` (default), ``start_metrics_server()`` is a
no-op and all metric objects are thin stubs that accept calls but discard
values (zero overhead path).
"""

from __future__ import annotations

import os
from typing import Any

_ENABLED = os.getenv("METRICS_ENABLED", "false").lower() == "true"
_PORT = int(os.getenv("METRICS_PORT", "9100"))

# ── Real or stub metrics ───────────────────────────────────────────────────

if _ENABLED:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    EQUITY = Gauge("algo_equity", "Current equity ($)")
    CASH = Gauge("algo_cash", "Current cash ($)")
    OPEN_POSITIONS = Gauge("algo_open_positions", "Open positions count")
    UNREALIZED_PNL = Gauge("algo_unrealized_pnl", "Unrealized P&L ($)")
    DAILY_PNL = Gauge("algo_daily_pnl", "Intraday realized P&L ($)")

    TRADES = Counter("algo_trades_total", "Total trades",
                     ["side", "strategy", "symbol"])
    ORDERS_REJECTED = Counter("algo_orders_rejected_total",
                              "Rejected orders", ["reason"])
    ALERTS = Counter("algo_alerts_total", "Alerts fired", ["level"])

    CYCLE_DURATION = Histogram("algo_cycle_duration_seconds",
                               "Engine cycle duration",
                               buckets=[0.1, 0.5, 1, 2, 5, 10, 30])
    BROKER_LATENCY = Histogram("algo_broker_latency_seconds",
                                "Broker API latency",
                                ["method"],
                                buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5])
else:
    class _Stub:
        """Drop-in replacement that accepts any call but does nothing."""
        def set(self, *a: Any, **kw: Any) -> None: pass
        def inc(self, *a: Any, **kw: Any) -> None: pass
        def dec(self, *a: Any, **kw: Any) -> None: pass
        def observe(self, *a: Any, **kw: Any) -> None: pass
        def labels(self, *a: Any, **kw: Any) -> "_Stub": return self
        def time(self) -> "_StubTimer": return _StubTimer()

    class _StubTimer:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    _s = _Stub()
    EQUITY = _s
    CASH = _s
    OPEN_POSITIONS = _s
    UNREALIZED_PNL = _s
    DAILY_PNL = _s
    TRADES = _s
    ORDERS_REJECTED = _s
    ALERTS = _s
    CYCLE_DURATION = _s
    BROKER_LATENCY = _s

    start_http_server = None  # type: ignore


def start_metrics_server(port: int | None = None) -> bool:
    """Start the ``/metrics`` HTTP server on a background thread.

    Returns True if actually started, False if metrics are disabled or
    already running.
    """
    if not _ENABLED or start_http_server is None:
        return False
    try:
        start_http_server(port or _PORT)
        return True
    except OSError:
        # Port already in use (previous run didn't exit cleanly).
        return False
