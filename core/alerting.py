"""Alerting system — sends trade notifications via Slack, Discord, or email.

Provides asynchronous, non-blocking alert delivery with deduplication and rate limiting.
All HTTP calls run in a daemon thread so the trading loop is never blocked.
"""

import json
import time
import threading
from collections import deque
from datetime import datetime
from enum import Enum

from config import Config
from utils.logger import log


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# Slack sidebar colors by level
_SLACK_COLORS = {
    AlertLevel.INFO: "#36a64f",       # green
    AlertLevel.WARNING: "#daa038",    # yellow
    AlertLevel.CRITICAL: "#cc0000",   # red
}

# Discord embed colors by level (decimal)
_DISCORD_COLORS = {
    AlertLevel.INFO: 3582784,        # green
    AlertLevel.WARNING: 14328888,    # yellow
    AlertLevel.CRITICAL: 13369344,   # red
}


class AlertChannel:
    """Base class for alert delivery channels."""

    def send(self, message: str, level: AlertLevel, data: dict | None = None) -> bool:
        """Send an alert. Returns True on success, False on failure."""
        raise NotImplementedError


class SlackChannel(AlertChannel):
    """Sends alerts to a Slack webhook URL."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, message: str, level: AlertLevel, data: dict | None = None) -> bool:
        try:
            import requests
            payload = {
                "attachments": [{
                    "color": _SLACK_COLORS.get(level, "#36a64f"),
                    "title": f"Algo-Trader [{level.value.upper()}]",
                    "text": message,
                    "ts": int(time.time()),
                    "footer": "Algo-Trader Bot",
                }]
            }
            if data:
                fields = [{"title": k, "value": str(v), "short": True} for k, v in data.items()]
                payload["attachments"][0]["fields"] = fields

            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            log.warning(f"Slack alert failed: {e}")
            return False


class DiscordChannel(AlertChannel):
    """Sends alerts to a Discord webhook URL."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, message: str, level: AlertLevel, data: dict | None = None) -> bool:
        try:
            import requests
            embed = {
                "title": f"Algo-Trader [{level.value.upper()}]",
                "description": message,
                "color": _DISCORD_COLORS.get(level, 3582784),
                "timestamp": datetime.now(tz=None).isoformat(),
                "footer": {"text": "Algo-Trader Bot"},
            }
            if data:
                embed["fields"] = [{"name": k, "value": str(v), "inline": True} for k, v in data.items()]

            payload = {"embeds": [embed]}
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            return resp.status_code in (200, 204)
        except Exception as e:
            log.warning(f"Discord alert failed: {e}")
            return False


# ── Rate Limiting & Deduplication ────────────────────────────────────

_DEDUP_WINDOW_SECONDS = 300   # 5 minutes
_RATE_LIMIT_PER_HOUR = 60


class AlertManager:
    """Central alert dispatcher with deduplication, rate limiting, and async delivery."""

    def __init__(self):
        self.channels: list[AlertChannel] = []
        self._dedup_cache: dict[str, float] = {}   # event_key -> last_sent_timestamp
        self._rate_window: deque[float] = deque()   # timestamps of recent alerts
        self._lock = threading.Lock()

        # Build channels from config
        if Config.ALERTING_ENABLED:
            if Config.SLACK_WEBHOOK_URL:
                self.channels.append(SlackChannel(Config.SLACK_WEBHOOK_URL))
            if Config.DISCORD_WEBHOOK_URL:
                self.channels.append(DiscordChannel(Config.DISCORD_WEBHOOK_URL))

    def _is_duplicate(self, event_key: str) -> bool:
        """Check if the same event was sent within the dedup window."""
        now = time.time()
        # Prune expired entries periodically (every 100th check)
        if len(self._dedup_cache) > 100:
            expired = [k for k, v in self._dedup_cache.items()
                       if (now - v) >= _DEDUP_WINDOW_SECONDS]
            for k in expired:
                del self._dedup_cache[k]
        last_sent = self._dedup_cache.get(event_key)
        if last_sent and (now - last_sent) < _DEDUP_WINDOW_SECONDS:
            return True
        return False

    def _is_rate_limited(self) -> bool:
        """Check if we've exceeded the hourly rate limit."""
        now = time.time()
        cutoff = now - 3600
        # Prune old entries
        while self._rate_window and self._rate_window[0] < cutoff:
            self._rate_window.popleft()
        return len(self._rate_window) >= _RATE_LIMIT_PER_HOUR

    def alert(self, event_type: str, message: str, level: AlertLevel,
              data: dict | None = None) -> bool:
        """Send an alert to all configured channels.

        Returns True if the alert was dispatched, False if suppressed or no channels.
        """
        if not self.channels:
            return False

        event_key = f"{event_type}:{message}"

        with self._lock:
            if self._is_duplicate(event_key):
                return False
            if self._is_rate_limited():
                log.warning("Alert rate limit exceeded, suppressing alert")
                return False
            # Record this alert
            now = time.time()
            self._dedup_cache[event_key] = now
            self._rate_window.append(now)

        # Send asynchronously in a daemon thread
        thread = threading.Thread(
            target=self._send_to_all,
            args=(message, level, data),
            daemon=True,
        )
        thread.start()
        return True

    def _send_to_all(self, message: str, level: AlertLevel, data: dict | None):
        """Send to all channels (runs in background thread)."""
        for channel in self.channels:
            try:
                channel.send(message, level, data)
            except Exception as e:
                # Never let a channel failure propagate
                try:
                    log.warning(f"Alert channel {channel.__class__.__name__} failed: {e}")
                except Exception:
                    pass

    # ── Convenience Methods ──────────────────────────────────────────

    def trade_executed(self, symbol: str, side: str, amount: float,
                       price: float, reason: str):
        """Alert on trade execution."""
        msg = f"{side.upper()} {symbol} — ${amount:.2f} @ ${price:.2f}"
        data = {"Symbol": symbol, "Side": side, "Amount": f"${amount:.2f}",
                "Price": f"${price:.2f}", "Reason": reason}
        self.alert("trade", msg, AlertLevel.INFO, data)

    def stop_loss_hit(self, symbol: str, entry_price: float,
                      stop_price: float, loss_pct: float):
        """Alert when trailing stop is triggered."""
        msg = (f"Stop loss hit for {symbol} — "
               f"entry ${entry_price:.2f}, stop ${stop_price:.2f} ({loss_pct:.1%} loss)")
        data = {"Symbol": symbol, "Entry": f"${entry_price:.2f}",
                "Stop": f"${stop_price:.2f}", "Loss": f"{loss_pct:.1%}"}
        self.alert("stop_loss", msg, AlertLevel.WARNING, data)

    def drawdown_halt(self, drawdown_pct: float, limit_pct: float):
        """Alert when daily drawdown halts trading."""
        msg = (f"TRADING HALTED — daily drawdown {drawdown_pct:.1%} "
               f"exceeds limit {limit_pct:.1%}")
        data = {"Drawdown": f"{drawdown_pct:.1%}", "Limit": f"{limit_pct:.1%}"}
        self.alert("drawdown_halt", msg, AlertLevel.CRITICAL, data)

    def agent_error(self, agent_name: str, error_msg: str):
        """Alert on agent failure."""
        msg = f"Agent '{agent_name}' error: {error_msg}"
        data = {"Agent": agent_name, "Error": error_msg[:200]}
        self.alert("agent_error", msg, AlertLevel.CRITICAL, data)

    def position_mismatch(self, mismatches: list[dict]):
        """Alert on position reconciliation mismatches."""
        details = "; ".join(f"{m['symbol']}: {m['issue']}" for m in mismatches[:5])
        msg = f"Position mismatch detected ({len(mismatches)} issues): {details}"
        data = {"Count": str(len(mismatches)), "Details": details}
        self.alert("position_mismatch", msg, AlertLevel.WARNING, data)

    def config_reloaded(self, changes: list[str]):
        """Alert on configuration reload."""
        msg = f"Configuration reloaded: {', '.join(changes[:5])}"
        self.alert("config_reload", msg, AlertLevel.INFO)

    def bot_shutdown(self, reason: str):
        """Alert on bot shutdown."""
        msg = f"Trading bot shutting down (reason: {reason})"
        self.alert("shutdown", msg, AlertLevel.WARNING, {"Reason": reason})

    def order_rejected(self, symbol: str, side: str, amount: float, reason: str):
        """Alert when a broker rejects an order."""
        msg = f"Order rejected: {side.upper()} {symbol} ${amount:.2f} — {reason}"
        data = {"Symbol": symbol, "Side": side, "Amount": f"${amount:.2f}",
                "Rejection reason": reason[:200]}
        self.alert("order_rejected", msg, AlertLevel.WARNING, data)

    def broker_api_failure(self, method: str, consecutive_count: int, error: str):
        """Alert when the broker API fails repeatedly (3+ in a row)."""
        msg = (f"Broker API failure — {method} failed {consecutive_count} times in a row "
               f"({error[:100]})")
        data = {"Method": method, "Consecutive failures": str(consecutive_count),
                "Last error": error[:200]}
        self.alert("broker_api_failure", msg, AlertLevel.CRITICAL, data)

    def external_position(self, symbol: str, qty: float, value: float):
        """Alert when an unauthorized external position is detected."""
        msg = (f"External position detected — {symbol} qty={qty:.4f} "
               f"value=${value:.2f} (not opened by this bot)")
        data = {"Symbol": symbol, "Quantity": f"{qty:.4f}",
                "Market value": f"${value:.2f}"}
        self.alert("external_position", msg, AlertLevel.WARNING, data)

    def symbol_degraded(self, symbol: str, recent_sharpe: float, lookback_days: int):
        """Alert when drift detection marks a symbol's recent performance degraded."""
        msg = (f"Symbol degraded — {symbol} recent Sharpe {recent_sharpe:.2f} "
               f"over {lookback_days}d. Position size will be reduced 70%.")
        data = {"Symbol": symbol, "Recent Sharpe": f"{recent_sharpe:.2f}",
                "Lookback days": str(lookback_days)}
        self.alert("symbol_degraded", msg, AlertLevel.WARNING, data)

    def intraday_halt(self, realized_pnl: float, unrealized_pnl: float,
                     threshold_usd: float):
        """Alert when the intraday P&L circuit breaker halts trading."""
        msg = (f"TRADING HALTED — intraday loss breached threshold. "
               f"Realized=${realized_pnl:.2f}, unrealized=${unrealized_pnl:.2f}, "
               f"threshold=${threshold_usd:.2f}")
        data = {"Realized PnL": f"${realized_pnl:.2f}",
                "Unrealized PnL": f"${unrealized_pnl:.2f}",
                "Threshold": f"${threshold_usd:.2f}"}
        self.alert("intraday_halt", msg, AlertLevel.CRITICAL, data)

    def liquidity_skip(self, symbol: str, spread_bps: float, limit_bps: float):
        """Alert (throttled) when a buy is skipped due to wide spread."""
        msg = (f"Liquidity skip: {symbol} spread {spread_bps:.1f} bps > "
               f"limit {limit_bps:.1f} bps")
        data = {"Symbol": symbol, "Spread (bps)": f"{spread_bps:.1f}",
                "Limit (bps)": f"{limit_bps:.1f}"}
        self.alert("liquidity_skip", msg, AlertLevel.INFO, data)

    def live_mode_override(self):
        """Alert when the paper->live safety gate is overridden."""
        msg = ("Paper->live safety gate OVERRIDDEN via I_UNDERSTAND_THE_RISK. "
               "Bot is now trading with real money.")
        self.alert("live_mode_override", msg, AlertLevel.CRITICAL)
