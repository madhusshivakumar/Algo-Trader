"""Tests for core/alerting.py — alert system with Slack/Discord channels."""

import time
from collections import deque

import pytest
from unittest.mock import patch, MagicMock


# ── SlackChannel Tests ──────────────────────────────────────────────


class TestSlackChannel:
    def test_sends_post_to_webhook_url(self):
        from core.alerting import SlackChannel, AlertLevel
        ch = SlackChannel("https://hooks.slack.com/test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = ch.send("test message", AlertLevel.INFO)
        assert result is True
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://hooks.slack.com/test"
        assert kwargs["timeout"] == 10

    def test_slack_payload_format(self):
        from core.alerting import SlackChannel, AlertLevel
        ch = SlackChannel("https://hooks.slack.com/test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.post", return_value=mock_resp) as mock_post:
            ch.send("buy signal", AlertLevel.WARNING, {"Symbol": "TSLA"})
        payload = mock_post.call_args[1]["json"]
        assert "attachments" in payload
        att = payload["attachments"][0]
        assert att["title"] == "Algo-Trader [WARNING]"
        assert att["text"] == "buy signal"
        assert att["color"] == "#daa038"
        assert any(f["title"] == "Symbol" and f["value"] == "TSLA" for f in att["fields"])

    def test_slack_returns_false_on_non_200(self):
        from core.alerting import SlackChannel, AlertLevel
        ch = SlackChannel("https://hooks.slack.com/test")
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("requests.post", return_value=mock_resp):
            result = ch.send("test", AlertLevel.INFO)
        assert result is False

    def test_slack_returns_false_on_exception(self):
        from core.alerting import SlackChannel, AlertLevel
        ch = SlackChannel("https://hooks.slack.com/test")
        with patch("requests.post", side_effect=ConnectionError("timeout")):
            result = ch.send("test", AlertLevel.INFO)
        assert result is False

    def test_slack_no_fields_without_data(self):
        from core.alerting import SlackChannel, AlertLevel
        ch = SlackChannel("https://hooks.slack.com/test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.post", return_value=mock_resp) as mock_post:
            ch.send("test", AlertLevel.INFO)
        payload = mock_post.call_args[1]["json"]
        assert "fields" not in payload["attachments"][0]

    def test_slack_critical_color(self):
        from core.alerting import SlackChannel, AlertLevel
        ch = SlackChannel("https://hooks.slack.com/test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.post", return_value=mock_resp) as mock_post:
            ch.send("critical!", AlertLevel.CRITICAL)
        payload = mock_post.call_args[1]["json"]
        assert payload["attachments"][0]["color"] == "#cc0000"


# ── DiscordChannel Tests ────────────────────────────────────────────


class TestDiscordChannel:
    def test_sends_post_to_webhook_url(self):
        from core.alerting import DiscordChannel, AlertLevel
        ch = DiscordChannel("https://discord.com/api/webhooks/test")
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = ch.send("test message", AlertLevel.INFO)
        assert result is True
        mock_post.assert_called_once()

    def test_discord_payload_format(self):
        from core.alerting import DiscordChannel, AlertLevel
        ch = DiscordChannel("https://discord.com/api/webhooks/test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.post", return_value=mock_resp) as mock_post:
            ch.send("sell signal", AlertLevel.CRITICAL, {"PnL": "-$50"})
        payload = mock_post.call_args[1]["json"]
        assert "embeds" in payload
        embed = payload["embeds"][0]
        assert embed["title"] == "Algo-Trader [CRITICAL]"
        assert embed["description"] == "sell signal"
        assert any(f["name"] == "PnL" and f["value"] == "-$50" for f in embed["fields"])

    def test_discord_returns_false_on_error(self):
        from core.alerting import DiscordChannel, AlertLevel
        ch = DiscordChannel("https://discord.com/api/webhooks/test")
        with patch("requests.post", side_effect=Exception("network error")):
            result = ch.send("test", AlertLevel.INFO)
        assert result is False

    def test_discord_accepts_200_and_204(self):
        from core.alerting import DiscordChannel, AlertLevel
        ch = DiscordChannel("https://discord.com/api/webhooks/test")
        for code in [200, 204]:
            mock_resp = MagicMock()
            mock_resp.status_code = code
            with patch("requests.post", return_value=mock_resp):
                assert ch.send("test", AlertLevel.INFO) is True


# ── AlertManager Tests ──────────────────────────────────────────────


class TestAlertManager:
    def _make_manager(self, channels=None):
        """Create an AlertManager with custom channels (bypassing Config)."""
        from core.alerting import AlertManager
        with patch("core.alerting.Config") as mock_cfg:
            mock_cfg.ALERTING_ENABLED = False
            mgr = AlertManager()
        if channels:
            mgr.channels = channels
        return mgr

    def test_no_channels_returns_false(self):
        mgr = self._make_manager(channels=[])
        from core.alerting import AlertLevel
        assert mgr.alert("test", "msg", AlertLevel.INFO) is False

    def test_alert_dispatches_to_channel(self):
        from core.alerting import AlertLevel
        mock_ch = MagicMock()
        mgr = self._make_manager(channels=[mock_ch])
        # Patch Thread to call _send_to_all synchronously
        with patch("core.alerting.threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock()
            mock_thread_cls.return_value.start = lambda: mgr._send_to_all("hello", AlertLevel.INFO, None)
            result = mgr.alert("test", "hello", AlertLevel.INFO)
        assert result is True
        mock_ch.send.assert_called_once_with("hello", AlertLevel.INFO, None)

    def test_alert_dispatches_to_multiple_channels(self):
        from core.alerting import AlertLevel
        ch1 = MagicMock()
        ch2 = MagicMock()
        mgr = self._make_manager(channels=[ch1, ch2])
        with patch("core.alerting.threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock()
            mock_thread_cls.return_value.start = lambda: mgr._send_to_all("msg", AlertLevel.WARNING, {"key": "val"})
            mgr.alert("test", "msg", AlertLevel.WARNING, {"key": "val"})
        ch1.send.assert_called_once_with("msg", AlertLevel.WARNING, {"key": "val"})
        ch2.send.assert_called_once_with("msg", AlertLevel.WARNING, {"key": "val"})

    def test_deduplication_suppresses_repeat(self):
        from core.alerting import AlertLevel
        mock_ch = MagicMock()
        mgr = self._make_manager(channels=[mock_ch])
        assert mgr.alert("trade", "BUY TSLA", AlertLevel.INFO) is True
        # Same event within 5 min should be suppressed
        assert mgr.alert("trade", "BUY TSLA", AlertLevel.INFO) is False

    def test_deduplication_allows_different_events(self):
        from core.alerting import AlertLevel
        mock_ch = MagicMock()
        mgr = self._make_manager(channels=[mock_ch])
        assert mgr.alert("trade", "BUY TSLA", AlertLevel.INFO) is True
        assert mgr.alert("trade", "SELL AAPL", AlertLevel.INFO) is True

    def test_deduplication_expires(self):
        from core.alerting import AlertLevel, _DEDUP_WINDOW_SECONDS
        mock_ch = MagicMock()
        mgr = self._make_manager(channels=[mock_ch])
        # Send first alert
        mgr.alert("trade", "BUY TSLA", AlertLevel.INFO)
        # Manually expire the dedup cache
        key = "trade:BUY TSLA"
        mgr._dedup_cache[key] = time.time() - _DEDUP_WINDOW_SECONDS - 1
        # Should now be allowed
        assert mgr.alert("trade", "BUY TSLA", AlertLevel.INFO) is True

    def test_rate_limiting(self):
        from core.alerting import AlertLevel, _RATE_LIMIT_PER_HOUR
        mock_ch = MagicMock()
        mgr = self._make_manager(channels=[mock_ch])
        # Fill the rate window
        now = time.time()
        mgr._rate_window = deque([now] * _RATE_LIMIT_PER_HOUR)
        # Next alert should be suppressed
        result = mgr.alert("new_event", "message", AlertLevel.INFO)
        assert result is False

    def test_rate_limiting_prunes_old_entries(self):
        from core.alerting import AlertLevel
        mock_ch = MagicMock()
        mgr = self._make_manager(channels=[mock_ch])
        # Add old entries (> 1 hour ago)
        old_time = time.time() - 3700
        mgr._rate_window = deque([old_time] * 100)
        # Should succeed since old entries are pruned
        result = mgr.alert("test", "msg", AlertLevel.INFO)
        assert result is True

    def test_channel_failure_doesnt_crash(self):
        from core.alerting import AlertLevel
        failing_ch = MagicMock()
        failing_ch.send.side_effect = Exception("boom")
        mgr = self._make_manager(channels=[failing_ch])
        # Call _send_to_all directly to test failure handling synchronously
        mgr._send_to_all("msg", AlertLevel.INFO, None)
        failing_ch.send.assert_called_once()

    def test_alert_with_data(self):
        from core.alerting import AlertLevel
        mock_ch = MagicMock()
        mgr = self._make_manager(channels=[mock_ch])
        data = {"Symbol": "TSLA", "Price": "150.00"}
        with patch("core.alerting.threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock()
            mock_thread_cls.return_value.start = lambda: mgr._send_to_all("msg", AlertLevel.INFO, data)
            mgr.alert("test", "msg", AlertLevel.INFO, data)
        mock_ch.send.assert_called_once_with("msg", AlertLevel.INFO, data)

    def test_disabled_creates_no_channels(self):
        from core.alerting import AlertManager
        with patch("core.alerting.Config") as mock_cfg:
            mock_cfg.ALERTING_ENABLED = False
            mgr = AlertManager()
        assert len(mgr.channels) == 0

    def test_enabled_with_slack_url(self):
        from core.alerting import AlertManager, SlackChannel
        with patch("core.alerting.Config") as mock_cfg:
            mock_cfg.ALERTING_ENABLED = True
            mock_cfg.SLACK_WEBHOOK_URL = "https://hooks.slack.com/test"
            mock_cfg.DISCORD_WEBHOOK_URL = ""
            mgr = AlertManager()
        assert len(mgr.channels) == 1
        assert isinstance(mgr.channels[0], SlackChannel)

    def test_enabled_with_both_urls(self):
        from core.alerting import AlertManager, SlackChannel, DiscordChannel
        with patch("core.alerting.Config") as mock_cfg:
            mock_cfg.ALERTING_ENABLED = True
            mock_cfg.SLACK_WEBHOOK_URL = "https://hooks.slack.com/test"
            mock_cfg.DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/test"
            mgr = AlertManager()
        assert len(mgr.channels) == 2

    def test_enabled_with_no_urls(self):
        from core.alerting import AlertManager
        with patch("core.alerting.Config") as mock_cfg:
            mock_cfg.ALERTING_ENABLED = True
            mock_cfg.SLACK_WEBHOOK_URL = ""
            mock_cfg.DISCORD_WEBHOOK_URL = ""
            mgr = AlertManager()
        assert len(mgr.channels) == 0

    def test_threading_is_daemon(self):
        from core.alerting import AlertLevel
        mock_ch = MagicMock()
        mgr = self._make_manager(channels=[mock_ch])
        with patch("core.alerting.threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            mgr.alert("test", "msg", AlertLevel.INFO)
            mock_thread_cls.assert_called_once()
            assert mock_thread_cls.call_args[1]["daemon"] is True
            mock_thread.start.assert_called_once()

    def test_dedup_cache_pruning(self):
        from core.alerting import AlertLevel, _DEDUP_WINDOW_SECONDS
        mock_ch = MagicMock()
        mgr = self._make_manager(channels=[mock_ch])
        # Add 150 expired entries
        expired_time = time.time() - _DEDUP_WINDOW_SECONDS - 10
        for i in range(150):
            mgr._dedup_cache[f"event_{i}:msg_{i}"] = expired_time
        assert len(mgr._dedup_cache) == 150
        # Trigger alert which should prune expired entries
        mgr.alert("new", "new_msg", AlertLevel.INFO)
        # Expired entries should be cleaned up
        assert len(mgr._dedup_cache) < 150

    def test_position_mismatch_empty_list(self):
        """Empty mismatch list still formats correctly."""
        mgr = self._make_manager(channels=[MagicMock()])
        mgr.position_mismatch([])
        # Should not crash — dedup check handles it


# ── Convenience Method Tests (synchronous via _send_to_all) ─────────


class TestConvenienceMethods:
    def _make_manager(self):
        from core.alerting import AlertManager
        with patch("core.alerting.Config") as mock_cfg:
            mock_cfg.ALERTING_ENABLED = False
            mgr = AlertManager()
        mock_ch = MagicMock()
        mgr.channels = [mock_ch]
        return mgr, mock_ch

    def _sync_alert(self, mgr):
        """Patch threading to run send synchronously."""
        import core.alerting as alerting_mod
        original_alert = mgr.alert

        def sync_alert(event_type, message, level, data=None):
            """Override alert to run _send_to_all synchronously."""
            if not mgr.channels:
                return False
            event_key = f"{event_type}:{message}"
            with mgr._lock:
                if mgr._is_duplicate(event_key):
                    return False
                if mgr._is_rate_limited():
                    return False
                now = time.time()
                mgr._dedup_cache[event_key] = now
                mgr._rate_window.append(now)
            mgr._send_to_all(message, level, data)
            return True

        mgr.alert = sync_alert
        return mgr

    def test_trade_executed(self):
        mgr, ch = self._make_manager()
        self._sync_alert(mgr)
        mgr.trade_executed("TSLA", "buy", 1500.0, 150.0, "momentum breakout")
        ch.send.assert_called_once()
        msg = ch.send.call_args[0][0]
        assert "BUY" in msg
        assert "TSLA" in msg
        assert "$1500.00" in msg

    def test_stop_loss_hit(self):
        from core.alerting import AlertLevel
        mgr, ch = self._make_manager()
        self._sync_alert(mgr)
        mgr.stop_loss_hit("AAPL", 150.0, 147.0, 0.02)
        ch.send.assert_called_once()
        msg = ch.send.call_args[0][0]
        level = ch.send.call_args[0][1]
        assert "AAPL" in msg
        assert level == AlertLevel.WARNING

    def test_drawdown_halt(self):
        from core.alerting import AlertLevel
        mgr, ch = self._make_manager()
        self._sync_alert(mgr)
        mgr.drawdown_halt(0.12, 0.10)
        ch.send.assert_called_once()
        msg = ch.send.call_args[0][0]
        level = ch.send.call_args[0][1]
        assert "HALTED" in msg
        assert level == AlertLevel.CRITICAL

    def test_agent_error(self):
        from core.alerting import AlertLevel
        mgr, ch = self._make_manager()
        self._sync_alert(mgr)
        mgr.agent_error("llm_analyst", "API key expired")
        ch.send.assert_called_once()
        level = ch.send.call_args[0][1]
        assert level == AlertLevel.CRITICAL

    def test_position_mismatch(self):
        mgr, ch = self._make_manager()
        self._sync_alert(mgr)
        mismatches = [
            {"symbol": "TSLA", "issue": "phantom_position", "details": "Bot tracks, broker doesn't"},
            {"symbol": "AAPL", "issue": "untracked_position", "details": "Broker has, bot doesn't"},
        ]
        mgr.position_mismatch(mismatches)
        ch.send.assert_called_once()
        msg = ch.send.call_args[0][0]
        assert "2 issues" in msg
        assert "TSLA" in msg

    def test_config_reloaded(self):
        from core.alerting import AlertLevel
        mgr, ch = self._make_manager()
        self._sync_alert(mgr)
        mgr.config_reloaded(["STOP_LOSS_PCT", "MAX_POSITION_PCT"])
        ch.send.assert_called_once()
        level = ch.send.call_args[0][1]
        assert level == AlertLevel.INFO
