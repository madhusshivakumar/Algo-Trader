"""Sprint 7: tests for monitoring modules (structured log, Sentry, Prometheus)."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest


# ── utils/structured_log.py ────────────────────────────────────────────────

class TestStructuredLog:
    def test_slog_imports_without_error(self):
        from utils.structured_log import slog
        assert slog is not None

    def test_new_trace_unique(self):
        from utils.structured_log import new_trace
        a, b = new_trace(), new_trace()
        assert a != b
        assert len(a) == 12

    def test_slog_info_does_not_raise(self):
        from utils.structured_log import slog
        slog.info("test event", key="value")

    def test_file_writer_disabled_by_default(self, tmp_path, monkeypatch):
        """When STRUCTURED_LOG is not true, the file writer is a no-op."""
        from utils import structured_log as sl
        monkeypatch.setattr(sl, "_ENABLED", False)
        writer = sl._JSONLFileWriter(str(tmp_path / "test.jsonl"))
        writer.write({"event": "test"})
        assert not (tmp_path / "test.jsonl").exists()

    def test_file_writer_writes_when_enabled(self, tmp_path, monkeypatch):
        from utils import structured_log as sl
        monkeypatch.setattr(sl, "_ENABLED", True)
        writer = sl._JSONLFileWriter(str(tmp_path / "test.jsonl"))
        writer.write({"event": "hello"})
        with open(tmp_path / "test.jsonl") as f:
            line = json.loads(f.readline())
        assert line["event"] == "hello"

    def test_file_writer_swallows_os_error(self, monkeypatch):
        from utils import structured_log as sl
        monkeypatch.setattr(sl, "_ENABLED", True)
        writer = sl._JSONLFileWriter("/dev/null/impossible/path.jsonl")
        writer.write({"event": "nope"})  # should not raise


# ── monitoring/sentry.py ───────────────────────────────────────────────────

class TestSentry:
    def test_init_sentry_noop_without_dsn(self):
        from monitoring.sentry import init_sentry, _initialized
        # Reset state
        import monitoring.sentry as ms
        ms._initialized = False
        assert init_sentry("") is False
        assert init_sentry() is False

    def test_is_initialized_returns_bool(self):
        from monitoring.sentry import is_initialized
        assert isinstance(is_initialized(), bool)


# ── monitoring/metrics.py (disabled path) ──────────────────────────────────

class TestMetricsDisabled:
    """All metric stubs must silently accept calls and return stubs."""

    def test_equity_stub_set(self):
        from monitoring.metrics import EQUITY
        EQUITY.set(999.0)  # must not raise

    def test_trades_stub_labels_and_inc(self):
        from monitoring.metrics import TRADES
        TRADES.labels("buy", "mr", "TSLA").inc()

    def test_cycle_duration_stub_observe(self):
        from monitoring.metrics import CYCLE_DURATION
        CYCLE_DURATION.observe(1.5)

    def test_broker_latency_stub_labels_observe(self):
        from monitoring.metrics import BROKER_LATENCY
        BROKER_LATENCY.labels("get_positions").observe(0.2)

    def test_start_metrics_server_noop(self):
        from monitoring.metrics import start_metrics_server
        assert start_metrics_server() is False
