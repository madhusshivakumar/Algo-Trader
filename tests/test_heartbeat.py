"""Tests for core/heartbeat.py and scripts/check_heartbeat.py — Sprint 8."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from core import heartbeat as hb


class TestHeartbeatPath:
    def test_default_path_resolves(self):
        p = hb.heartbeat_path()
        assert p.endswith("heartbeat.json")

    def test_env_override(self, monkeypatch, tmp_path):
        custom = str(tmp_path / "custom_hb.json")
        monkeypatch.setenv("HEARTBEAT_PATH", custom)
        assert hb.heartbeat_path() == custom


class TestWriteHeartbeat:
    def test_writes_valid_json(self, tmp_path, monkeypatch):
        path = str(tmp_path / "hb.json")
        monkeypatch.setenv("HEARTBEAT_PATH", path)
        ok = hb.write_heartbeat(
            cycle_count=42, positions_evaluated=5, signals_produced=2,
            equity=10_000.50, halted=False,
        )
        assert ok is True
        with open(path) as f:
            data = json.load(f)
        assert data["cycle_count"] == 42
        assert data["positions_evaluated"] == 5
        assert data["signals_produced"] == 2
        assert data["equity"] == 10_000.50
        assert data["halted"] is False
        assert "ts" in data

    def test_atomic_write_no_partial_file(self, tmp_path, monkeypatch):
        """Simulate partial-write: tmp file should be cleaned on failure."""
        path = str(tmp_path / "hb.json")
        monkeypatch.setenv("HEARTBEAT_PATH", path)

        # Force os.replace to fail
        with patch("os.replace", side_effect=OSError("disk full")):
            ok = hb.write_heartbeat(cycle_count=1)
        assert ok is False
        # No temp files left behind
        tmps = [f for f in os.listdir(tmp_path)
                if f.startswith(".heartbeat.")]
        assert tmps == []

    def test_extra_fields_included(self, tmp_path, monkeypatch):
        path = str(tmp_path / "hb.json")
        monkeypatch.setenv("HEARTBEAT_PATH", path)
        hb.write_heartbeat(cycle_count=1, extra={"regime": "normal"})
        with open(path) as f:
            data = json.load(f)
        assert data["extra"] == {"regime": "normal"}

    def test_write_failure_returns_false(self, monkeypatch):
        # Unwritable path → graceful False, not exception
        monkeypatch.setenv("HEARTBEAT_PATH",
                           "/dev/null/cannot/create/here.json")
        assert hb.write_heartbeat(cycle_count=1) is False

    def test_equity_none_serializes(self, tmp_path, monkeypatch):
        path = str(tmp_path / "hb.json")
        monkeypatch.setenv("HEARTBEAT_PATH", path)
        hb.write_heartbeat(cycle_count=1, equity=None)
        with open(path) as f:
            data = json.load(f)
        assert data["equity"] is None


class TestReadHeartbeat:
    def test_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEARTBEAT_PATH",
                           str(tmp_path / "nope.json"))
        assert hb.read_heartbeat() is None

    def test_malformed_returns_none(self, tmp_path, monkeypatch):
        path = tmp_path / "hb.json"
        path.write_text("{not valid")
        monkeypatch.setenv("HEARTBEAT_PATH", str(path))
        assert hb.read_heartbeat() is None

    def test_roundtrip(self, tmp_path, monkeypatch):
        path = str(tmp_path / "hb.json")
        monkeypatch.setenv("HEARTBEAT_PATH", path)
        hb.write_heartbeat(cycle_count=99)
        out = hb.read_heartbeat()
        assert out is not None
        assert out["cycle_count"] == 99


class TestSecondsSince:
    def test_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEARTBEAT_PATH",
                           str(tmp_path / "nope.json"))
        assert hb.seconds_since_heartbeat() is None

    def test_fresh_returns_small_value(self, tmp_path, monkeypatch):
        path = str(tmp_path / "hb.json")
        monkeypatch.setenv("HEARTBEAT_PATH", path)
        hb.write_heartbeat(cycle_count=1)
        elapsed = hb.seconds_since_heartbeat()
        assert elapsed is not None
        assert 0 <= elapsed < 5  # just wrote

    def test_stale_returns_large_value(self, tmp_path, monkeypatch):
        """Inject an old timestamp into a heartbeat and confirm the delta."""
        path = str(tmp_path / "hb.json")
        monkeypatch.setenv("HEARTBEAT_PATH", path)
        old = (datetime.now() - timedelta(minutes=30)).isoformat(
            timespec="seconds")
        with open(path, "w") as f:
            json.dump({"ts": old, "cycle_count": 1}, f)
        elapsed = hb.seconds_since_heartbeat()
        assert elapsed is not None
        assert elapsed >= 1790  # ~30 min

    def test_bad_timestamp_returns_none(self, tmp_path, monkeypatch):
        path = str(tmp_path / "hb.json")
        monkeypatch.setenv("HEARTBEAT_PATH", path)
        with open(path, "w") as f:
            json.dump({"ts": "not-a-timestamp"}, f)
        assert hb.seconds_since_heartbeat() is None

    def test_naive_iso_interpreted_as_utc(self, tmp_path, monkeypatch):
        """Regression for the Apr 21 bug: container-UTC vs host-PT mismatch.

        An old heartbeat with a naive timestamp must be interpreted as UTC
        so the host's local-time clock doesn't compute a negative delta.
        """
        path = str(tmp_path / "hb.json")
        monkeypatch.setenv("HEARTBEAT_PATH", path)
        # Write a naive ISO string representing 30 min ago in UTC
        from datetime import datetime, timezone, timedelta
        ts_naive = (datetime.now(timezone.utc)
                    - timedelta(minutes=30)).replace(tzinfo=None)
        with open(path, "w") as f:
            json.dump({"ts": ts_naive.isoformat(timespec="seconds")}, f)
        elapsed = hb.seconds_since_heartbeat()
        assert elapsed is not None
        # Should be ~1800s. Must NOT be negative.
        assert elapsed > 0
        assert 1700 <= elapsed <= 1900

    def test_ts_epoch_preferred_over_iso(self, tmp_path, monkeypatch):
        """ts_epoch is the preferred field; use it even if ts is weird."""
        path = str(tmp_path / "hb.json")
        monkeypatch.setenv("HEARTBEAT_PATH", path)
        # Weird iso but correct epoch — numeric field must win
        with open(path, "w") as f:
            json.dump({"ts": "bogus-string",
                       "ts_epoch": int(time.time()) - 600}, f)
        elapsed = hb.seconds_since_heartbeat()
        assert elapsed is not None
        assert 590 <= elapsed <= 610


# ── CLI tests ─────────────────────────────────────────────────────────

class TestCheckHeartbeatCLI:
    def _run(self, monkeypatch, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "check_hb",
            os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "scripts", "check_heartbeat.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_missing_file_exits_1_and_alerts(self, tmp_path, monkeypatch,
                                             capsys):
        monkeypatch.setenv("HEARTBEAT_PATH",
                           str(tmp_path / "missing.json"))
        mod = self._run(monkeypatch, tmp_path)
        fake_alert = MagicMock()
        with patch.object(mod, "_fire_alert", fake_alert):
            rc = mod.main(["--silent"])
        assert rc == 1
        assert fake_alert.called
        assert fake_alert.call_args.kwargs["event_type"] == "heartbeat_missing"

    def test_stale_heartbeat_alerts(self, tmp_path, monkeypatch):
        path = str(tmp_path / "hb.json")
        monkeypatch.setenv("HEARTBEAT_PATH", path)
        old = (datetime.now() - timedelta(minutes=10)).isoformat(
            timespec="seconds")
        with open(path, "w") as f:
            json.dump({"ts": old, "cycle_count": 5,
                       "halted": False, "equity": 1000.0}, f)
        mod = self._run(monkeypatch, tmp_path)
        fake_alert = MagicMock()
        with patch.object(mod, "_fire_alert", fake_alert):
            rc = mod.main(["--silent", "--stale-seconds", "300"])
        assert rc == 1
        assert fake_alert.call_args.kwargs["event_type"] == "heartbeat_stale"

    def test_fresh_heartbeat_exits_0(self, tmp_path, monkeypatch):
        path = str(tmp_path / "hb.json")
        monkeypatch.setenv("HEARTBEAT_PATH", path)
        hb.write_heartbeat(cycle_count=10, equity=1000)
        mod = self._run(monkeypatch, tmp_path)
        fake_alert = MagicMock()
        with patch.object(mod, "_fire_alert", fake_alert):
            rc = mod.main(["--silent"])
        assert rc == 0
        assert not fake_alert.called

    def test_halted_but_fresh_exits_2(self, tmp_path, monkeypatch):
        path = str(tmp_path / "hb.json")
        monkeypatch.setenv("HEARTBEAT_PATH", path)
        hb.write_heartbeat(cycle_count=10, equity=1000, halted=True)
        mod = self._run(monkeypatch, tmp_path)
        fake_alert = MagicMock()
        with patch.object(mod, "_fire_alert", fake_alert):
            rc = mod.main(["--silent"])
        assert rc == 2
        assert fake_alert.call_args.kwargs["event_type"] == "engine_halted"

    def test_malformed_exits_1(self, tmp_path, monkeypatch):
        path = tmp_path / "hb.json"
        path.write_text("{not valid")
        monkeypatch.setenv("HEARTBEAT_PATH", str(path))
        mod = self._run(monkeypatch, tmp_path)
        fake_alert = MagicMock()
        with patch.object(mod, "_fire_alert", fake_alert):
            rc = mod.main(["--silent"])
        # Missing heartbeat (file is unreadable) — treated as missing
        assert rc == 1
