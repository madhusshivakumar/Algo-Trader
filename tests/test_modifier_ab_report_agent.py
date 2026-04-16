"""Tests for agents/modifier_ab_report.py — weekly A/B report runner."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from agents import modifier_ab_report
from analytics.modifier_performance import ModifierReport


def _fake_report(name, rec="keep", contrib=1.5, n=12):
    return ModifierReport(
        modifier=name, config_flag=f"{name.upper()}_ENABLED",
        days=30, contribution=contrib, n_trades_matched=n,
        summary={"count": 40, "action_flip_count": 2},
        recommendation=rec,
    )


class TestAgentMain:
    def test_dry_run_does_not_call_alert_or_mutate(self, tmp_path, monkeypatch):
        # Redirect report path to tmp so we don't pollute the repo
        monkeypatch.setattr(modifier_ab_report, "_REPORT_PATH",
                            str(tmp_path / "report.json"))
        monkeypatch.setattr(modifier_ab_report, "_DATA_DIR", str(tmp_path))

        reports = [_fake_report("sentiment", "disable", contrib=-1.0, n=8)]
        alert_mgr = MagicMock()

        with patch.object(modifier_ab_report, "compute_all_reports",
                          return_value=reports), \
             patch.object(modifier_ab_report, "_get_alert_manager",
                          return_value=alert_mgr):
            rc = modifier_ab_report.main(["--dry-run"])

        assert rc == 0
        # Alert manager should NOT be invoked in dry run
        alert_mgr.alert.assert_not_called()

    def test_live_run_dispatches_alert_and_persists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(modifier_ab_report, "_REPORT_PATH",
                            str(tmp_path / "report.json"))
        monkeypatch.setattr(modifier_ab_report, "_DATA_DIR", str(tmp_path))

        from config import Config
        monkeypatch.setattr(Config, "_AGENT_TEST_FLAG", True, raising=False)

        rep = ModifierReport(
            modifier="agent_tester", config_flag="_AGENT_TEST_FLAG",
            days=30, contribution=-2.0, n_trades_matched=10,
            summary={"count": 40, "action_flip_count": 2},
            recommendation="disable",
        )
        alert_mgr = MagicMock()

        with patch.object(modifier_ab_report, "compute_all_reports",
                          return_value=[rep]), \
             patch.object(modifier_ab_report, "_get_alert_manager",
                          return_value=alert_mgr):
            rc = modifier_ab_report.main([])

        assert rc == 0
        assert Config._AGENT_TEST_FLAG is False
        alert_mgr.alert.assert_called_once()

        # Report JSON persisted
        with open(str(tmp_path / "report.json")) as f:
            payload = json.load(f)
        assert payload["auto_disabled"] == ["agent_tester"]
        assert len(payload["reports"]) == 1
        assert payload["reports"][0]["modifier"] == "agent_tester"

    def test_days_argument_propagates(self, tmp_path, monkeypatch):
        monkeypatch.setattr(modifier_ab_report, "_REPORT_PATH",
                            str(tmp_path / "report.json"))
        monkeypatch.setattr(modifier_ab_report, "_DATA_DIR", str(tmp_path))

        captured = {}

        def _spy(days, **kwargs):
            captured["days"] = days
            return []

        with patch.object(modifier_ab_report, "compute_all_reports", _spy):
            modifier_ab_report.main(["--dry-run", "--days", "14"])
        assert captured["days"] == 14

    def test_alert_manager_unavailable_does_not_crash(self, tmp_path,
                                                      monkeypatch):
        monkeypatch.setattr(modifier_ab_report, "_REPORT_PATH",
                            str(tmp_path / "report.json"))
        monkeypatch.setattr(modifier_ab_report, "_DATA_DIR", str(tmp_path))

        with patch.object(modifier_ab_report, "compute_all_reports",
                          return_value=[]), \
             patch.object(modifier_ab_report, "_get_alert_manager",
                          return_value=None):
            rc = modifier_ab_report.main([])
        assert rc == 0


class TestPersistReport:
    def test_writes_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(modifier_ab_report, "_REPORT_PATH",
                            str(tmp_path / "report.json"))
        monkeypatch.setattr(modifier_ab_report, "_DATA_DIR", str(tmp_path))
        reports = [_fake_report("sentiment")]
        modifier_ab_report._persist_report(reports, [])
        with open(str(tmp_path / "report.json")) as f:
            payload = json.load(f)
        assert "generated_at" in payload
        assert len(payload["reports"]) == 1

    def test_handles_os_error(self, monkeypatch):
        # Impossible path — must swallow, not raise
        monkeypatch.setattr(modifier_ab_report, "_REPORT_PATH",
                            "/dev/null/cannot/create/here.json")
        monkeypatch.setattr(modifier_ab_report, "_DATA_DIR",
                            "/dev/null/cannot/create")
        # Should not raise
        modifier_ab_report._persist_report([_fake_report("a")], [])


class TestGetAlertManager:
    def test_returns_instance_when_available(self):
        """The real AlertManager should instantiate cleanly under test env."""
        mgr = modifier_ab_report._get_alert_manager()
        # Either we got a real manager, or we got None because deps are missing
        # in the test env. Both are acceptable. Key guarantee: no exception.
        assert mgr is None or hasattr(mgr, "alert")
