"""Tests for core/dashboard_views.py — Sprint 6H."""

from __future__ import annotations

import json
import os

import pytest

from core.dashboard_views import (
    VALID_VIEWS,
    build_view_payload,
    protection_status,
    read_latest_alerts,
    read_modifier_report,
)
from core import dashboard_views


# ── protection_status ──────────────────────────────────────────────────────

class TestProtectionStatus:
    def test_green_when_no_issues(self):
        s = protection_status()
        assert s["level"] == "green"
        assert "healthy" in s["label"].lower()

    def test_red_when_halted(self):
        s = protection_status(halted=True)
        assert s["level"] == "red"

    def test_yellow_on_high_drawdown(self):
        s = protection_status(daily_dd_pct=-4.5)
        assert s["level"] == "yellow"
        assert "-4.5" in s["label"]

    def test_yellow_on_degraded_symbols(self):
        s = protection_status(drift_degraded_symbols=["TSLA", "NVDA"])
        assert s["level"] == "yellow"
        assert "2 symbols" in s["label"]

    def test_halted_takes_precedence(self):
        s = protection_status(halted=True, daily_dd_pct=-99.0,
                              drift_degraded_symbols=["SPY"])
        assert s["level"] == "red"

    def test_mild_drawdown_stays_green(self):
        s = protection_status(daily_dd_pct=-1.5)
        assert s["level"] == "green"

    def test_single_degraded_symbol_singular_text(self):
        s = protection_status(drift_degraded_symbols=["AAPL"])
        assert "1 symbol " in s["label"]


# ── build_view_payload (Simple) ───────────────────────────────────────────

class TestSimpleView:
    def test_default_is_simple(self):
        p = build_view_payload("simple")
        assert p["view"] == "simple"

    def test_invalid_view_falls_back_to_simple(self):
        p = build_view_payload("xlarge")
        assert p["view"] == "simple"

    def test_contains_balance_and_pnl(self):
        p = build_view_payload(
            "simple",
            account={"equity": 9800, "cash": 5000},
            stats={"total_pnl": -200, "win_rate": 55.0},
        )
        assert p["balance"] == 9800
        assert p["total_pnl"] == -200
        assert p["win_rate"] == 55.0

    def test_last_5_trades_only(self):
        trades = [{"symbol": f"T{i}", "side": "buy", "amount": 100,
                    "timestamp": f"2026-04-14T0{i}:00"}
                   for i in range(8)]
        p = build_view_payload("simple", trades=trades)
        assert len(p["recent_trades"]) == 5

    def test_trade_fields_trimmed(self):
        trades = [{"symbol": "TSLA", "side": "buy", "amount": 100,
                    "price": 120.5, "pnl": 3.2, "strategy": "mr",
                    "explanation": "Price dipped below average",
                    "timestamp": "2026-04-14T10:00"}]
        p = build_view_payload("simple", trades=trades)
        t = p["recent_trades"][0]
        assert "explanation" in t
        assert "price" not in t  # stripped for Simple
        assert "strategy" not in t

    def test_protection_badge_included(self):
        p = build_view_payload("simple", halted=True)
        assert p["protection"]["level"] == "red"

    def test_framing_passthrough(self):
        framing = {"sharpe_line": "Backtest Sharpe 1.20",
                    "disclaimer_line": "IMPORTANT: ..."}
        p = build_view_payload("simple", framing=framing)
        assert p["framing"]["sharpe_line"] == "Backtest Sharpe 1.20"

    def test_regime_passthrough(self):
        p = build_view_payload("simple", regime="high_vol")
        assert p["regime"] == "high_vol"

    def test_no_positions_in_simple(self):
        p = build_view_payload(
            "simple",
            account={"positions": [{"symbol": "TSLA"}]},
        )
        assert "positions" not in p

    def test_no_equity_curve_in_simple(self):
        p = build_view_payload("simple",
                               equity_curve=[{"equity": 10000}])
        assert "equity_curve" not in p


# ── build_view_payload (Standard) ─────────────────────────────────────────

class TestStandardView:
    def test_includes_positions(self):
        p = build_view_payload(
            "standard",
            account={"positions": [{"symbol": "TSLA"}]},
        )
        assert len(p["positions"]) == 1

    def test_includes_equity_curve(self):
        p = build_view_payload("standard",
                               equity_curve=[{"equity": 10000}])
        assert len(p["equity_curve"]) == 1

    def test_up_to_50_trades(self):
        trades = [{"symbol": f"T{i}", "side": "buy"}
                   for i in range(60)]
        p = build_view_payload("standard", trades=trades)
        assert len(p["recent_trades"]) == 50

    def test_includes_win_loss_stats(self):
        p = build_view_payload(
            "standard",
            stats={"wins": 10, "losses": 5, "avg_win": 2.5,
                    "avg_loss": -1.0, "total_trades": 15},
        )
        assert p["wins"] == 10
        assert p["losses"] == 5


# ── build_view_payload (Advanced) ─────────────────────────────────────────

class TestAdvancedView:
    def test_includes_raw_stats(self):
        raw = {"sharpe": 1.2, "sortino": 1.5}
        p = build_view_payload("advanced", stats=raw)
        assert p["stats_raw"] == raw

    def test_includes_alerts(self, tmp_path, monkeypatch):
        log = tmp_path / "log.jsonl"
        log.write_text('{"level":"warning","msg":"drawdown"}\n'
                       '{"level":"error","msg":"broker fail"}\n')
        monkeypatch.setattr(dashboard_views, "_ALERTS_DIR", str(tmp_path))
        p = build_view_payload("advanced")
        assert len(p["alerts"]) == 2

    def test_includes_modifier_report(self, tmp_path, monkeypatch):
        rpt = tmp_path / "report.json"
        rpt.write_text(json.dumps({"reports": [{"modifier": "sentiment"}]}))
        monkeypatch.setattr(dashboard_views, "_MODIFIER_REPORT",
                            str(rpt))
        p = build_view_payload("advanced")
        assert p["modifier_report"]["reports"][0]["modifier"] == "sentiment"


# ── read_latest_alerts ────────────────────────────────────────────────────

class TestReadAlerts:
    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dashboard_views, "_ALERTS_DIR", str(tmp_path))
        assert read_latest_alerts() == []

    def test_reads_lines_newest_first(self, tmp_path, monkeypatch):
        log = tmp_path / "log.jsonl"
        log.write_text('{"n":1}\n{"n":2}\n{"n":3}\n')
        monkeypatch.setattr(dashboard_views, "_ALERTS_DIR", str(tmp_path))
        out = read_latest_alerts(limit=10)
        assert out[0]["n"] == 3  # newest first

    def test_respects_limit(self, tmp_path, monkeypatch):
        log = tmp_path / "log.jsonl"
        log.write_text("\n".join(json.dumps({"n": i}) for i in range(100)))
        monkeypatch.setattr(dashboard_views, "_ALERTS_DIR", str(tmp_path))
        assert len(read_latest_alerts(limit=5)) == 5

    def test_skips_malformed_lines(self, tmp_path, monkeypatch):
        log = tmp_path / "log.jsonl"
        log.write_text('{"ok":1}\nnot json\n{"ok":2}\n')
        monkeypatch.setattr(dashboard_views, "_ALERTS_DIR", str(tmp_path))
        out = read_latest_alerts()
        assert len(out) == 2

    def test_os_error_returns_empty(self, monkeypatch):
        monkeypatch.setattr(dashboard_views, "_ALERTS_DIR",
                            "/dev/null/nope")
        assert read_latest_alerts() == []


# ── read_modifier_report ──────────────────────────────────────────────────

class TestReadModifierReport:
    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dashboard_views, "_MODIFIER_REPORT",
                            str(tmp_path / "nope.json"))
        assert read_modifier_report() is None

    def test_reads_valid_json(self, tmp_path, monkeypatch):
        rpt = tmp_path / "report.json"
        rpt.write_text('{"reports": []}')
        monkeypatch.setattr(dashboard_views, "_MODIFIER_REPORT",
                            str(rpt))
        assert read_modifier_report() == {"reports": []}

    def test_malformed_json_returns_none(self, tmp_path, monkeypatch):
        rpt = tmp_path / "report.json"
        rpt.write_text("{bad")
        monkeypatch.setattr(dashboard_views, "_MODIFIER_REPORT",
                            str(rpt))
        assert read_modifier_report() is None


# ── Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_nones_does_not_crash(self):
        p = build_view_payload("advanced")
        assert p["view"] == "advanced"
        assert p["recent_trades"] == []
        assert p["protection"]["level"] == "green"

    def test_valid_views_constant_covers_all(self):
        assert set(VALID_VIEWS) == {"simple", "standard", "advanced"}
