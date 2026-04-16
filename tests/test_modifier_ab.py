"""Tests for analytics/modifier_ab.py — signal modifier A/B logging.

Sprint 6D. Covers:
    - log_delta: JSONL append, flag gating, path override, error swallowing
    - read_deltas: iteration, since-filter, malformed-line skip, missing-file
    - read_deltas_for_modifier: 30-day default window, modifier filter
    - summarize_deltas: count, noops, flips, mean/abs mean, unique symbols
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from analytics import modifier_ab


@pytest.fixture
def tmp_log(tmp_path):
    """Return a fresh log path inside pytest's tmp_path."""
    return str(tmp_path / "ab_log.jsonl")


def _read_lines(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── log_delta ────────────────────────────────────────────────────────────

class TestLogDelta:
    def test_writes_record(self, tmp_log):
        before = {"action": "buy", "strength": 0.5}
        after = {"action": "buy", "strength": 0.7}
        ok = modifier_ab.log_delta("AAPL", "sentiment", before, after,
                                   log_path=tmp_log)
        assert ok is True
        assert os.path.exists(tmp_log)
        records = _read_lines(tmp_log)
        assert len(records) == 1
        r = records[0]
        assert r["symbol"] == "AAPL"
        assert r["modifier"] == "sentiment"
        assert r["before"]["action"] == "buy"
        assert r["before"]["strength"] == 0.5
        assert r["after"]["strength"] == 0.7
        assert r["delta"] == pytest.approx(0.2, abs=1e-4)
        assert r["action_changed"] is False
        assert "ts" in r

    def test_action_flip_recorded(self, tmp_log):
        before = {"action": "buy", "strength": 0.5}
        after = {"action": "hold", "strength": 0.0}
        modifier_ab.log_delta("AAPL", "earnings_blackout",
                              before, after, log_path=tmp_log)
        records = _read_lines(tmp_log)
        assert records[0]["action_changed"] is True
        assert records[0]["delta"] == pytest.approx(-0.5, abs=1e-4)

    def test_respects_disabled_flag(self, tmp_log):
        with patch("analytics.modifier_ab.Config") as mock_cfg:
            mock_cfg.MODIFIER_AB_ENABLED = False
            ok = modifier_ab.log_delta("AAPL", "sentiment",
                                       {"action": "buy", "strength": 0.5},
                                       {"action": "buy", "strength": 0.6},
                                       log_path=tmp_log)
        assert ok is False
        assert not os.path.exists(tmp_log)

    def test_respects_enabled_flag_missing_attr(self, tmp_log):
        """Default should be True when attr missing (getattr default)."""
        class FakeCfg:
            pass
        with patch("analytics.modifier_ab.Config", FakeCfg):
            ok = modifier_ab.log_delta("AAPL", "sentiment",
                                       {"action": "buy", "strength": 0.5},
                                       {"action": "buy", "strength": 0.6},
                                       log_path=tmp_log)
        assert ok is True

    def test_swallows_disk_error(self):
        # Point to a path under a non-existent directory whose parent we
        # cannot create (conflicting file).
        ok = modifier_ab.log_delta(
            "AAPL", "sentiment",
            {"action": "buy", "strength": 0.5},
            {"action": "buy", "strength": 0.6},
            log_path="/dev/null/impossible/path.jsonl",
        )
        assert ok is False

    def test_timestamp_override(self, tmp_log):
        ts = datetime(2026, 4, 1, 10, 0, 0)
        modifier_ab.log_delta("AAPL", "sentiment",
                              {"action": "buy", "strength": 0.5},
                              {"action": "buy", "strength": 0.6},
                              timestamp=ts, log_path=tmp_log)
        records = _read_lines(tmp_log)
        assert records[0]["ts"] == "2026-04-01T10:00:00"

    def test_appends_multiple(self, tmp_log):
        for i in range(3):
            modifier_ab.log_delta(
                f"SYM{i}", "llm",
                {"action": "buy", "strength": 0.4},
                {"action": "buy", "strength": 0.4 + i * 0.05},
                log_path=tmp_log,
            )
        assert len(_read_lines(tmp_log)) == 3

    def test_numeric_coercion(self, tmp_log):
        # Signal may contain strength as int or None
        modifier_ab.log_delta(
            "AAPL", "sentiment",
            {"action": "buy", "strength": 1},
            {"action": "buy", "strength": None},
            log_path=tmp_log,
        )
        r = _read_lines(tmp_log)[0]
        assert r["before"]["strength"] == 1.0
        assert r["after"]["strength"] == 0.0


# ── read_deltas ──────────────────────────────────────────────────────────

class TestReadDeltas:
    def test_returns_empty_when_missing(self, tmp_path):
        path = str(tmp_path / "missing.jsonl")
        assert list(modifier_ab.read_deltas(log_path=path)) == []

    def test_yields_all_records(self, tmp_log):
        modifier_ab.log_delta("A", "sentiment",
                              {"action": "buy", "strength": 0.4},
                              {"action": "buy", "strength": 0.5},
                              log_path=tmp_log)
        modifier_ab.log_delta("B", "llm",
                              {"action": "sell", "strength": 0.6},
                              {"action": "sell", "strength": 0.5},
                              log_path=tmp_log)
        records = list(modifier_ab.read_deltas(log_path=tmp_log))
        assert len(records) == 2

    def test_since_filter(self, tmp_log):
        old_ts = datetime(2020, 1, 1)
        new_ts = datetime(2026, 4, 1)
        modifier_ab.log_delta("A", "sentiment",
                              {"action": "buy", "strength": 0.4},
                              {"action": "buy", "strength": 0.5},
                              timestamp=old_ts, log_path=tmp_log)
        modifier_ab.log_delta("B", "sentiment",
                              {"action": "buy", "strength": 0.4},
                              {"action": "buy", "strength": 0.6},
                              timestamp=new_ts, log_path=tmp_log)
        records = list(modifier_ab.read_deltas(
            since=datetime(2026, 1, 1), log_path=tmp_log))
        assert len(records) == 1
        assert records[0]["symbol"] == "B"

    def test_skips_malformed_lines(self, tmp_log):
        # Write one bad line and one good line
        with open(tmp_log, "w") as f:
            f.write("not a json line\n")
            f.write(json.dumps({"ts": "2026-04-01T10:00:00",
                                "symbol": "X", "modifier": "sentiment",
                                "before": {"action": "buy", "strength": 0.4},
                                "after": {"action": "buy", "strength": 0.5},
                                "delta": 0.1, "action_changed": False}) + "\n")
            f.write("\n")  # blank line
        records = list(modifier_ab.read_deltas(log_path=tmp_log))
        assert len(records) == 1
        assert records[0]["symbol"] == "X"


# ── read_deltas_for_modifier ─────────────────────────────────────────────

class TestReadDeltasForModifier:
    def test_filters_by_modifier(self, tmp_log):
        modifier_ab.log_delta("A", "sentiment",
                              {"action": "buy", "strength": 0.4},
                              {"action": "buy", "strength": 0.5},
                              log_path=tmp_log)
        modifier_ab.log_delta("B", "llm",
                              {"action": "sell", "strength": 0.6},
                              {"action": "sell", "strength": 0.5},
                              log_path=tmp_log)
        sent = modifier_ab.read_deltas_for_modifier("sentiment",
                                                    log_path=tmp_log)
        assert len(sent) == 1
        assert sent[0]["symbol"] == "A"

    def test_respects_days_window(self, tmp_log):
        old = datetime.now() - timedelta(days=100)
        recent = datetime.now() - timedelta(days=1)
        modifier_ab.log_delta("A", "sentiment",
                              {"action": "buy", "strength": 0.4},
                              {"action": "buy", "strength": 0.5},
                              timestamp=old, log_path=tmp_log)
        modifier_ab.log_delta("B", "sentiment",
                              {"action": "buy", "strength": 0.4},
                              {"action": "buy", "strength": 0.5},
                              timestamp=recent, log_path=tmp_log)
        res = modifier_ab.read_deltas_for_modifier("sentiment", days=30,
                                                   log_path=tmp_log)
        assert len(res) == 1
        assert res[0]["symbol"] == "B"


# ── summarize_deltas ─────────────────────────────────────────────────────

class TestSummarizeDeltas:
    def test_empty(self):
        out = modifier_ab.summarize_deltas([])
        assert out["count"] == 0
        assert out["noop_count"] == 0
        assert out["action_flip_count"] == 0
        assert out["mean_delta"] == 0.0
        assert out["abs_mean_delta"] == 0.0
        assert out["unique_symbols"] == 0

    def test_aggregates(self):
        records = [
            {"symbol": "A", "delta": 0.1, "action_changed": False},
            {"symbol": "A", "delta": -0.05, "action_changed": False},
            {"symbol": "B", "delta": 0.0, "action_changed": True},
            {"symbol": "B", "delta": 0.0, "action_changed": False},
        ]
        s = modifier_ab.summarize_deltas(records)
        assert s["count"] == 4
        assert s["noop_count"] == 2  # deltas == 0
        assert s["action_flip_count"] == 1
        assert s["mean_delta"] == pytest.approx(0.0125, abs=1e-4)
        assert s["abs_mean_delta"] == pytest.approx(0.0375, abs=1e-4)
        assert s["unique_symbols"] == 2

    def test_bad_delta_treated_as_zero(self):
        """Corrupt / missing deltas must sum to zero, not crash the report."""
        records = [
            {"symbol": "A", "delta": "oops", "action_changed": False},
            {"symbol": "B", "delta": None, "action_changed": False},
            {"symbol": "C", "delta": 0.1, "action_changed": False},
        ]
        s = modifier_ab.summarize_deltas(records)
        assert s["count"] == 3
        # Only the 0.1 contributes → mean = 0.1/3
        assert s["mean_delta"] == pytest.approx(0.1 / 3, abs=1e-4)
        # noops: the two coerced-to-zero entries
        assert s["noop_count"] == 2
        assert s["unique_symbols"] == 3
