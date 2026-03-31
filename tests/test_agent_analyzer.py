"""Tests for agents/trade_analyzer.py — covers utility functions and main() with mocking."""

import json
import os
import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

import numpy as np


class TestGetTodaysTrades:
    def test_returns_trades(self, tmp_db):
        from agents.trade_analyzer import get_todays_trades
        with patch("agents.trade_analyzer.DB_PATH", tmp_db):
            trades = get_todays_trades()
        assert len(trades) == 8
        assert trades[0]["symbol"] == "AAPL"

    def test_no_db_returns_empty(self, tmp_path):
        from agents.trade_analyzer import get_todays_trades
        with patch("agents.trade_analyzer.DB_PATH", str(tmp_path / "nope.db")):
            assert get_todays_trades() == []


class TestGetEquitySnapshots:
    def test_returns_snapshots(self, tmp_db):
        from agents.trade_analyzer import get_equity_snapshots
        with patch("agents.trade_analyzer.DB_PATH", tmp_db):
            snaps = get_equity_snapshots()
        assert len(snaps) == 1
        assert snaps[0]["equity"] == 100000

    def test_no_db_returns_empty(self, tmp_path):
        from agents.trade_analyzer import get_equity_snapshots
        with patch("agents.trade_analyzer.DB_PATH", str(tmp_path / "nope.db")):
            assert get_equity_snapshots() == []


class TestAnalyzeBySymbol:
    def test_basic_analysis(self):
        from agents.trade_analyzer import analyze_by_symbol
        trades = [
            {"symbol": "AAPL", "side": "buy", "pnl": 0},
            {"symbol": "AAPL", "side": "sell", "pnl": 50},
            {"symbol": "AAPL", "side": "sell", "pnl": -20},
            {"symbol": "NVDA", "side": "buy", "pnl": 0},
            {"symbol": "NVDA", "side": "sell", "pnl": 100},
        ]
        result = analyze_by_symbol(trades)
        assert result["AAPL"]["trades"] == 3
        assert result["AAPL"]["wins"] == 1
        assert result["AAPL"]["losses"] == 1
        assert result["AAPL"]["pnl"] == 30
        assert result["NVDA"]["win_rate"] == 100.0

    def test_empty_trades(self):
        from agents.trade_analyzer import analyze_by_symbol
        assert analyze_by_symbol([]) == {}


class TestAnalyzeByStrategy:
    def test_basic_analysis(self):
        from agents.trade_analyzer import analyze_by_strategy
        trades = [
            {"strategy": "momentum", "side": "sell", "pnl": 50},
            {"strategy": "momentum", "side": "sell", "pnl": -10},
            {"strategy": "scalper", "side": "sell", "pnl": 20},
            {"strategy": "", "side": "buy", "pnl": 0},
        ]
        result = analyze_by_strategy(trades)
        assert result["momentum"]["wins"] == 1
        assert result["momentum"]["losses"] == 1
        assert result["scalper"]["pnl"] == 20
        assert "unknown" in result  # empty strategy -> "unknown"


class TestAnalyzeTimeOfDay:
    def test_time_buckets(self):
        from agents.trade_analyzer import analyze_time_of_day
        today = datetime.now().strftime("%Y-%m-%d")
        trades = [
            {"side": "sell", "timestamp": f"{today} 08:00:00", "pnl": 10},   # pre_market
            {"side": "sell", "timestamp": f"{today} 10:00:00", "pnl": -5},   # morning
            {"side": "sell", "timestamp": f"{today} 12:00:00", "pnl": 20},   # midday
            {"side": "sell", "timestamp": f"{today} 15:00:00", "pnl": -15},  # afternoon
            {"side": "sell", "timestamp": f"{today} 17:00:00", "pnl": 30},   # after_hours
            {"side": "buy", "timestamp": f"{today} 10:00:00", "pnl": 0},     # buy ignored
        ]
        result = analyze_time_of_day(trades)
        assert len(result) == 5
        assert result["pre_market (6:00-9:30)"]["pnl"] == 10
        assert result["after_hours (16:00+)"]["pnl"] == 30

    def test_invalid_timestamp(self):
        from agents.trade_analyzer import analyze_time_of_day
        trades = [{"side": "sell", "timestamp": "invalid", "pnl": 10}]
        result = analyze_time_of_day(trades)
        assert result == {}


class TestDetectPatterns:
    def test_overtrading_detected(self):
        from agents.trade_analyzer import detect_patterns
        by_symbol = {"AAPL": {"trades": 8, "win_rate": 30, "pnl": -50}}
        patterns = detect_patterns(by_symbol, {}, [])
        assert any("overtrading" in p for p in patterns)

    def test_consecutive_losses(self):
        from agents.trade_analyzer import detect_patterns
        trades = [
            {"side": "sell", "pnl": -10},
            {"side": "sell", "pnl": -5},
            {"side": "sell", "pnl": -20},
        ]
        patterns = detect_patterns({}, {}, trades)
        assert any("consecutive" in p.lower() for p in patterns)

    def test_underperforming_strategy(self):
        from agents.trade_analyzer import detect_patterns
        by_strategy = {"momentum": {"sells": 5, "win_rate": 20}}
        patterns = detect_patterns({}, by_strategy, [])
        assert any("underperforming" in p for p in patterns)

    def test_well_performing_strategy(self):
        from agents.trade_analyzer import detect_patterns
        by_strategy = {"scalper": {"sells": 5, "win_rate": 80}}
        patterns = detect_patterns({}, by_strategy, [])
        assert any("performing well" in p for p in patterns)

    def test_large_single_loss(self):
        from agents.trade_analyzer import detect_patterns
        trades = [{"side": "sell", "pnl": -100, "symbol": "TSLA", "reason": "stop loss"}]
        patterns = detect_patterns({}, {}, trades)
        assert any("Large loss" in p for p in patterns)

    def test_no_patterns(self):
        from agents.trade_analyzer import detect_patterns
        by_symbol = {"AAPL": {"trades": 2, "win_rate": 60, "pnl": 10}}
        by_strategy = {"momentum": {"sells": 2, "win_rate": 50}}
        trades = [{"side": "sell", "pnl": 5}]
        patterns = detect_patterns(by_symbol, by_strategy, trades)
        assert patterns == []


class TestComputeRiskAdjustments:
    def test_no_sells_no_adjustments(self):
        from agents.trade_analyzer import compute_risk_adjustments
        assert compute_risk_adjustments({}, [], {}) == {}

    def test_low_win_rate_widens_stop(self):
        from agents.trade_analyzer import compute_risk_adjustments
        sells = [{"pnl": -10}] * 4 + [{"pnl": 5}]  # 20% win rate
        result = compute_risk_adjustments({}, sells, {"stop_loss_pct": 0.025})
        assert "stop_loss_pct" in result
        assert result["stop_loss_pct"]["suggested"] > 0.025

    def test_high_win_rate_bad_ratio_tightens(self):
        from agents.trade_analyzer import compute_risk_adjustments
        sells = [{"pnl": 5}] * 4 + [{"pnl": -50}]  # 80% win rate, avg_loss > avg_win
        result = compute_risk_adjustments({}, sells, {"stop_loss_pct": 0.03})
        assert "stop_loss_pct" in result
        assert result["stop_loss_pct"]["suggested"] < 0.03

    def test_stop_loss_at_max_no_change(self):
        from agents.trade_analyzer import compute_risk_adjustments
        sells = [{"pnl": -10}] * 4 + [{"pnl": 5}]
        result = compute_risk_adjustments({}, sells, {"stop_loss_pct": 0.05})
        # At max, no change
        assert result == {}


class TestApplyRiskAdjustments:
    def test_apply_to_env(self, tmp_path):
        from agents.trade_analyzer import apply_risk_adjustments
        env = tmp_path / ".env"
        env.write_text("STOP_LOSS_PCT=0.025\nOTHER=val\n")
        adjustments = {
            "stop_loss_pct": {"current": 0.025, "suggested": 0.03, "reason": "test"},
        }
        with patch("agents.trade_analyzer.ENV_FILE", str(env)):
            apply_risk_adjustments(adjustments)
        content = env.read_text()
        assert "STOP_LOSS_PCT=0.03" in content

    def test_out_of_bounds_skipped(self, tmp_path):
        from agents.trade_analyzer import apply_risk_adjustments
        env = tmp_path / ".env"
        env.write_text("STOP_LOSS_PCT=0.025\n")
        adjustments = {
            "stop_loss_pct": {"current": 0.025, "suggested": 0.99, "reason": "test"},
        }
        with patch("agents.trade_analyzer.ENV_FILE", str(env)):
            apply_risk_adjustments(adjustments)
        content = env.read_text()
        assert "STOP_LOSS_PCT=0.025" in content

    def test_no_adjustments(self, tmp_path):
        from agents.trade_analyzer import apply_risk_adjustments
        apply_risk_adjustments({})  # Should not raise

    def test_add_new_param(self, tmp_path):
        from agents.trade_analyzer import apply_risk_adjustments
        env = tmp_path / ".env"
        env.write_text("OTHER=val\n")
        adjustments = {
            "stop_loss_pct": {"current": 0.025, "suggested": 0.03, "reason": "test"},
        }
        with patch("agents.trade_analyzer.ENV_FILE", str(env)):
            apply_risk_adjustments(adjustments)
        content = env.read_text()
        assert "STOP_LOSS_PCT=0.03" in content


class TestUpdateLearnings:
    def test_append_new_entry(self, tmp_path):
        from agents.trade_analyzer import update_learnings
        lf = str(tmp_path / "learnings.json")
        with open(lf, "w") as f:
            json.dump({"version": 1, "entries": []}, f)
        with patch("agents.trade_analyzer.LEARNINGS_FILE", lf):
            update_learnings({"best_performing_symbols": ["AAPL"]})
        data = json.loads(open(lf).read())
        assert len(data["entries"]) == 1

    def test_idempotent_same_day(self, tmp_path):
        from agents.trade_analyzer import update_learnings
        lf = str(tmp_path / "learnings.json")
        today = datetime.now().strftime("%Y-%m-%d")
        with open(lf, "w") as f:
            json.dump({"version": 1, "entries": [
                {"date": today, "source": "trade_analyzer", "findings": {"old": True}}
            ]}, f)
        with patch("agents.trade_analyzer.LEARNINGS_FILE", lf):
            update_learnings({"new": True})
        data = json.loads(open(lf).read())
        assert len(data["entries"]) == 1
        assert data["entries"][0]["findings"] == {"new": True}

    def test_prune_to_90(self, tmp_path):
        from agents.trade_analyzer import update_learnings
        lf = str(tmp_path / "learnings.json")
        entries = [{"date": f"2025-{i:02d}-01", "findings": {}} for i in range(1, 13)] * 10  # 120 entries
        with open(lf, "w") as f:
            json.dump({"version": 1, "entries": entries}, f)
        with patch("agents.trade_analyzer.LEARNINGS_FILE", lf):
            update_learnings({"test": True})
        data = json.loads(open(lf).read())
        assert len(data["entries"]) <= 90

    def test_missing_file_creates_new(self, tmp_path):
        from agents.trade_analyzer import update_learnings
        lf = str(tmp_path / "learnings.json")
        with patch("agents.trade_analyzer.LEARNINGS_FILE", lf):
            update_learnings({"test": True})
        assert os.path.exists(lf)


class TestUpdateFallbackConfig:
    def test_profitable_day_updates(self, tmp_path):
        from agents.trade_analyzer import update_fallback_config
        env = tmp_path / ".env"
        env.write_text("STOP_LOSS_PCT=0.03\nMAX_POSITION_PCT=0.40\n")
        ff = str(tmp_path / "fallback.json")
        mock_config = MagicMock()
        mock_config.EQUITY_SYMBOLS = ["AAPL"]
        mock_router = MagicMock(STRATEGY_MAP={"AAPL": "momentum"})
        with patch("agents.trade_analyzer.ENV_FILE", str(env)), \
             patch("agents.trade_analyzer.FALLBACK_FILE", ff), \
             patch.dict("sys.modules", {"config": MagicMock(Config=mock_config),
                                        "strategies.router": mock_router}):
            update_fallback_config(50.0)
        data = json.loads(open(ff).read())
        assert data["equity_symbols"] == ["AAPL"]
        assert data["risk_params"]["stop_loss_pct"] == 0.03

    def test_negative_pnl_no_update(self, tmp_path):
        from agents.trade_analyzer import update_fallback_config
        ff = str(tmp_path / "fallback.json")
        with patch("agents.trade_analyzer.FALLBACK_FILE", ff):
            update_fallback_config(-10.0)
        assert not os.path.exists(ff)


class TestAnalyzerUpdateAgentState:
    def test_write_state(self, tmp_path):
        from agents.trade_analyzer import update_agent_state
        sf = str(tmp_path / "agent_state.json")
        with patch("agents.trade_analyzer.AGENT_STATE_FILE", sf):
            update_agent_state("success", trades_analyzed=20, duration=5)
        data = json.loads(open(sf).read())
        assert data["trade_analyzer"]["status"] == "success"


class TestAnalyzerMain:
    def _setup_db(self, tmp_path, trades=None):
        """Helper to create a test DB."""
        db_path = str(tmp_path / "trades.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, symbol TEXT, side TEXT, amount REAL,
            price REAL, reason TEXT, pnl REAL DEFAULT 0, strategy TEXT DEFAULT ''
        )""")
        conn.execute("""CREATE TABLE equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, equity REAL, cash REAL
        )""")
        if trades:
            conn.executemany(
                "INSERT INTO trades (timestamp, symbol, side, amount, price, reason, pnl, strategy) "
                "VALUES (?,?,?,?,?,?,?,?)", trades
            )
        conn.commit()
        conn.close()
        return db_path

    def test_main_with_trades(self, tmp_path):
        from agents import trade_analyzer as ta_mod

        today = datetime.now().strftime("%Y-%m-%d")
        trades = [
            (f"{today} 09:35:00", "AAPL", "buy", 5000, 248, "buy signal", 0, "momentum"),
            (f"{today} 10:15:00", "AAPL", "sell", 5000, 250, "sell signal", 40, "momentum"),
            (f"{today} 10:30:00", "NVDA", "buy", 8000, 880, "buy signal", 0, "scalper"),
            (f"{today} 11:30:00", "NVDA", "sell", 8000, 875, "trailing stop", -45, "scalper"),
        ]
        db_path = self._setup_db(tmp_path, trades)
        reports_dir = str(tmp_path / "reports")
        env_file = tmp_path / ".env"
        env_file.write_text("STOP_LOSS_PCT=0.025\n")

        with patch.object(ta_mod, "DB_PATH", db_path), \
             patch.object(ta_mod, "REPORTS_DIR", reports_dir), \
             patch.object(ta_mod, "LEARNINGS_FILE", str(tmp_path / "learnings.json")), \
             patch.object(ta_mod, "FALLBACK_FILE", str(tmp_path / "fallback.json")), \
             patch.object(ta_mod, "AGENT_STATE_FILE", str(tmp_path / "agent_state.json")), \
             patch.object(ta_mod, "ENV_FILE", str(env_file)):
            ta_mod.main()

        assert os.path.exists(os.path.join(reports_dir, f"{today}.json"))

    def test_main_no_trades(self, tmp_path):
        from agents import trade_analyzer as ta_mod
        db_path = self._setup_db(tmp_path)

        with patch.object(ta_mod, "DB_PATH", db_path), \
             patch.object(ta_mod, "AGENT_STATE_FILE", str(tmp_path / "agent_state.json")):
            ta_mod.main()

    def test_main_no_db(self, tmp_path):
        from agents import trade_analyzer as ta_mod

        with patch.object(ta_mod, "DB_PATH", str(tmp_path / "nope.db")), \
             patch.object(ta_mod, "AGENT_STATE_FILE", str(tmp_path / "agent_state.json")):
            ta_mod.main()

    def test_main_with_many_trades_triggers_patterns(self, tmp_path):
        """Test main with enough trades to trigger overtrading and consecutive losses."""
        from agents import trade_analyzer as ta_mod

        today = datetime.now().strftime("%Y-%m-%d")
        # Create overtrading scenario: 8 trades for AAPL with low win rate
        trades = []
        for i in range(4):
            trades.append((f"{today} 09:{30+i}:00", "AAPL", "buy", 1000, 150, "buy", 0, "momentum"))
            trades.append((f"{today} 09:{30+i}:30", "AAPL", "sell", 1000, 148, "stop", -20, "momentum"))
        # Add winning trades for another symbol
        trades.append((f"{today} 10:00:00", "NVDA", "buy", 5000, 900, "buy", 0, "scalper"))
        trades.append((f"{today} 10:30:00", "NVDA", "sell", 5000, 910, "sell", 55, "scalper"))
        # Add a large loss
        trades.append((f"{today} 11:00:00", "TSLA", "buy", 3000, 250, "buy", 0, "macd_crossover"))
        trades.append((f"{today} 11:30:00", "TSLA", "sell", 3000, 230, "stop", -60, "macd_crossover"))

        db_path = self._setup_db(tmp_path, trades)
        reports_dir = str(tmp_path / "reports")
        env_file = tmp_path / ".env"
        env_file.write_text("STOP_LOSS_PCT=0.025\n")

        with patch.object(ta_mod, "DB_PATH", db_path), \
             patch.object(ta_mod, "REPORTS_DIR", reports_dir), \
             patch.object(ta_mod, "LEARNINGS_FILE", str(tmp_path / "learnings.json")), \
             patch.object(ta_mod, "FALLBACK_FILE", str(tmp_path / "fallback.json")), \
             patch.object(ta_mod, "AGENT_STATE_FILE", str(tmp_path / "agent_state.json")), \
             patch.object(ta_mod, "ENV_FILE", str(env_file)):
            ta_mod.main()

        report = json.loads(open(os.path.join(reports_dir, f"{today}.json")).read())
        assert len(report["patterns"]) > 0  # Should detect patterns

    def test_main_exception(self, tmp_path):
        """Test main when an exception occurs."""
        from agents import trade_analyzer as ta_mod

        with patch.object(ta_mod, "DB_PATH", str(tmp_path / "nope.db")), \
             patch.object(ta_mod, "AGENT_STATE_FILE", str(tmp_path / "agent_state.json")), \
             patch("agents.trade_analyzer.get_todays_trades", side_effect=RuntimeError("test")):
            with pytest.raises(RuntimeError):
                ta_mod.main()
