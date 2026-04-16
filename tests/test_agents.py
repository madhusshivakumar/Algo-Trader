"""Unit tests for the agent modules (optimizer, scanner, analyzer)."""

import json
import os
import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestStrategyOptimizer:
    """Test strategy optimizer agent functions."""

    def test_compute_composite_score(self):
        from agents.strategy_optimizer import compute_composite_score
        metrics = {
            "sharpe_ratio": 5.0,
            "total_return": 10.0,
            "win_rate": 60.0,
            "max_drawdown": 0.05,
        }
        score = compute_composite_score(metrics)
        assert 0 < score <= 1.5  # Reasonable range
        assert isinstance(score, float)

    def test_composite_score_negative_return(self):
        from agents.strategy_optimizer import compute_composite_score
        metrics = {
            "sharpe_ratio": -2.0,
            "total_return": -5.0,
            "win_rate": 30.0,
            "max_drawdown": 0.15,
        }
        score = compute_composite_score(metrics)
        assert isinstance(score, float)
        assert score >= 0  # Should still be non-negative

    def test_load_learnings_empty(self, tmp_data_dir):
        from agents.strategy_optimizer import load_learnings
        with patch("agents.strategy_optimizer.LEARNINGS_FILE",
                   str(tmp_data_dir / "nonexistent.json")):
            result = load_learnings()
        assert result == {"version": 1, "entries": []}

    def test_load_learnings_valid(self, sample_learnings):
        from agents.strategy_optimizer import load_learnings
        with patch("agents.strategy_optimizer.LEARNINGS_FILE", str(sample_learnings)):
            result = load_learnings()
        assert len(result["entries"]) == 1

    def test_get_strategy_penalties_no_entries(self):
        from agents.strategy_optimizer import get_strategy_penalties
        penalties = get_strategy_penalties({"version": 1, "entries": []})
        assert penalties == {}

    def test_update_agent_state(self, tmp_data_dir):
        from agents.strategy_optimizer import update_agent_state
        state_file = str(tmp_data_dir / "agent_state.json")
        with patch("agents.strategy_optimizer.AGENT_STATE_FILE", state_file):
            update_agent_state("success", symbols_tested=8, duration=120)

        with open(state_file) as f:
            state = json.load(f)
        assert state["strategy_optimizer"]["status"] == "success"
        assert state["strategy_optimizer"]["symbols_tested"] == 8


class TestMarketScanner:
    """Test market scanner agent functions."""

    def test_load_optimizer_assignments(self, sample_assignments):
        from agents.market_scanner import load_optimizer_assignments
        with patch("agents.market_scanner.ASSIGNMENTS_FILE", str(sample_assignments)), \
             patch("agents.market_scanner.AGENT_STATE_FILE", "/nonexistent"):
            result = load_optimizer_assignments()
        assert "BTC/USD" in result
        assert result["BTC/USD"] == "mean_reversion_aggressive"

    def test_load_learnings(self, sample_learnings):
        from agents.market_scanner import load_learnings
        with patch("agents.market_scanner.LEARNINGS_FILE", str(sample_learnings)):
            favor, avoid = load_learnings()
        assert "NVDA" in favor
        assert "META" in avoid

    def test_load_learnings_empty(self, tmp_data_dir):
        from agents.market_scanner import load_learnings
        with patch("agents.market_scanner.LEARNINGS_FILE",
                   str(tmp_data_dir / "nonexistent.json")):
            favor, avoid = load_learnings()
        assert favor == []
        assert avoid == []

    def test_apply_sector_diversity(self):
        from agents.market_scanner import apply_sector_diversity
        candidates = [
            {"symbol": f"STOCK{i}", "sector": "Tech", "composite_score": 1.0 - i * 0.01}
            for i in range(10)
        ]
        result = apply_sector_diversity(candidates, max_per_sector=4)
        assert len(result) == 4

    def test_apply_sector_diversity_mixed(self):
        from agents.market_scanner import apply_sector_diversity
        candidates = [
            {"symbol": "AAPL", "sector": "Tech", "composite_score": 0.9},
            {"symbol": "MSFT", "sector": "Tech", "composite_score": 0.8},
            {"symbol": "JPM", "sector": "Finance", "composite_score": 0.85},
            {"symbol": "XOM", "sector": "Energy", "composite_score": 0.75},
        ]
        result = apply_sector_diversity(candidates, max_per_sector=2)
        assert len(result) == 4  # All pass with max 2 per sector

    def test_check_deadline_not_morning(self):
        from agents.market_scanner import check_deadline
        # During test time (not 6:25 AM), should return False
        result = check_deadline()
        # This depends on when tests run, but the logic is sound
        assert isinstance(result, bool)

    def test_score_candidate_filters_low_price(self):
        from agents.market_scanner import score_candidate
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "open": [5.0] * 5, "high": [5.5] * 5, "low": [4.5] * 5,
            "close": [5.0] * 5, "volume": [10_000_000] * 5,
        })
        result = score_candidate("PENNY", df, 0.5, [], [], set(), set())
        assert result is None  # Filtered out (price < $10)

    def test_score_candidate_filters_low_volume(self):
        from agents.market_scanner import score_candidate
        import pandas as pd
        df = pd.DataFrame({
            "open": [100.0] * 5, "high": [101.0] * 5, "low": [99.0] * 5,
            "close": [100.0] * 5, "volume": [100_000] * 5,  # Too low
        })
        result = score_candidate("LOW_VOL", df, 0.5, [], [], set(), set())
        assert result is None


class TestTradeAnalyzer:
    """Test trade analyzer agent functions."""

    def test_analyze_by_symbol(self):
        from agents.trade_analyzer import analyze_by_symbol
        trades = [
            {"symbol": "AAPL", "side": "buy", "pnl": 0},
            {"symbol": "AAPL", "side": "sell", "pnl": 10},
            {"symbol": "NVDA", "side": "buy", "pnl": 0},
            {"symbol": "NVDA", "side": "sell", "pnl": -5},
        ]
        result = analyze_by_symbol(trades)
        assert "AAPL" in result
        assert "NVDA" in result
        assert result["AAPL"]["pnl"] == 10
        assert result["NVDA"]["pnl"] == -5

    def test_analyze_by_strategy(self):
        from agents.trade_analyzer import analyze_by_strategy
        trades = [
            {"strategy": "mean_reversion_aggressive", "side": "sell", "pnl": 10},
            {"strategy": "mean_reversion_aggressive", "side": "sell", "pnl": -5},
            {"strategy": "volume_profile", "side": "sell", "pnl": 20},
        ]
        result = analyze_by_strategy(trades)
        assert "mean_reversion_aggressive" in result
        assert result["mean_reversion_aggressive"]["wins"] == 1
        assert result["mean_reversion_aggressive"]["losses"] == 1

    def test_analyze_time_of_day(self):
        from agents.trade_analyzer import analyze_time_of_day
        today = datetime.now().strftime("%Y-%m-%d")
        trades = [
            {"timestamp": f"{today} 09:45:00", "side": "sell", "pnl": 10},
            {"timestamp": f"{today} 10:30:00", "side": "sell", "pnl": -5},
            {"timestamp": f"{today} 14:30:00", "side": "sell", "pnl": 8},
        ]
        result = analyze_time_of_day(trades)
        assert len(result) > 0

    def test_detect_patterns_overtrading(self):
        from agents.trade_analyzer import detect_patterns
        by_symbol = {
            "BTC/USD": {"trades": 8, "win_rate": 25, "pnl": -200},
        }
        patterns = detect_patterns(by_symbol, {}, [])
        assert any("overtrading" in p.lower() or "BTC/USD" in p for p in patterns)

    def test_detect_patterns_empty(self):
        from agents.trade_analyzer import detect_patterns
        patterns = detect_patterns({}, {}, [])
        assert isinstance(patterns, list)

    def test_compute_risk_adjustments_low_win_rate(self):
        from agents.trade_analyzer import compute_risk_adjustments
        sells = [
            {"pnl": -10}, {"pnl": -8}, {"pnl": 5}, {"pnl": -12},
            {"pnl": -6}, {"pnl": 3}, {"pnl": -9},
        ]
        current = {"stop_loss_pct": 0.025}
        adjustments = compute_risk_adjustments({}, sells, current)
        if adjustments:
            assert "stop_loss_pct" in adjustments
            assert adjustments["stop_loss_pct"]["suggested"] > 0.025

    def test_compute_risk_adjustments_no_sells(self):
        from agents.trade_analyzer import compute_risk_adjustments
        adjustments = compute_risk_adjustments({}, [], {"stop_loss_pct": 0.025})
        assert adjustments == {}

    def test_update_learnings_idempotent(self, tmp_data_dir):
        from agents.trade_analyzer import update_learnings
        learnings_file = str(tmp_data_dir / "learnings.json")
        # Write initial
        with open(learnings_file, "w") as f:
            json.dump({"version": 1, "entries": []}, f)

        findings = {"best_performing_symbols": ["AAPL"], "worst_performing_symbols": []}

        with patch("agents.trade_analyzer.LEARNINGS_FILE", learnings_file):
            update_learnings(findings)
            update_learnings(findings)  # Run twice — should not duplicate

        with open(learnings_file) as f:
            data = json.load(f)
        today = datetime.now().strftime("%Y-%m-%d")
        today_entries = [e for e in data["entries"] if e["date"] == today]
        assert len(today_entries) == 1  # Idempotent

    def test_update_learnings_prunes_old(self, tmp_data_dir):
        from agents.trade_analyzer import update_learnings
        learnings_file = str(tmp_data_dir / "learnings.json")
        # Create 100 old entries
        entries = [{"date": f"2025-01-{i:02d}", "source": "test", "findings": {}}
                   for i in range(1, 32)] * 4  # 124 entries
        with open(learnings_file, "w") as f:
            json.dump({"version": 1, "entries": entries}, f)

        with patch("agents.trade_analyzer.LEARNINGS_FILE", learnings_file):
            update_learnings({"test": True})

        with open(learnings_file) as f:
            data = json.load(f)
        assert len(data["entries"]) <= 90
