"""Integration tests — verify modules work together end-to-end."""

import json
import os
import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

import pandas as pd
import numpy as np


class TestRouterToStrategy:
    """Test that the router correctly dispatches to strategies."""

    def test_all_configured_symbols_produce_signals(self, flat_market):
        from strategies.router import STRATEGY_MAP, compute_signals
        for symbol in STRATEGY_MAP:
            signal = compute_signals(symbol, flat_market)
            assert signal["action"] in ("buy", "sell", "hold"), \
                f"Symbol {symbol} produced invalid action"
            assert "strategy" in signal

    def test_different_strategies_for_different_symbols(self, flat_market):
        from strategies.router import compute_signals
        btc_signal = compute_signals("BTC/USD", flat_market)
        eth_signal = compute_signals("ETH/USD", flat_market)
        # BTC uses Mean Rev Aggressive, ETH uses Volume Profile
        assert btc_signal["strategy"] != eth_signal["strategy"]


class TestBacktestToOptimizer:
    """Test that backtest output feeds into optimizer scoring."""

    def test_backtest_result_has_optimizer_fields(self, flat_market):
        from compare_strategies import backtest_strategy, STRATEGIES
        from agents.strategy_optimizer import compute_composite_score

        fn = list(STRATEGIES.values())[0]
        metrics = backtest_strategy(fn, flat_market)

        # Optimizer needs these specific fields
        score = compute_composite_score(metrics)
        assert isinstance(score, float)
        assert score >= 0

    def test_all_strategies_produce_scoreable_results(self, flat_market):
        from compare_strategies import backtest_strategy, STRATEGIES
        from agents.strategy_optimizer import compute_composite_score

        for name, fn in STRATEGIES.items():
            metrics = backtest_strategy(fn, flat_market)
            score = compute_composite_score(metrics)
            assert isinstance(score, float), f"{name} produced non-float score"


class TestOptimizerToScanner:
    """Test that optimizer output is consumable by scanner."""

    def test_scanner_reads_optimizer_output(self, sample_assignments):
        from agents.market_scanner import load_optimizer_assignments
        with patch("agents.market_scanner.ASSIGNMENTS_FILE", str(sample_assignments)), \
             patch("agents.market_scanner.AGENT_STATE_FILE", "/nonexistent"):
            assignments = load_optimizer_assignments()

        from strategies.router import STRATEGY_REGISTRY
        for sym, strat in assignments.items():
            assert strat in STRATEGY_REGISTRY, \
                f"Optimizer assigned unknown strategy '{strat}' to {sym}"


class TestAnalyzerToLearnings:
    """Test that analyzer output feeds back into scanner/optimizer."""

    def test_analyzer_learnings_readable_by_scanner(self, sample_learnings):
        from agents.market_scanner import load_learnings
        with patch("agents.market_scanner.LEARNINGS_FILE", str(sample_learnings)):
            favor, avoid = load_learnings()

        assert isinstance(favor, list)
        assert isinstance(avoid, list)

    def test_analyzer_learnings_readable_by_optimizer(self, sample_learnings):
        from agents.strategy_optimizer import load_learnings, get_strategy_penalties
        with patch("agents.strategy_optimizer.LEARNINGS_FILE", str(sample_learnings)):
            learnings = load_learnings()
        penalties = get_strategy_penalties(learnings)
        assert isinstance(penalties, dict)


class TestDBIntegration:
    """Test database reads/writes across modules."""

    def test_logger_writes_analyzer_reads(self, tmp_db):
        """Logger writes trades, analyzer can read them."""
        import sqlite3
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        trades = conn.execute("SELECT * FROM trades").fetchall()
        conn.close()

        trades = [dict(t) for t in trades]
        assert len(trades) == 8

        # Analyzer functions should work on these
        from agents.trade_analyzer import analyze_by_symbol, analyze_by_strategy
        by_sym = analyze_by_symbol(trades)
        by_strat = analyze_by_strategy(trades)

        assert "AAPL" in by_sym
        assert "BTC/USD" in by_sym
        assert len(by_strat) > 0

    def test_trade_pnl_sums_correctly(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        total = conn.execute("SELECT SUM(pnl) FROM trades WHERE side='sell'").fetchone()[0]
        conn.close()
        # 32.0 + (-45.50) + 75.75 + (-112.50) = -50.25
        assert abs(total - (-50.25)) < 0.01


class TestAgentStateCommunication:
    """Test that agents can read each other's state."""

    def test_all_agents_in_state_file(self, sample_agent_state):
        with open(sample_agent_state) as f:
            state = json.load(f)
        assert "strategy_optimizer" in state
        assert "market_scanner" in state
        assert "trade_analyzer" in state

    def test_state_has_required_fields(self, sample_agent_state):
        with open(sample_agent_state) as f:
            state = json.load(f)
        for agent_name, agent_state in state.items():
            assert "last_run" in agent_state, f"{agent_name} missing last_run"
            assert "status" in agent_state, f"{agent_name} missing status"
            assert agent_state["status"] in ("success", "failed", "pending", "aborted")


class TestFallbackConfig:
    """Test fallback config works when agent files are missing."""

    def test_router_loads_without_agent_files(self):
        """Router should still work if data/ files don't exist."""
        from strategies.router import STRATEGY_REGISTRY, DEFAULT_STRATEGY, get_strategy
        name, fn = get_strategy("SOME_NEW_SYMBOL")
        assert name is not None
        assert callable(fn)

    def test_fallback_config_valid(self):
        fallback_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "fallback_config.json"
        )
        if os.path.exists(fallback_path):
            with open(fallback_path) as f:
                data = json.load(f)
            assert "equity_symbols" in data
            assert "strategy_map" in data
            assert "risk_params" in data
            assert len(data["equity_symbols"]) >= 6


class TestEndToEndSignalFlow:
    """Full signal flow: market data → strategy → signal → validation."""

    def test_full_signal_pipeline(self, flat_market):
        from strategies.router import compute_signals
        from core.risk_manager import RiskManager

        rm = RiskManager()
        rm.initialize(100000)

        signal = compute_signals("AAPL", flat_market)

        # Signal should be actionable
        if signal["action"] == "buy":
            size = rm.calculate_position_size(100000) * signal["strength"]
            assert size > 0
            rm.register_entry("AAPL", 100.0)
            assert "AAPL" in rm.trailing_stops

        elif signal["action"] == "sell":
            assert signal["strength"] > 0

        assert rm.can_trade(100000) is True
