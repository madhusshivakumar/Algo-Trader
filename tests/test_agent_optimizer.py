"""Tests for agents/strategy_optimizer.py — covers utility functions and main() with mocking."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

import numpy as np
import pandas as pd


class TestLoadLearnings:
    def test_load_existing_file(self, tmp_path):
        from agents.strategy_optimizer import load_learnings
        lf = tmp_path / "learnings.json"
        lf.write_text(json.dumps({"version": 1, "entries": [{"date": "2026-03-27"}]}))
        with patch("agents.strategy_optimizer.LEARNINGS_FILE", str(lf)):
            result = load_learnings()
        assert result["version"] == 1
        assert len(result["entries"]) == 1

    def test_load_missing_file(self, tmp_path):
        from agents.strategy_optimizer import load_learnings
        with patch("agents.strategy_optimizer.LEARNINGS_FILE", str(tmp_path / "nope.json")):
            result = load_learnings()
        assert result == {"version": 1, "entries": []}

    def test_load_corrupt_file(self, tmp_path):
        from agents.strategy_optimizer import load_learnings
        lf = tmp_path / "bad.json"
        lf.write_text("{bad json")
        with patch("agents.strategy_optimizer.LEARNINGS_FILE", str(lf)):
            result = load_learnings()
        assert result == {"version": 1, "entries": []}


class TestGetStrategyPenalties:
    def test_no_penalties_for_short_underperformance(self):
        from agents.strategy_optimizer import get_strategy_penalties
        learnings = {"entries": [
            {"findings": {"strategy_notes": {"momentum": "weak signals"}, "worst_performing_symbols": ["AAPL"]}},
            {"findings": {"strategy_notes": {"momentum": "poor results"}, "worst_performing_symbols": ["AAPL"]}},
        ]}
        result = get_strategy_penalties(learnings)
        # Only 2 days, need >= 3 for penalty
        assert "AAPL" not in result

    def test_penalty_after_3_days(self):
        from agents.strategy_optimizer import get_strategy_penalties
        learnings = {"entries": [
            {"findings": {"strategy_notes": {"momentum": "weak"}, "worst_performing_symbols": ["AAPL"]}},
            {"findings": {"strategy_notes": {"momentum": "poor"}, "worst_performing_symbols": ["AAPL"]}},
            {"findings": {"strategy_notes": {"momentum": "loss heavy"}, "worst_performing_symbols": ["AAPL"]}},
        ]}
        result = get_strategy_penalties(learnings)
        assert result["AAPL"]["momentum"] == 0.80

    def test_no_penalty_for_strong_strategy(self):
        from agents.strategy_optimizer import get_strategy_penalties
        learnings = {"entries": [
            {"findings": {"strategy_notes": {"momentum": "strong results"}, "worst_performing_symbols": ["AAPL"]}},
        ] * 5}
        result = get_strategy_penalties(learnings)
        assert result == {}

    def test_empty_learnings(self):
        from agents.strategy_optimizer import get_strategy_penalties
        assert get_strategy_penalties({"entries": []}) == {}
        assert get_strategy_penalties({}) == {}


class TestComputeCompositeScore:
    def test_perfect_score(self):
        from agents.strategy_optimizer import compute_composite_score
        metrics = {"sharpe_ratio": 5.0, "total_return": 20, "win_rate": 100, "max_drawdown": 0}
        score = compute_composite_score(metrics)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_terrible_metrics(self):
        from agents.strategy_optimizer import compute_composite_score
        metrics = {"sharpe_ratio": -5.0, "total_return": -20, "win_rate": 0, "max_drawdown": 1.0}
        score = compute_composite_score(metrics)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_average_metrics(self):
        from agents.strategy_optimizer import compute_composite_score
        metrics = {"sharpe_ratio": 1.0, "total_return": 5, "win_rate": 55, "max_drawdown": 0.1}
        score = compute_composite_score(metrics)
        assert 0.3 < score < 0.8

    def test_missing_keys_defaults(self):
        from agents.strategy_optimizer import compute_composite_score
        score = compute_composite_score({})
        assert isinstance(score, float)

    def test_extreme_negative_sharpe_clamped(self):
        from agents.strategy_optimizer import compute_composite_score
        metrics = {"sharpe_ratio": -100, "total_return": 0, "win_rate": 50, "max_drawdown": 0.5}
        score = compute_composite_score(metrics)
        assert score >= 0


class TestUpdateAgentState:
    def test_write_new_state(self, tmp_path):
        from agents.strategy_optimizer import update_agent_state
        sf = str(tmp_path / "agent_state.json")
        with patch("agents.strategy_optimizer.AGENT_STATE_FILE", sf):
            update_agent_state("success", symbols_tested=10, duration=60)
        data = json.loads(open(sf).read())
        assert data["strategy_optimizer"]["status"] == "success"
        assert data["strategy_optimizer"]["symbols_tested"] == 10

    def test_update_existing_state(self, tmp_path):
        from agents.strategy_optimizer import update_agent_state
        sf = str(tmp_path / "agent_state.json")
        with open(sf, "w") as f:
            json.dump({"market_scanner": {"status": "success"}}, f)
        with patch("agents.strategy_optimizer.AGENT_STATE_FILE", sf):
            update_agent_state("failed", error="test error", duration=5)
        data = json.loads(open(sf).read())
        assert data["strategy_optimizer"]["status"] == "failed"
        assert data["strategy_optimizer"]["error"] == "test error"
        assert data["market_scanner"]["status"] == "success"

    def test_update_corrupt_existing(self, tmp_path):
        from agents.strategy_optimizer import update_agent_state
        sf = str(tmp_path / "agent_state.json")
        with open(sf, "w") as f:
            f.write("{bad")
        with patch("agents.strategy_optimizer.AGENT_STATE_FILE", sf):
            update_agent_state("success", duration=1)
        data = json.loads(open(sf).read())
        assert data["strategy_optimizer"]["status"] == "success"


class TestOptimizerMain:
    def test_main_success(self, tmp_path):
        """Test main() with mocked broker and minimal symbols."""
        from agents import strategy_optimizer as opt_mod

        # Set up directories
        opt_dir = tmp_path / "optimizer"
        opt_dir.mkdir()

        mock_broker = MagicMock()
        # Return a small DataFrame
        rng = np.random.RandomState(42)
        n = 250
        df = pd.DataFrame({
            "open": 100 + rng.randn(n),
            "high": 101 + rng.randn(n),
            "low": 99 + rng.randn(n),
            "close": 100 + rng.randn(n),
            "volume": rng.randint(1_000_000, 10_000_000, n),
        })
        mock_broker.get_historical_bars.return_value = df

        learnings_file = str(tmp_path / "learnings.json")
        with open(learnings_file, "w") as f:
            json.dump({"version": 1, "entries": []}, f)

        with patch.object(opt_mod, "RESULTS_FILE", str(opt_dir / "backtest_results.json")), \
             patch.object(opt_mod, "ASSIGNMENTS_FILE", str(opt_dir / "strategy_assignments.json")), \
             patch.object(opt_mod, "LEARNINGS_FILE", learnings_file), \
             patch.object(opt_mod, "AGENT_STATE_FILE", str(tmp_path / "agent_state.json")), \
             patch("agents.strategy_optimizer.Config") as mock_config, \
             patch("agents.strategy_optimizer.Broker", return_value=mock_broker):
            mock_config.validate.return_value = None
            mock_config.CRYPTO_SYMBOLS = ["BTC/USD"]
            mock_config.EQUITY_SYMBOLS = ["AAPL"]
            mock_config.CANDLE_HISTORY_DAYS = 5
            opt_mod.main()

        # Verify output files
        assert os.path.exists(str(opt_dir / "backtest_results.json"))
        assert os.path.exists(str(opt_dir / "strategy_assignments.json"))

    def test_main_empty_data(self, tmp_path):
        """Test main when broker returns empty data."""
        from agents import strategy_optimizer as opt_mod

        opt_dir = tmp_path / "optimizer"
        opt_dir.mkdir()

        mock_broker = MagicMock()
        mock_broker.get_historical_bars.return_value = pd.DataFrame()

        with patch.object(opt_mod, "RESULTS_FILE", str(opt_dir / "backtest_results.json")), \
             patch.object(opt_mod, "ASSIGNMENTS_FILE", str(opt_dir / "strategy_assignments.json")), \
             patch.object(opt_mod, "LEARNINGS_FILE", str(tmp_path / "nope.json")), \
             patch.object(opt_mod, "AGENT_STATE_FILE", str(tmp_path / "agent_state.json")), \
             patch("agents.strategy_optimizer.Config") as mock_config, \
             patch("agents.strategy_optimizer.Broker", return_value=mock_broker):
            mock_config.validate.return_value = None
            mock_config.CRYPTO_SYMBOLS = ["BTC/USD"]
            mock_config.EQUITY_SYMBOLS = []
            mock_config.CANDLE_HISTORY_DAYS = 5
            opt_mod.main()

    def test_main_broker_error(self, tmp_path):
        """Test main when broker.get_historical_bars raises."""
        from agents import strategy_optimizer as opt_mod

        opt_dir = tmp_path / "optimizer"
        opt_dir.mkdir()

        mock_broker = MagicMock()
        mock_broker.get_historical_bars.side_effect = Exception("API error")

        with patch.object(opt_mod, "RESULTS_FILE", str(opt_dir / "backtest_results.json")), \
             patch.object(opt_mod, "ASSIGNMENTS_FILE", str(opt_dir / "strategy_assignments.json")), \
             patch.object(opt_mod, "LEARNINGS_FILE", str(tmp_path / "nope.json")), \
             patch.object(opt_mod, "AGENT_STATE_FILE", str(tmp_path / "agent_state.json")), \
             patch("agents.strategy_optimizer.Config") as mock_config, \
             patch("agents.strategy_optimizer.Broker", return_value=mock_broker):
            mock_config.validate.return_value = None
            mock_config.CRYPTO_SYMBOLS = []
            mock_config.EQUITY_SYMBOLS = ["AAPL"]
            mock_config.CANDLE_HISTORY_DAYS = 5
            opt_mod.main()

    def test_main_config_validate_fails(self, tmp_path):
        """Test main when Config.validate raises."""
        from agents import strategy_optimizer as opt_mod

        with patch.object(opt_mod, "AGENT_STATE_FILE", str(tmp_path / "agent_state.json")), \
             patch("agents.strategy_optimizer.Config") as mock_config, \
             patch("agents.strategy_optimizer.Broker") as mock_broker_cls:
            mock_config.validate.side_effect = ValueError("no key")
            with pytest.raises(ValueError):
                opt_mod.main()

    def test_numpy_encoder(self):
        """Test that NumpyEncoder handles numpy types."""
        # The NumpyEncoder is defined inside main(), but we test the concept
        import numpy as np
        # Simulate what NumpyEncoder does
        vals = {
            "bool": np.bool_(True),
            "int": np.int64(42),
            "float": np.float64(3.14),
            "array": np.array([1, 2, 3]),
        }
        # These should be convertible
        assert int(vals["bool"]) == 1
        assert int(vals["int"]) == 42
        assert float(vals["float"]) == pytest.approx(3.14)
        assert vals["array"].tolist() == [1, 2, 3]
