"""Tests for core/rl_strategy_selector.py — DQN strategy selection."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _make_df(n=60):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "close": close,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "volume": np.random.randint(1000, 5000, n),
    })


class TestRLStrategySelector:
    def test_no_model_file(self, tmp_path):
        from core.rl_strategy_selector import RLStrategySelector
        selector = RLStrategySelector(str(tmp_path / "nonexistent.zip"))
        assert not selector.is_ready()

    def test_select_strategy_not_ready(self, tmp_path):
        from core.rl_strategy_selector import RLStrategySelector
        selector = RLStrategySelector(str(tmp_path / "nonexistent.zip"))
        result = selector.select_strategy(_make_df())
        assert result is None

    @patch("core.rl_strategy_selector.os.path.exists", return_value=True)
    def test_loads_model(self, mock_exists):
        from core.rl_strategy_selector import RLStrategySelector
        mock_model = MagicMock()

        mock_dqn = MagicMock()
        mock_dqn.load.return_value = mock_model
        mock_sb3 = MagicMock()
        mock_sb3.DQN = mock_dqn

        with patch.dict("sys.modules", {"stable_baselines3": mock_sb3}):
            selector = RLStrategySelector("test.zip")
        assert selector.is_ready()

    @patch("core.rl_strategy_selector.os.path.exists", return_value=True)
    def test_select_returns_strategy(self, mock_exists):
        from core.rl_strategy_selector import RLStrategySelector
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(3), None)

        mock_dqn = MagicMock()
        mock_dqn.load.return_value = mock_model
        mock_sb3 = MagicMock()
        mock_sb3.DQN = mock_dqn

        with patch.dict("sys.modules", {"stable_baselines3": mock_sb3}):
            selector = RLStrategySelector("test.zip")
            result = selector.select_strategy(_make_df())

        from core.rl_features import STRATEGY_KEYS
        assert result == STRATEGY_KEYS[3]

    @patch("core.rl_strategy_selector.os.path.exists", return_value=True)
    def test_invalid_action_returns_none(self, mock_exists):
        from core.rl_strategy_selector import RLStrategySelector
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(99), None)

        mock_dqn = MagicMock()
        mock_dqn.load.return_value = mock_model
        mock_sb3 = MagicMock()
        mock_sb3.DQN = mock_dqn

        with patch.dict("sys.modules", {"stable_baselines3": mock_sb3}):
            selector = RLStrategySelector("test.zip")
            result = selector.select_strategy(_make_df())

        assert result is None

    def test_insufficient_data(self, tmp_path):
        from core.rl_strategy_selector import RLStrategySelector
        selector = RLStrategySelector.__new__(RLStrategySelector)
        selector.model = MagicMock()

        df = _make_df(10)  # Too short
        result = selector.select_strategy(df)
        assert result is None

    @patch("core.rl_strategy_selector.os.path.exists", return_value=True)
    def test_prediction_error(self, mock_exists):
        from core.rl_strategy_selector import RLStrategySelector
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("prediction error")

        mock_dqn = MagicMock()
        mock_dqn.load.return_value = mock_model
        mock_sb3 = MagicMock()
        mock_sb3.DQN = mock_dqn

        with patch.dict("sys.modules", {"stable_baselines3": mock_sb3}):
            selector = RLStrategySelector("test.zip")
            result = selector.select_strategy(_make_df())

        assert result is None


class TestModuleLevelFunctions:
    def test_select_strategy_disabled(self):
        from core.rl_strategy_selector import select_strategy
        import core.rl_strategy_selector as mod
        mod._selector = None

        with patch("core.rl_strategy_selector.Config") as mock_config:
            mock_config.RL_STRATEGY_ENABLED = False
            result = select_strategy(_make_df())
        assert result is None

    def test_get_selector_creates_when_enabled(self):
        from core.rl_strategy_selector import get_selector
        import core.rl_strategy_selector as mod
        mod._selector = None

        with patch("core.rl_strategy_selector.Config") as mock_config:
            mock_config.RL_STRATEGY_ENABLED = True
            mock_config.RL_MODEL_PATH = "/nonexistent/model.zip"
            selector = get_selector()
        assert selector is not None
        mod._selector = None  # cleanup
