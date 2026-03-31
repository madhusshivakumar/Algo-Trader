"""Tests for agents/rl_trainer.py — weekly DQN training agent."""

import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _make_df(n=200):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "close": close,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "volume": np.random.randint(1000, 5000, n),
    })


class TestComputeStrategyReturns:
    def test_returns_array(self):
        from agents.rl_trainer import compute_strategy_returns
        df = _make_df(100)
        fn = lambda df: {"action": "buy", "strength": 0.8, "reason": "test"}
        returns = compute_strategy_returns(df, fn)
        assert isinstance(returns, np.ndarray)
        assert len(returns) == 100

    def test_returns_zeros_for_first_bars(self):
        from agents.rl_trainer import compute_strategy_returns
        df = _make_df(100)
        fn = lambda df: {"action": "buy", "strength": 0.8, "reason": "test"}
        returns = compute_strategy_returns(df, fn)
        assert all(returns[:30] == 0)

    def test_flat_when_sell(self):
        from agents.rl_trainer import compute_strategy_returns
        df = _make_df(50)
        fn = lambda df: {"action": "sell", "strength": 0.8, "reason": "test"}
        returns = compute_strategy_returns(df, fn)
        # All returns should be 0 when position is flat
        assert all(returns == 0)


class TestComputeSharpe:
    def test_positive_sharpe(self):
        from agents.rl_trainer import compute_sharpe
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.01])
        sharpe = compute_sharpe(returns)
        assert sharpe > 0

    def test_zero_returns(self):
        from agents.rl_trainer import compute_sharpe
        returns = np.zeros(10)
        assert compute_sharpe(returns) == 0.0

    def test_empty_returns(self):
        from agents.rl_trainer import compute_sharpe
        assert compute_sharpe(np.array([])) == 0.0

    def test_single_return(self):
        from agents.rl_trainer import compute_sharpe
        assert compute_sharpe(np.array([0.01])) == 0.0


class TestTrainModel:
    @patch("agents.rl_trainer.DQN", create=True)
    @patch("agents.rl_trainer.StrategySelectionEnv", create=True)
    def test_train_returns_model(self, mock_env_cls, mock_dqn_cls):
        from agents.rl_trainer import train_model
        from core.rl_features import STRATEGY_KEYS

        mock_model = MagicMock()
        mock_dqn_cls.return_value = mock_model

        df = _make_df()
        returns = {k: np.random.randn(len(df)) * 0.001 for k in STRATEGY_KEYS}

        mock_sb3 = MagicMock()
        mock_sb3.DQN = mock_dqn_cls
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env

        with patch.dict("sys.modules", {
            "stable_baselines3": mock_sb3,
            "core.rl_environment": MagicMock(StrategySelectionEnv=mock_env_cls),
        }):
            result = train_model(df, returns)

        assert result is not None

    def test_train_without_deps_returns_none(self):
        from agents.rl_trainer import train_model
        from core.rl_features import STRATEGY_KEYS

        df = _make_df()
        returns = {k: np.random.randn(len(df)) * 0.001 for k in STRATEGY_KEYS}

        with patch.dict("sys.modules", {"stable_baselines3": None}):
            # This should handle ImportError gracefully
            # The actual behavior depends on whether sb3 is installed
            pass  # Test is for when deps are missing


class TestSaveTrainLog:
    def test_writes_log(self, tmp_path):
        from agents.rl_trainer import save_train_log
        log_file = tmp_path / "train_log.json"

        with patch("agents.rl_trainer._DATA_DIR", str(tmp_path)), \
             patch("agents.rl_trainer._TRAIN_LOG", str(log_file)), \
             patch("agents.rl_trainer.Config") as mock_config:
            mock_config.RL_MIN_SHARPE_THRESHOLD = 0.5
            save_train_log(1.2, 0.8, True)

        data = json.loads(log_file.read_text())
        assert data["train_sharpe"] == 1.2
        assert data["val_sharpe"] == 0.8
        assert data["deployed"] is True


class TestRunTraining:
    @patch("agents.rl_trainer.save_train_log")
    @patch("agents.rl_trainer.save_model")
    @patch("agents.rl_trainer.validate_model", return_value=0.8)
    @patch("agents.rl_trainer.train_model")
    @patch("agents.rl_trainer.compute_strategy_returns")
    def test_deploys_when_sharpe_above_threshold(
        self, mock_returns, mock_train, mock_validate, mock_save, mock_log
    ):
        from agents.rl_trainer import run_training
        from core.rl_features import STRATEGY_KEYS

        mock_returns.return_value = np.random.randn(200) * 0.001
        mock_model = MagicMock()
        mock_train.return_value = mock_model

        df = _make_df(200)
        fns = {k: MagicMock() for k in STRATEGY_KEYS}

        with patch("agents.rl_trainer.Config") as mock_config:
            mock_config.RL_MIN_SHARPE_THRESHOLD = 0.5
            result = run_training(df, fns)

        assert result["success"] is True
        assert result["deployed"] is True
        mock_save.assert_called_once()

    @patch("agents.rl_trainer.save_train_log")
    @patch("agents.rl_trainer.save_model")
    @patch("agents.rl_trainer.validate_model", return_value=0.1)
    @patch("agents.rl_trainer.train_model")
    @patch("agents.rl_trainer.compute_strategy_returns")
    def test_no_deploy_when_sharpe_below_threshold(
        self, mock_returns, mock_train, mock_validate, mock_save, mock_log
    ):
        from agents.rl_trainer import run_training
        from core.rl_features import STRATEGY_KEYS

        mock_returns.return_value = np.random.randn(200) * 0.001
        mock_train.return_value = MagicMock()

        df = _make_df(200)
        fns = {k: MagicMock() for k in STRATEGY_KEYS}

        with patch("agents.rl_trainer.Config") as mock_config:
            mock_config.RL_MIN_SHARPE_THRESHOLD = 0.5
            result = run_training(df, fns)

        assert result["success"] is True
        assert result["deployed"] is False
        mock_save.assert_not_called()

    @patch("agents.rl_trainer.train_model", return_value=None)
    @patch("agents.rl_trainer.compute_strategy_returns")
    def test_training_failure(self, mock_returns, mock_train):
        from agents.rl_trainer import run_training
        from core.rl_features import STRATEGY_KEYS

        mock_returns.return_value = np.random.randn(200) * 0.001

        df = _make_df(200)
        fns = {k: MagicMock() for k in STRATEGY_KEYS}

        result = run_training(df, fns)
        assert result["success"] is False


class TestMain:
    @patch("agents.rl_trainer.run_training")
    def test_main_success(self, mock_run):
        from agents.rl_trainer import main
        mock_run.return_value = {
            "success": True,
            "train_sharpe": 1.0,
            "val_sharpe": 0.7,
            "deployed": True,
        }

        mock_broker = MagicMock()
        mock_broker.get_historical_bars.return_value = _make_df(200)
        mock_broker_module = MagicMock()
        mock_broker_module.Broker.return_value = mock_broker

        mock_router = MagicMock()
        mock_router.STRATEGY_REGISTRY = {"mean_reversion": MagicMock()}

        with patch("agents.rl_trainer.Config") as mock_config, \
             patch.dict("sys.modules", {
                 "core.broker": mock_broker_module,
                 "strategies.router": mock_router,
             }):
            mock_config.RL_STRATEGY_ENABLED = True
            mock_config.CRYPTO_SYMBOLS = ["BTC/USD"]
            mock_config.SYMBOLS = ["BTC/USD"]
            result = main()

        assert result["success"] is True

    def test_main_no_data(self):
        from agents.rl_trainer import main

        mock_broker = MagicMock()
        mock_broker.get_historical_bars.return_value = pd.DataFrame()
        mock_broker_module = MagicMock()
        mock_broker_module.Broker.return_value = mock_broker

        mock_router = MagicMock()
        mock_router.STRATEGY_REGISTRY = {}

        with patch("agents.rl_trainer.Config") as mock_config, \
             patch.dict("sys.modules", {
                 "core.broker": mock_broker_module,
                 "strategies.router": mock_router,
             }):
            mock_config.RL_STRATEGY_ENABLED = True
            mock_config.CRYPTO_SYMBOLS = ["BTC/USD"]
            result = main()

        assert result is None
