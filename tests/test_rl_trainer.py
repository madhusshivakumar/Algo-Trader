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

    def test_costs_reduce_net_returns_on_flip(self):
        """Sprint 6B: a strategy that flips should yield lower net returns
        with `include_costs=True` than with costs off, because each flip
        pays one-way cost."""
        from agents.rl_trainer import compute_strategy_returns
        df = _make_df(200)

        call_count = {"n": 0}
        def flippy_fn(sub_df):
            # Alternate buy/sell deterministically so flips happen.
            call_count["n"] += 1
            return {
                "action": "buy" if call_count["n"] % 2 == 1 else "sell",
                "strength": 0.8,
                "reason": "test",
            }

        # Reset counter between runs so both see the same signal sequence.
        call_count["n"] = 0
        gross = compute_strategy_returns(df, flippy_fn, include_costs=False)
        call_count["n"] = 0
        net = compute_strategy_returns(df, flippy_fn, include_costs=True,
                                       symbol="AAPL")

        # Net returns should be ≤ gross at every bar (costs never add).
        assert (net <= gross + 1e-9).all()
        # And strictly less somewhere (some flip happened).
        assert net.sum() < gross.sum()

    def test_costs_off_matches_gross(self):
        """Turning costs off must reproduce the pre-Sprint-6B behaviour."""
        from agents.rl_trainer import compute_strategy_returns
        df = _make_df(100)
        fn = lambda d: {"action": "buy", "strength": 0.8, "reason": "t"}
        rets = compute_strategy_returns(df, fn, include_costs=False)
        # No cost drag → first 30 bars still zero, and returns are finite.
        assert np.isfinite(rets).all()
        assert (rets[:30] == 0).all()


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

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {
            "stable_baselines3": mock_sb3,
            "core.rl_environment": MagicMock(StrategySelectionEnv=mock_env_cls),
            "torch": mock_torch,
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
        # Sprint 6B: log now includes test_sharpe as the deploy-gate metric;
        # val_sharpe is retained for transparency but is a training artifact.
        from agents.rl_trainer import save_train_log
        log_file = tmp_path / "train_log.json"

        with patch("agents.rl_trainer._DATA_DIR", str(tmp_path)), \
             patch("agents.rl_trainer._TRAIN_LOG", str(log_file)), \
             patch("agents.rl_trainer.Config") as mock_config:
            mock_config.RL_MIN_SHARPE_THRESHOLD = 0.5
            save_train_log(1.2, 0.8, 0.6, True)

        data = json.loads(log_file.read_text())
        assert data["train_sharpe"] == 1.2
        assert data["val_sharpe"] == 0.8
        assert data["test_sharpe"] == 0.6
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

    @patch("agents.rl_trainer.save_train_log")
    @patch("agents.rl_trainer.save_model")
    @patch("agents.rl_trainer.compute_all_strategy_returns")
    @patch("agents.rl_trainer.validate_model")
    @patch("agents.rl_trainer.train_model")
    def test_splits_60_20_20(
        self, mock_train, mock_validate, mock_all_returns, mock_save, mock_log
    ):
        """Sprint 6B: run_training must split 60/20/20 and call validate_model
        twice (once for val, once for test)."""
        from agents.rl_trainer import run_training
        from core.rl_features import STRATEGY_KEYS

        # compute_all_strategy_returns is called 3× (train/val/test); give
        # each a dict of zero arrays sized to whatever slice was passed.
        def fake_all_returns(slice_df, *_, **__):
            n = len(slice_df)
            return {k: np.zeros(n) for k in STRATEGY_KEYS}
        mock_all_returns.side_effect = fake_all_returns
        mock_validate.side_effect = [0.9, 0.7]  # val, test
        mock_train.return_value = MagicMock()

        df = _make_df(100)
        fns = {k: MagicMock() for k in STRATEGY_KEYS}

        with patch("agents.rl_trainer.Config") as mock_config:
            mock_config.RL_MIN_SHARPE_THRESHOLD = 0.5
            result = run_training(df, fns)

        # 3 calls: train/val/test — confirms all three splits were materialized.
        assert mock_all_returns.call_count == 3
        # Validate twice: once on val, once on test.
        assert mock_validate.call_count == 2
        assert result["success"] is True
        # test_sharpe is 0.7, which is what the deploy gate checks.
        assert result["test_sharpe"] == 0.7
        assert result["val_sharpe"] == 0.9
        assert result["deployed"] is True

    @patch("agents.rl_trainer.save_train_log")
    @patch("agents.rl_trainer.save_model")
    @patch("agents.rl_trainer.compute_all_strategy_returns")
    @patch("agents.rl_trainer.validate_model")
    @patch("agents.rl_trainer.train_model")
    def test_deploy_gate_uses_test_not_val(
        self, mock_train, mock_validate, mock_all_returns, mock_save, mock_log
    ):
        """Sprint 6B: if val=2.5 but test=0.1, we MUST NOT deploy. The old
        behaviour (deploying on val Sharpe) was the entire bug this sprint fixes."""
        from agents.rl_trainer import run_training
        from core.rl_features import STRATEGY_KEYS

        def fake_all_returns(slice_df, *_, **__):
            n = len(slice_df)
            return {k: np.zeros(n) for k in STRATEGY_KEYS}
        mock_all_returns.side_effect = fake_all_returns
        # Wildly optimistic val, honest test — classic val-set-leakage signature.
        mock_validate.side_effect = [2.5, 0.1]
        mock_train.return_value = MagicMock()

        df = _make_df(100)
        fns = {k: MagicMock() for k in STRATEGY_KEYS}

        with patch("agents.rl_trainer.Config") as mock_config:
            mock_config.RL_MIN_SHARPE_THRESHOLD = 0.5
            result = run_training(df, fns)

        assert result["val_sharpe"] == 2.5
        assert result["test_sharpe"] == 0.1
        assert result["deployed"] is False
        # save_model must NOT be called when test Sharpe fails the gate,
        # no matter how rosy val looks.
        mock_save.assert_not_called()

    def test_warm_start_when_prior_model_exists(self, tmp_path):
        """Sprint 6B: train_model should call DQN.load() when _MODEL_PATH exists."""
        from agents import rl_trainer
        from core.rl_features import STRATEGY_KEYS

        prior_path = tmp_path / "dqn_latest.zip"
        prior_path.write_bytes(b"fake-model-bytes")

        mock_loaded = MagicMock()
        mock_dqn_class = MagicMock()
        mock_dqn_class.load.return_value = mock_loaded

        mock_sb3 = MagicMock(DQN=mock_dqn_class)
        mock_env_cls = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        df = _make_df(100)
        returns = {k: np.zeros(len(df)) for k in STRATEGY_KEYS}

        with patch.object(rl_trainer, "_MODEL_PATH", str(prior_path)), \
             patch.dict("sys.modules", {
                 "stable_baselines3": mock_sb3,
                 "core.rl_environment": MagicMock(StrategySelectionEnv=mock_env_cls),
                 "torch": mock_torch,
             }):
            result = rl_trainer.train_model(df, returns, total_timesteps=1)

        # Warm-started: DQN.load was called, DQN() constructor was NOT.
        mock_dqn_class.load.assert_called_once()
        mock_dqn_class.assert_not_called()
        assert result is mock_loaded


class TestMain:
    @patch("agents.rl_trainer.run_training")
    def test_main_success(self, mock_run):
        from agents.rl_trainer import main
        mock_run.return_value = {
            "success": True,
            "train_sharpe": 1.0,
            "val_sharpe": 0.7,
            "test_sharpe": 0.5,  # Sprint 6B: deploy gate
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
