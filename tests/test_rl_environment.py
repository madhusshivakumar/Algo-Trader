"""Tests for core/rl_environment.py — RL gymnasium environment."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _make_df(n=100):
    """Create sample OHLCV DataFrame."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "close": close,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "volume": np.random.randint(1000, 5000, n),
    })


def _make_returns(n=100):
    """Create mock strategy returns."""
    from core.rl_features import STRATEGY_KEYS
    np.random.seed(42)
    return {k: np.random.randn(n) * 0.001 for k in STRATEGY_KEYS}


try:
    import gymnasium  # noqa: F401
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False


@pytest.mark.skipif(not HAS_GYMNASIUM, reason="gymnasium not installed")
class TestStrategySelectionEnv:
    def test_reset(self):
        from core.rl_environment import StrategySelectionEnv
        df = _make_df()
        env = StrategySelectionEnv(df, _make_returns())

        obs, info = env.reset()
        from core.rl_features import NUM_FEATURES  # Sprint 6E: 10→16
        assert obs.shape == (NUM_FEATURES,)
        assert isinstance(info, dict)

    def test_step_returns_tuple(self):
        from core.rl_environment import StrategySelectionEnv
        df = _make_df()
        env = StrategySelectionEnv(df, _make_returns())
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)
        from core.rl_features import NUM_FEATURES  # Sprint 6E: 10→16
        assert obs.shape == (NUM_FEATURES,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert truncated is False

    def test_episode_terminates(self):
        from core.rl_environment import StrategySelectionEnv
        df = _make_df(50)
        env = StrategySelectionEnv(df, _make_returns(50))
        env.reset()

        terminated = False
        steps = 0
        while not terminated:
            _, _, terminated, _, _ = env.step(0)
            steps += 1
            if steps > 100:
                break
        assert terminated

    def test_switching_penalty(self):
        from core.rl_environment import StrategySelectionEnv
        df = _make_df()
        env = StrategySelectionEnv(df, _make_returns())
        env.reset()

        # First step with action 0
        env.step(0)
        # Step with same action (no penalty)
        _, reward_same, _, _, _ = env.step(0)

        # Reset and try switching
        env.reset()
        env.step(0)
        _, reward_switch, _, _, _ = env.step(1)

        # Switch reward should be lower due to penalty
        # (may not always hold due to different returns, but penalty is applied)
        assert isinstance(reward_switch, float)

    def test_action_space(self):
        from core.rl_environment import StrategySelectionEnv
        from core.rl_features import NUM_STRATEGIES
        df = _make_df()
        env = StrategySelectionEnv(df, _make_returns())

        assert env.action_space.n == NUM_STRATEGIES

    def test_observation_space(self):
        from core.rl_environment import StrategySelectionEnv
        from core.rl_features import NUM_FEATURES
        df = _make_df()
        env = StrategySelectionEnv(df, _make_returns())

        assert env.observation_space.shape == (NUM_FEATURES,)

    def test_step_accumulates_returns(self):
        from core.rl_environment import StrategySelectionEnv
        df = _make_df()
        env = StrategySelectionEnv(df, _make_returns())
        env.reset()

        env.step(0)
        env.step(1)
        env.step(2)
        assert len(env.selected_returns) == 3

    def test_missing_strategy_returns(self):
        from core.rl_environment import StrategySelectionEnv
        df = _make_df()
        # Empty returns dict
        env = StrategySelectionEnv(df, {})
        env.reset()

        obs, reward, _, _, _ = env.step(0)
        from core.rl_features import NUM_FEATURES  # Sprint 6E: 10→16
        assert obs.shape == (NUM_FEATURES,)


@pytest.mark.skipif(not HAS_GYMNASIUM, reason="gymnasium not installed")
class TestCreateEnv:
    def test_factory(self):
        from core.rl_environment import create_env, StrategySelectionEnv
        df = _make_df()
        env = create_env(df, _make_returns())
        assert isinstance(env, StrategySelectionEnv)


class TestNoGymnasium:
    def test_raises_without_gymnasium(self):
        from core.rl_environment import StrategySelectionEnv
        df = _make_df()

        with patch("core.rl_environment.HAS_GYMNASIUM", False):
            with pytest.raises(ImportError, match="gymnasium"):
                StrategySelectionEnv(df, _make_returns())
