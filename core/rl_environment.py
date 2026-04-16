"""RL Environment — gymnasium env for strategy selection.

The agent selects one of 9 strategies at each timestep.
Reward is based on rolling Sharpe ratio with a switching penalty.

Sprint 6B: reward window raised from 20 → 252 bars to match the standard
daily Sharpe horizon. A 20-bar window was lying to us — Sharpe over 20
samples is extremely noisy, which is why the trainer was overfitting to
short-run luck (val Sharpe 2.65 is not a real number, it's a 20-bar
fluke). 252 makes the reward signal match the metric we actually care
about.
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

from core.rl_features import NUM_FEATURES, NUM_STRATEGIES, STRATEGY_KEYS, extract_features

# Reward parameters
# Sprint 6B: Match standard daily Sharpe horizon. Previous value (20) was
# 20-sample variance — extremely noisy, biased toward short-run luck.
SHARPE_WINDOW = 252
SWITCHING_PENALTY = -0.001
RISK_FREE_RATE = 0.0

# Base class: gym.Env when gymnasium is installed, otherwise plain object
_BaseEnv = gym.Env if HAS_GYMNASIUM else object


class StrategySelectionEnv(_BaseEnv):
    """Gymnasium-compatible environment for RL strategy selection.

    State: 10-dimensional feature vector from rl_features
    Action: Discrete(9) — index of strategy to use
    Reward: Rolling Sharpe of selected strategy + switching penalty
    """

    def __init__(self, df, strategy_returns: dict[str, np.ndarray]):
        """
        Args:
            df: DataFrame with OHLCV data (used for feature extraction).
            strategy_returns: dict mapping strategy_key → array of per-bar returns.
                Each array should have length == len(df).
        """
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium is required for RL training. Install with: pip install gymnasium")

        self.df = df
        self.strategy_returns = strategy_returns
        self.n_steps = len(df)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(NUM_FEATURES,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_STRATEGIES)

        # State
        self.current_step = 0
        self.prev_action = 0
        self.selected_returns = []

    def reset(self, seed=None):
        """Reset environment to initial state."""
        self.current_step = 30  # Need 30 bars for feature extraction
        self.prev_action = 0
        self.selected_returns = []
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        """Take a step: select strategy, compute reward, advance."""
        # Get the return for the selected strategy at this timestep
        strategy_key = STRATEGY_KEYS[action]
        returns = self.strategy_returns.get(strategy_key, np.zeros(self.n_steps))

        if self.current_step < len(returns):
            step_return = returns[self.current_step]
        else:
            step_return = 0.0

        self.selected_returns.append(step_return)

        # Compute reward: rolling Sharpe + switching penalty
        reward = self._compute_reward(action)

        self.prev_action = action
        self.current_step += 1

        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(NUM_FEATURES, dtype=np.float32)

        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        """Extract feature vector for current timestep."""
        end = self.current_step + 1
        start = max(0, end - 60)  # Use up to 60 bars for feature computation
        sub_df = self.df.iloc[start:end]

        features = extract_features(sub_df)
        if features is None:
            return np.zeros(NUM_FEATURES, dtype=np.float32)
        return features

    def _compute_reward(self, action: int) -> float:
        """Compute reward: rolling Sharpe + switching penalty."""
        # Switching penalty
        penalty = SWITCHING_PENALTY if action != self.prev_action else 0.0

        # Rolling Sharpe
        if len(self.selected_returns) < 2:
            return penalty

        window = self.selected_returns[-SHARPE_WINDOW:]
        mean_ret = np.mean(window)
        std_ret = np.std(window)

        if std_ret < 1e-8:
            sharpe = 0.0
        else:
            sharpe = (mean_ret - RISK_FREE_RATE) / std_ret

        # Scale sharpe to reasonable reward magnitude
        return sharpe * 0.1 + penalty


def create_env(df, strategy_returns: dict[str, np.ndarray]) -> StrategySelectionEnv:
    """Factory function to create the RL environment."""
    return StrategySelectionEnv(df, strategy_returns)
