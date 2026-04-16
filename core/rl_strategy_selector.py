"""RL Strategy Selector — uses trained DQN to pick optimal strategy.

Loads a pre-trained model and predicts the best strategy given current
market features. Falls back gracefully if model is missing or fails.
"""

import os
import numpy as np

from config import Config
from core.rl_features import extract_features, STRATEGY_KEYS, NUM_STRATEGIES
from utils.logger import log


class RLStrategySelector:
    """Selects trading strategy using a trained DQN model."""

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.model_path = model_path or Config.RL_MODEL_PATH
        self._load_model()

    def _load_model(self):
        """Load the trained DQN model."""
        if not os.path.exists(self.model_path):
            log.warning(f"RL model not found: {self.model_path}")
            return

        try:
            from stable_baselines3 import DQN
            self.model = DQN.load(self.model_path)
            log.success(f"Loaded RL model from {self.model_path}")
        except Exception as e:
            log.error(f"Failed to load RL model: {e}")
            self.model = None

    def is_ready(self) -> bool:
        """Check if model is loaded and ready for predictions."""
        return self.model is not None

    def select_strategy(self, df, regime: str | None = None) -> str | None:
        """Select the best strategy for current market conditions.

        Args:
            df: DataFrame with OHLCV data.
            regime: Current market regime from `core.regime_detector`, one of
                'low_vol' / 'normal' / 'high_vol' / 'crisis'. Sprint 6E plumbs
                this into `extract_features` so the policy can condition its
                pick on whether the market is trending vs stressed.

        Returns:
            Strategy key string, or None if prediction fails.
        """
        if not self.is_ready():
            return None

        features = extract_features(df, regime=regime)
        if features is None:
            log.warning("Insufficient data for RL feature extraction")
            return None

        try:
            action, _ = self.model.predict(features, deterministic=True)
            action_idx = int(action)
            if 0 <= action_idx < NUM_STRATEGIES:
                strategy = STRATEGY_KEYS[action_idx]
                log.info(f"RL selected strategy: {strategy}")
                return strategy
            else:
                log.warning(f"RL predicted invalid action: {action_idx}")
                return None
        except Exception as e:
            # A shape mismatch here is expected right after the 10→16 dim
            # migration until the next weekly retrain deploys a 16-dim model.
            # Fallback is the default strategy map, so trading continues.
            log.error(f"RL prediction failed: {e}")
            return None


# Module-level singleton (lazy-loaded)
_selector = None


def get_selector() -> RLStrategySelector | None:
    """Get or create the module-level RL selector."""
    global _selector
    if _selector is None and Config.RL_STRATEGY_ENABLED:
        _selector = RLStrategySelector()
    return _selector


def select_strategy(df, regime: str | None = None) -> str | None:
    """Convenience function: select strategy using the global selector.

    Sprint 6E: regime is plumbed through to extract_features so the model
    can see the live regime bits alongside the market features.
    """
    selector = get_selector()
    if selector is None or not selector.is_ready():
        return None
    return selector.select_strategy(df, regime=regime)
