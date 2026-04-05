"""RL Trainer Agent — weekly DQN retraining for strategy selection.

Runs weekly (Sunday 2 AM):
  1. Fetch 90 days of historical bars for all symbols
  2. Compute per-strategy returns via backtesting
  3. Train DQN in StrategySelectionEnv
  4. Walk-forward validation (70/30 split)
  5. Deploy only if validation Sharpe > threshold
  6. Save model to data/rl_models/dqn_latest.zip
"""

import json
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

from config import Config
from core.rl_features import STRATEGY_KEYS, extract_features
from utils.logger import log

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rl_models")
_MODEL_PATH = os.path.join(_DATA_DIR, "dqn_latest.zip")
_TRAIN_LOG = os.path.join(_DATA_DIR, "train_log.json")


def _ensure_data_dir():
    os.makedirs(_DATA_DIR, exist_ok=True)


def compute_strategy_returns(df: pd.DataFrame, strategy_fn) -> np.ndarray:
    """Compute per-bar returns for a strategy on the given DataFrame.

    Simulates the strategy: if buy signal, long; if sell, flat.
    Returns array of per-bar returns.
    """
    n = len(df)
    returns = np.zeros(n)
    price_returns = df["close"].pct_change().fillna(0).values

    position = 0  # 0 = flat, 1 = long
    for i in range(30, n):  # Need 30 bars for indicators
        sub_df = df.iloc[:i + 1]
        try:
            signal = strategy_fn(sub_df)
            if signal.get("action") == "buy" and signal.get("strength", 0) > 0.4:
                position = 1
            elif signal.get("action") == "sell":
                position = 0
        except Exception:
            pass

        returns[i] = price_returns[i] * position

    return returns


def compute_sharpe(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio from an array of returns."""
    if len(returns) < 2:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret < 1e-8:
        return 0.0
    # Annualize (assume 1-min bars, ~390 per day, ~252 trading days)
    daily_sharpe = mean_ret / std_ret * np.sqrt(390)
    return float(daily_sharpe)


def train_model(df: pd.DataFrame, strategy_returns: dict[str, np.ndarray],
                total_timesteps: int = 10000) -> object | None:
    """Train a DQN model on the given data.

    Returns the trained model or None if training fails.
    """
    try:
        from stable_baselines3 import DQN
        from core.rl_environment import StrategySelectionEnv
    except ImportError as e:
        log.error(f"RL training requires stable-baselines3 and gymnasium: {e}")
        return None

    env = StrategySelectionEnv(df, strategy_returns)

    try:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=10000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            verbose=0,
        )
        model.learn(total_timesteps=total_timesteps)
        return model
    except Exception as e:
        log.error(f"DQN training failed: {e}")
        return None


def validate_model(model, df: pd.DataFrame,
                   strategy_returns: dict[str, np.ndarray]) -> float:
    """Validate model on held-out data using walk-forward evaluation.

    Returns the validation Sharpe ratio.
    """
    from core.rl_features import extract_features, STRATEGY_KEYS

    selected_returns = []
    for i in range(30, len(df)):
        sub_df = df.iloc[:i + 1]
        features = extract_features(sub_df)
        if features is None:
            continue

        try:
            action, _ = model.predict(features, deterministic=True)
            strategy_key = STRATEGY_KEYS[int(action)]
            rets = strategy_returns.get(strategy_key, np.zeros(len(df)))
            if i < len(rets):
                selected_returns.append(rets[i])
        except Exception:
            selected_returns.append(0.0)

    return compute_sharpe(np.array(selected_returns))


def save_model(model, path: str | None = None):
    """Save model to disk."""
    _ensure_data_dir()
    path = path or _MODEL_PATH
    model.save(path)
    log.success(f"Saved RL model to {path}")


def save_train_log(train_sharpe: float, val_sharpe: float, deployed: bool):
    """Save training metrics log."""
    _ensure_data_dir()
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "train_sharpe": round(train_sharpe, 4),
        "val_sharpe": round(val_sharpe, 4),
        "deployed": deployed,
        "threshold": Config.RL_MIN_SHARPE_THRESHOLD,
    }
    os.makedirs(os.path.dirname(_TRAIN_LOG), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(_TRAIN_LOG), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(log_data, f, indent=2)
        os.replace(tmp_path, _TRAIN_LOG)
    except Exception:
        os.unlink(tmp_path)
        raise


def run_training(df: pd.DataFrame, strategy_fns: dict,
                 total_timesteps: int = 10000) -> dict:
    """Full training pipeline with walk-forward validation.

    Args:
        df: Full OHLCV DataFrame (90 days).
        strategy_fns: dict mapping strategy_key → compute_signals function.
        total_timesteps: DQN training steps.

    Returns:
        dict with training results.
    """
    # Split: 70% train, 30% validation
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    log.info(f"Training on {len(train_df)} bars, validating on {len(val_df)} bars")

    # Compute strategy returns for training data
    train_returns = {}
    for key in STRATEGY_KEYS:
        fn = strategy_fns.get(key)
        if fn:
            train_returns[key] = compute_strategy_returns(train_df, fn)
        else:
            train_returns[key] = np.zeros(len(train_df))

    # Train
    model = train_model(train_df, train_returns, total_timesteps)
    if model is None:
        return {"success": False, "error": "Training failed"}

    train_sharpe = compute_sharpe(
        np.concatenate([train_returns[k] for k in STRATEGY_KEYS])
    )

    # Validate
    val_returns = {}
    for key in STRATEGY_KEYS:
        fn = strategy_fns.get(key)
        if fn:
            val_returns[key] = compute_strategy_returns(val_df, fn)
        else:
            val_returns[key] = np.zeros(len(val_df))

    val_sharpe = validate_model(model, val_df, val_returns)

    # Deploy decision
    deployed = val_sharpe >= Config.RL_MIN_SHARPE_THRESHOLD
    if deployed:
        save_model(model)
        log.success(f"Model deployed (val Sharpe: {val_sharpe:.4f} >= {Config.RL_MIN_SHARPE_THRESHOLD})")
    else:
        log.warning(
            f"Model NOT deployed (val Sharpe: {val_sharpe:.4f} < {Config.RL_MIN_SHARPE_THRESHOLD})"
        )

    save_train_log(train_sharpe, val_sharpe, deployed)

    return {
        "success": True,
        "train_sharpe": train_sharpe,
        "val_sharpe": val_sharpe,
        "deployed": deployed,
    }


def main():
    """Main entry point — full RL training pipeline."""
    log.info("=" * 50)
    log.info("RL Trainer Agent starting")
    log.info("=" * 50)

    if not Config.RL_STRATEGY_ENABLED:
        log.warning("RL_STRATEGY_ENABLED=false, running anyway (agent invoked directly)")

    # Import strategy functions
    from strategies.router import STRATEGY_REGISTRY

    # Fetch historical data
    try:
        from core.broker import Broker
        broker = Broker()
        # Use first crypto symbol for training (24/7 data availability)
        symbol = Config.CRYPTO_SYMBOLS[0] if Config.CRYPTO_SYMBOLS else Config.SYMBOLS[0]
        df = broker.get_historical_bars(symbol, days=90)
        if df is None or df.empty:
            log.error("No historical data available for training")
            return None
    except Exception as e:
        log.error(f"Failed to fetch training data: {e}")
        return None

    result = run_training(df, STRATEGY_REGISTRY, total_timesteps=50000)

    if result.get("success"):
        log.success(
            f"Training complete — train Sharpe: {result['train_sharpe']:.4f}, "
            f"val Sharpe: {result['val_sharpe']:.4f}, deployed: {result['deployed']}"
        )
    else:
        log.error(f"Training failed: {result.get('error')}")

    return result


if __name__ == "__main__":
    main()
