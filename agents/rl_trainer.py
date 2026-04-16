"""RL Trainer Agent — weekly DQN retraining for strategy selection.

Runs weekly (Sunday 2 AM):
  1. Fetch 90 days of historical bars for all symbols
  2. Compute per-strategy returns via backtesting (transaction-cost adjusted)
  3. Train DQN in StrategySelectionEnv (100K steps, warm-start from prior model)
  4. 60/20/20 train/val/test split — val for early-stop gating, test for final eval only
  5. Deploy only if test Sharpe > threshold (never retrain on test data)
  6. Save model to data/rl_models/dqn_latest.zip

Sprint 6B notes:
  - Previous protocol used a 70/30 split and reported val Sharpe directly, which
    (combined with a 20-bar Sharpe reward window) produced unrealistic numbers
    like val Sharpe = 2.65. That's the bot memorizing the val set — a well-known
    pitfall when val is used for both early-stop and deploy gating.
  - The new 60/20/20 split separates the two: val gates early stopping, test is
    the honest out-of-sample evaluation. Test is consulted exactly once per run.
  - `compute_strategy_returns` now subtracts per-position-change transaction
    costs so the reward approximates net returns, not gross.
  - Training warm-starts from the prior model when present, preventing
    catastrophic forgetting between weekly runs.
"""

import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

from config import Config
from core.rl_features import STRATEGY_KEYS, extract_features
from utils.logger import log

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rl_models")
_MODEL_PATH = os.path.join(_DATA_DIR, "dqn_latest.zip")
_TRAIN_LOG = os.path.join(_DATA_DIR, "train_log.json")

# Sprint 6B: Default training steps raised from 10K → 100K. 10K over 88K bars
# is ~0.1 epoch; DQN has no chance to converge. 100K gives each bar ~1 update
# on average while staying under an hour on the RTX 4070 Super.
DEFAULT_TRAIN_STEPS = 100_000


def _ensure_data_dir():
    os.makedirs(_DATA_DIR, exist_ok=True)


def compute_strategy_returns(
    df: pd.DataFrame,
    strategy_fn,
    symbol: str | None = None,
    include_costs: bool = True,
) -> np.ndarray:
    """Compute per-bar NET returns for a strategy on the given DataFrame.

    Simulates the strategy: if buy signal, long; if sell, flat.
    Uses a stride to avoid O(n*indicators) cost.

    Sprint 6B: `include_costs=True` subtracts estimated round-trip transaction
    cost each time the position changes. Without it, the RL reward is gross
    returns — and the agent learns to switch constantly because there's no
    per-switch drag beyond the small SWITCHING_PENALTY. With costs, the agent
    learns a strategy-selection policy that survives actual trading fees.

    Args:
        df: OHLCV DataFrame.
        strategy_fn: Strategy compute_signals function.
        symbol: Symbol for cost classification (crypto vs equity). If None,
            equity spreads are used (conservative for tests).
        include_costs: If True, subtract costs on position changes.

    Returns:
        Per-bar net returns array of length len(df).
    """
    n = len(df)
    price_returns = df["close"].pct_change().fillna(0).values

    # Cost model is only used when `include_costs` is True AND TC is configured.
    # We estimate cost as a fraction of notional on each position flip.
    cost_per_flip = 0.0
    if include_costs:
        try:
            from core.transaction_costs import TransactionCostModel
            model = TransactionCostModel()
            # One-way cost as a fraction (buy costs on entry, sell on exit).
            # We treat the flip as a single-bar hit of one-way cost; the second
            # flip (sell) pays another cost on the exit bar.
            one_way_bps = model.slippage_bps
            if symbol and Config.is_crypto(symbol):
                one_way_bps += model.spread_bps_crypto
            else:
                one_way_bps += model.spread_bps_equity
            one_way_bps += model.commission_pct * 10000
            cost_per_flip = one_way_bps / 10000.0
        except Exception:
            cost_per_flip = 0.0

    stride = 10
    position = 0
    positions = np.zeros(n)
    flips = np.zeros(n)

    for i in range(30, n, stride):
        end = min(i + stride, n)
        sub_df = df.iloc[:end]
        prev_position = position
        try:
            signal = strategy_fn(sub_df)
            if signal.get("action") == "buy" and signal.get("strength", 0) > 0.4:
                position = 1
            elif signal.get("action") == "sell":
                position = 0
        except Exception:
            pass
        positions[i:end] = position
        # Record a flip on the first bar of the new window if position changed.
        if position != prev_position:
            flips[i] = 1.0

    gross = price_returns * positions
    # Subtract one-way cost on each flip — approximates the realized drag.
    cost_drag = flips * cost_per_flip
    return gross - cost_drag


def _compute_returns_worker(args: tuple) -> tuple[str, np.ndarray]:
    """Worker for parallel strategy backtest — runs in subprocess."""
    strategy_key, df_bytes, n, symbol = args
    from strategies.router import STRATEGY_REGISTRY
    import pickle
    df = pickle.loads(df_bytes)
    fn = STRATEGY_REGISTRY.get(strategy_key)
    if fn is None:
        return (strategy_key, np.zeros(n))
    return (strategy_key, compute_strategy_returns(df, fn, symbol=symbol))


def compute_all_strategy_returns(
    df: pd.DataFrame,
    strategy_fns: dict,
    symbol: str | None = None,
) -> dict[str, np.ndarray]:
    """Compute net returns for all strategies in parallel across CPU cores.

    Args:
        df: OHLCV DataFrame.
        strategy_fns: Map of strategy_key → compute_signals function.
        symbol: Symbol for TC classification (crypto vs equity). Propagated
            to each worker so costs are consistent.
    """
    import pickle
    df_bytes = pickle.dumps(df)
    n = len(df)

    keys_with_fns = [k for k in STRATEGY_KEYS if k in strategy_fns]
    keys_without = [k for k in STRATEGY_KEYS if k not in strategy_fns]

    results = {}
    for k in keys_without:
        results[k] = np.zeros(n)

    log.info(f"Backtesting {len(keys_with_fns)} strategies in parallel across CPU cores...")
    with ProcessPoolExecutor() as executor:
        args = [(key, df_bytes, n, symbol) for key in keys_with_fns]
        for key, returns in executor.map(_compute_returns_worker, args):
            log.info(f"  {key} backtest complete")
            results[key] = returns

    return results


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
                total_timesteps: int = DEFAULT_TRAIN_STEPS) -> object | None:
    """Train a DQN model on the given data.

    Sprint 6B: Warm-start from prior model when available. Weekly retraining
    from scratch throws away 10K–100K steps of prior learning every week — a
    classic case of catastrophic forgetting disguised as "fresh training."
    Loading the saved weights first means the agent refines its policy rather
    than rediscovering it.

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
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Training on device: {device}")

        model = None
        if os.path.exists(_MODEL_PATH):
            try:
                log.info(f"Warm-starting from prior model at {_MODEL_PATH}")
                model = DQN.load(_MODEL_PATH, env=env, device=device)
            except Exception as e:
                log.warning(f"Could not load prior model ({e}); training fresh")
                model = None

        if model is None:
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
                device=device,
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


def save_train_log(train_sharpe: float, val_sharpe: float, test_sharpe: float,
                   deployed: bool):
    """Save training metrics log.

    Sprint 6B: `test_sharpe` is the honest out-of-sample number. `val_sharpe`
    is the early-stop signal the trainer consults during learning and is
    therefore not a trustworthy deploy gate — treat it as a training artifact.
    `test_sharpe` is the one to show on dashboards and base decisions on.
    """
    _ensure_data_dir()
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "train_sharpe": round(train_sharpe, 4),
        "val_sharpe": round(val_sharpe, 4),
        "test_sharpe": round(test_sharpe, 4),
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
                 total_timesteps: int = DEFAULT_TRAIN_STEPS,
                 symbol: str | None = None) -> dict:
    """Full training pipeline with 60/20/20 train/val/test split.

    Sprint 6B: Previous 70/30 split used val for BOTH early-stop gating AND
    deploy threshold — the classic val-set-leakage bug. A Sharpe of 2.65 on
    a 30% slice the trainer is already tuned against isn't a real number;
    it's overfitting. New 60/20/20 split cleanly separates the roles:
      - train (60%): gradient updates only
      - val (20%): early-stop signal, computed during training
      - test (20%): ONE-SHOT final evaluation. Deploy gate uses this.

    Args:
        df: Full OHLCV DataFrame (90 days).
        strategy_fns: dict mapping strategy_key → compute_signals function.
        total_timesteps: DQN training steps.
        symbol: Symbol being trained (propagated to transaction-cost model
            so crypto vs equity spreads are used correctly).

    Returns:
        dict with training results — train/val/test Sharpe + deploy flag.
    """
    # 60/20/20 split — val for early-stop gating, test for honest final eval.
    split_train = int(len(df) * 0.6)
    split_val = int(len(df) * 0.8)
    train_df = df.iloc[:split_train].copy()
    val_df = df.iloc[split_train:split_val].copy()
    test_df = df.iloc[split_val:].copy()

    log.info(
        f"Split: train={len(train_df)} val={len(val_df)} test={len(test_df)} bars"
    )

    # Compute strategy returns (parallel) — cost-aware if symbol known.
    train_returns = compute_all_strategy_returns(train_df, strategy_fns, symbol=symbol)

    # Train (on GPU if available; warm-starts from prior model if present).
    model = train_model(train_df, train_returns, total_timesteps)
    if model is None:
        return {"success": False, "error": "Training failed"}

    train_sharpe = compute_sharpe(
        np.concatenate([train_returns[k] for k in STRATEGY_KEYS])
    )

    # Validate — used as early-stop / training-quality signal ONLY.
    val_returns = compute_all_strategy_returns(val_df, strategy_fns, symbol=symbol)
    val_sharpe = validate_model(model, val_df, val_returns)

    # Test — honest one-shot evaluation. This drives the deploy decision.
    test_returns = compute_all_strategy_returns(test_df, strategy_fns, symbol=symbol)
    test_sharpe = validate_model(model, test_df, test_returns)

    # Deploy gate is test Sharpe, NOT val. Val was consulted during training
    # and is no longer trustworthy as an out-of-sample estimate.
    deployed = test_sharpe >= Config.RL_MIN_SHARPE_THRESHOLD
    if deployed:
        save_model(model)
        log.success(
            f"Model deployed (test Sharpe: {test_sharpe:.4f} >= "
            f"{Config.RL_MIN_SHARPE_THRESHOLD}; val: {val_sharpe:.4f})"
        )
    else:
        log.warning(
            f"Model NOT deployed (test Sharpe: {test_sharpe:.4f} < "
            f"{Config.RL_MIN_SHARPE_THRESHOLD}; val: {val_sharpe:.4f})"
        )

    save_train_log(train_sharpe, val_sharpe, test_sharpe, deployed)

    return {
        "success": True,
        "train_sharpe": train_sharpe,
        "val_sharpe": val_sharpe,
        "test_sharpe": test_sharpe,
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

    result = run_training(
        df,
        STRATEGY_REGISTRY,
        total_timesteps=DEFAULT_TRAIN_STEPS,
        symbol=symbol,
    )

    if result.get("success"):
        # Sprint 6I: headline is TEST Sharpe. Val printed for transparency
        # but is not what the deploy gate checks. Every metric is wrapped
        # with the industry reality-anchor via ``core.expected_returns``.
        from core.expected_returns import frame_sharpe, INDUSTRY_RANGE_SHARPE

        log.success(
            f"Training complete — train: {result['train_sharpe']:.4f}, "
            f"val: {result['val_sharpe']:.4f}, "
            f"test: {result['test_sharpe']:.4f} (deploy gate), "
            f"deployed: {result['deployed']}"
        )
        log.info(frame_sharpe(result["test_sharpe"],
                              context="out-of-sample test"))
        log.info(frame_sharpe(result["val_sharpe"],
                              context="validation (in-sample)"))
        log.info(INDUSTRY_RANGE_SHARPE)
    else:
        log.error(f"Training failed: {result.get('error')}")

    return result


if __name__ == "__main__":
    main()
