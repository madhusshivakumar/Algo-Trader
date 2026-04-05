#!/usr/bin/env python3
"""Agent 1: Strategy Optimizer — backtests all strategies × all symbols and writes optimal assignments.

Runs daily at 5:30 AM weekdays. Outputs to data/optimizer/.
Does NOT modify router.py or .env — only writes recommendations for the Market Scanner.
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings("ignore")

from core.broker import Broker
from config import Config
from compare_strategies import STRATEGIES, backtest_strategy

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
AGENT_STATE_FILE = os.path.join(DATA_DIR, "agent_state.json")
RESULTS_FILE = os.path.join(DATA_DIR, "optimizer", "backtest_results.json")
ASSIGNMENTS_FILE = os.path.join(DATA_DIR, "optimizer", "strategy_assignments.json")
LEARNINGS_FILE = os.path.join(DATA_DIR, "learnings.json")


def load_learnings() -> dict:
    """Load learnings for penalty adjustments."""
    if os.path.exists(LEARNINGS_FILE):
        try:
            with open(LEARNINGS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass
    return {"version": 1, "entries": []}


def get_strategy_penalties(learnings: dict) -> dict[str, dict[str, float]]:
    """Extract strategy penalties from recent learnings.

    Returns: {symbol: {strategy_name: penalty_multiplier}}
    If a strategy has been flagged as underperforming on a symbol for 3+ days,
    apply a 20% penalty to its composite score.
    """
    penalties = {}
    recent_entries = learnings.get("entries", [])[-7:]  # Last 7 days

    # Count how many days each strategy underperformed per symbol
    underperform_counts: dict[str, dict[str, int]] = {}
    for entry in recent_entries:
        findings = entry.get("findings", {})
        strategy_notes = findings.get("strategy_notes", {})
        for strat_name, note in strategy_notes.items():
            if any(word in note.lower() for word in ["weak", "underperform", "poor", "loss"]):
                # Check which symbols this applies to
                worst = findings.get("worst_performing_symbols", [])
                for sym in worst:
                    underperform_counts.setdefault(sym, {})
                    underperform_counts[sym][strat_name] = underperform_counts[sym].get(strat_name, 0) + 1

    for sym, strats in underperform_counts.items():
        for strat_name, count in strats.items():
            if count >= 3:
                penalties.setdefault(sym, {})[strat_name] = 0.80  # 20% penalty

    return penalties


def compute_composite_score(metrics: dict) -> float:
    """Compute a composite score for ranking strategies.

    Weights: 40% Sharpe, 30% Return, 20% Win Rate, 10% (1 - Max Drawdown)
    """
    sharpe = max(metrics.get("sharpe_ratio", 0), -5)  # Clamp extreme negatives
    sharpe_norm = (sharpe + 5) / 10  # Normalize to ~0-1 range

    ret = metrics.get("total_return", 0)
    ret_norm = max(min((ret + 20) / 40, 1), 0)  # -20% to +20% → 0-1

    wr = metrics.get("win_rate", 0) / 100  # Already 0-1

    dd = 1 - metrics.get("max_drawdown", 0)  # Lower DD = higher score

    return 0.4 * sharpe_norm + 0.3 * ret_norm + 0.2 * wr + 0.1 * dd


def update_agent_state(status: str, symbols_tested: int = 0, error: str = None, duration: float = 0):
    """Update agent_state.json with our run status."""
    state = {}
    if os.path.exists(AGENT_STATE_FILE):
        try:
            with open(AGENT_STATE_FILE) as f:
                state = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    state["strategy_optimizer"] = {
        "last_run": datetime.now().isoformat(),
        "status": status,
        "duration_seconds": round(duration),
        "symbols_tested": symbols_tested,
        "error": error,
    }

    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(AGENT_STATE_FILE), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, AGENT_STATE_FILE)
    except Exception:
        os.unlink(tmp_path)
        raise


def main():
    start_time = time.time()
    print(f"[Strategy Optimizer] Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        Config.validate()
        broker = Broker()

        # Load learnings for penalty adjustments
        learnings = load_learnings()
        penalties = get_strategy_penalties(learnings)

        # Get all symbols to test
        all_symbols = Config.CRYPTO_SYMBOLS + Config.EQUITY_SYMBOLS
        print(f"[Strategy Optimizer] Testing {len(STRATEGIES)} strategies × {len(all_symbols)} symbols")

        # Strategy name mapping (display name → key for registry)
        STRATEGY_KEY_MAP = {
            "Momentum Breakout": "momentum",
            "Mean Reversion (RSI+BB)": "mean_reversion",
            "Mean Rev Aggressive": "mean_reversion_aggressive",
            "Ensemble (Mom+MR)": "ensemble",
            "Scalper (EMA+VWAP)": "scalper",
            "MACD Crossover": "macd_crossover",
            "Triple EMA Trend": "triple_ema",
            "RSI Divergence": "rsi_divergence",
            "Volume Profile": "volume_profile",
        }

        all_results = {}
        assignments = {}

        for symbol in all_symbols:
            print(f"\n  Fetching data for {symbol}...")
            try:
                df = broker.get_historical_bars(symbol, days=Config.CANDLE_HISTORY_DAYS)
            except Exception as e:
                print(f"  ERROR fetching {symbol}: {e}")
                continue

            if df.empty or len(df) < 200:
                print(f"  Skipping {symbol} — insufficient data ({len(df)} bars)")
                continue

            print(f"  Got {len(df)} bars. Testing strategies...")
            symbol_results = {}

            for strat_name, strat_fn in STRATEGIES.items():
                try:
                    metrics = backtest_strategy(strat_fn, df)
                    score = compute_composite_score(metrics)

                    # Apply penalty if this strategy has been underperforming
                    strat_key = STRATEGY_KEY_MAP.get(strat_name, strat_name)
                    penalty = penalties.get(symbol, {}).get(strat_key, 1.0)
                    adjusted_score = score * penalty

                    symbol_results[strat_name] = {
                        "total_return": round(metrics["total_return"], 2),
                        "sharpe": round(metrics["sharpe_ratio"], 2),
                        "win_rate": round(metrics["win_rate"], 1),
                        "max_drawdown": round(metrics["max_drawdown"], 4),
                        "num_trades": metrics["num_trades"],
                        "profit_factor": round(metrics["profit_factor"], 2) if metrics["profit_factor"] < 100 else 999,
                        "composite_score": round(adjusted_score, 4),
                        "penalty_applied": penalty < 1.0,
                    }

                    icon = "+" if metrics["total_return"] > 0 else "-"
                    print(f"    {icon} {strat_name}: {metrics['total_return']:+.2f}% "
                          f"(Sharpe: {metrics['sharpe_ratio']:.2f}, score: {adjusted_score:.3f})"
                          + (" [PENALIZED]" if penalty < 1.0 else ""))

                except Exception as e:
                    print(f"    ERROR {strat_name}: {e}")
                    continue

            all_results[symbol] = symbol_results

            # Pick the best strategy for this symbol
            if symbol_results:
                best = max(symbol_results.items(), key=lambda x: x[1]["composite_score"])
                best_name = best[0]
                best_key = STRATEGY_KEY_MAP.get(best_name, best_name)
                best_metrics = best[1]

                assignments[symbol] = {
                    "strategy": best_key,
                    "display_name": best_name,
                    "reason": f"Highest composite score ({best_metrics['composite_score']:.3f}): "
                              f"return {best_metrics['total_return']:+.2f}%, "
                              f"Sharpe {best_metrics['sharpe']:.2f}, "
                              f"win rate {best_metrics['win_rate']:.0f}%",
                    "difficult": best_metrics["total_return"] < 0,
                }
                print(f"  → Best: {best_name} (score: {best_metrics['composite_score']:.3f})")

        duration = time.time() - start_time

        # Write results
        results_output = {
            "run_date": datetime.now().strftime("%Y-%m-%d"),
            "run_timestamp": datetime.now().isoformat(),
            "lookback_days": Config.CANDLE_HISTORY_DAYS,
            "strategies_tested": len(STRATEGIES),
            "results": all_results,
        }

        assignments_output = {
            "run_date": datetime.now().strftime("%Y-%m-%d"),
            "run_timestamp": datetime.now().isoformat(),
            "assignments": assignments,
        }

        # Atomic writes (use custom encoder for numpy types)
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, (np.bool_, np.integer)):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(RESULTS_FILE), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(results_output, f, indent=2, cls=NumpyEncoder)
            os.replace(tmp_path, RESULTS_FILE)
        except Exception:
            os.unlink(tmp_path)
            raise

        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(ASSIGNMENTS_FILE), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(assignments_output, f, indent=2, cls=NumpyEncoder)
            os.replace(tmp_path, ASSIGNMENTS_FILE)
        except Exception:
            os.unlink(tmp_path)
            raise

        update_agent_state("success", symbols_tested=len(all_results), duration=duration)

        print(f"\n[Strategy Optimizer] Complete in {duration:.0f}s")
        print(f"  Symbols tested: {len(all_results)}")
        print(f"  Results: {RESULTS_FILE}")
        print(f"  Assignments: {ASSIGNMENTS_FILE}")

        # Print summary
        print("\n  Strategy Assignments:")
        for sym, info in assignments.items():
            flag = " ⚠️ DIFFICULT" if info["difficult"] else ""
            print(f"    {sym:8s} → {info['display_name']}{flag}")

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n[Strategy Optimizer] FAILED: {e}")
        update_agent_state("failed", error=str(e), duration=duration)
        raise


if __name__ == "__main__":
    main()
