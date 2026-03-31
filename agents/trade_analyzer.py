#!/usr/bin/env python3
"""Agent 3: Trade Analyzer — reviews the day's trades, generates reports, updates learnings.

Runs daily at 5:00 PM weekdays.
Reads from trades.db, writes reports and learnings.
Can adjust risk parameters within bounded limits.
"""

import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(PROJECT_ROOT, "trades.db")
AGENT_STATE_FILE = os.path.join(DATA_DIR, "agent_state.json")
LEARNINGS_FILE = os.path.join(DATA_DIR, "learnings.json")
FALLBACK_FILE = os.path.join(DATA_DIR, "fallback_config.json")
REPORTS_DIR = os.path.join(DATA_DIR, "analyzer", "reports")
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

# Risk parameter bounds (safety guardrails)
RISK_BOUNDS = {
    "stop_loss_pct":       {"min": 0.015, "max": 0.05, "max_step": 0.005},
    "max_position_pct":    {"min": 0.10,  "max": 0.50, "max_step": 0.05},
    "daily_drawdown_limit": {"min": 0.05, "max": 0.15, "max_step": 0.02},
}


def get_todays_trades() -> list[dict]:
    """Query trades.db for today's trades."""
    if not os.path.exists(DB_PATH):
        return []

    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    trades = conn.execute(
        "SELECT * FROM trades WHERE timestamp LIKE ? ORDER BY id",
        (f"{today}%",)
    ).fetchall()
    conn.close()
    return [dict(t) for t in trades]


def get_equity_snapshots() -> list[dict]:
    """Get today's equity snapshots."""
    if not os.path.exists(DB_PATH):
        return []

    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM equity_snapshots WHERE timestamp LIKE ? ORDER BY id",
        (f"{today}%",)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def analyze_by_symbol(trades: list[dict]) -> dict:
    """Group and analyze trades by symbol."""
    by_symbol = defaultdict(list)
    for t in trades:
        by_symbol[t["symbol"]].append(t)

    results = {}
    for sym, sym_trades in by_symbol.items():
        sells = [t for t in sym_trades if t["side"] == "sell"]
        buys = [t for t in sym_trades if t["side"] == "buy"]
        wins = [t for t in sells if (t.get("pnl") or 0) > 0]
        losses = [t for t in sells if (t.get("pnl") or 0) < 0]
        total_pnl = sum(t.get("pnl") or 0 for t in sells)

        results[sym] = {
            "trades": len(sym_trades),
            "buys": len(buys),
            "sells": len(sells),
            "wins": len(wins),
            "losses": len(losses),
            "pnl": round(total_pnl, 2),
            "win_rate": round(len(wins) / len(sells) * 100, 1) if sells else 0,
            "avg_win": round(sum(t["pnl"] for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else 0,
        }

    return results


def analyze_by_strategy(trades: list[dict]) -> dict:
    """Group and analyze trades by strategy."""
    by_strategy = defaultdict(list)
    for t in trades:
        strat = t.get("strategy") or "unknown"
        by_strategy[strat].append(t)

    results = {}
    for strat, strat_trades in by_strategy.items():
        sells = [t for t in strat_trades if t["side"] == "sell"]
        wins = [t for t in sells if (t.get("pnl") or 0) > 0]
        losses = [t for t in sells if (t.get("pnl") or 0) < 0]
        total_pnl = sum(t.get("pnl") or 0 for t in sells)

        results[strat] = {
            "trades": len(strat_trades),
            "sells": len(sells),
            "wins": len(wins),
            "losses": len(losses),
            "pnl": round(total_pnl, 2),
            "win_rate": round(len(wins) / len(sells) * 100, 1) if sells else 0,
        }

    return results


def analyze_time_of_day(trades: list[dict]) -> dict:
    """Analyze win rates by time of day."""
    buckets = {
        "pre_market (6:00-9:30)": [],
        "morning (9:30-11:00)": [],
        "midday (11:00-14:00)": [],
        "afternoon (14:00-16:00)": [],
        "after_hours (16:00+)": [],
    }

    for t in trades:
        if t["side"] != "sell":
            continue
        try:
            ts = datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S")
            hour_min = ts.hour * 60 + ts.minute
        except (ValueError, KeyError):
            continue

        if hour_min < 570:        # 9:30
            buckets["pre_market (6:00-9:30)"].append(t)
        elif hour_min < 660:      # 11:00
            buckets["morning (9:30-11:00)"].append(t)
        elif hour_min < 840:      # 14:00
            buckets["midday (11:00-14:00)"].append(t)
        elif hour_min < 960:      # 16:00
            buckets["afternoon (14:00-16:00)"].append(t)
        else:
            buckets["after_hours (16:00+)"].append(t)

    results = {}
    for period, period_trades in buckets.items():
        if not period_trades:
            continue
        wins = [t for t in period_trades if (t.get("pnl") or 0) > 0]
        total_pnl = sum(t.get("pnl") or 0 for t in period_trades)
        results[period] = {
            "sells": len(period_trades),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(period_trades) * 100, 1),
            "pnl": round(total_pnl, 2),
        }

    return results


def detect_patterns(by_symbol: dict, by_strategy: dict, trades: list[dict]) -> list[str]:
    """Detect notable patterns in today's trading."""
    patterns = []

    # High-frequency losers
    for sym, stats in by_symbol.items():
        if stats["trades"] >= 6 and stats["win_rate"] < 40:
            patterns.append(
                f"{sym} overtrading: {stats['trades']} trades, {stats['win_rate']:.0f}% win rate, "
                f"${stats['pnl']:.2f} P&L"
            )

    # Consecutive losses
    sells = [t for t in trades if t["side"] == "sell"]
    consecutive_losses = 0
    max_consecutive = 0
    for t in sells:
        if (t.get("pnl") or 0) < 0:
            consecutive_losses += 1
            max_consecutive = max(max_consecutive, consecutive_losses)
        else:
            consecutive_losses = 0

    if max_consecutive >= 3:
        patterns.append(f"Had {max_consecutive} consecutive losing trades")

    # Strategy performance notes
    for strat, stats in by_strategy.items():
        if stats["sells"] >= 3 and stats["win_rate"] < 35:
            patterns.append(f"Strategy '{strat}' underperforming: {stats['win_rate']:.0f}% win rate on {stats['sells']} sells")
        elif stats["sells"] >= 3 and stats["win_rate"] > 65:
            patterns.append(f"Strategy '{strat}' performing well: {stats['win_rate']:.0f}% win rate on {stats['sells']} sells")

    # Large single losses
    for t in sells:
        pnl = t.get("pnl") or 0
        if pnl < -50:
            patterns.append(f"Large loss: {t['symbol']} lost ${abs(pnl):.2f} ({t.get('reason', 'unknown')})")

    return patterns


def analyze_sentiment_correlation(trades: list[dict]) -> dict | None:
    """Analyze correlation between sentiment scores and trade outcomes."""
    sells_with_sentiment = [
        t for t in trades
        if t["side"] == "sell" and t.get("sentiment_score") is not None
    ]
    if len(sells_with_sentiment) < 3:
        return None

    aligned = 0
    opposed = 0
    for t in sells_with_sentiment:
        pnl = t.get("pnl") or 0
        sent = t["sentiment_score"]
        # Buy that profited with positive sentiment, or sold at loss with negative
        if (pnl > 0 and sent > 0) or (pnl < 0 and sent < 0):
            aligned += 1
        elif (pnl > 0 and sent < 0) or (pnl < 0 and sent > 0):
            opposed += 1

    total = aligned + opposed
    return {
        "trades_with_sentiment": len(sells_with_sentiment),
        "aligned": aligned,
        "opposed": opposed,
        "alignment_rate": round(aligned / total * 100, 1) if total > 0 else 0,
        "useful": aligned > opposed,
    }


def analyze_rl_performance(trades: list[dict]) -> dict | None:
    """Compare RL-selected vs optimizer-selected strategy performance."""
    rl_trades = [t for t in trades if t.get("rl_selected")]
    optimizer_trades = [t for t in trades if not t.get("rl_selected")]

    if not rl_trades and not optimizer_trades:
        return None

    def _stats(trade_list):
        sells = [t for t in trade_list if t["side"] == "sell"]
        if not sells:
            return {"trades": len(trade_list), "sells": 0, "pnl": 0, "win_rate": 0}
        wins = [t for t in sells if (t.get("pnl") or 0) > 0]
        total_pnl = sum(t.get("pnl") or 0 for t in sells)
        return {
            "trades": len(trade_list),
            "sells": len(sells),
            "pnl": round(total_pnl, 2),
            "win_rate": round(len(wins) / len(sells) * 100, 1),
        }

    return {
        "rl_selected": _stats(rl_trades),
        "optimizer_selected": _stats(optimizer_trades),
        "rl_better": (
            _stats(rl_trades).get("pnl", 0) > _stats(optimizer_trades).get("pnl", 0)
            if rl_trades and optimizer_trades else None
        ),
    }


def compute_risk_adjustments(by_symbol: dict, sells: list[dict],
                             current_params: dict) -> dict:
    """Compute bounded risk parameter adjustments based on patterns."""
    adjustments = {}
    total_sells = len(sells)
    if total_sells == 0:
        return adjustments

    wins = [t for t in sells if (t.get("pnl") or 0) > 0]
    losses = [t for t in sells if (t.get("pnl") or 0) < 0]
    overall_win_rate = len(wins) / total_sells

    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses)) if losses else 0

    current_sl = current_params.get("stop_loss_pct", 0.025)

    # If win rate < 40%, losses may be stop-loss whipsaws — widen slightly
    if overall_win_rate < 0.40 and total_sells >= 5:
        new_sl = min(current_sl + 0.005, RISK_BOUNDS["stop_loss_pct"]["max"])
        if new_sl != current_sl:
            adjustments["stop_loss_pct"] = {
                "current": current_sl,
                "suggested": new_sl,
                "reason": f"Low win rate ({overall_win_rate:.0%}) — widening stop-loss to reduce whipsaws",
            }

    # If win rate > 60% but avg_win < avg_loss, tighten stops
    elif overall_win_rate > 0.60 and avg_win < avg_loss and total_sells >= 5:
        new_sl = max(current_sl - 0.005, RISK_BOUNDS["stop_loss_pct"]["min"])
        if new_sl != current_sl:
            adjustments["stop_loss_pct"] = {
                "current": current_sl,
                "suggested": new_sl,
                "reason": f"High win rate ({overall_win_rate:.0%}) but avg loss (${avg_loss:.2f}) > avg win (${avg_win:.2f})",
            }

    return adjustments


def apply_risk_adjustments(adjustments: dict):
    """Apply risk parameter changes to .env (bounded and validated)."""
    if not adjustments or not os.path.exists(ENV_FILE):
        return

    with open(ENV_FILE) as f:
        content = f.read()

    changed = False
    for param, info in adjustments.items():
        env_key = param.upper()
        new_val = info["suggested"]
        bounds = RISK_BOUNDS.get(param, {})

        # Validate bounds
        if new_val < bounds.get("min", 0) or new_val > bounds.get("max", 1):
            print(f"  ⚠️ Skipping {param}: {new_val} out of bounds [{bounds['min']}, {bounds['max']}]")
            continue

        if f"{env_key}=" in content:
            content = re.sub(rf'{env_key}=[\d.]+', f'{env_key}={new_val}', content)
        else:
            content += f"\n{env_key}={new_val}\n"

        print(f"  Adjusted {param}: {info['current']} → {new_val} ({info['reason']})")
        changed = True

    if changed:
        tmp = ENV_FILE + ".tmp"
        with open(tmp, "w") as f:
            f.write(content)
        os.rename(tmp, ENV_FILE)


def update_learnings(findings: dict):
    """Append today's findings to learnings.json (prune to 90 days)."""
    learnings = {"version": 1, "entries": []}
    if os.path.exists(LEARNINGS_FILE):
        try:
            with open(LEARNINGS_FILE) as f:
                learnings = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    today = datetime.now().strftime("%Y-%m-%d")

    # Remove any existing entry for today (idempotent)
    learnings["entries"] = [e for e in learnings.get("entries", []) if e.get("date") != today]

    # Append today's findings
    learnings["entries"].append({
        "date": today,
        "source": "trade_analyzer",
        "findings": findings,
    })

    # Prune entries older than 90 days
    cutoff = datetime.now().strftime("%Y-%m-%d")
    # Simple approach: keep last 90 entries
    learnings["entries"] = learnings["entries"][-90:]

    tmp = LEARNINGS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(learnings, f, indent=2)
    os.rename(tmp, LEARNINGS_FILE)


def update_fallback_config(net_pnl: float):
    """Update fallback config only if today was profitable."""
    if net_pnl <= 0:
        print("  Net P&L negative — keeping existing fallback config")
        return

    # Read current .env for risk params
    current_params = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    current_params[key.strip()] = val.strip()

    from config import Config

    fallback = {
        "updated": datetime.now().isoformat(),
        "equity_symbols": Config.EQUITY_SYMBOLS,
        "strategy_map": {},
        "risk_params": {
            "stop_loss_pct": float(current_params.get("STOP_LOSS_PCT", "0.025")),
            "max_position_pct": float(current_params.get("MAX_POSITION_PCT", "0.50")),
            "daily_drawdown_limit": float(current_params.get("DAILY_DRAWDOWN_LIMIT", "0.10")),
            "cooldown_equity": 300,
        },
    }

    # Load current strategy map
    try:
        from strategies.router import STRATEGY_MAP
        fallback["strategy_map"] = dict(STRATEGY_MAP)
    except ImportError:
        pass

    tmp = FALLBACK_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(fallback, f, indent=2)
    os.rename(tmp, FALLBACK_FILE)
    print(f"  Updated fallback config (profitable day: ${net_pnl:.2f})")


def update_agent_state(status: str, trades_analyzed: int = 0, error: str = None, duration: float = 0):
    """Update agent_state.json."""
    state = {}
    if os.path.exists(AGENT_STATE_FILE):
        try:
            with open(AGENT_STATE_FILE) as f:
                state = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    state["trade_analyzer"] = {
        "last_run": datetime.now().isoformat(),
        "status": status,
        "duration_seconds": round(duration),
        "trades_analyzed": trades_analyzed,
        "error": error,
    }

    with open(AGENT_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def main():
    start_time = time.time()
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"[Trade Analyzer] Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Get today's trades
        trades = get_todays_trades()
        print(f"  Found {len(trades)} trades for {today}")

        if not trades:
            print("  No trades to analyze")
            update_agent_state("success", trades_analyzed=0)
            return

        sells = [t for t in trades if t["side"] == "sell"]
        buys = [t for t in trades if t["side"] == "buy"]

        # Compute metrics
        gross_pnl = sum(t.get("pnl") or 0 for t in sells)
        wins = [t for t in sells if (t.get("pnl") or 0) > 0]
        losses = [t for t in sells if (t.get("pnl") or 0) < 0]
        win_rate = len(wins) / len(sells) * 100 if sells else 0

        best_trade = max(sells, key=lambda t: t.get("pnl") or 0) if sells else None
        worst_trade = min(sells, key=lambda t: t.get("pnl") or 0) if sells else None

        by_symbol = analyze_by_symbol(trades)
        by_strategy = analyze_by_strategy(trades)
        time_analysis = analyze_time_of_day(trades)
        patterns = detect_patterns(by_symbol, by_strategy, trades)

        # Get current risk params
        current_risk = {
            "stop_loss_pct": float(os.getenv("STOP_LOSS_PCT", "0.025")),
            "max_position_pct": float(os.getenv("MAX_POSITION_PCT", "0.50")),
            "daily_drawdown_limit": float(os.getenv("DAILY_DRAWDOWN_LIMIT", "0.10")),
        }
        risk_adjustments = compute_risk_adjustments(by_symbol, sells, current_risk)

        # Build report
        report = {
            "date": today,
            "summary": {
                "total_trades": len(trades),
                "buys": len(buys),
                "sells": len(sells),
                "gross_pnl": round(gross_pnl, 2),
                "win_rate": round(win_rate, 1),
                "wins": len(wins),
                "losses": len(losses),
                "best_trade": {
                    "symbol": best_trade["symbol"],
                    "pnl": round(best_trade.get("pnl") or 0, 2),
                    "reason": best_trade.get("reason", ""),
                } if best_trade else None,
                "worst_trade": {
                    "symbol": worst_trade["symbol"],
                    "pnl": round(worst_trade.get("pnl") or 0, 2),
                    "reason": worst_trade.get("reason", ""),
                } if worst_trade else None,
            },
            "by_symbol": by_symbol,
            "by_strategy": by_strategy,
            "time_of_day": time_analysis,
            "patterns": patterns,
            "risk_adjustments": risk_adjustments,
        }

        # Write report
        os.makedirs(REPORTS_DIR, exist_ok=True)
        report_file = os.path.join(REPORTS_DIR, f"{today}.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report: {report_file}")

        # Build learnings
        best_symbols = [sym for sym, stats in by_symbol.items()
                        if stats["pnl"] > 0 and stats["win_rate"] >= 50]
        worst_symbols = [sym for sym, stats in by_symbol.items()
                         if stats["pnl"] < -20 or (stats["trades"] >= 4 and stats["win_rate"] < 30)]

        strategy_notes = {}
        for strat, stats in by_strategy.items():
            if stats["sells"] >= 3:
                if stats["win_rate"] >= 60:
                    strategy_notes[strat] = f"Strong: {stats['win_rate']:.0f}% win rate, ${stats['pnl']:.2f} P&L"
                elif stats["win_rate"] < 40:
                    strategy_notes[strat] = f"Weak: {stats['win_rate']:.0f}% win rate, ${stats['pnl']:.2f} P&L"

        avg_drawdown = abs(min((t.get("pnl") or 0) for t in sells)) / 100 if sells else 0
        max_single_loss = min((t.get("pnl") or 0) for t in sells) if sells else 0

        # v2: Sentiment correlation analysis
        sentiment_correlation = analyze_sentiment_correlation(trades)
        if sentiment_correlation:
            report["sentiment_analysis"] = sentiment_correlation

        # v2: RL vs optimizer comparison
        rl_comparison = analyze_rl_performance(trades)
        if rl_comparison:
            report["rl_comparison"] = rl_comparison

        findings = {
            "best_performing_symbols": best_symbols,
            "worst_performing_symbols": worst_symbols,
            "strategy_notes": strategy_notes,
            "risk_observations": {
                "avg_drawdown_pct": round(avg_drawdown, 4),
                "max_single_loss": round(max_single_loss, 2),
                "overall_win_rate": round(win_rate, 1),
                "total_pnl": round(gross_pnl, 2),
            },
            "patterns": patterns,
            "symbols_to_avoid": worst_symbols,
            "symbols_to_favor": best_symbols,
        }

        if sentiment_correlation:
            findings["sentiment_correlation"] = sentiment_correlation
        if rl_comparison:
            findings["rl_comparison"] = rl_comparison

        # Update learnings
        update_learnings(findings)
        print(f"  Learnings updated: {len(best_symbols)} favored, {len(worst_symbols)} to avoid")

        # Apply risk adjustments (bounded)
        if risk_adjustments:
            print(f"  Risk adjustments:")
            apply_risk_adjustments(risk_adjustments)
        else:
            print("  No risk adjustments needed")

        # Update fallback config if profitable day
        update_fallback_config(gross_pnl)

        duration = time.time() - start_time
        update_agent_state("success", trades_analyzed=len(trades), duration=duration)

        # Print summary
        print(f"\n[Trade Analyzer] Complete in {duration:.1f}s")
        print(f"  {'='*50}")
        print(f"  Daily P&L: ${gross_pnl:.2f}")
        print(f"  Trades: {len(trades)} ({len(buys)} buys, {len(sells)} sells)")
        print(f"  Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
        if best_trade:
            print(f"  Best:  {best_trade['symbol']} ${best_trade.get('pnl', 0):.2f}")
        if worst_trade:
            print(f"  Worst: {worst_trade['symbol']} ${worst_trade.get('pnl', 0):.2f}")
        print(f"  {'='*50}")

        print("\n  By Symbol:")
        for sym, stats in sorted(by_symbol.items(), key=lambda x: x[1]["pnl"], reverse=True):
            icon = "🟢" if stats["pnl"] > 0 else "🔴"
            print(f"    {icon} {sym:8s} | {stats['trades']:2d} trades | "
                  f"{stats['win_rate']:5.1f}% WR | ${stats['pnl']:+8.2f}")

        if patterns:
            print("\n  Patterns Detected:")
            for p in patterns:
                print(f"    ⚠️  {p}")

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n[Trade Analyzer] FAILED: {e}")
        import traceback
        traceback.print_exc()
        update_agent_state("failed", error=str(e), duration=duration)
        raise


if __name__ == "__main__":
    main()
