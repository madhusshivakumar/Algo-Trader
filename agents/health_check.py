#!/usr/bin/env python3
"""Agent 4: Health Check — runs tests, validates system integrity, reports issues.

Runs daily at 5:00 AM weekdays (before all other agents).
Also runnable on-demand via: ./bot.sh health
"""

import json
import os
import sys
import subprocess
import time
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
AGENT_STATE_FILE = os.path.join(DATA_DIR, "agent_state.json")
HEALTH_REPORT_FILE = os.path.join(DATA_DIR, "health_report.json")
DB_PATH = os.path.join(PROJECT_ROOT, "trades.db")


def run_tests() -> dict:
    """Run the full pytest suite and capture results."""
    print("  Running pytest suite...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    output = result.stdout + result.stderr
    passed = output.count(" PASSED")
    failed = output.count(" FAILED")
    errors = output.count(" ERROR")

    # Also try to parse the summary line (e.g., "230 passed, 1 failed in 3.5s")
    import re
    summary_match = re.search(r'(\d+) passed', output)
    if summary_match and passed == 0:
        passed = int(summary_match.group(1))
    fail_match = re.search(r'(\d+) failed', output)
    if fail_match and failed == 0:
        failed = int(fail_match.group(1))

    # Extract failure details
    failures = []
    if failed > 0 or errors > 0:
        for line in output.split("\n"):
            if "FAILED" in line or "ERROR" in line:
                failures.append(line.strip())

    return {
        "exit_code": result.returncode,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "failures": failures,
        "output": output[-2000:] if len(output) > 2000 else output,  # Last 2000 chars
    }


def check_file_integrity() -> list[dict]:
    """Verify all critical files exist and are valid."""
    issues = []

    # Critical files
    critical_files = {
        "config.py": "Configuration",
        "core/engine.py": "Trading engine",
        "core/broker.py": "Broker interface",
        "core/risk_manager.py": "Risk manager",
        "strategies/router.py": "Strategy router",
        "utils/logger.py": "Logger",
        "main.py": "Entry point",
        ".env": "Environment config",
    }

    for filepath, description in critical_files.items():
        full_path = os.path.join(PROJECT_ROOT, filepath)
        if not os.path.exists(full_path):
            issues.append({
                "severity": "critical",
                "file": filepath,
                "issue": f"Missing: {description}",
            })
        elif os.path.getsize(full_path) == 0:
            issues.append({
                "severity": "critical",
                "file": filepath,
                "issue": f"Empty file: {description}",
            })

    # Strategy files
    strategy_files = [
        "strategies/mean_reversion_aggressive.py",
        "strategies/mean_reversion.py",
        "strategies/volume_profile.py",
        "strategies/momentum.py",
        "strategies/macd_crossover.py",
        "strategies/triple_ema.py",
        "strategies/rsi_divergence.py",
        "strategies/scalper.py",
        "strategies/ensemble.py",
    ]

    for sf in strategy_files:
        full_path = os.path.join(PROJECT_ROOT, sf)
        if not os.path.exists(full_path):
            issues.append({
                "severity": "high",
                "file": sf,
                "issue": "Missing strategy file",
            })

    # Agent files
    agent_files = [
        "agents/strategy_optimizer.py",
        "agents/market_scanner.py",
        "agents/trade_analyzer.py",
    ]

    for af in agent_files:
        full_path = os.path.join(PROJECT_ROOT, af)
        if not os.path.exists(full_path):
            issues.append({
                "severity": "high",
                "file": af,
                "issue": "Missing agent file",
            })

    # v2 agent files
    v2_agent_files = [
        "agents/sentiment_agent.py",
        "agents/llm_analyst.py",
        "agents/rl_trainer.py",
    ]
    for af in v2_agent_files:
        full_path = os.path.join(PROJECT_ROOT, af)
        if not os.path.exists(full_path):
            issues.append({
                "severity": "medium",
                "file": af,
                "issue": "Missing v2 agent file",
            })

    # v2 core files
    v2_core_files = [
        "core/signal_modifiers.py",
        "core/news_client.py",
        "core/sentiment_analyzer.py",
        "core/llm_client.py",
        "core/rl_features.py",
        "core/rl_environment.py",
        "core/rl_strategy_selector.py",
    ]
    for cf in v2_core_files:
        full_path = os.path.join(PROJECT_ROOT, cf)
        if not os.path.exists(full_path):
            issues.append({
                "severity": "medium",
                "file": cf,
                "issue": "Missing v2 core file",
            })

    # Data directory structure
    data_dirs = [
        "data", "data/scanner", "data/optimizer",
        "data/analyzer", "data/analyzer/reports",
        "data/history", "data/history/symbols", "data/history/assignments",
        "data/sentiment", "data/llm_analyst", "data/rl_models",
    ]

    for dd in data_dirs:
        full_path = os.path.join(PROJECT_ROOT, dd)
        if not os.path.isdir(full_path):
            issues.append({
                "severity": "medium",
                "file": dd,
                "issue": "Missing data directory",
            })

    return issues


def check_imports() -> list[dict]:
    """Verify all critical imports work."""
    issues = []

    import_checks = [
        ("config", "from config import Config"),
        ("broker", "from core.broker import Broker"),
        ("engine", "from core.engine import TradingEngine"),
        ("risk_manager", "from core.risk_manager import RiskManager"),
        ("router", "from strategies.router import compute_signals, STRATEGY_REGISTRY, STRATEGY_MAP"),
        ("logger", "from utils.logger import log"),
        ("backtest", "from compare_strategies import backtest_strategy, STRATEGIES"),
        ("optimizer", "from agents.strategy_optimizer import compute_composite_score"),
        ("scanner", "from agents.market_scanner import apply_sector_diversity"),
        ("analyzer", "from agents.trade_analyzer import analyze_by_symbol"),
        ("signal_modifiers", "from core.signal_modifiers import apply_sentiment, apply_llm_conviction"),
        ("news_client", "from core.news_client import fetch_news, fetch_headlines"),
        ("sentiment_analyzer", "from core.sentiment_analyzer import score_text, score_headlines"),
        ("llm_client", "from core.llm_client import call_llm, get_daily_spend"),
        ("rl_features", "from core.rl_features import extract_features, STRATEGY_KEYS"),
        ("rl_selector", "from core.rl_strategy_selector import RLStrategySelector"),
    ]

    for name, import_stmt in import_checks:
        try:
            exec(import_stmt)
        except Exception as e:
            issues.append({
                "severity": "critical",
                "module": name,
                "issue": f"Import failed: {e}",
            })

    return issues


def check_data_files() -> list[dict]:
    """Validate JSON data files are parseable and have expected structure."""
    issues = []

    json_files = {
        "data/agent_state.json": ["strategy_optimizer", "market_scanner", "trade_analyzer"],
        "data/learnings.json": ["version", "entries"],
        "data/fallback_config.json": ["equity_symbols", "strategy_map", "risk_params"],
    }

    for filepath, required_keys in json_files.items():
        full_path = os.path.join(PROJECT_ROOT, filepath)
        if not os.path.exists(full_path):
            issues.append({
                "severity": "medium",
                "file": filepath,
                "issue": "File does not exist",
            })
            continue

        try:
            with open(full_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            issues.append({
                "severity": "high",
                "file": filepath,
                "issue": f"Invalid JSON: {e}",
            })
            continue

        for key in required_keys:
            if key not in data:
                issues.append({
                    "severity": "medium",
                    "file": filepath,
                    "issue": f"Missing key: '{key}'",
                })

    return issues


def check_database() -> list[dict]:
    """Validate the trades database."""
    issues = []

    if not os.path.exists(DB_PATH):
        issues.append({
            "severity": "medium",
            "file": "trades.db",
            "issue": "Database does not exist",
        })
        return issues

    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)

        # Check tables exist
        tables = {t[0] for t in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        if "trades" not in tables:
            issues.append({"severity": "critical", "file": "trades.db",
                           "issue": "Missing 'trades' table"})
        if "equity_snapshots" not in tables:
            issues.append({"severity": "high", "file": "trades.db",
                           "issue": "Missing 'equity_snapshots' table"})

        # Check trades table has required columns (including v2)
        if "trades" in tables:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()]
            for req_col in ["strategy", "sentiment_score", "llm_conviction", "rl_selected"]:
                if req_col not in cols:
                    issues.append({"severity": "high", "file": "trades.db",
                                   "issue": f"Missing '{req_col}' column in trades table"})

        conn.close()
    except Exception as e:
        issues.append({"severity": "critical", "file": "trades.db",
                       "issue": f"Database error: {e}"})

    return issues


def check_router_consistency() -> list[dict]:
    """Verify the router's strategy map is consistent with the registry."""
    issues = []

    try:
        from strategies.router import STRATEGY_MAP, STRATEGY_REGISTRY, DEFAULT_STRATEGY

        if DEFAULT_STRATEGY not in STRATEGY_REGISTRY:
            issues.append({"severity": "critical", "module": "router",
                           "issue": f"Default strategy '{DEFAULT_STRATEGY}' not in registry"})

        for sym, strat in STRATEGY_MAP.items():
            if strat not in STRATEGY_REGISTRY:
                issues.append({"severity": "high", "module": "router",
                               "issue": f"Symbol '{sym}' assigned to unknown strategy '{strat}'"})

        if len(STRATEGY_MAP) == 0:
            issues.append({"severity": "high", "module": "router",
                           "issue": "Strategy map is empty"})

    except Exception as e:
        issues.append({"severity": "critical", "module": "router",
                       "issue": f"Router check failed: {e}"})

    return issues


def check_agent_staleness() -> list[dict]:
    """Check if any agents haven't run recently."""
    issues = []

    if not os.path.exists(AGENT_STATE_FILE):
        return issues

    try:
        with open(AGENT_STATE_FILE) as f:
            state = json.load(f)

        now = datetime.now()
        stale_threshold = timedelta(days=3)

        for agent_name, agent_state in state.items():
            last_run = agent_state.get("last_run")
            status = agent_state.get("status", "unknown")

            if last_run is None:
                issues.append({"severity": "low", "agent": agent_name,
                               "issue": "Agent has never run"})
                continue

            try:
                last_dt = datetime.fromisoformat(last_run)
                if now - last_dt > stale_threshold:
                    days_ago = (now - last_dt).days
                    issues.append({"severity": "medium", "agent": agent_name,
                                   "issue": f"Last ran {days_ago} days ago"})
            except ValueError:
                pass

            if status == "failed":
                error = agent_state.get("error", "unknown")
                issues.append({"severity": "high", "agent": agent_name,
                               "issue": f"Last run failed: {error}"})

    except (json.JSONDecodeError, KeyError):
        issues.append({"severity": "medium", "file": "agent_state.json",
                       "issue": "Agent state file is corrupt"})

    return issues


def check_env_config() -> list[dict]:
    """Verify .env has required keys."""
    issues = []
    env_path = os.path.join(PROJECT_ROOT, ".env")

    if not os.path.exists(env_path):
        issues.append({"severity": "critical", "file": ".env", "issue": "Missing .env file"})
        return issues

    with open(env_path) as f:
        content = f.read()

    required_keys = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "EQUITY_SYMBOLS"]
    for key in required_keys:
        if key not in content:
            issues.append({"severity": "critical" if "API" in key else "high",
                           "file": ".env", "issue": f"Missing {key}"})

    # Check EQUITY_SYMBOLS has reasonable count
    for line in content.split("\n"):
        if line.startswith("EQUITY_SYMBOLS="):
            symbols = [s.strip() for s in line.split("=", 1)[1].split(",") if s.strip()]
            if len(symbols) < 3:
                issues.append({"severity": "high", "file": ".env",
                               "issue": f"Only {len(symbols)} equity symbols configured"})
            elif len(symbols) > 25:
                issues.append({"severity": "medium", "file": ".env",
                               "issue": f"{len(symbols)} equity symbols — may be too many"})

    return issues


def update_agent_state(status: str, duration: float, issues_count: int):
    """Update agent_state.json."""
    state = {}
    if os.path.exists(AGENT_STATE_FILE):
        try:
            with open(AGENT_STATE_FILE) as f:
                state = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    state["health_check"] = {
        "last_run": datetime.now().isoformat(),
        "status": status,
        "duration_seconds": round(duration),
        "issues_found": issues_count,
        "error": None,
    }

    with open(AGENT_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def main():
    start_time = time.time()
    print(f"[Health Check] Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    all_issues = []

    # 1. Run pytest
    print("\n📋 Running test suite...")
    test_results = run_tests()
    print(f"   Tests: {test_results['passed']} passed, {test_results['failed']} failed, {test_results['errors']} errors")
    if test_results["failures"]:
        for f in test_results["failures"][:10]:
            print(f"   ❌ {f}")
            all_issues.append({"severity": "high", "source": "tests", "issue": f})

    # 2. File integrity
    print("\n📁 Checking file integrity...")
    file_issues = check_file_integrity()
    if file_issues:
        for issue in file_issues:
            print(f"   ⚠️  [{issue['severity']}] {issue.get('file', '')}: {issue['issue']}")
    else:
        print("   ✓ All files present")
    all_issues.extend(file_issues)

    # 3. Import checks
    print("\n📦 Checking imports...")
    import_issues = check_imports()
    if import_issues:
        for issue in import_issues:
            print(f"   ❌ [{issue['severity']}] {issue['module']}: {issue['issue']}")
    else:
        print("   ✓ All imports OK")
    all_issues.extend(import_issues)

    # 4. Data files
    print("\n📊 Checking data files...")
    data_issues = check_data_files()
    if data_issues:
        for issue in data_issues:
            print(f"   ⚠️  [{issue['severity']}] {issue.get('file', '')}: {issue['issue']}")
    else:
        print("   ✓ Data files valid")
    all_issues.extend(data_issues)

    # 5. Database
    print("\n🗄️  Checking database...")
    db_issues = check_database()
    if db_issues:
        for issue in db_issues:
            print(f"   ⚠️  [{issue['severity']}] {issue['issue']}")
    else:
        print("   ✓ Database OK")
    all_issues.extend(db_issues)

    # 6. Router consistency
    print("\n🔀 Checking router consistency...")
    router_issues = check_router_consistency()
    if router_issues:
        for issue in router_issues:
            print(f"   ⚠️  [{issue['severity']}] {issue['issue']}")
    else:
        print("   ✓ Router consistent")
    all_issues.extend(router_issues)

    # 7. Agent staleness
    print("\n🤖 Checking agent status...")
    agent_issues = check_agent_staleness()
    if agent_issues:
        for issue in agent_issues:
            print(f"   ⚠️  [{issue['severity']}] {issue.get('agent', '')}: {issue['issue']}")
    else:
        print("   ✓ Agents up to date")
    all_issues.extend(agent_issues)

    # 8. Env config
    print("\n🔧 Checking environment config...")
    env_issues = check_env_config()
    if env_issues:
        for issue in env_issues:
            print(f"   ⚠️  [{issue['severity']}] {issue['issue']}")
    else:
        print("   ✓ Environment config valid")
    all_issues.extend(env_issues)

    # Summary
    duration = time.time() - start_time
    critical = sum(1 for i in all_issues if i.get("severity") == "critical")
    high = sum(1 for i in all_issues if i.get("severity") == "high")
    medium = sum(1 for i in all_issues if i.get("severity") == "medium")
    low = sum(1 for i in all_issues if i.get("severity") == "low")

    print(f"\n{'='*60}")
    if not all_issues:
        print("✅ SYSTEM HEALTHY — all checks passed")
        status = "healthy"
    elif critical > 0:
        print(f"🔴 CRITICAL ISSUES: {critical} critical, {high} high, {medium} medium, {low} low")
        status = "critical"
    elif high > 0:
        print(f"🟠 ISSUES FOUND: {high} high, {medium} medium, {low} low")
        status = "degraded"
    else:
        print(f"🟡 MINOR ISSUES: {medium} medium, {low} low")
        status = "warning"

    print(f"  Duration: {duration:.1f}s")
    print(f"  Test results: {test_results['passed']} passed, {test_results['failed']} failed")

    # Write report
    report = {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "duration_seconds": round(duration, 1),
        "test_results": {
            "passed": test_results["passed"],
            "failed": test_results["failed"],
            "errors": test_results["errors"],
            "failures": test_results["failures"],
        },
        "issues": all_issues,
        "summary": {
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
            "total": len(all_issues),
        },
    }

    with open(HEALTH_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {HEALTH_REPORT_FILE}")

    update_agent_state(status, duration, len(all_issues))

    return 0 if critical == 0 and test_results["failed"] == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
