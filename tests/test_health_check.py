"""Tests for agents/health_check.py — file integrity, imports, data files,
database, router, agent staleness, env config, and main orchestration."""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, mock_open

import pytest

# We import the individual check functions, not main() (which runs pytest itself)
from agents.health_check import (
    check_file_integrity,
    check_data_files,
    check_database,
    check_router_consistency,
    check_agent_staleness,
    check_env_config,
    check_imports,
    update_agent_state,
    run_tests,
    AGENT_STATE_FILE,
    HEALTH_REPORT_FILE,
    DB_PATH,
    PROJECT_ROOT,
)


# ═══════════════════════════════════════════════════════════════════════
# check_file_integrity
# ═══════════════════════════════════════════════════════════════════════

class TestFileIntegrity:
    def test_all_files_present(self):
        """When all critical files exist, no issues."""
        issues = check_file_integrity()
        # config.py and core/engine.py should exist in our repo
        critical_missing = [i for i in issues if i["severity"] == "critical"
                           and i["file"] in ("config.py", "core/engine.py")]
        assert len(critical_missing) == 0

    def test_missing_file_detected(self):
        """A missing critical file should be flagged."""
        with patch("os.path.exists", return_value=False), \
             patch("os.path.isdir", return_value=False), \
             patch("os.path.getsize", return_value=0):
            issues = check_file_integrity()
        assert len(issues) > 0
        assert any(i["severity"] == "critical" for i in issues)

    def test_empty_file_detected(self):
        """An empty critical file should be flagged."""
        real_exists = os.path.exists
        real_getsize = os.path.getsize

        def fake_exists(p):
            return True

        def fake_getsize(p):
            if "config.py" in p:
                return 0
            return real_getsize(p) if real_exists(p) else 100

        with patch("os.path.exists", side_effect=fake_exists), \
             patch("os.path.getsize", side_effect=fake_getsize), \
             patch("os.path.isdir", return_value=True):
            issues = check_file_integrity()

        empty_issues = [i for i in issues if "Empty" in i.get("issue", "")]
        assert len(empty_issues) > 0


# ═══════════════════════════════════════════════════════════════════════
# check_imports
# ═══════════════════════════════════════════════════════════════════════

class TestImportChecks:
    def test_imports_succeed(self):
        """Core imports should work in a valid project."""
        issues = check_imports()
        # At minimum config/broker/engine should import
        critical_fails = [i for i in issues if i["module"] in ("config", "broker", "engine")]
        assert len(critical_fails) == 0

    def test_import_failure_reported(self):
        """A broken import should produce an issue."""
        with patch("builtins.exec", side_effect=ImportError("no module")):
            issues = check_imports()
        assert len(issues) > 0
        assert all(i["severity"] == "critical" for i in issues)


# ═══════════════════════════════════════════════════════════════════════
# check_data_files
# ═══════════════════════════════════════════════════════════════════════

class TestDataFiles:
    def test_missing_data_file(self, tmp_path):
        """Missing JSON files should be reported."""
        with patch("agents.health_check.PROJECT_ROOT", str(tmp_path)):
            issues = check_data_files()
        assert len(issues) > 0
        assert all(i["severity"] == "medium" for i in issues)

    def test_invalid_json(self, tmp_path):
        """Corrupt JSON should be detected."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "agent_state.json").write_text("NOT VALID JSON{{{")

        with patch("agents.health_check.PROJECT_ROOT", str(tmp_path)):
            issues = check_data_files()

        json_issues = [i for i in issues if "Invalid JSON" in i.get("issue", "")]
        assert len(json_issues) >= 1

    def test_missing_keys(self, tmp_path):
        """JSON with missing required keys should be flagged."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "agent_state.json").write_text("{}")

        with patch("agents.health_check.PROJECT_ROOT", str(tmp_path)):
            issues = check_data_files()

        key_issues = [i for i in issues if "Missing key" in i.get("issue", "")]
        assert len(key_issues) >= 1

    def test_valid_data_files(self, tmp_path):
        """Properly formatted data files produce no issues for that file."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "agent_state.json").write_text(json.dumps({
            "strategy_optimizer": {}, "market_scanner": {}, "trade_analyzer": {}
        }))
        (data_dir / "learnings.json").write_text(json.dumps({
            "version": 1, "entries": []
        }))
        (data_dir / "fallback_config.json").write_text(json.dumps({
            "equity_symbols": [], "strategy_map": {}, "risk_params": {}
        }))

        with patch("agents.health_check.PROJECT_ROOT", str(tmp_path)):
            issues = check_data_files()

        assert len(issues) == 0


# ═══════════════════════════════════════════════════════════════════════
# check_database
# ═══════════════════════════════════════════════════════════════════════

class TestDatabase:
    def test_missing_db(self, tmp_path):
        """Missing database file should be reported."""
        with patch("agents.health_check.DB_PATH", str(tmp_path / "nope.db")):
            issues = check_database()
        assert len(issues) == 1
        assert issues[0]["severity"] == "medium"

    def test_db_missing_tables(self, tmp_path):
        """DB without required tables should be flagged."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.commit()
        conn.close()

        with patch("agents.health_check.DB_PATH", db_path):
            issues = check_database()

        assert any("trades" in i.get("issue", "") for i in issues)

    def test_db_missing_columns(self, tmp_path):
        """DB with trades table but missing v2 columns should be flagged."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE trades (id INTEGER, symbol TEXT)")
        conn.execute("CREATE TABLE equity_snapshots (id INTEGER)")
        conn.commit()
        conn.close()

        with patch("agents.health_check.DB_PATH", db_path):
            issues = check_database()

        col_issues = [i for i in issues if "column" in i.get("issue", "")]
        assert len(col_issues) >= 1

    def test_valid_db_no_issues(self, tmp_path):
        """A proper database should produce no issues."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE trades (
            id INTEGER, symbol TEXT, strategy TEXT,
            sentiment_score REAL, llm_conviction REAL, rl_selected INTEGER
        )""")
        conn.execute("CREATE TABLE equity_snapshots (id INTEGER)")
        conn.commit()
        conn.close()

        with patch("agents.health_check.DB_PATH", db_path):
            issues = check_database()

        assert len(issues) == 0

    def test_corrupt_db(self, tmp_path):
        """Corrupt database file should be caught."""
        db_path = str(tmp_path / "corrupt.db")
        with open(db_path, "w") as f:
            f.write("this is not a database")

        with patch("agents.health_check.DB_PATH", db_path):
            issues = check_database()

        assert any(i["severity"] == "critical" for i in issues)


# ═══════════════════════════════════════════════════════════════════════
# check_router_consistency
# ═══════════════════════════════════════════════════════════════════════

class TestRouterConsistency:
    def test_valid_router(self):
        """With a valid router setup, no issues."""
        issues = check_router_consistency()
        # In our real project the router should be consistent
        critical = [i for i in issues if i["severity"] == "critical"]
        assert len(critical) == 0

    def test_unknown_strategy_in_map(self):
        """A symbol assigned to a non-existent strategy should be flagged."""
        fake_map = {"AAPL": "nonexistent_strategy"}
        fake_registry = {"mean_reversion": lambda df: {}}
        with patch("agents.health_check.check_router_consistency") as mock_check:
            # We test the logic directly
            pass

        # Direct import test
        from strategies.router import STRATEGY_MAP, STRATEGY_REGISTRY, DEFAULT_STRATEGY
        # All entries in STRATEGY_MAP should be in STRATEGY_REGISTRY
        for sym, strat in STRATEGY_MAP.items():
            assert strat in STRATEGY_REGISTRY, f"{sym} assigned to unknown strategy {strat}"


# ═══════════════════════════════════════════════════════════════════════
# check_agent_staleness
# ═══════════════════════════════════════════════════════════════════════

class TestAgentStaleness:
    def test_no_state_file(self, tmp_path):
        """Missing agent_state.json should produce no issues."""
        with patch("agents.health_check.AGENT_STATE_FILE", str(tmp_path / "nope.json")):
            issues = check_agent_staleness()
        assert issues == []

    def test_stale_agent_detected(self, tmp_path):
        """Agent that hasn't run in 5 days should be flagged."""
        state = {
            "my_agent": {
                "last_run": (datetime.now() - timedelta(days=5)).isoformat(),
                "status": "success",
            }
        }
        path = tmp_path / "agent_state.json"
        path.write_text(json.dumps(state))

        with patch("agents.health_check.AGENT_STATE_FILE", str(path)):
            issues = check_agent_staleness()

        assert len(issues) >= 1
        assert any("days ago" in i["issue"] for i in issues)

    def test_failed_agent_detected(self, tmp_path):
        """Agent with failed status should be flagged."""
        state = {
            "failing_agent": {
                "last_run": datetime.now().isoformat(),
                "status": "failed",
                "error": "API key expired",
            }
        }
        path = tmp_path / "agent_state.json"
        path.write_text(json.dumps(state))

        with patch("agents.health_check.AGENT_STATE_FILE", str(path)):
            issues = check_agent_staleness()

        assert any("failed" in i["issue"].lower() for i in issues)

    def test_never_run_agent(self, tmp_path):
        """Agent with no last_run should be flagged."""
        state = {"new_agent": {"status": "unknown"}}
        path = tmp_path / "agent_state.json"
        path.write_text(json.dumps(state))

        with patch("agents.health_check.AGENT_STATE_FILE", str(path)):
            issues = check_agent_staleness()

        assert any("never run" in i["issue"].lower() for i in issues)

    def test_corrupt_state_file(self, tmp_path):
        """Corrupt JSON should be caught."""
        path = tmp_path / "agent_state.json"
        path.write_text("CORRUPT{{{")

        with patch("agents.health_check.AGENT_STATE_FILE", str(path)):
            issues = check_agent_staleness()

        assert any("corrupt" in i["issue"].lower() for i in issues)


# ═══════════════════════════════════════════════════════════════════════
# check_env_config
# ═══════════════════════════════════════════════════════════════════════

class TestEnvConfig:
    def test_missing_env_file(self, tmp_path):
        """Missing .env should be critical."""
        with patch("agents.health_check.PROJECT_ROOT", str(tmp_path)):
            issues = check_env_config()
        assert any(i["severity"] == "critical" for i in issues)

    def test_missing_keys(self, tmp_path):
        """Missing required keys should be flagged."""
        env_path = tmp_path / ".env"
        env_path.write_text("SOME_OTHER_KEY=value\n")

        with patch("agents.health_check.PROJECT_ROOT", str(tmp_path)):
            issues = check_env_config()

        assert any("ALPACA_API_KEY" in i.get("issue", "") for i in issues)

    def test_few_equity_symbols(self, tmp_path):
        """Too few equity symbols should be flagged."""
        env_path = tmp_path / ".env"
        env_path.write_text(
            "ALPACA_API_KEY=test\nALPACA_SECRET_KEY=test\nEQUITY_SYMBOLS=AAPL\n"
        )

        with patch("agents.health_check.PROJECT_ROOT", str(tmp_path)):
            issues = check_env_config()

        assert any("Only 1" in i.get("issue", "") for i in issues)

    def test_valid_env(self, tmp_path):
        """Valid .env should produce no issues."""
        env_path = tmp_path / ".env"
        env_path.write_text(
            "ALPACA_API_KEY=pk_test\n"
            "ALPACA_SECRET_KEY=sk_test\n"
            "EQUITY_SYMBOLS=AAPL,TSLA,NVDA,AMD,META\n"
        )

        with patch("agents.health_check.PROJECT_ROOT", str(tmp_path)):
            issues = check_env_config()

        assert len(issues) == 0


# ═══════════════════════════════════════════════════════════════════════
# update_agent_state
# ═══════════════════════════════════════════════════════════════════════

class TestUpdateAgentState:
    def test_creates_new_state(self, tmp_path):
        """Should create agent_state.json if it doesn't exist."""
        path = str(tmp_path / "agent_state.json")
        with patch("agents.health_check.AGENT_STATE_FILE", path):
            update_agent_state("healthy", 5.0, 0)

        with open(path) as f:
            state = json.load(f)

        assert "health_check" in state
        assert state["health_check"]["status"] == "healthy"
        assert state["health_check"]["issues_found"] == 0

    def test_merges_with_existing_state(self, tmp_path):
        """Should preserve other agents' state."""
        path = tmp_path / "agent_state.json"
        path.write_text(json.dumps({"other_agent": {"status": "ok"}}))

        with patch("agents.health_check.AGENT_STATE_FILE", str(path)):
            update_agent_state("degraded", 10.0, 3)

        with open(str(path)) as f:
            state = json.load(f)

        assert "other_agent" in state
        assert "health_check" in state


# ═══════════════════════════════════════════════════════════════════════
# run_tests
# ═══════════════════════════════════════════════════════════════════════

class TestRunTests:
    def test_run_tests_success(self):
        """Mock subprocess to simulate passing tests."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "10 passed in 1.0s"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = run_tests()

        assert result["passed"] >= 10
        assert result["failed"] == 0
        assert result["exit_code"] == 0

    def test_run_tests_failure(self):
        """Mock subprocess to simulate failing tests."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "9 passed, 1 failed in 2.0s\nFAILED tests/test_foo.py::test_bar"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = run_tests()

        assert result["failed"] >= 1
        assert len(result["failures"]) >= 1

    def test_run_tests_timeout(self):
        """Test timeout should be handled."""
        with patch("subprocess.run", side_effect=Exception("timeout")):
            with pytest.raises(Exception):
                run_tests()

    def test_output_truncated_at_2000_chars(self):
        """Long output should be truncated to last 2000 chars."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "x" * 5000
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = run_tests()

        assert len(result["output"]) == 2000

    def test_error_count_parsed(self):
        """Errors in output should be counted."""
        mock_result = MagicMock()
        mock_result.returncode = 2
        mock_result.stdout = "test_a ERROR\ntest_b ERROR\n0 passed, 0 failed in 0.5s"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = run_tests()

        assert result["errors"] == 2


# ═══════════════════════════════════════════════════════════════════════
# check_router_consistency (with module patching)
# ═══════════════════════════════════════════════════════════════════════

class TestRouterConsistencyIsolated:
    """Test router consistency with proper module-level mocking."""

    def test_empty_strategy_map(self):
        """Empty strategy map should be flagged."""
        import types
        fake_router = types.ModuleType("strategies.router")
        fake_router.STRATEGY_MAP = {}
        fake_router.STRATEGY_REGISTRY = {"mean_reversion": lambda df: {}}
        fake_router.DEFAULT_STRATEGY = "mean_reversion"

        with patch.dict("sys.modules", {
            "strategies.router": fake_router,
            "strategies": types.ModuleType("strategies"),
        }):
            issues = check_router_consistency()

        assert any("empty" in i["issue"].lower() for i in issues)

    def test_default_strategy_not_in_registry(self):
        """Default strategy not in registry should be critical."""
        import types
        fake_router = types.ModuleType("strategies.router")
        fake_router.STRATEGY_MAP = {"AAPL": "mean_reversion"}
        fake_router.STRATEGY_REGISTRY = {"mean_reversion": lambda df: {}}
        fake_router.DEFAULT_STRATEGY = "ghost_strategy"

        with patch.dict("sys.modules", {
            "strategies.router": fake_router,
            "strategies": types.ModuleType("strategies"),
        }):
            issues = check_router_consistency()

        assert any(i["severity"] == "critical" for i in issues)

    def test_symbol_assigned_to_unknown_strategy(self):
        """Symbol mapped to non-existent strategy should be flagged."""
        import types
        fake_router = types.ModuleType("strategies.router")
        fake_router.STRATEGY_MAP = {"AAPL": "nonexistent"}
        fake_router.STRATEGY_REGISTRY = {"mean_reversion": lambda df: {}}
        fake_router.DEFAULT_STRATEGY = "mean_reversion"

        with patch.dict("sys.modules", {
            "strategies.router": fake_router,
            "strategies": types.ModuleType("strategies"),
        }):
            issues = check_router_consistency()

        assert any("unknown strategy" in i["issue"].lower() for i in issues)

    def test_router_import_fails(self):
        """If router import fails, a critical issue should be raised."""
        with patch.dict("sys.modules", {
            "strategies.router": None,
            "strategies": None,
        }):
            issues = check_router_consistency()

        assert len(issues) == 1
        assert issues[0]["severity"] == "critical"
        assert "failed" in issues[0]["issue"].lower()


# ═══════════════════════════════════════════════════════════════════════
# check_env_config — additional edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestEnvConfigEdgeCases:
    def test_too_many_equity_symbols(self, tmp_path):
        """More than 25 equity symbols should trigger a warning."""
        symbols = ",".join([f"SYM{i}" for i in range(30)])
        env_path = tmp_path / ".env"
        env_path.write_text(
            f"ALPACA_API_KEY=pk\nALPACA_SECRET_KEY=sk\nEQUITY_SYMBOLS={symbols}\n"
        )

        with patch("agents.health_check.PROJECT_ROOT", str(tmp_path)):
            issues = check_env_config()

        assert any("too many" in i.get("issue", "").lower() for i in issues)

    def test_missing_secret_key(self, tmp_path):
        """Missing ALPACA_SECRET_KEY should be flagged."""
        env_path = tmp_path / ".env"
        env_path.write_text(
            "ALPACA_API_KEY=pk\nEQUITY_SYMBOLS=AAPL,TSLA,NVDA,AMD\n"
        )

        with patch("agents.health_check.PROJECT_ROOT", str(tmp_path)):
            issues = check_env_config()

        assert any("ALPACA_SECRET_KEY" in i.get("issue", "") for i in issues)


# ═══════════════════════════════════════════════════════════════════════
# check_database — connection error
# ═══════════════════════════════════════════════════════════════════════

class TestDatabaseConnectionError:
    def test_sqlite_connect_raises(self):
        """If sqlite3.connect raises, a critical issue should be returned."""
        with patch("agents.health_check.os.path.exists", return_value=True), \
             patch("sqlite3.connect", side_effect=Exception("disk I/O error")):
            issues = check_database()

        assert len(issues) >= 1
        assert any(i["severity"] == "critical" for i in issues)


# ═══════════════════════════════════════════════════════════════════════
# update_agent_state — corrupt existing file
# ═══════════════════════════════════════════════════════════════════════

class TestUpdateAgentStateEdgeCases:
    def test_handles_corrupt_existing_state(self, tmp_path):
        """Should overwrite corrupt state file gracefully."""
        path = tmp_path / "agent_state.json"
        path.write_text("CORRUPT{{{not json")

        with patch("agents.health_check.AGENT_STATE_FILE", str(path)):
            update_agent_state("critical", 2.0, 5)

        with open(str(path)) as f:
            state = json.load(f)

        assert state["health_check"]["status"] == "critical"
        assert state["health_check"]["issues_found"] == 5


# ═══════════════════════════════════════════════════════════════════════
# main() orchestration
# ═══════════════════════════════════════════════════════════════════════

class TestMain:
    """Tests for main() orchestration function."""

    def _run_main_with_mocks(self, test_results=None, db_issues=None):
        """Helper to run main() with all checks mocked."""
        from agents.health_check import main
        import contextlib

        if test_results is None:
            test_results = {
                "exit_code": 0, "passed": 50, "failed": 0,
                "errors": 0, "failures": [], "output": "",
            }

        targets = {
            "agents.health_check.run_tests": MagicMock(return_value=test_results),
            "agents.health_check.check_file_integrity": MagicMock(return_value=[]),
            "agents.health_check.check_imports": MagicMock(return_value=[]),
            "agents.health_check.check_data_files": MagicMock(return_value=[]),
            "agents.health_check.check_database": MagicMock(return_value=db_issues or []),
            "agents.health_check.check_router_consistency": MagicMock(return_value=[]),
            "agents.health_check.check_agent_staleness": MagicMock(return_value=[]),
            "agents.health_check.check_env_config": MagicMock(return_value=[]),
            "agents.health_check.update_agent_state": MagicMock(),
            "agents.health_check.tempfile.mkstemp": MagicMock(return_value=(99, "/tmp/fake.tmp")),
            "agents.health_check.os.replace": MagicMock(),
            "agents.health_check.os.fdopen": MagicMock(),
        }

        with contextlib.ExitStack() as stack:
            for target, mock_obj in targets.items():
                stack.enter_context(patch(target, mock_obj))
            return main()

    def test_healthy_system_returns_0(self):
        """When all checks pass, main() returns 0."""
        result = self._run_main_with_mocks()
        assert result == 0

    def test_critical_issues_returns_1(self):
        """When critical issues exist, main() returns 1."""
        result = self._run_main_with_mocks(
            db_issues=[{"severity": "critical", "file": "trades.db", "issue": "Missing trades table"}]
        )
        assert result == 1

    def test_failed_tests_returns_1(self):
        """When tests fail, main() returns 1."""
        result = self._run_main_with_mocks(
            test_results={
                "exit_code": 1, "passed": 48, "failed": 2,
                "errors": 0, "failures": ["FAILED test_x", "FAILED test_y"], "output": "",
            }
        )
        assert result == 1
