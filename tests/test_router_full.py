"""Full coverage tests for the strategy router, including file-loading paths."""

import json
import os
import importlib
import pytest
from unittest.mock import patch


class TestLoadStrategyMapFromFiles:
    """Test _load_strategy_map with various file states."""

    def test_load_from_assignments_file(self, tmp_path):
        assignments = {
            "run_date": "2026-03-28",
            "assignments": {
                "AAPL": {"strategy": "mean_reversion_aggressive", "reason": "best"},
                "BTC/USD": {"strategy": "volume_profile", "reason": "best"},
            }
        }
        af = tmp_path / "strategy_assignments.json"
        af.write_text(json.dumps(assignments))

        from strategies.router import _load_strategy_map, STRATEGY_REGISTRY
        with patch("strategies.router._ASSIGNMENTS_FILE", str(af)), \
             patch("strategies.router._FALLBACK_FILE", "/nonexistent"):
            result = _load_strategy_map()

        assert result["AAPL"] == "mean_reversion_aggressive"
        assert result["BTC/USD"] == "volume_profile"

    def test_load_from_assignments_string_format(self, tmp_path):
        """Assignments where value is a plain string, not a dict."""
        assignments = {
            "run_date": "2026-03-28",
            "assignments": {
                "AAPL": "momentum",
                "NVDA": "scalper",
            }
        }
        af = tmp_path / "strategy_assignments.json"
        af.write_text(json.dumps(assignments))

        from strategies.router import _load_strategy_map
        with patch("strategies.router._ASSIGNMENTS_FILE", str(af)), \
             patch("strategies.router._FALLBACK_FILE", "/nonexistent"):
            result = _load_strategy_map()

        assert result["AAPL"] == "momentum"
        assert result["NVDA"] == "scalper"

    def test_load_invalid_strategy_falls_back_to_default(self, tmp_path):
        assignments = {
            "assignments": {
                "AAPL": {"strategy": "nonexistent_strategy"},
            }
        }
        af = tmp_path / "strategy_assignments.json"
        af.write_text(json.dumps(assignments))

        from strategies.router import _load_strategy_map, DEFAULT_STRATEGY
        with patch("strategies.router._ASSIGNMENTS_FILE", str(af)), \
             patch("strategies.router._FALLBACK_FILE", "/nonexistent"):
            result = _load_strategy_map()

        assert result["AAPL"] == DEFAULT_STRATEGY

    def test_load_from_fallback_when_assignments_missing(self, tmp_path):
        fallback = {
            "strategy_map": {
                "TSLA": "mean_reversion",
                "SPY": "momentum",
            }
        }
        ff = tmp_path / "fallback_config.json"
        ff.write_text(json.dumps(fallback))

        from strategies.router import _load_strategy_map
        with patch("strategies.router._ASSIGNMENTS_FILE", "/nonexistent"), \
             patch("strategies.router._FALLBACK_FILE", str(ff)):
            result = _load_strategy_map()

        assert result["TSLA"] == "mean_reversion"
        assert result["SPY"] == "momentum"

    def test_load_from_fallback_when_assignments_corrupt(self, tmp_path):
        af = tmp_path / "strategy_assignments.json"
        af.write_text("{invalid json")

        fallback = {"strategy_map": {"AAPL": "scalper"}}
        ff = tmp_path / "fallback_config.json"
        ff.write_text(json.dumps(fallback))

        from strategies.router import _load_strategy_map
        with patch("strategies.router._ASSIGNMENTS_FILE", str(af)), \
             patch("strategies.router._FALLBACK_FILE", str(ff)):
            result = _load_strategy_map()

        assert result["AAPL"] == "scalper"

    def test_load_defaults_when_all_files_missing(self):
        from strategies.router import _load_strategy_map, _DEFAULT_MAP
        with patch("strategies.router._ASSIGNMENTS_FILE", "/nonexistent"), \
             patch("strategies.router._FALLBACK_FILE", "/nonexistent"):
            result = _load_strategy_map()

        assert result == _DEFAULT_MAP

    def test_load_from_fallback_with_invalid_strategies(self, tmp_path):
        fallback = {"strategy_map": {"AAPL": "bad_strategy_name"}}
        ff = tmp_path / "fallback_config.json"
        ff.write_text(json.dumps(fallback))

        from strategies.router import _load_strategy_map, _DEFAULT_MAP
        with patch("strategies.router._ASSIGNMENTS_FILE", "/nonexistent"), \
             patch("strategies.router._FALLBACK_FILE", str(ff)):
            result = _load_strategy_map()

        # Should fall through to defaults since all strategies are invalid
        assert result == _DEFAULT_MAP

    def test_load_empty_assignments(self, tmp_path):
        af = tmp_path / "strategy_assignments.json"
        af.write_text(json.dumps({"assignments": {}}))

        fallback = {"strategy_map": {"AAPL": "momentum"}}
        ff = tmp_path / "fallback_config.json"
        ff.write_text(json.dumps(fallback))

        from strategies.router import _load_strategy_map
        with patch("strategies.router._ASSIGNMENTS_FILE", str(af)), \
             patch("strategies.router._FALLBACK_FILE", str(ff)):
            result = _load_strategy_map()

        # Empty result should fall through to fallback
        assert "AAPL" in result

    def test_load_fallback_corrupt_json(self, tmp_path):
        ff = tmp_path / "fallback_config.json"
        ff.write_text("not json")

        from strategies.router import _load_strategy_map, _DEFAULT_MAP
        with patch("strategies.router._ASSIGNMENTS_FILE", "/nonexistent"), \
             patch("strategies.router._FALLBACK_FILE", str(ff)):
            result = _load_strategy_map()

        assert result == _DEFAULT_MAP
