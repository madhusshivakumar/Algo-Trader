"""Tests for the pattern discoverer — LLM-powered pattern learning."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import date


def _mock_llm_json(parsed):
    return {"parsed": parsed, "text": json.dumps(parsed),
            "model": "test", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.001}


class TestLoadRecentFeedback:
    def test_loads_recent_files(self, tmp_path):
        from agents.pattern_discoverer import load_recent_feedback
        today = date.today().isoformat()
        feedback = {"date": today, "overall_direction_accuracy": 0.65}
        path = tmp_path / f"{today}.json"
        with open(path, "w") as f:
            json.dump(feedback, f)

        with patch("agents.pattern_discoverer._FEEDBACK_DIR", str(tmp_path)):
            result = load_recent_feedback(days=7)
        assert len(result) == 1
        assert result[0]["date"] == today

    def test_empty_dir(self, tmp_path):
        from agents.pattern_discoverer import load_recent_feedback
        with patch("agents.pattern_discoverer._FEEDBACK_DIR", str(tmp_path)):
            result = load_recent_feedback()
        assert result == []

    def test_skips_old_files(self, tmp_path):
        from agents.pattern_discoverer import load_recent_feedback
        # Old file
        old_path = tmp_path / "2020-01-01.json"
        with open(old_path, "w") as f:
            json.dump({"date": "2020-01-01"}, f)

        with patch("agents.pattern_discoverer._FEEDBACK_DIR", str(tmp_path)):
            result = load_recent_feedback(days=7)
        assert len(result) == 0

    def test_skips_corrupt_files(self, tmp_path):
        from agents.pattern_discoverer import load_recent_feedback
        today = date.today().isoformat()
        path = tmp_path / f"{today}.json"
        path.write_text("not json")

        with patch("agents.pattern_discoverer._FEEDBACK_DIR", str(tmp_path)):
            result = load_recent_feedback(days=7)
        assert result == []


class TestDiscoverPatterns:
    @patch("agents.pattern_discoverer.get_daily_spend", return_value=0.0)
    @patch("agents.pattern_discoverer.call_llm_json")
    def test_success(self, mock_llm, mock_spend):
        from agents.pattern_discoverer import discover_patterns
        discovery = {
            "patterns": [
                {
                    "trigger": "Oil spike >2%",
                    "observation": "Predicted Energy +0.5 but actual -0.2",
                    "suggested_adjustment": "Energy may drop on recession fears",
                    "affected_sectors": ["Energy"],
                    "confidence": 0.8,
                }
            ],
            "overall_calibration": "too_aggressive",
            "biggest_miss": "Energy on oil spike",
            "summary": "Model overestimates Energy on oil spikes.",
        }
        mock_llm.return_value = _mock_llm_json(discovery)

        result = discover_patterns([{"date": "2026-03-31", "symbols": {}}])
        assert len(result["patterns"]) == 1
        assert result["overall_calibration"] == "too_aggressive"

    @patch("agents.pattern_discoverer.get_daily_spend", return_value=999.0)
    def test_budget_exceeded(self, mock_spend):
        from agents.pattern_discoverer import discover_patterns
        result = discover_patterns([{"date": "2026-03-31"}])
        assert result is None

    def test_empty_feedback(self):
        from agents.pattern_discoverer import discover_patterns
        result = discover_patterns([])
        assert result is None

    @patch("agents.pattern_discoverer.get_daily_spend", return_value=0.0)
    @patch("agents.pattern_discoverer.call_llm_json")
    def test_api_failure(self, mock_llm, mock_spend):
        from agents.pattern_discoverer import discover_patterns
        mock_llm.side_effect = RuntimeError("fail")
        result = discover_patterns([{"date": "2026-03-31"}])
        assert result is None


class TestSavePatterns:
    def test_saves_new_patterns(self, tmp_path):
        from agents.pattern_discoverer import save_patterns
        patterns_file = tmp_path / "patterns.json"

        with patch("agents.pattern_discoverer._PATTERNS_FILE", str(patterns_file)), \
             patch("agents.pattern_discoverer._DATA_DIR", str(tmp_path)):
            discovery = {
                "patterns": [
                    {"trigger": "Oil spike", "affected_sectors": ["Energy"], "confidence": 0.8}
                ],
                "overall_calibration": "well_calibrated",
            }
            save_patterns(discovery)

        with open(patterns_file) as f:
            data = json.load(f)
        assert len(data["patterns"]) == 1
        assert data["patterns"][0]["trigger"] == "Oil spike"
        assert "discovered" in data["patterns"][0]

    def test_merges_with_existing(self, tmp_path):
        from agents.pattern_discoverer import save_patterns
        patterns_file = tmp_path / "patterns.json"

        # Write existing
        existing = {
            "patterns": [
                {"trigger": "Rate hike", "affected_sectors": ["Finance"], "confidence": 0.7}
            ],
        }
        with open(patterns_file, "w") as f:
            json.dump(existing, f)

        with patch("agents.pattern_discoverer._PATTERNS_FILE", str(patterns_file)), \
             patch("agents.pattern_discoverer._DATA_DIR", str(tmp_path)):
            save_patterns({
                "patterns": [
                    {"trigger": "Oil spike", "affected_sectors": ["Energy"], "confidence": 0.8}
                ],
            })

        with open(patterns_file) as f:
            data = json.load(f)
        assert len(data["patterns"]) == 2

    def test_deduplicates_by_trigger(self, tmp_path):
        from agents.pattern_discoverer import save_patterns
        patterns_file = tmp_path / "patterns.json"

        existing = {
            "patterns": [
                {"trigger": "Oil spike", "affected_sectors": ["Energy"], "confidence": 0.5}
            ],
        }
        with open(patterns_file, "w") as f:
            json.dump(existing, f)

        with patch("agents.pattern_discoverer._PATTERNS_FILE", str(patterns_file)), \
             patch("agents.pattern_discoverer._DATA_DIR", str(tmp_path)):
            save_patterns({
                "patterns": [
                    {"trigger": "Oil spike", "affected_sectors": ["Energy"], "confidence": 0.9}
                ],
            })

        with open(patterns_file) as f:
            data = json.load(f)
        # Should be 1, not 2 (deduplicated)
        assert len(data["patterns"]) == 1
        # New one should win (it's first in merged list)
        assert data["patterns"][0]["confidence"] == 0.9

    def test_caps_at_max_patterns(self, tmp_path):
        from agents.pattern_discoverer import save_patterns, MAX_PATTERNS
        patterns_file = tmp_path / "patterns.json"

        with patch("agents.pattern_discoverer._PATTERNS_FILE", str(patterns_file)), \
             patch("agents.pattern_discoverer._DATA_DIR", str(tmp_path)):
            save_patterns({
                "patterns": [
                    {"trigger": f"pattern {i}", "confidence": 0.5}
                    for i in range(MAX_PATTERNS + 10)
                ],
            })

        with open(patterns_file) as f:
            data = json.load(f)
        assert len(data["patterns"]) == MAX_PATTERNS


class TestGetDiscoveredPatternsForSector:
    def test_returns_relevant_patterns(self, tmp_path):
        from agents.pattern_discoverer import get_discovered_patterns_for_sector
        patterns_data = {
            "patterns": [
                {"trigger": "Oil spike", "observation": "test", "suggested_adjustment": "adj",
                 "affected_sectors": ["Energy"], "confidence": 0.8},
                {"trigger": "Rate hike", "observation": "test2", "suggested_adjustment": "adj2",
                 "affected_sectors": ["Finance"], "confidence": 0.7},
            ],
        }
        patterns_file = tmp_path / "patterns.json"
        with open(patterns_file, "w") as f:
            json.dump(patterns_data, f)

        with patch("agents.pattern_discoverer._PATTERNS_FILE", str(patterns_file)):
            result = get_discovered_patterns_for_sector("Energy")
        assert "Oil spike" in result
        assert "Rate hike" not in result

    def test_filters_low_confidence(self, tmp_path):
        from agents.pattern_discoverer import get_discovered_patterns_for_sector
        patterns_data = {
            "patterns": [
                {"trigger": "Weak pattern", "affected_sectors": ["Tech"], "confidence": 0.2},
            ],
        }
        patterns_file = tmp_path / "patterns.json"
        with open(patterns_file, "w") as f:
            json.dump(patterns_data, f)

        with patch("agents.pattern_discoverer._PATTERNS_FILE", str(patterns_file)):
            result = get_discovered_patterns_for_sector("Tech")
        assert result == ""

    def test_no_patterns_file(self, tmp_path):
        from agents.pattern_discoverer import get_discovered_patterns_for_sector
        with patch("agents.pattern_discoverer._PATTERNS_FILE", str(tmp_path / "nope.json")):
            result = get_discovered_patterns_for_sector("Tech")
        assert result == ""


class TestBuildDiscoveryPrompt:
    def test_includes_sector_summary(self):
        from agents.pattern_discoverer import _build_discovery_prompt
        feedback = [{
            "date": "2026-03-31",
            "overall_direction_accuracy": 0.6,
            "sector_summary": {
                "Tech": {"direction_accuracy": 0.75, "avg_predicted": 0.4, "avg_actual_pct": 0.8},
            },
            "symbols": {},
        }]
        prompt = _build_discovery_prompt(feedback)
        assert "Tech" in prompt
        assert "60%" in prompt or "0.6" in prompt

    def test_includes_misses(self):
        from agents.pattern_discoverer import _build_discovery_prompt
        feedback = [{
            "date": "2026-03-31",
            "overall_direction_accuracy": 0.5,
            "sector_summary": {},
            "symbols": {
                "AAPL": {
                    "predicted_score": 0.8,
                    "predicted_bias": "bullish",
                    "actual_pct_change": -2.0,
                    "direction_correct": False,
                },
            },
        }]
        prompt = _build_discovery_prompt(feedback)
        assert "MISS" in prompt
        assert "AAPL" in prompt
