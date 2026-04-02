"""Tests for the conviction scorer — prediction archiving, scoring, feedback."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import date


class TestArchivePredictions:
    def test_archives_to_correct_path(self, tmp_path):
        from agents.conviction_scorer import archive_predictions
        with patch("agents.conviction_scorer._PREDICTION_DIR", str(tmp_path)):
            data = {
                "timestamp": "2026-03-31T06:15:00",
                "convictions": {"AAPL": {"score": 0.5}},
                "portfolio_risk": "low",
                "overall_bias": "bullish",
            }
            archive_predictions(data)

            today = date.today().isoformat()
            path = tmp_path / f"{today}.json"
            assert path.exists()

            with open(path) as f:
                loaded = json.load(f)
            assert loaded["convictions"]["AAPL"]["score"] == 0.5

    def test_archive_creates_directory(self, tmp_path):
        from agents.conviction_scorer import archive_predictions
        new_dir = tmp_path / "subdir"
        with patch("agents.conviction_scorer._PREDICTION_DIR", str(new_dir)):
            archive_predictions({"convictions": {}})
            assert new_dir.exists()


class TestLoadPredictions:
    def test_load_existing(self, tmp_path):
        from agents.conviction_scorer import load_predictions
        data = {"date": "2026-03-31", "convictions": {"AAPL": {"score": 0.5}}}
        path = tmp_path / "2026-03-31.json"
        with open(path, "w") as f:
            json.dump(data, f)

        with patch("agents.conviction_scorer._PREDICTION_DIR", str(tmp_path)):
            result = load_predictions("2026-03-31")
        assert result["convictions"]["AAPL"]["score"] == 0.5

    def test_load_missing_returns_none(self, tmp_path):
        from agents.conviction_scorer import load_predictions
        with patch("agents.conviction_scorer._PREDICTION_DIR", str(tmp_path)):
            result = load_predictions("2099-01-01")
        assert result is None

    def test_load_corrupt_returns_none(self, tmp_path):
        from agents.conviction_scorer import load_predictions
        path = tmp_path / "2026-03-31.json"
        path.write_text("not json!!!")
        with patch("agents.conviction_scorer._PREDICTION_DIR", str(tmp_path)):
            result = load_predictions("2026-03-31")
        assert result is None


class TestScorePredictions:
    def test_direction_correct_bullish(self):
        from agents.conviction_scorer import score_predictions
        predictions = {
            "date": "2026-03-31",
            "convictions": {"AAPL": {"score": 0.5, "bias": "bullish"}},
        }
        price_changes = {"AAPL": 1.2}

        result = score_predictions(predictions, price_changes)
        assert result["symbols"]["AAPL"]["direction_correct"] is True
        assert result["overall_direction_accuracy"] == 1.0

    def test_direction_wrong(self):
        from agents.conviction_scorer import score_predictions
        predictions = {
            "date": "2026-03-31",
            "convictions": {"AAPL": {"score": 0.5, "bias": "bullish"}},
        }
        price_changes = {"AAPL": -2.0}

        result = score_predictions(predictions, price_changes)
        assert result["symbols"]["AAPL"]["direction_correct"] is False

    def test_neutral_prediction_small_move_is_correct(self):
        from agents.conviction_scorer import score_predictions
        predictions = {
            "date": "2026-03-31",
            "convictions": {"AAPL": {"score": 0.0, "bias": "neutral"}},
        }
        price_changes = {"AAPL": 0.1}

        result = score_predictions(predictions, price_changes)
        assert result["symbols"]["AAPL"]["direction_correct"] is True

    def test_missing_symbol_in_prices_skipped(self):
        from agents.conviction_scorer import score_predictions
        predictions = {
            "date": "2026-03-31",
            "convictions": {
                "AAPL": {"score": 0.5},
                "MISSING": {"score": -0.3},
            },
        }
        price_changes = {"AAPL": 1.0}

        result = score_predictions(predictions, price_changes)
        assert "AAPL" in result["symbols"]
        assert "MISSING" not in result["symbols"]
        assert result["total_scored"] == 1

    def test_sector_aggregation(self):
        from agents.conviction_scorer import score_predictions
        predictions = {
            "date": "2026-03-31",
            "convictions": {
                "AAPL": {"score": 0.5, "bias": "bullish"},
                "MSFT": {"score": 0.3, "bias": "bullish"},
                "XOM": {"score": -0.5, "bias": "bearish"},
            },
        }
        price_changes = {"AAPL": 1.0, "MSFT": -0.5, "XOM": -1.0}

        result = score_predictions(predictions, price_changes)
        assert "Tech" in result["sector_summary"]
        assert "Energy" in result["sector_summary"]
        assert result["sector_summary"]["Energy"]["direction_accuracy"] == 1.0

    def test_magnitude_error_computed(self):
        from agents.conviction_scorer import score_predictions
        predictions = {
            "date": "2026-03-31",
            "convictions": {"AAPL": {"score": 0.5}},
        }
        price_changes = {"AAPL": 0.5}

        result = score_predictions(predictions, price_changes)
        assert "magnitude_error" in result["symbols"]["AAPL"]
        assert result["symbols"]["AAPL"]["magnitude_error"] >= 0

    def test_empty_convictions(self):
        from agents.conviction_scorer import score_predictions
        result = score_predictions({"convictions": {}}, {})
        assert result["total_scored"] == 0
        assert result["overall_direction_accuracy"] == 0


class TestWriteFeedback:
    def test_writes_to_correct_path(self, tmp_path):
        from agents.conviction_scorer import write_feedback
        feedback = {
            "date": "2026-03-31",
            "total_correct": 3,
            "total_scored": 5,
            "overall_direction_accuracy": 0.6,
            "symbols": {},
            "sector_summary": {},
        }
        with patch("agents.conviction_scorer._FEEDBACK_DIR", str(tmp_path)):
            write_feedback(feedback)
        assert (tmp_path / "2026-03-31.json").exists()


class TestUpdateSummary:
    def test_updates_from_feedback_files(self, tmp_path):
        from agents.conviction_scorer import update_summary, load_summary
        feedback_dir = tmp_path / "feedback"
        feedback_dir.mkdir()
        summary_file = tmp_path / "summary.json"

        feedback = {
            "date": date.today().isoformat(),
            "total_correct": 3,
            "total_scored": 5,
            "overall_direction_accuracy": 0.6,
            "symbols": {
                "AAPL": {"predicted_score": 0.5, "actual_pct_change": 1.0, "direction_correct": True},
            },
            "sector_summary": {
                "Tech": {"direction_accuracy": 1.0, "num_symbols": 1, "avg_predicted": 0.5, "avg_actual_pct": 1.0},
            },
        }
        with open(feedback_dir / f"{date.today().isoformat()}.json", "w") as f:
            json.dump(feedback, f)

        with patch("agents.conviction_scorer._FEEDBACK_DIR", str(feedback_dir)), \
             patch("agents.conviction_scorer._SUMMARY_FILE", str(summary_file)):
            update_summary()

            summary = load_summary()
        assert summary["total_days"] == 1
        assert summary["overall_accuracy"] == 0.6

    def test_empty_feedback_dir(self, tmp_path):
        from agents.conviction_scorer import update_summary
        feedback_dir = tmp_path / "feedback"
        feedback_dir.mkdir()
        summary_file = tmp_path / "summary.json"

        with patch("agents.conviction_scorer._FEEDBACK_DIR", str(feedback_dir)), \
             patch("agents.conviction_scorer._SUMMARY_FILE", str(summary_file)):
            update_summary()

        with open(summary_file) as f:
            summary = json.load(f)
        assert summary["total_days"] == 0


class TestGetFeedback:
    def test_sector_feedback_with_data(self, tmp_path):
        from agents.conviction_scorer import get_sector_feedback
        summary = {
            "total_days": 5,
            "overall_accuracy": 0.65,
            "sector_accuracy": {"Tech": 0.75},
            "recent_days": [{"date": "2026-03-31", "accuracy": 0.7, "scored": 10}],
        }
        summary_file = tmp_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f)

        with patch("agents.conviction_scorer._SUMMARY_FILE", str(summary_file)):
            result = get_sector_feedback("Tech")
        assert "75%" in result
        assert "calibrate" in result.lower()

    def test_sector_feedback_no_data(self, tmp_path):
        from agents.conviction_scorer import get_sector_feedback
        with patch("agents.conviction_scorer._SUMMARY_FILE", str(tmp_path / "nope.json")):
            result = get_sector_feedback("Tech")
        assert result == ""

    def test_symbol_feedback_with_history(self, tmp_path):
        from agents.conviction_scorer import get_symbol_feedback
        summary = {
            "symbol_accuracy": {
                "AAPL": [
                    {"date": "2026-03-29", "predicted": 0.5, "actual_pct": 1.0, "correct": True},
                    {"date": "2026-03-30", "predicted": -0.3, "actual_pct": -0.5, "correct": True},
                ],
            },
        }
        summary_file = tmp_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f)

        with patch("agents.conviction_scorer._SUMMARY_FILE", str(summary_file)):
            result = get_symbol_feedback("AAPL")
        assert "AAPL" in result
        assert "Correct" in result

    def test_symbol_feedback_no_history(self, tmp_path):
        from agents.conviction_scorer import get_symbol_feedback
        with patch("agents.conviction_scorer._SUMMARY_FILE", str(tmp_path / "nope.json")):
            result = get_symbol_feedback("AAPL")
        assert result == ""


class TestFetchPriceChanges:
    def test_computes_pct_change(self):
        import pandas as pd
        from agents.conviction_scorer import fetch_price_changes

        mock_broker = MagicMock()
        df = pd.DataFrame({"close": [100.0, 102.0]})
        mock_broker.get_historical_bars.return_value = df

        result = fetch_price_changes(["AAPL"], broker=mock_broker)
        assert "AAPL" in result
        assert abs(result["AAPL"] - 2.0) < 0.01

    def test_handles_broker_error(self):
        from agents.conviction_scorer import fetch_price_changes

        mock_broker = MagicMock()
        mock_broker.get_historical_bars.side_effect = Exception("API error")

        result = fetch_price_changes(["AAPL"], broker=mock_broker)
        assert "AAPL" not in result

    def test_insufficient_data(self):
        import pandas as pd
        from agents.conviction_scorer import fetch_price_changes

        mock_broker = MagicMock()
        df = pd.DataFrame({"close": [100.0]})  # only 1 bar
        mock_broker.get_historical_bars.return_value = df

        result = fetch_price_changes(["AAPL"], broker=mock_broker)
        assert "AAPL" not in result
