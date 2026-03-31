"""Tests for core/llm_client.py — Anthropic Claude API wrapper."""

import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import date


class TestEstimateCost:
    def test_haiku_cost(self):
        from core.llm_client import _estimate_cost
        cost = _estimate_cost("claude-haiku-4-20250514", 1000, 500)
        expected = (1000 / 1000 * 0.00025) + (500 / 1000 * 0.00125)
        assert cost == pytest.approx(expected)

    def test_sonnet_cost(self):
        from core.llm_client import _estimate_cost
        cost = _estimate_cost("claude-sonnet-4-20250514", 1000, 500)
        expected = (1000 / 1000 * 0.003) + (500 / 1000 * 0.015)
        assert cost == pytest.approx(expected)

    def test_unknown_model_uses_default(self):
        from core.llm_client import _estimate_cost
        cost = _estimate_cost("unknown-model", 1000, 500)
        # Should use _DEFAULT_COST (same as sonnet)
        expected = (1000 / 1000 * 0.003) + (500 / 1000 * 0.015)
        assert cost == pytest.approx(expected)


class TestSpendTracking:
    def test_load_empty_spend_log(self, tmp_path):
        from core.llm_client import _load_spend_log
        with patch("core.llm_client._SPEND_LOG", str(tmp_path / "nope.json")):
            result = _load_spend_log()
        assert result == {"date": "", "total_usd": 0.0, "calls": []}

    def test_load_existing_spend_log(self, tmp_path):
        from core.llm_client import _load_spend_log
        f = tmp_path / "spend.json"
        data = {"date": "2024-01-01", "total_usd": 0.5, "calls": []}
        f.write_text(json.dumps(data))
        with patch("core.llm_client._SPEND_LOG", str(f)):
            result = _load_spend_log()
        assert result["total_usd"] == 0.5

    def test_load_corrupt_spend_log(self, tmp_path):
        from core.llm_client import _load_spend_log
        f = tmp_path / "spend.json"
        f.write_text("{bad json")
        with patch("core.llm_client._SPEND_LOG", str(f)):
            result = _load_spend_log()
        assert result == {"date": "", "total_usd": 0.0, "calls": []}

    def test_get_daily_spend_today(self, tmp_path):
        from core.llm_client import get_daily_spend
        f = tmp_path / "spend.json"
        data = {"date": str(date.today()), "total_usd": 0.75, "calls": []}
        f.write_text(json.dumps(data))
        with patch("core.llm_client._SPEND_LOG", str(f)):
            result = get_daily_spend()
        assert result == 0.75

    def test_get_daily_spend_stale_date(self, tmp_path):
        from core.llm_client import get_daily_spend
        f = tmp_path / "spend.json"
        data = {"date": "2020-01-01", "total_usd": 5.0, "calls": []}
        f.write_text(json.dumps(data))
        with patch("core.llm_client._SPEND_LOG", str(f)):
            result = get_daily_spend()
        assert result == 0.0

    def test_record_spend(self, tmp_path):
        from core.llm_client import _record_spend, _load_spend_log
        f = tmp_path / "spend.json"
        with patch("core.llm_client._SPEND_LOG", str(f)), \
             patch("core.llm_client._DATA_DIR", str(tmp_path)):
            _record_spend("claude-haiku-4-20250514", 500, 200, 0.001)
            result = _load_spend_log()
        assert result["date"] == str(date.today())
        assert result["total_usd"] == pytest.approx(0.001)
        assert len(result["calls"]) == 1

    def test_record_spend_accumulates(self, tmp_path):
        from core.llm_client import _record_spend, _load_spend_log
        f = tmp_path / "spend.json"
        with patch("core.llm_client._SPEND_LOG", str(f)), \
             patch("core.llm_client._DATA_DIR", str(tmp_path)):
            _record_spend("claude-haiku-4-20250514", 500, 200, 0.001)
            _record_spend("claude-haiku-4-20250514", 500, 200, 0.002)
            result = _load_spend_log()
        assert result["total_usd"] == pytest.approx(0.003)
        assert len(result["calls"]) == 2


class TestCallLLM:
    def _mock_anthropic(self, mock_response):
        """Helper to create a mock anthropic module for import patching."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_module = MagicMock()
        mock_module.Anthropic.return_value = mock_client
        return mock_module, mock_client

    @patch("core.llm_client.get_daily_spend", return_value=0.0)
    @patch("core.llm_client._record_spend")
    def test_successful_call(self, mock_record, mock_spend):
        from core.llm_client import call_llm
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello world")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_module, mock_client = self._mock_anthropic(mock_response)
        with patch.dict("sys.modules", {"anthropic": mock_module}):
            result = call_llm("test prompt", model="claude-haiku-4-20250514")

        assert result["text"] == "Hello world"
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        mock_record.assert_called_once()

    @patch("core.llm_client.get_daily_spend", return_value=5.0)
    def test_budget_exceeded(self, mock_spend):
        from core.llm_client import call_llm
        with patch("core.llm_client.Config") as mock_config:
            mock_config.LLM_BUDGET_DAILY = 1.0
            with pytest.raises(RuntimeError, match="budget exceeded"):
                call_llm("test")

    @patch("core.llm_client.get_daily_spend", return_value=0.0)
    @patch("core.llm_client._record_spend")
    @patch("core.llm_client.time.sleep")
    def test_retries_on_failure(self, mock_sleep, mock_record, mock_spend):
        from core.llm_client import call_llm
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="OK")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            Exception("rate limit"),
            mock_response,
        ]
        mock_module.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            result = call_llm("test", model="claude-haiku-4-20250514")

        assert result["text"] == "OK"
        assert mock_client.messages.create.call_count == 2
        mock_sleep.assert_called_once()

    @patch("core.llm_client.get_daily_spend", return_value=0.0)
    @patch("core.llm_client.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep, mock_spend):
        from core.llm_client import call_llm
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("persistent error")
        mock_module.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            with pytest.raises(Exception, match="persistent error"):
                call_llm("test", model="claude-haiku-4-20250514")

        assert mock_client.messages.create.call_count == 3

    @patch("core.llm_client.get_daily_spend", return_value=0.0)
    @patch("core.llm_client._record_spend")
    def test_passes_system_prompt(self, mock_record, mock_spend):
        from core.llm_client import call_llm
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        mock_module, mock_client = self._mock_anthropic(mock_response)
        with patch.dict("sys.modules", {"anthropic": mock_module}):
            call_llm("test", system="You are helpful", model="claude-haiku-4-20250514")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"


class TestCallLLMJson:
    @patch("core.llm_client.call_llm")
    def test_parses_json_response(self, mock_call):
        from core.llm_client import call_llm_json
        mock_call.return_value = {
            "text": '{"key": "value"}',
            "model": "test",
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_usd": 0.001,
        }

        result = call_llm_json("test")
        assert result["parsed"] == {"key": "value"}

    @patch("core.llm_client.call_llm")
    def test_strips_markdown_fences(self, mock_call):
        from core.llm_client import call_llm_json
        mock_call.return_value = {
            "text": '```json\n{"key": "value"}\n```',
            "model": "test",
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_usd": 0.001,
        }

        result = call_llm_json("test")
        assert result["parsed"] == {"key": "value"}

    @patch("core.llm_client.call_llm")
    def test_raises_on_invalid_json(self, mock_call):
        from core.llm_client import call_llm_json
        mock_call.return_value = {
            "text": "not valid json at all",
            "model": "test",
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_usd": 0.001,
        }

        with pytest.raises(ValueError, match="Failed to parse"):
            call_llm_json("test")
