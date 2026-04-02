"""Anthropic Claude API client — handles retries, rate limits, and cost tracking.

Provides a thin wrapper around the Anthropic SDK with:
  - Automatic retries with exponential backoff
  - Daily budget enforcement
  - Cost tracking per call
"""

import fcntl
import json
import os
import tempfile
import time
from datetime import datetime, date

from config import Config
from utils.logger import log

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "llm_analyst")
_SPEND_LOG = os.path.join(_DATA_DIR, "spend_log.json")

# Approximate cost per 1K tokens (input/output) for Claude models
_COST_PER_1K = {
    "claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
}
_DEFAULT_COST = {"input": 0.003, "output": 0.015}

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


def _ensure_data_dir():
    os.makedirs(_DATA_DIR, exist_ok=True)


def _load_spend_log() -> dict:
    """Load daily spend tracking."""
    if os.path.exists(_SPEND_LOG):
        try:
            with open(_SPEND_LOG) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"date": "", "total_usd": 0.0, "calls": []}


def _save_spend_log(data: dict):
    _ensure_data_dir()
    # Atomic write: write to temp file then rename (POSIX atomic)
    fd, tmp_path = tempfile.mkstemp(dir=_DATA_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, _SPEND_LOG)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def get_daily_spend() -> float:
    """Get total spend for today in USD."""
    log_data = _load_spend_log()
    if log_data.get("date") == str(date.today()):
        return log_data.get("total_usd", 0.0)
    return 0.0


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a given call."""
    rates = _COST_PER_1K.get(model, _DEFAULT_COST)
    return (input_tokens / 1000 * rates["input"] +
            output_tokens / 1000 * rates["output"])


_SPEND_LOCK = os.path.join(_DATA_DIR, ".spend_log.lock")


def _record_spend(model: str, input_tokens: int, output_tokens: int, cost: float):
    """Record a spend entry in the daily log (file-locked for concurrency)."""
    _ensure_data_dir()
    lock_fd = open(_SPEND_LOCK, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        log_data = _load_spend_log()
        today = str(date.today())

        if log_data.get("date") != today:
            log_data = {"date": today, "total_usd": 0.0, "calls": []}

        log_data["total_usd"] += cost
        log_data["calls"].append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        })
        # Cap call log to prevent unbounded growth
        if len(log_data["calls"]) > 500:
            log_data["calls"] = log_data["calls"][-500:]
        _save_spend_log(log_data)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def call_llm(prompt: str, model: str | None = None,
             system: str = "", max_tokens: int = 1024) -> dict:
    """Call Claude API with retry logic and budget enforcement.

    Args:
        prompt: User message content
        model: Model ID (defaults to Config.LLM_QUICK_MODEL)
        system: Optional system prompt
        max_tokens: Max response tokens

    Returns:
        dict with keys: text, model, input_tokens, output_tokens, cost_usd

    Raises:
        RuntimeError: If daily budget exceeded
        Exception: If all retries exhausted
    """
    if model is None:
        model = Config.LLM_QUICK_MODEL

    # Budget check
    current_spend = get_daily_spend()
    if current_spend >= Config.LLM_BUDGET_DAILY:
        raise RuntimeError(
            f"Daily LLM budget exceeded: ${current_spend:.2f} >= ${Config.LLM_BUDGET_DAILY:.2f}"
        )

    import anthropic
    client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system

            response = client.messages.create(**kwargs)

            text = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = _estimate_cost(model, input_tokens, output_tokens)

            _record_spend(model, input_tokens, output_tokens, cost)

            return {
                "text": text,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost, 6),
            }

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                log.warning(f"LLM call failed (attempt {attempt + 1}): {e}, retrying in {delay}s")
                time.sleep(delay)
            else:
                log.error(f"LLM call failed after {MAX_RETRIES} attempts: {e}")

    raise last_error


def call_llm_json(prompt: str, model: str | None = None,
                  system: str = "", max_tokens: int = 1024) -> dict:
    """Call Claude and parse the response as JSON.

    Extracts JSON from the response text (handles markdown code blocks).
    Returns parsed dict, or raises ValueError if parsing fails.
    """
    result = call_llm(prompt, model=model, system=system, max_tokens=max_tokens)
    text = result["text"].strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {text[:200]}")

    result["parsed"] = parsed
    return result
