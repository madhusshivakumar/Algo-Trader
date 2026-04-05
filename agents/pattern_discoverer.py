"""Pattern Discoverer — uses LLM to find new patterns from prediction feedback.

Runs weekly (Sundays) or on-demand. Uses 1 Haiku call to analyze the past week's
prediction accuracy and discover systematic biases or new patterns.

Discovered patterns are injected into future sector analysis prompts,
creating a self-improving feedback loop.
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.llm_client import call_llm_json, get_daily_spend
from utils.logger import log

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "llm_analyst")
_FEEDBACK_DIR = os.path.join(_DATA_DIR, "feedback")
_PATTERNS_FILE = os.path.join(_DATA_DIR, "discovered_patterns.json")

MAX_PATTERNS = 20  # keep last N discovered patterns


_DISCOVERY_SYSTEM = """You are a quantitative analyst reviewing a trading model's prediction accuracy over the past week.

Given the model's predictions vs actual outcomes, identify:
1. SYSTEMATIC BIASES — does the model consistently over/under predict certain sectors or conditions?
2. NEW PATTERNS — when condition X occurred, the model predicted Y but actual was Z. Why?
3. INFLUENCER EFFECTS — did statements from key figures cause unexpected moves?
4. CALIBRATION ISSUES — is the model too aggressive or too conservative?

Output 2-5 discovered patterns that would improve future predictions.

Respond in JSON:
{
  "patterns": [
    {
      "trigger": "description of the condition/event",
      "observation": "what the model predicted vs what actually happened",
      "suggested_adjustment": "how to improve predictions when this happens again",
      "affected_sectors": ["sector1", "sector2"],
      "confidence": <float 0.0-1.0>
    }
  ],
  "overall_calibration": "too_aggressive|too_conservative|well_calibrated",
  "biggest_miss": "description of the worst prediction error this week",
  "summary": "2-3 sentence summary of findings"
}"""


def load_recent_feedback(days: int = 7) -> list[dict]:
    """Load the last N days of scored feedback."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    feedback_list = []

    if not os.path.exists(_FEEDBACK_DIR):
        return []

    for fname in sorted(os.listdir(_FEEDBACK_DIR)):
        if not fname.endswith(".json"):
            continue
        fdate = fname.replace(".json", "")
        if fdate < cutoff:
            continue
        try:
            with open(os.path.join(_FEEDBACK_DIR, fname)) as f:
                feedback_list.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue

    return feedback_list


def load_existing_patterns() -> list[dict]:
    """Load previously discovered patterns."""
    if not os.path.exists(_PATTERNS_FILE):
        return []
    try:
        with open(_PATTERNS_FILE) as f:
            data = json.load(f)
        return data.get("patterns", [])
    except (json.JSONDecodeError, OSError):
        return []


def _build_discovery_prompt(feedback_list: list[dict]) -> str:
    """Build the prompt for pattern discovery."""
    parts = ["## Weekly Prediction Accuracy Review\n"]

    for feedback in feedback_list:
        fdate = feedback.get("date", "unknown")
        accuracy = feedback.get("overall_direction_accuracy", 0)
        parts.append(f"### {fdate} — {accuracy:.0%} direction accuracy")

        # Sector summary
        for sector, stats in feedback.get("sector_summary", {}).items():
            parts.append(
                f"  {sector}: accuracy={stats.get('direction_accuracy', 0):.0%}, "
                f"avg_predicted={stats.get('avg_predicted', 0):+.2f}, "
                f"avg_actual={stats.get('avg_actual_pct', 0):+.2f}%"
            )

        # Biggest misses (wrong direction with high conviction)
        misses = []
        for sym, data in feedback.get("symbols", {}).items():
            if not data.get("direction_correct") and abs(data.get("predicted_score", 0)) > 0.3:
                misses.append(
                    f"  MISS: {sym} — predicted {data['predicted_score']:+.2f} "
                    f"({data.get('predicted_bias', 'N/A')}), "
                    f"actual {data['actual_pct_change']:+.2f}%"
                )
        if misses:
            parts.append("  Notable misses:")
            parts.extend(misses[:5])

        parts.append("")

    return "\n".join(parts)


def discover_patterns(feedback_list: list[dict]) -> dict | None:
    """Run LLM-powered pattern discovery on recent feedback.

    Args:
        feedback_list: List of scored feedback dicts.

    Returns:
        Discovery result dict or None if budget exceeded or no data.
    """
    if not feedback_list:
        log.warning("No feedback data for pattern discovery")
        return None

    if get_daily_spend() >= Config.LLM_BUDGET_DAILY:
        log.warning("Budget exceeded, skipping pattern discovery")
        return None

    prompt = _build_discovery_prompt(feedback_list)

    try:
        result = call_llm_json(
            prompt,
            model=Config.LLM_QUICK_MODEL,
            system=_DISCOVERY_SYSTEM,
            max_tokens=1024,
        )
        parsed = result.get("parsed", {})
        return parsed
    except Exception as e:
        log.warning(f"Pattern discovery failed: {e}")
        return None


def save_patterns(discovery_result: dict):
    """Save discovered patterns, merging with existing ones.

    Keeps the most recent MAX_PATTERNS patterns, deduplicating by trigger.
    """
    os.makedirs(_DATA_DIR, exist_ok=True)

    existing = load_existing_patterns()
    new_patterns = discovery_result.get("patterns", [])

    # Add discovery date to new patterns
    today = date.today().isoformat()
    for p in new_patterns:
        p["discovered"] = today

    # Merge: new patterns first, then existing (newest takes precedence)
    merged = new_patterns + existing

    # Deduplicate by trigger (keep most recent)
    seen_triggers = set()
    deduped = []
    for p in merged:
        trigger = p.get("trigger", "").lower().strip()
        if trigger and trigger not in seen_triggers:
            seen_triggers.add(trigger)
            deduped.append(p)

    # Keep only the most recent MAX_PATTERNS
    deduped = deduped[:MAX_PATTERNS]

    data = {
        "updated": datetime.now().isoformat(),
        "patterns": deduped,
        "last_discovery": {
            "date": today,
            "calibration": discovery_result.get("overall_calibration", "unknown"),
            "biggest_miss": discovery_result.get("biggest_miss", ""),
            "summary": discovery_result.get("summary", ""),
        },
    }

    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(_PATTERNS_FILE), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, _PATTERNS_FILE)
    except Exception:
        os.unlink(tmp_path)
        raise
    log.info(f"Saved {len(deduped)} patterns ({len(new_patterns)} new)")


def get_discovered_patterns_for_sector(sector: str) -> str:
    """Get formatted discovered patterns for a sector prompt.

    Returns patterns relevant to the given sector, formatted for prompt injection.
    """
    patterns = load_existing_patterns()
    if not patterns:
        return ""

    relevant = [
        p for p in patterns
        if sector in p.get("affected_sectors", [])
        and p.get("confidence", 0) >= 0.5
    ]

    if not relevant:
        return ""

    lines = ["## Learned Patterns (from recent accuracy data)"]
    for p in relevant[:3]:  # cap at 3 to control tokens
        lines.append(f"- When: {p.get('trigger', 'unknown')}")
        lines.append(f"  Observed: {p.get('observation', 'N/A')}")
        lines.append(f"  Adjustment: {p.get('suggested_adjustment', 'N/A')}")

    return "\n".join(lines)


def run_discovery() -> dict | None:
    """Run the full pattern discovery pipeline."""
    feedback_list = load_recent_feedback(days=7)

    if not feedback_list:
        log.warning("No recent feedback data available")
        return None

    log.info(f"Analyzing {len(feedback_list)} days of feedback")
    result = discover_patterns(feedback_list)

    if result:
        save_patterns(result)
        log.success(
            f"Discovery complete: {len(result.get('patterns', []))} patterns found, "
            f"calibration: {result.get('overall_calibration', 'unknown')}"
        )
    return result


def main():
    """Entry point — run pattern discovery."""
    log.info("=" * 50)
    log.info("Pattern Discoverer — Learning from Mistakes")
    log.info("=" * 50)

    if not Config.LLM_PATTERN_DISCOVERY_ENABLED:
        log.warning("LLM_PATTERN_DISCOVERY_ENABLED=false, running anyway")

    result = run_discovery()
    return result


if __name__ == "__main__":
    main()
