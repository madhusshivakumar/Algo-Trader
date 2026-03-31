"""LLM Multi-Agent Analyst — Claude-powered analysis providing conviction scores.

Runs daily at 6:15 AM ET (after sentiment agent).
For each symbol, runs a 3-step analysis:
  1. News Analyst (Haiku) — summarizes recent news sentiment
  2. Technical Analyst (Haiku) — describes RSI/MACD/BB picture
  3. Debate Synthesizer (Sonnet) — outputs conviction score [-1, 1]

Writes results to data/llm_analyst/convictions.json for signal_modifiers to read.
"""

import json
import os
from datetime import datetime

from config import Config
from core.llm_client import call_llm_json, get_daily_spend
from utils.logger import log

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "llm_analyst")
_CONVICTIONS_FILE = os.path.join(_DATA_DIR, "convictions.json")


def _ensure_data_dir():
    os.makedirs(_DATA_DIR, exist_ok=True)


_NEWS_ANALYST_SYSTEM = """You are a financial news analyst. Analyze the provided news headlines
and summarize the key sentiment drivers. Focus on:
- Major events (earnings, product launches, lawsuits, executive changes)
- Market-moving news vs noise
- Overall sentiment direction (bullish/bearish/neutral)
Respond in JSON: {"summary": "...", "sentiment": "bullish|bearish|neutral", "key_events": [...]}"""

_TECHNICAL_ANALYST_SYSTEM = """You are a technical analysis expert. Analyze the provided technical
indicators and describe the current market picture. Focus on:
- Trend direction and strength
- Support/resistance levels
- Momentum indicators (RSI, MACD)
- Volume patterns
Respond in JSON: {"summary": "...", "trend": "up|down|sideways", "key_signals": [...]}"""

_SYNTHESIZER_SYSTEM = """You are a senior investment analyst synthesizing multiple perspectives.
Given a news analysis and technical analysis for a stock, provide a final conviction assessment.
Weigh both perspectives and output a balanced judgment.
Respond in JSON:
{
  "conviction_score": <float -1.0 to 1.0, positive=bullish>,
  "bias": "bullish|bearish|neutral",
  "confidence": <float 0.0 to 1.0>,
  "key_factors": ["factor1", "factor2"],
  "risk_flags": ["risk1", "risk2"],
  "reasoning": "brief explanation"
}"""


def analyze_news(symbol: str, headlines: list[str]) -> dict:
    """Step 1: News analyst (Haiku) — summarize news sentiment."""
    if not headlines:
        return {"summary": "No recent news", "sentiment": "neutral", "key_events": []}

    prompt = (
        f"Analyze these recent headlines for {symbol}:\n\n"
        + "\n".join(f"- {h}" for h in headlines[:20])
    )

    try:
        result = call_llm_json(
            prompt,
            model=Config.LLM_QUICK_MODEL,
            system=_NEWS_ANALYST_SYSTEM,
            max_tokens=512,
        )
        return result.get("parsed", {})
    except Exception as e:
        log.warning(f"News analysis failed for {symbol}: {e}")
        return {"summary": "Analysis failed", "sentiment": "neutral", "key_events": []}


def analyze_technicals(symbol: str, indicators: dict) -> dict:
    """Step 2: Technical analyst (Haiku) — describe technical picture."""
    if not indicators:
        return {"summary": "No indicator data", "trend": "sideways", "key_signals": []}

    prompt = (
        f"Analyze these technical indicators for {symbol}:\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in indicators.items())
    )

    try:
        result = call_llm_json(
            prompt,
            model=Config.LLM_QUICK_MODEL,
            system=_TECHNICAL_ANALYST_SYSTEM,
            max_tokens=512,
        )
        return result.get("parsed", {})
    except Exception as e:
        log.warning(f"Technical analysis failed for {symbol}: {e}")
        return {"summary": "Analysis failed", "trend": "sideways", "key_signals": []}


def synthesize(symbol: str, news_analysis: dict, tech_analysis: dict) -> dict:
    """Step 3: Debate synthesizer (Sonnet) — produce conviction score."""
    prompt = (
        f"Symbol: {symbol}\n\n"
        f"## News Analysis\n{json.dumps(news_analysis, indent=2)}\n\n"
        f"## Technical Analysis\n{json.dumps(tech_analysis, indent=2)}\n\n"
        "Synthesize these perspectives into a final conviction assessment."
    )

    try:
        result = call_llm_json(
            prompt,
            model=Config.LLM_DEEP_MODEL,
            system=_SYNTHESIZER_SYSTEM,
            max_tokens=1024,
        )
        parsed = result.get("parsed", {})
        # Ensure conviction_score is valid
        score = parsed.get("conviction_score", 0)
        if not isinstance(score, (int, float)):
            score = 0
        parsed["conviction_score"] = max(-1.0, min(1.0, score))
        return parsed
    except Exception as e:
        log.warning(f"Synthesis failed for {symbol}: {e}")
        return {
            "conviction_score": 0,
            "bias": "neutral",
            "confidence": 0,
            "key_factors": [],
            "risk_flags": ["analysis_failed"],
            "reasoning": str(e),
        }


def extract_indicators(df) -> dict:
    """Extract key technical indicators from a DataFrame for LLM analysis."""
    if df is None or df.empty:
        return {}

    latest = df.iloc[-1]
    indicators = {}

    for col in ["rsi", "RSI", "rsi_14"]:
        if col in df.columns:
            indicators["RSI"] = round(float(latest[col]), 2)
            break

    for col in ["macd", "MACD"]:
        if col in df.columns:
            indicators["MACD"] = round(float(latest[col]), 4)
            break

    for col in ["macd_signal", "MACD_signal"]:
        if col in df.columns:
            indicators["MACD_Signal"] = round(float(latest[col]), 4)
            break

    if "close" in df.columns:
        indicators["Price"] = round(float(latest["close"]), 2)
        # Simple momentum
        if len(df) >= 20:
            indicators["5d_change"] = round(
                float((latest["close"] - df.iloc[-5]["close"]) / df.iloc[-5]["close"] * 100), 2
            )
            indicators["20d_change"] = round(
                float((latest["close"] - df.iloc[-20]["close"]) / df.iloc[-20]["close"] * 100), 2
            )

    if "volume" in df.columns and len(df) >= 20:
        avg_vol = df["volume"].tail(20).mean()
        indicators["Volume_vs_avg"] = round(float(latest["volume"] / avg_vol), 2)

    return indicators


def analyze_symbol(symbol: str, headlines: list[str] | None = None,
                   df=None) -> dict | None:
    """Run full 3-step analysis for a single symbol.

    Returns conviction dict or None if budget exceeded.
    """
    # Budget check
    if get_daily_spend() >= Config.LLM_BUDGET_DAILY:
        log.warning(f"Budget exceeded, skipping LLM analysis for {symbol}")
        return None

    news_result = analyze_news(symbol, headlines or [])
    tech_indicators = extract_indicators(df)
    tech_result = analyze_technicals(symbol, tech_indicators)
    conviction = synthesize(symbol, news_result, tech_result)

    score = conviction.get("conviction_score", 0)
    bias = conviction.get("bias", "neutral")
    log.info(f"{symbol}: LLM conviction={score:+.2f} ({bias})")

    return {
        "score": score,
        "bias": bias,
        "confidence": conviction.get("confidence", 0),
        "key_factors": conviction.get("key_factors", []),
        "risk_flags": conviction.get("risk_flags", []),
        "reasoning": conviction.get("reasoning", ""),
    }


def run_analysis(symbols: list[str] | None = None) -> dict:
    """Run LLM analysis for all symbols. Returns full convictions dict."""
    if symbols is None:
        symbols = Config.SYMBOLS

    convictions = {}
    for symbol in symbols:
        if get_daily_spend() >= Config.LLM_BUDGET_DAILY:
            log.warning(f"Daily budget reached, stopping analysis at {symbol}")
            break

        result = analyze_symbol(symbol)
        if result is not None:
            convictions[symbol] = result

    return {
        "timestamp": datetime.now().isoformat(),
        "convictions": convictions,
        "daily_spend": round(get_daily_spend(), 4),
    }


def write_convictions(data: dict):
    """Write convictions to JSON file for signal_modifiers to read."""
    _ensure_data_dir()
    with open(_CONVICTIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    log.success(f"Wrote LLM convictions for {len(data.get('convictions', {}))} symbols")


def main():
    """Main entry point — run full LLM analysis pipeline."""
    log.info("=" * 50)
    log.info("LLM Analyst Agent starting")
    log.info("=" * 50)

    if not Config.LLM_ANALYST_ENABLED:
        log.warning("LLM_ANALYST_ENABLED=false, running anyway (agent invoked directly)")

    if not Config.ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY not set, cannot run LLM analysis")
        return None

    data = run_analysis()
    write_convictions(data)

    scored = len(data.get("convictions", {}))
    total = len(Config.SYMBOLS)
    log.success(
        f"LLM analysis complete: {scored}/{total} symbols analyzed, "
        f"daily spend: ${data.get('daily_spend', 0):.4f}"
    )
    return data


if __name__ == "__main__":
    main()
