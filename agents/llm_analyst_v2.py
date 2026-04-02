"""LLM Analyst v2 — Big-picture macro → sector → symbol analysis pipeline.

Replaces the flat per-symbol loop with a 4-stage cascade:
  Stage 1: Macro scan (1 Haiku call) — global market regime
  Stage 2: Sector analysis (1 Haiku per sector) — sector outlook + cross-dependencies
  Stage 3: Symbol conviction (1 Haiku per symbol) — context-enriched conviction
  Stage 4: Portfolio synthesis (1 Sonnet call) — risk-adjusted final scores

Estimated cost: ~$0.06/day (vs ~$0.49/day for v1).
Writes to the same convictions.json so signal_modifiers.py works unchanged.

When LLM_FEEDBACK_LOOP_ENABLED: injects accuracy feedback into prompts.
When LLM_INFLUENCER_TRACKING_ENABLED: injects influencer activity context.
When LLM_PATTERN_DISCOVERY_ENABLED: injects learned patterns from feedback data.
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.llm_client import call_llm_json, get_daily_spend
from core.news_client import fetch_headlines
from utils.logger import log

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "llm_analyst")
_CONVICTIONS_FILE = os.path.join(_DATA_DIR, "convictions.json")
_MACRO_FILE = os.path.join(_DATA_DIR, "macro_report.json")
_SECTOR_FILE = os.path.join(_DATA_DIR, "sector_report.json")


def _ensure_data_dir():
    os.makedirs(_DATA_DIR, exist_ok=True)


# ── Sector mappings ────────────────────────────────────────────────

# Map sectors to ETFs for broad news fetching
SECTOR_ETF_MAP = {
    "Tech": "XLK",
    "Finance": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer": "XLY",
    "Industrial": "XLI",
    "Fintech": "XLF",  # reuses Finance ETF
    "ETF": "SPY",
}

# Map symbols to sectors (imported from scanner, replicated here for independence)
SYMBOL_SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech", "META": "Tech",
    "NVDA": "Tech", "TSM": "Tech", "AVGO": "Tech", "ORCL": "Tech", "CRM": "Tech",
    "AMD": "Tech", "INTC": "Tech", "QCOM": "Tech", "AMAT": "Tech", "MU": "Tech",
    "LRCX": "Tech", "KLAC": "Tech", "MRVL": "Tech", "SNPS": "Tech", "CDNS": "Tech",
    "TSLA": "Consumer", "NFLX": "Consumer", "DIS": "Consumer", "BABA": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "MCD": "Consumer", "COST": "Consumer",
    "WMT": "Consumer", "TGT": "Consumer",
    "AMGN": "Healthcare", "GILD": "Healthcare", "ISRG": "Healthcare", "DXCM": "Healthcare",
    "VRTX": "Healthcare", "REGN": "Healthcare", "MRNA": "Healthcare", "BIIB": "Healthcare",
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "MS": "Finance", "WFC": "Finance",
    "C": "Finance", "BLK": "Finance", "SCHW": "Finance", "AXP": "Finance", "V": "Finance",
    "MA": "Finance",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "OXY": "Energy",
    "EOG": "Energy", "MPC": "Energy", "VLO": "Energy", "PSX": "Energy",
    "BA": "Industrial", "CAT": "Industrial", "DE": "Industrial", "HON": "Industrial",
    "GE": "Industrial", "RTX": "Industrial", "LMT": "Industrial", "UNP": "Industrial",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF", "XLF": "ETF",
    "XLE": "ETF", "XLK": "ETF", "ARKK": "ETF",
    "COIN": "Fintech", "SQ": "Fintech", "SHOP": "Fintech", "PLTR": "Fintech",
    "SOFI": "Fintech", "RIVN": "Consumer", "LCID": "Consumer", "UBER": "Consumer",
    "LYFT": "Consumer", "SNAP": "Tech",
    # Crypto
    "BTC/USD": "Crypto", "ETH/USD": "Crypto",
}

# Cross-sector dependency relationships
# Format: {source_sector: {affected_sector: impact_direction}}
# Positive = same direction, Negative = inverse
SECTOR_DEPENDENCIES = {
    "Energy": {
        "Industrial": -0.5,    # Energy costs up → industrial margins squeezed
        "Consumer": -0.3,      # Higher fuel/goods costs → consumer spending down
        "Finance": 0.2,        # Energy lending benefits
    },
    "Finance": {
        "Tech": -0.4,          # Rate hikes → growth/tech valuations compressed
        "Consumer": -0.3,      # Tighter credit → less consumer spending
        "Fintech": -0.3,       # Same as tech, rate-sensitive
        "Industrial": -0.2,    # Higher borrowing costs
    },
    "Tech": {
        "Consumer": 0.4,       # Tech supply chain, consumer electronics
        "Fintech": 0.6,        # Fintech closely tied to tech sentiment
    },
    "Healthcare": {
        # Largely independent, minor defensive rotation effects
    },
    "Consumer": {
        "Industrial": 0.3,     # Consumer demand → industrial production
    },
    "Industrial": {
        "Energy": 0.3,         # Industrial activity → energy demand
    },
}

# Historical macro patterns — injected into sector prompts as context
HISTORICAL_PATTERNS = [
    {
        "trigger": "Fed raises interest rates or signals hawkish stance",
        "effects": "Finance +20-30%, Tech -15-25%, Consumer -10-15%, Fintech -20%. Growth stocks suffer most. Value/dividend stocks outperform.",
    },
    {
        "trigger": "Fed cuts rates or signals dovish stance",
        "effects": "Tech +20-30%, Consumer +15%, Fintech +25%, Finance -5-10%. Growth stocks rally. Bond proxies decline.",
    },
    {
        "trigger": "Oil prices spike >5% (supply disruption, OPEC cuts, geopolitical)",
        "effects": "Energy +15-25%, Industrial -10-15%, Consumer -5-10%, Airlines/Transport -15-20%. Inflationary pressure across sectors.",
    },
    {
        "trigger": "Oil prices drop >5% (demand concerns, supply glut)",
        "effects": "Energy -15-25%, Industrial +5-10%, Consumer +5%, Transport +10-15%. Deflationary signal.",
    },
    {
        "trigger": "Strong jobs report / low unemployment",
        "effects": "Finance +5%, Consumer +5-10%, Industrial +5%. May trigger rate hike fears → Tech -5%.",
    },
    {
        "trigger": "Weak jobs report / rising unemployment",
        "effects": "Defensive sectors (Healthcare, Utilities) +5%. Cyclicals (Consumer, Industrial) -10%. May trigger rate cut hopes → Tech +5%.",
    },
    {
        "trigger": "Geopolitical escalation (war, sanctions, trade war)",
        "effects": "Defense/Aerospace +10-15%, Energy +5-10%, Tech -10% (supply chain), Consumer -5%. Flight to safety (gold, bonds).",
    },
    {
        "trigger": "Semiconductor shortage or chip export restrictions",
        "effects": "Semiconductor stocks mixed (+/-20%), Consumer electronics -10%, Auto -10-15%, Cloud/AI companies -5%.",
    },
    {
        "trigger": "Banking crisis or credit stress",
        "effects": "Finance -20-30%, Fintech -15%, all sectors -5-10% (contagion fear). Flight to quality, defensive rotation.",
    },
    {
        "trigger": "Major tech earnings beat (FAANG+)",
        "effects": "Tech +5-10%, Fintech +5%, broad market sentiment boost +2-3%. AI/cloud sub-sector rally.",
    },
    {
        "trigger": "Recession fears / yield curve inversion",
        "effects": "Cyclicals (Consumer, Industrial, Energy) -10-15%. Defensives (Healthcare, Utilities) +5%. Finance mixed. Tech -5-10%.",
    },
    {
        "trigger": "Crypto regulatory crackdown or exchange failure",
        "effects": "Crypto -15-30%, Fintech (COIN, SQ) -10-20%, broader market minimal impact unless systemic.",
    },
]


# ── System prompts ─────────────────────────────────────────────────

_MACRO_SYSTEM = """You are a senior macro strategist. Analyze today's market headlines and identify:
1. Market regime: risk-on (bullish, buying dips) or risk-off (bearish, selling rallies) or neutral
2. Fed policy stance: hawkish (tightening), dovish (easing), or neutral
3. Key macro events that will move markets today
4. Geopolitical risk level
5. Sector-level implications — which sectors benefit/suffer from today's news

Be specific and actionable. Focus on what matters TODAY, not general market knowledge.

Respond in JSON:
{
  "regime": "risk-on|risk-off|neutral",
  "fed_stance": "hawkish|dovish|neutral",
  "key_events": ["event1", "event2", ...],
  "geopolitical_risk": "low|medium|high",
  "summary": "2-3 sentence macro summary",
  "sector_implications": {"Tech": "bullish|bearish|neutral", "Finance": "...", ...}
}"""

_SECTOR_SYSTEM = """You are a sector analyst. Given the macro context and sector-specific news, analyze this sector's outlook for today.

Consider:
1. Direct impact of macro events on this sector
2. Cross-sector dependencies (e.g., energy costs affecting industrials)
3. Historical patterns — when similar conditions occurred, how did this sector perform?
4. Supply chain and demand chain effects

Respond in JSON:
{
  "sector": "<sector_name>",
  "outlook": "bullish|bearish|neutral",
  "conviction": <float -1.0 to 1.0>,
  "drivers": ["driver1", "driver2"],
  "cross_sector_effects": {"affected_sector": <float impact>},
  "risk_level": "low|medium|high",
  "reasoning": "2-3 sentence explanation"
}"""

_SYMBOL_SYSTEM = """You are an equity analyst. Given the macro outlook, sector analysis, and symbol-specific data, provide a conviction score for this stock TODAY.

Consider:
1. How macro conditions affect this specific company
2. The sector outlook and where this company sits within it
3. Symbol-specific news and catalysts
4. Technical indicators (trend, momentum, volume)
5. Whether the stock is likely to outperform or underperform its sector today

Respond in JSON:
{
  "conviction_score": <float -1.0 to 1.0, positive=bullish>,
  "bias": "bullish|bearish|neutral",
  "confidence": <float 0.0 to 1.0>,
  "key_factors": ["factor1", "factor2"],
  "risk_flags": ["risk1", "risk2"],
  "reasoning": "1-2 sentence explanation"
}"""

_PORTFOLIO_SYSTEM = """You are a portfolio risk manager reviewing today's trading plan. Given all symbol convictions, macro context, and sector analysis:

1. Check for CONCENTRATION RISK — too many correlated bets in one sector
2. Check for CONTRADICTIONS — bullish on a sector but bearish on its key stocks
3. Adjust conviction scores if portfolio is too one-directional
4. Flag any positions that should be reduced or avoided
5. Provide an overall portfolio risk assessment

You may adjust individual conviction scores up or down by at most 0.3 to account for portfolio-level risk.

Respond in JSON:
{
  "adjusted_convictions": {"SYMBOL": <adjusted_score>, ...},
  "portfolio_risk": "low|medium|high",
  "concentration_warnings": ["warning1", ...],
  "recommendations": ["rec1", "rec2"],
  "overall_bias": "bullish|bearish|neutral",
  "reasoning": "2-3 sentence portfolio assessment"
}"""


# ── Stage 1: Macro Scan ──────────────────────────────────────────

def analyze_macro(fetch_fn=None) -> dict:
    """Fetch broad market news and analyze macro conditions.

    Args:
        fetch_fn: Optional override for headline fetching (for testing).

    Returns:
        Macro analysis dict with regime, fed_stance, sector_implications, etc.
    """
    _fetch = fetch_fn or fetch_headlines
    default_result = {
        "regime": "neutral",
        "fed_stance": "neutral",
        "key_events": [],
        "geopolitical_risk": "low",
        "summary": "No macro data available",
        "sector_implications": {},
    }

    # Fetch broad market headlines
    headlines = []
    for symbol in ["SPY", "QQQ", "DIA"]:
        headlines.extend(_fetch(symbol, lookback_hours=24))

    if not headlines:
        log.warning("No macro headlines found, using neutral defaults")
        return default_result

    # Deduplicate headlines
    headlines = list(dict.fromkeys(headlines))[:30]

    # Build prompt with optional influencer context
    prompt_parts = [
        "Today's broad market headlines:\n",
        "\n".join(f"- {h}" for h in headlines),
    ]

    if Config.LLM_INFLUENCER_TRACKING_ENABLED:
        from core.influencer_registry import get_influencer_summary
        influencer_ctx = get_influencer_summary(headlines)
        if influencer_ctx:
            prompt_parts.append(f"\n\n{influencer_ctx}")

    prompt = "\n".join(prompt_parts)

    try:
        result = call_llm_json(
            prompt,
            model=Config.LLM_QUICK_MODEL,
            system=_MACRO_SYSTEM,
            max_tokens=768,
        )
        parsed = result.get("parsed", default_result)
        log.info(f"Macro scan: regime={parsed.get('regime')}, "
                 f"fed={parsed.get('fed_stance')}, "
                 f"geo_risk={parsed.get('geopolitical_risk')}")
        return parsed
    except Exception as e:
        log.warning(f"Macro analysis failed: {e}")
        return default_result


# ── Stage 2: Sector Analysis ─────────────────────────────────────

def _build_sector_prompt(sector: str, macro_context: dict,
                         headlines: list[str]) -> str:
    """Build the prompt for a sector analysis call."""
    # Macro summary
    macro_summary = macro_context.get("summary", "No macro data")
    macro_implications = macro_context.get("sector_implications", {})
    macro_view = macro_implications.get(sector, "neutral")

    # Cross-sector dependencies affecting this sector
    dependencies = []
    for source_sector, impacts in SECTOR_DEPENDENCIES.items():
        if sector in impacts:
            direction = "positively" if impacts[sector] > 0 else "negatively"
            dependencies.append(
                f"- {source_sector} movements {direction} affect {sector} "
                f"(correlation: {impacts[sector]:+.1f})"
            )

    # Relevant historical patterns
    patterns = []
    for p in HISTORICAL_PATTERNS:
        if sector.lower() in p["effects"].lower():
            patterns.append(f"- When: {p['trigger']}\n  Then: {p['effects']}")

    parts = [
        f"## Macro Context\n{macro_summary}\nMacro view on {sector}: {macro_view}",
        f"\n## {sector} Sector Headlines\n"
        + ("\n".join(f"- {h}" for h in headlines[:15]) if headlines else "No sector-specific news"),
    ]

    if dependencies:
        parts.append(
            "\n## Cross-Sector Dependencies\n" + "\n".join(dependencies)
        )

    if patterns:
        parts.append(
            "\n## Historical Patterns (when similar conditions occurred)\n"
            + "\n".join(patterns[:5])
        )

    # Inject influencer context (if enabled)
    if Config.LLM_INFLUENCER_TRACKING_ENABLED:
        from core.influencer_registry import get_influencer_context_for_sector
        influencer_ctx = get_influencer_context_for_sector(sector, headlines)
        if influencer_ctx:
            parts.append(f"\n{influencer_ctx}")

    # Inject feedback (if enabled)
    if Config.LLM_FEEDBACK_LOOP_ENABLED:
        from agents.conviction_scorer import get_sector_feedback
        feedback_ctx = get_sector_feedback(sector)
        if feedback_ctx:
            parts.append(f"\n{feedback_ctx}")

    # Inject discovered patterns (if enabled)
    if Config.LLM_PATTERN_DISCOVERY_ENABLED:
        from agents.pattern_discoverer import get_discovered_patterns_for_sector
        learned_ctx = get_discovered_patterns_for_sector(sector)
        if learned_ctx:
            parts.append(f"\n{learned_ctx}")

    return "\n".join(parts)


def analyze_sector(sector: str, macro_context: dict,
                   fetch_fn=None) -> dict:
    """Analyze a single sector given macro context.

    Args:
        sector: Sector name (e.g., "Tech", "Energy").
        macro_context: Output from analyze_macro().
        fetch_fn: Optional override for headline fetching.

    Returns:
        Sector analysis dict.
    """
    _fetch = fetch_fn or fetch_headlines
    default_result = {
        "sector": sector,
        "outlook": "neutral",
        "conviction": 0.0,
        "drivers": [],
        "cross_sector_effects": {},
        "risk_level": "medium",
        "reasoning": "Analysis unavailable",
    }

    # Fetch sector ETF headlines
    etf = SECTOR_ETF_MAP.get(sector, "SPY")
    headlines = _fetch(etf, lookback_hours=24)

    prompt = _build_sector_prompt(sector, macro_context, headlines)

    try:
        result = call_llm_json(
            prompt,
            model=Config.LLM_QUICK_MODEL,
            system=_SECTOR_SYSTEM,
            max_tokens=512,
        )
        parsed = result.get("parsed", default_result)
        parsed["sector"] = sector  # ensure sector name is correct
        conviction = parsed.get("conviction", 0)
        if isinstance(conviction, (int, float)):
            parsed["conviction"] = max(-1.0, min(1.0, conviction))
        log.info(f"Sector {sector}: outlook={parsed.get('outlook')}, "
                 f"conviction={parsed.get('conviction', 0):+.2f}")
        return parsed
    except Exception as e:
        log.warning(f"Sector analysis failed for {sector}: {e}")
        return default_result


def analyze_all_sectors(macro_context: dict, symbols: list[str],
                        fetch_fn=None) -> dict[str, dict]:
    """Analyze all sectors relevant to the given symbols.

    Returns:
        Dict mapping sector name to sector analysis.
    """
    # Determine which sectors we actually need
    active_sectors = set()
    for sym in symbols:
        sector = SYMBOL_SECTOR_MAP.get(sym)
        if sector and sector != "Crypto":  # Crypto doesn't have an ETF
            active_sectors.add(sector)

    sector_results = {}
    for sector in sorted(active_sectors):
        if get_daily_spend() >= Config.LLM_BUDGET_DAILY:
            log.warning(f"Budget reached during sector analysis at {sector}")
            break
        sector_results[sector] = analyze_sector(sector, macro_context, fetch_fn)

    return sector_results


# ── Stage 3: Symbol Conviction ───────────────────────────────────

def _build_symbol_prompt(symbol: str, macro_context: dict,
                         sector_context: dict | None,
                         headlines: list[str],
                         indicators: dict) -> str:
    """Build the prompt for a symbol conviction call."""
    sector = SYMBOL_SECTOR_MAP.get(symbol, "Unknown")
    macro_summary = macro_context.get("summary", "No macro data")

    parts = [
        f"## Symbol: {symbol} (Sector: {sector})",
        f"\n## Macro Context\n{macro_summary}",
    ]

    # Sector context
    if sector_context:
        parts.append(
            f"\n## Sector Outlook ({sector})\n"
            f"Outlook: {sector_context.get('outlook', 'neutral')}, "
            f"Conviction: {sector_context.get('conviction', 0):+.2f}\n"
            f"Drivers: {', '.join(sector_context.get('drivers', []))}\n"
            f"Reasoning: {sector_context.get('reasoning', 'N/A')}"
        )

        # Cross-sector effects impacting this symbol's sector
        effects = sector_context.get("cross_sector_effects", {})
        if effects:
            parts.append(
                "\nCross-sector effects: "
                + ", ".join(f"{k}: {v:+.2f}" for k, v in effects.items())
            )

    # Symbol-specific headlines
    if headlines:
        parts.append(
            f"\n## {symbol} Recent Headlines\n"
            + "\n".join(f"- {h}" for h in headlines[:10])
        )
    else:
        parts.append(f"\n## {symbol} Recent Headlines\nNo symbol-specific news")

    # Technical indicators
    if indicators:
        parts.append(
            "\n## Technical Indicators\n"
            + "\n".join(f"- {k}: {v}" for k, v in indicators.items())
        )

    # Inject influencer context (if enabled)
    if Config.LLM_INFLUENCER_TRACKING_ENABLED:
        from core.influencer_registry import get_influencer_context_for_symbol
        influencer_ctx = get_influencer_context_for_symbol(symbol, headlines)
        if influencer_ctx:
            parts.append(f"\n{influencer_ctx}")

    # Inject feedback (if enabled)
    if Config.LLM_FEEDBACK_LOOP_ENABLED:
        from agents.conviction_scorer import get_symbol_feedback
        feedback_ctx = get_symbol_feedback(symbol)
        if feedback_ctx:
            parts.append(f"\n{feedback_ctx}")

    return "\n".join(parts)


def analyze_symbol(symbol: str, macro_context: dict,
                   sector_contexts: dict[str, dict],
                   headlines: list[str] | None = None,
                   indicators: dict | None = None,
                   fetch_fn=None) -> dict | None:
    """Analyze a single symbol with full macro + sector context.

    Args:
        symbol: Trading symbol.
        macro_context: Output from analyze_macro().
        sector_contexts: Output from analyze_all_sectors().
        headlines: Pre-fetched headlines, or None to fetch.
        indicators: Technical indicators dict.
        fetch_fn: Optional override for headline fetching.

    Returns:
        Conviction dict or None if budget exceeded.
    """
    if get_daily_spend() >= Config.LLM_BUDGET_DAILY:
        log.warning(f"Budget exceeded, skipping LLM analysis for {symbol}")
        return None

    _fetch = fetch_fn or fetch_headlines

    # Get sector context for this symbol
    sector = SYMBOL_SECTOR_MAP.get(symbol, "Unknown")
    sector_ctx = sector_contexts.get(sector)

    # Fetch symbol-specific headlines if not provided
    if headlines is None:
        headlines = _fetch(symbol, lookback_hours=24)

    prompt = _build_symbol_prompt(
        symbol, macro_context, sector_ctx, headlines or [], indicators or {}
    )

    try:
        result = call_llm_json(
            prompt,
            model=Config.LLM_QUICK_MODEL,
            system=_SYMBOL_SYSTEM,
            max_tokens=512,
        )
        parsed = result.get("parsed", {})
        score = parsed.get("conviction_score", 0)
        if not isinstance(score, (int, float)):
            score = 0
        parsed["conviction_score"] = max(-1.0, min(1.0, score))
        return parsed
    except Exception as e:
        log.warning(f"Symbol analysis failed for {symbol}: {e}")
        return {
            "conviction_score": 0,
            "bias": "neutral",
            "confidence": 0,
            "key_factors": [],
            "risk_flags": ["analysis_failed"],
            "reasoning": str(e),
        }


# ── Stage 4: Portfolio Synthesis ─────────────────────────────────

def synthesize_portfolio(macro_context: dict,
                         sector_contexts: dict[str, dict],
                         symbol_convictions: dict[str, dict]) -> dict:
    """Run portfolio-level risk assessment and adjust convictions.

    Args:
        macro_context: Output from analyze_macro().
        sector_contexts: Output from analyze_all_sectors().
        symbol_convictions: Dict mapping symbol to conviction dict.

    Returns:
        Dict with adjusted_convictions, portfolio_risk, recommendations.
    """
    default_result = {
        "adjusted_convictions": {
            sym: conv.get("conviction_score", 0)
            for sym, conv in symbol_convictions.items()
        },
        "portfolio_risk": "medium",
        "concentration_warnings": [],
        "recommendations": [],
        "overall_bias": "neutral",
        "reasoning": "Synthesis unavailable, using raw convictions",
    }

    if not symbol_convictions:
        return default_result

    # Build portfolio summary for Sonnet
    macro_summary = macro_context.get("summary", "No macro data")

    sector_summary = "\n".join(
        f"- {s}: {ctx.get('outlook', 'neutral')} (conviction: {ctx.get('conviction', 0):+.2f})"
        for s, ctx in sector_contexts.items()
    )

    conviction_lines = []
    for sym, conv in symbol_convictions.items():
        sector = SYMBOL_SECTOR_MAP.get(sym, "Unknown")
        score = conv.get("conviction_score", 0)
        bias = conv.get("bias", "neutral")
        conviction_lines.append(f"- {sym} ({sector}): {score:+.2f} ({bias})")

    prompt = (
        f"## Macro Context\n{macro_summary}\n\n"
        f"## Sector Outlooks\n{sector_summary}\n\n"
        f"## Individual Convictions\n" + "\n".join(conviction_lines) + "\n\n"
        "Review this portfolio for concentration risk, contradictions, and overall balance. "
        "Adjust conviction scores if needed (max +/-0.3 adjustment)."
    )

    try:
        result = call_llm_json(
            prompt,
            model=Config.LLM_DEEP_MODEL,
            system=_PORTFOLIO_SYSTEM,
            max_tokens=1024,
        )
        parsed = result.get("parsed", default_result)

        # Validate adjusted convictions
        adjusted = parsed.get("adjusted_convictions", {})
        for sym in symbol_convictions:
            if sym in adjusted:
                score = adjusted[sym]
                if isinstance(score, (int, float)):
                    adjusted[sym] = max(-1.0, min(1.0, score))
                else:
                    adjusted[sym] = symbol_convictions[sym].get("conviction_score", 0)
            else:
                adjusted[sym] = symbol_convictions[sym].get("conviction_score", 0)
        parsed["adjusted_convictions"] = adjusted

        log.info(f"Portfolio synthesis: risk={parsed.get('portfolio_risk')}, "
                 f"bias={parsed.get('overall_bias')}")
        return parsed
    except Exception as e:
        log.warning(f"Portfolio synthesis failed: {e}")
        return default_result


# ── Orchestrator ─────────────────────────────────────────────────

def run_analysis(symbols: list[str] | None = None,
                 fetch_fn=None) -> dict:
    """Run the full 4-stage analysis pipeline.

    Args:
        symbols: List of symbols to analyze. Defaults to Config.SYMBOLS.
        fetch_fn: Optional override for headline fetching (for testing).

    Returns:
        Full convictions dict ready for signal_modifiers.
    """
    if symbols is None:
        symbols = Config.SYMBOLS

    log.info("Stage 1/4: Macro scan...")
    macro_context = analyze_macro(fetch_fn)

    log.info("Stage 2/4: Sector analysis...")
    sector_contexts = analyze_all_sectors(macro_context, symbols, fetch_fn)

    log.info("Stage 3/4: Symbol convictions...")
    symbol_convictions = {}
    for symbol in symbols:
        if get_daily_spend() >= Config.LLM_BUDGET_DAILY:
            log.warning(f"Budget reached at {symbol}, stopping symbol analysis")
            break
        result = analyze_symbol(symbol, macro_context, sector_contexts,
                                fetch_fn=fetch_fn)
        if result is not None:
            symbol_convictions[symbol] = result

    log.info("Stage 4/4: Portfolio synthesis...")
    portfolio = synthesize_portfolio(macro_context, sector_contexts,
                                     symbol_convictions)

    # Build final convictions using adjusted scores
    adjusted_scores = portfolio.get("adjusted_convictions", {})
    convictions = {}
    for sym, conv in symbol_convictions.items():
        adjusted_score = adjusted_scores.get(sym, conv.get("conviction_score", 0))
        convictions[sym] = {
            "score": adjusted_score,
            "raw_score": conv.get("conviction_score", 0),
            "bias": conv.get("bias", "neutral"),
            "confidence": conv.get("confidence", 0),
            "key_factors": conv.get("key_factors", []),
            "risk_flags": conv.get("risk_flags", []),
            "reasoning": conv.get("reasoning", ""),
        }

    return {
        "timestamp": datetime.now().isoformat(),
        "version": "v2",
        "convictions": convictions,
        "portfolio_risk": portfolio.get("portfolio_risk", "medium"),
        "concentration_warnings": portfolio.get("concentration_warnings", []),
        "recommendations": portfolio.get("recommendations", []),
        "overall_bias": portfolio.get("overall_bias", "neutral"),
        "daily_spend": round(get_daily_spend(), 4),
    }


def write_outputs(data: dict, macro_context: dict,
                  sector_contexts: dict[str, dict]):
    """Write all output files."""
    _ensure_data_dir()

    # Convictions (same path as v1 — signal_modifiers reads this)
    with open(_CONVICTIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    log.success(f"Wrote convictions for {len(data.get('convictions', {}))} symbols")

    # Macro report (bonus output for observability)
    macro_report = {
        "timestamp": datetime.now().isoformat(),
        **macro_context,
    }
    with open(_MACRO_FILE, "w") as f:
        json.dump(macro_report, f, indent=2)
    log.success("Wrote macro report")

    # Sector report (bonus output for observability)
    sector_report = {
        "timestamp": datetime.now().isoformat(),
        "sectors": sector_contexts,
    }
    with open(_SECTOR_FILE, "w") as f:
        json.dump(sector_report, f, indent=2)
    log.success("Wrote sector report")

    # Archive predictions for feedback loop (if enabled)
    if Config.LLM_FEEDBACK_LOOP_ENABLED:
        from agents.conviction_scorer import archive_predictions
        archive_predictions(data)


def main():
    """Main entry point — run full v2 LLM analysis pipeline."""
    log.info("=" * 50)
    log.info("LLM Analyst v2 — Big Picture Pipeline")
    log.info("=" * 50)

    if not Config.ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY not set, cannot run LLM analysis")
        return None

    # Run stages 1-3, capturing intermediate results
    macro_context = analyze_macro()

    sector_contexts = analyze_all_sectors(macro_context, Config.SYMBOLS)

    # Run stages 3-4 via orchestrator (which also does portfolio synthesis)
    symbol_convictions = {}
    for symbol in Config.SYMBOLS:
        if get_daily_spend() >= Config.LLM_BUDGET_DAILY:
            log.warning(f"Budget reached at {symbol}")
            break
        result = analyze_symbol(symbol, macro_context, sector_contexts)
        if result is not None:
            symbol_convictions[symbol] = result

    portfolio = synthesize_portfolio(macro_context, sector_contexts,
                                     symbol_convictions)

    # Build output
    adjusted_scores = portfolio.get("adjusted_convictions", {})
    convictions = {}
    for sym, conv in symbol_convictions.items():
        adjusted_score = adjusted_scores.get(sym, conv.get("conviction_score", 0))
        convictions[sym] = {
            "score": adjusted_score,
            "raw_score": conv.get("conviction_score", 0),
            "bias": conv.get("bias", "neutral"),
            "confidence": conv.get("confidence", 0),
            "key_factors": conv.get("key_factors", []),
            "risk_flags": conv.get("risk_flags", []),
            "reasoning": conv.get("reasoning", ""),
        }

    data = {
        "timestamp": datetime.now().isoformat(),
        "version": "v2",
        "convictions": convictions,
        "portfolio_risk": portfolio.get("portfolio_risk", "medium"),
        "concentration_warnings": portfolio.get("concentration_warnings", []),
        "recommendations": portfolio.get("recommendations", []),
        "overall_bias": portfolio.get("overall_bias", "neutral"),
        "daily_spend": round(get_daily_spend(), 4),
    }

    write_outputs(data, macro_context, sector_contexts)

    scored = len(convictions)
    total = len(Config.SYMBOLS)
    log.success(
        f"v2 analysis complete: {scored}/{total} symbols, "
        f"portfolio_risk={data.get('portfolio_risk')}, "
        f"spend=${data.get('daily_spend', 0):.4f}"
    )
    return data


if __name__ == "__main__":
    main()
