"""Influencer Registry — tracks key market-moving figures and their sector impact.

Maps influential people to their sectors of influence and historical impact patterns.
Provides headline matching to detect influencer activity in news feeds.
No LLM calls — pure string matching and static data.
"""

import re
from dataclasses import dataclass, field


@dataclass
class InfluencerMatch:
    """A matched influencer mention in a headline."""
    name: str
    role: str
    headline: str
    sectors: list[str]
    symbols: list[str]
    patterns: list[dict]


# ── Influencer Registry ────────────────────────────────────────

INFLUENCER_REGISTRY = {
    # ── Federal Reserve & Monetary Policy ──
    "Jerome Powell": {
        "role": "Federal Reserve Chair",
        "sectors": ["Finance", "Tech", "Consumer", "Industrial", "Energy"],
        "symbols": [],
        "keywords": ["jerome powell", "fed chair powell", "powell said", "powell's", "chairman powell"],
        "historical_patterns": [
            {"event": "hawkish speech / rate hike signal",
             "typical_impact": {"Finance": +0.2, "Tech": -0.3, "Consumer": -0.2, "Fintech": -0.3}},
            {"event": "dovish speech / rate cut signal",
             "typical_impact": {"Finance": -0.1, "Tech": +0.3, "Consumer": +0.2, "Fintech": +0.3}},
            {"event": "inflation concerns raised",
             "typical_impact": {"Energy": +0.1, "Tech": -0.2, "Consumer": -0.15}},
        ],
    },
    "Janet Yellen": {
        "role": "Treasury Secretary",
        "sectors": ["Finance", "Tech", "Consumer"],
        "symbols": [],
        "keywords": ["janet yellen", "treasury secretary yellen", "yellen said", "yellen's"],
        "historical_patterns": [
            {"event": "fiscal stimulus / spending signal",
             "typical_impact": {"Consumer": +0.2, "Industrial": +0.2, "Finance": +0.1}},
            {"event": "debt ceiling / fiscal concern",
             "typical_impact": {"Finance": -0.2, "Tech": -0.1}},
        ],
    },

    # ── Tech Leaders ──
    "Elon Musk": {
        "role": "CEO, Tesla & SpaceX; Owner, X",
        "sectors": ["Consumer", "Tech", "Crypto"],
        "symbols": ["TSLA"],
        "keywords": ["elon musk", "musk said", "musk's", "tesla ceo"],
        "historical_patterns": [
            {"event": "Tesla product / delivery announcement",
             "typical_impact": {"TSLA": +0.3, "Consumer": +0.1}},
            {"event": "controversial statement / regulatory concern",
             "typical_impact": {"TSLA": -0.2}},
            {"event": "crypto endorsement or comment",
             "typical_impact": {"BTC/USD": +0.15, "Crypto": +0.1}},
            {"event": "Tesla earnings beat",
             "typical_impact": {"TSLA": +0.4, "Consumer": +0.1}},
            {"event": "Tesla earnings miss / delivery miss",
             "typical_impact": {"TSLA": -0.4, "Consumer": -0.1}},
        ],
    },
    "Jensen Huang": {
        "role": "CEO, NVIDIA",
        "sectors": ["Tech"],
        "symbols": ["NVDA", "AMD", "TSM", "AVGO", "MRVL"],
        "keywords": ["jensen huang", "nvidia ceo", "huang said"],
        "historical_patterns": [
            {"event": "AI/GPU demand announcement or product launch",
             "typical_impact": {"NVDA": +0.3, "AMD": +0.15, "TSM": +0.1, "Tech": +0.1}},
            {"event": "supply constraint or guidance cut",
             "typical_impact": {"NVDA": -0.3, "AMD": -0.1, "Tech": -0.1}},
        ],
    },
    "Tim Cook": {
        "role": "CEO, Apple",
        "sectors": ["Tech", "Consumer"],
        "symbols": ["AAPL"],
        "keywords": ["tim cook", "apple ceo", "cook said"],
        "historical_patterns": [
            {"event": "new product launch (iPhone, Vision Pro, etc.)",
             "typical_impact": {"AAPL": +0.2, "Tech": +0.05}},
            {"event": "China/supply chain concerns",
             "typical_impact": {"AAPL": -0.2, "Tech": -0.1}},
        ],
    },
    "Satya Nadella": {
        "role": "CEO, Microsoft",
        "sectors": ["Tech"],
        "symbols": ["MSFT", "ORCL", "CRM"],
        "keywords": ["satya nadella", "microsoft ceo", "nadella said"],
        "historical_patterns": [
            {"event": "AI / cloud growth announcement",
             "typical_impact": {"MSFT": +0.2, "Tech": +0.1}},
            {"event": "antitrust or regulatory concern",
             "typical_impact": {"MSFT": -0.15, "Tech": -0.05}},
        ],
    },
    "Mark Zuckerberg": {
        "role": "CEO, Meta",
        "sectors": ["Tech"],
        "symbols": ["META"],
        "keywords": ["mark zuckerberg", "zuckerberg", "meta ceo"],
        "historical_patterns": [
            {"event": "ad revenue growth / user growth",
             "typical_impact": {"META": +0.25, "Tech": +0.05}},
            {"event": "metaverse spending concerns / Reality Labs losses",
             "typical_impact": {"META": -0.2}},
        ],
    },
    "Andy Jassy": {
        "role": "CEO, Amazon",
        "sectors": ["Tech", "Consumer"],
        "symbols": ["AMZN"],
        "keywords": ["andy jassy", "amazon ceo", "jassy said"],
        "historical_patterns": [
            {"event": "AWS growth acceleration",
             "typical_impact": {"AMZN": +0.25, "Tech": +0.1}},
            {"event": "layoffs or cost-cutting signal",
             "typical_impact": {"AMZN": +0.1, "Tech": -0.05}},
        ],
    },

    # ── Investors & Market Movers ──
    "Warren Buffett": {
        "role": "CEO, Berkshire Hathaway",
        "sectors": ["Finance", "Energy", "Consumer"],
        "symbols": [],
        "keywords": ["warren buffett", "buffett said", "buffett's", "berkshire hathaway"],
        "historical_patterns": [
            {"event": "major stock purchase disclosed",
             "typical_impact": {"Finance": +0.1}},
            {"event": "cash pile increase / market caution",
             "typical_impact": {"Finance": -0.1}},
            {"event": "annual letter / Omaha meeting commentary",
             "typical_impact": {}},  # varies by content
        ],
    },
    "Cathie Wood": {
        "role": "CEO, ARK Invest",
        "sectors": ["Tech", "Fintech", "Consumer"],
        "symbols": ["ARKK", "TSLA", "COIN", "PLTR", "SQ"],
        "keywords": ["cathie wood", "ark invest", "wood said"],
        "historical_patterns": [
            {"event": "large buy in innovation stock",
             "typical_impact": {"ARKK": +0.1, "Fintech": +0.05}},
            {"event": "bearish macro warning",
             "typical_impact": {"Tech": -0.05}},
        ],
    },
    "Michael Burry": {
        "role": "Investor, Scion Asset Management",
        "sectors": ["Finance", "Tech"],
        "symbols": [],
        "keywords": ["michael burry", "burry said", "scion asset"],
        "historical_patterns": [
            {"event": "large short position disclosed",
             "typical_impact": {"Tech": -0.1}},
            {"event": "market crash warning",
             "typical_impact": {"Tech": -0.05, "Finance": -0.05}},
        ],
    },
    "Jamie Dimon": {
        "role": "CEO, JPMorgan Chase",
        "sectors": ["Finance"],
        "symbols": ["JPM", "BAC", "GS"],
        "keywords": ["jamie dimon", "jpmorgan ceo", "dimon said", "dimon's"],
        "historical_patterns": [
            {"event": "economic outlook positive",
             "typical_impact": {"Finance": +0.1, "Consumer": +0.05}},
            {"event": "recession warning / economic storm",
             "typical_impact": {"Finance": -0.15, "Consumer": -0.1}},
        ],
    },

    # ── Crypto ──
    "Michael Saylor": {
        "role": "Executive Chairman, MicroStrategy",
        "sectors": ["Crypto", "Fintech"],
        "symbols": ["BTC/USD"],
        "keywords": ["michael saylor", "saylor said", "microstrategy"],
        "historical_patterns": [
            {"event": "bitcoin purchase announcement",
             "typical_impact": {"BTC/USD": +0.1}},
            {"event": "bullish bitcoin commentary",
             "typical_impact": {"BTC/USD": +0.05}},
        ],
    },

    # ── Political / Trade ──
    "US President": {
        "role": "President of the United States",
        "sectors": ["Industrial", "Energy", "Consumer", "Tech", "Finance"],
        "symbols": [],
        "keywords": ["president said", "white house", "executive order",
                     "presidential", "oval office"],
        "historical_patterns": [
            {"event": "tariff announcement / trade war escalation",
             "typical_impact": {"Industrial": -0.2, "Tech": -0.15, "Consumer": -0.1}},
            {"event": "infrastructure spending / stimulus",
             "typical_impact": {"Industrial": +0.3, "Energy": +0.1}},
            {"event": "tech regulation / antitrust action",
             "typical_impact": {"Tech": -0.2, "Fintech": -0.15}},
        ],
    },

    # ── OPEC / Oil ──
    "OPEC": {
        "role": "Oil cartel",
        "sectors": ["Energy", "Industrial", "Consumer"],
        "symbols": ["XOM", "CVX", "OXY", "SLB", "XLE"],
        "keywords": ["opec", "opec+", "oil cartel", "saudi aramco", "oil production cut",
                     "oil output"],
        "historical_patterns": [
            {"event": "production cut announcement",
             "typical_impact": {"Energy": +0.3, "Industrial": -0.15, "Consumer": -0.1}},
            {"event": "production increase / no cut",
             "typical_impact": {"Energy": -0.25, "Industrial": +0.1, "Consumer": +0.1}},
        ],
    },
}


def match_influencers(headlines: list[str]) -> list[InfluencerMatch]:
    """Scan headlines for mentions of tracked influencers.

    Uses case-insensitive multi-word keyword matching to reduce false positives.

    Args:
        headlines: List of news headline strings.

    Returns:
        List of InfluencerMatch objects, deduplicated by (influencer, headline).
    """
    matches = []
    seen = set()

    for headline in headlines:
        headline_lower = headline.lower()
        for name, info in INFLUENCER_REGISTRY.items():
            for keyword in info["keywords"]:
                if keyword.lower() in headline_lower:
                    key = (name, headline)
                    if key not in seen:
                        seen.add(key)
                        matches.append(InfluencerMatch(
                            name=name,
                            role=info["role"],
                            headline=headline,
                            sectors=info["sectors"],
                            symbols=info.get("symbols", []),
                            patterns=info["historical_patterns"],
                        ))
                    break  # one keyword match per influencer per headline is enough

    return matches


def get_influencer_context_for_sector(sector: str,
                                       headlines: list[str]) -> str:
    """Get formatted influencer context for a sector prompt.

    Args:
        sector: Sector name (e.g., "Tech", "Energy").
        headlines: All available headlines to scan.

    Returns:
        Formatted text block for prompt injection. Empty string if no matches.
    """
    matches = match_influencers(headlines)

    # Filter to matches relevant to this sector
    relevant = [m for m in matches if sector in m.sectors]
    if not relevant:
        return ""

    lines = ["## Influencer Activity"]
    seen_influencers = set()

    for match in relevant[:5]:  # cap at 5 to control token count
        if match.name not in seen_influencers:
            seen_influencers.add(match.name)
            lines.append(f"- {match.name} ({match.role}): \"{match.headline}\"")

            # Add most relevant historical pattern
            for pattern in match.patterns:
                impact = pattern.get("typical_impact", {})
                if sector in impact or any(s in impact for s in [sector]):
                    lines.append(
                        f"  Historical: {pattern['event']} → "
                        + ", ".join(f"{k} {v:+.1f}" for k, v in impact.items())
                    )
                    break

    return "\n".join(lines) if len(lines) > 1 else ""


def get_influencer_context_for_symbol(symbol: str,
                                       headlines: list[str]) -> str:
    """Get formatted influencer context for a symbol prompt.

    Args:
        symbol: Trading symbol (e.g., "TSLA", "NVDA").
        headlines: All available headlines to scan.

    Returns:
        Formatted text block for prompt injection. Empty string if no matches.
    """
    matches = match_influencers(headlines)

    # Filter to matches that directly affect this symbol
    relevant = [m for m in matches if symbol in m.symbols]
    if not relevant:
        return ""

    lines = ["## Influencer Activity"]
    for match in relevant[:3]:  # cap at 3
        lines.append(f"- {match.name} ({match.role}): \"{match.headline}\"")
        for pattern in match.patterns:
            impact = pattern.get("typical_impact", {})
            if symbol in impact:
                lines.append(
                    f"  Historical: {pattern['event']} → {symbol} {impact[symbol]:+.1f}"
                )
                break

    return "\n".join(lines) if len(lines) > 1 else ""


def get_influencer_summary(headlines: list[str]) -> str:
    """Get a brief influencer summary for the macro prompt.

    Returns a 2-3 line summary of all detected influencer activity.
    """
    matches = match_influencers(headlines)
    if not matches:
        return ""

    # Deduplicate by influencer name
    by_name = {}
    for m in matches:
        if m.name not in by_name:
            by_name[m.name] = m

    lines = ["## Key Figure Activity"]
    for name, match in list(by_name.items())[:5]:
        lines.append(f"- {name} ({match.role}): \"{match.headline}\"")

    return "\n".join(lines)
