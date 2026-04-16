"""
Sector Expert Agent - Base class for domain-specific investment research.

Each sector expert:
1. Identifies top companies in its sector by market cap
2. Analyzes fundamentals, competitive positioning, growth catalysts
3. Scores and ranks companies on investment merit
4. Produces a justified recommendation with bull/bear cases
"""

import json
import os
import asyncio
import aiohttp
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime
from enum import Enum


class InvestmentRating(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class CompanyAnalysis:
    ticker: str
    name: str
    sector: str
    market_cap_b: float  # billions
    pe_ratio: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    profit_margin_pct: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe_pct: Optional[float] = None
    dividend_yield_pct: Optional[float] = None
    moat_score: int = 0  # 1-10
    management_score: int = 0  # 1-10
    growth_score: int = 0  # 1-10
    valuation_score: int = 0  # 1-10
    overall_score: float = 0.0
    rating: str = "HOLD"
    bull_case: str = ""
    bear_case: str = ""
    catalyst: str = ""
    risk: str = ""
    thesis: str = ""


@dataclass
class SectorReport:
    sector: str
    agent_name: str
    timestamp: str
    total_companies_analyzed: int
    top_picks: list[CompanyAnalysis] = field(default_factory=list)
    sector_outlook: str = ""
    key_themes: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    iteration: int = 1
    judge_feedback: str = ""
    confidence_score: float = 0.0

    def to_dict(self):
        d = asdict(self)
        return d


# Sector definitions with subsectors and key metrics to focus on
SECTORS = {
    "technology": {
        "name": "Information Technology",
        "subsectors": ["Software", "Semiconductors", "IT Services", "Hardware", "Cloud/SaaS"],
        "key_metrics": ["revenue_growth", "r_and_d_spend", "tam_expansion", "recurring_revenue", "margin_expansion"],
        "focus_areas": [
            "AI/ML adoption and monetization",
            "Cloud infrastructure spending trends",
            "Semiconductor cycle positioning",
            "Enterprise software stickiness",
            "Cybersecurity spending acceleration",
        ],
    },
    "healthcare": {
        "name": "Healthcare",
        "subsectors": ["Pharma", "Biotech", "Medical Devices", "Managed Care", "Life Sciences Tools"],
        "key_metrics": ["pipeline_value", "patent_cliff_exposure", "r_and_d_productivity", "pricing_power", "demographic_tailwinds"],
        "focus_areas": [
            "GLP-1 and obesity drug market expansion",
            "Gene therapy commercialization",
            "Medicare/Medicaid policy changes",
            "Biosimilar competition dynamics",
            "MedTech innovation cycles",
        ],
    },
    "financials": {
        "name": "Financials",
        "subsectors": ["Banks", "Insurance", "Capital Markets", "Fintech", "REITs (Financial)"],
        "key_metrics": ["net_interest_margin", "credit_quality", "roa", "efficiency_ratio", "book_value_growth"],
        "focus_areas": [
            "Interest rate trajectory impact",
            "Credit cycle positioning",
            "Digital banking disruption",
            "Insurance underwriting profitability",
            "Wealth management growth",
        ],
    },
    "consumer_discretionary": {
        "name": "Consumer Discretionary",
        "subsectors": ["E-commerce", "Automotive", "Luxury", "Restaurants", "Homebuilders"],
        "key_metrics": ["same_store_sales", "consumer_confidence", "brand_strength", "digital_penetration", "margin_resilience"],
        "focus_areas": [
            "Consumer spending resilience",
            "EV transition economics",
            "Luxury demand in emerging markets",
            "Restaurant traffic recovery",
            "Housing affordability dynamics",
        ],
    },
    "consumer_staples": {
        "name": "Consumer Staples",
        "subsectors": ["Food & Beverage", "Household Products", "Personal Care", "Tobacco", "Retail (Staples)"],
        "key_metrics": ["organic_growth", "pricing_power", "volume_trends", "private_label_competition", "dividend_sustainability"],
        "focus_areas": [
            "Pricing elasticity in inflationary environment",
            "Health and wellness product trends",
            "Emerging market penetration",
            "Supply chain optimization",
            "ESG and sustainability positioning",
        ],
    },
    "energy": {
        "name": "Energy",
        "subsectors": ["Integrated Oil", "E&P", "Midstream", "Refining", "Renewables"],
        "key_metrics": ["free_cash_flow_yield", "reserve_replacement", "breakeven_price", "capital_discipline", "transition_readiness"],
        "focus_areas": [
            "OPEC+ production dynamics",
            "Energy transition investment",
            "LNG demand growth",
            "Capital return programs",
            "Carbon capture economics",
        ],
    },
    "industrials": {
        "name": "Industrials",
        "subsectors": ["Aerospace & Defense", "Machinery", "Transportation", "Construction", "Electrical Equipment"],
        "key_metrics": ["backlog_growth", "book_to_bill", "margin_expansion", "aftermarket_revenue", "capex_cycle_positioning"],
        "focus_areas": [
            "Defense spending trajectory",
            "Infrastructure bill beneficiaries",
            "Reshoring and supply chain localization",
            "Automation and robotics adoption",
            "Commercial aerospace recovery",
        ],
    },
    "materials": {
        "name": "Materials",
        "subsectors": ["Chemicals", "Metals & Mining", "Construction Materials", "Packaging", "Paper & Forest"],
        "key_metrics": ["commodity_price_leverage", "cost_curve_position", "inventory_cycle", "pricing_power", "sustainability_premium"],
        "focus_areas": [
            "Critical mineral supply constraints",
            "Battery material demand growth",
            "Construction spending trends",
            "Specialty chemical pricing",
            "Circular economy opportunities",
        ],
    },
    "utilities": {
        "name": "Utilities",
        "subsectors": ["Electric Utilities", "Gas Utilities", "Water", "Renewable IPPs", "Multi-Utilities"],
        "key_metrics": ["rate_base_growth", "regulatory_environment", "renewable_mix", "dividend_growth", "capex_recovery"],
        "focus_areas": [
            "Data center power demand surge",
            "Grid modernization spending",
            "Renewable energy cost curves",
            "Regulatory rate case outcomes",
            "Nuclear renaissance potential",
        ],
    },
    "real_estate": {
        "name": "Real Estate",
        "subsectors": ["Data Centers", "Industrial REITs", "Residential", "Retail REITs", "Healthcare REITs"],
        "key_metrics": ["ffo_growth", "occupancy_rates", "same_property_noi", "cap_rates", "development_pipeline"],
        "focus_areas": [
            "Data center demand explosion",
            "Industrial/logistics secular growth",
            "Office market restructuring",
            "Interest rate sensitivity",
            "Housing supply shortage",
        ],
    },
    "communication_services": {
        "name": "Communication Services",
        "subsectors": ["Social Media", "Streaming", "Telecom", "Gaming", "Advertising"],
        "key_metrics": ["arpu_growth", "subscriber_trends", "engagement_metrics", "content_roi", "ad_market_share"],
        "focus_areas": [
            "AI-driven content and advertising",
            "Streaming profitability inflection",
            "5G/fiber monetization",
            "Gaming market consolidation",
            "Regulatory and antitrust risk",
        ],
    },
}


def get_sector_config(sector_key: str) -> dict:
    return SECTORS.get(sector_key, {})


def get_all_sectors() -> list[str]:
    return list(SECTORS.keys())
