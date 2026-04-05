"""Adversarial Judge Agent — harsh critic of sector expert research.

Reads sector analysis reports, identifies weaknesses, biases, and gaps,
then produces a critique with specific improvement demands.

Usage:
    python -m agents.sectors.sector_judge
    python -m agents.sectors.sector_judge --sector technology
"""

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sector_research"


def _pct(val):
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def load_sector_report(sector_key: str) -> dict:
    path = DATA_DIR / f"{sector_key}_analysis.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def critique_methodology(report: dict) -> list:
    """Attack the scoring methodology itself."""
    issues = []
    results = report.get("results", [])
    if not results:
        return ["FATAL: No results to evaluate — analysis produced nothing."]

    # Check for score clustering (lazy differentiation)
    scores = [r["score"]["total"] for r in results]
    if scores:
        avg = sum(scores) / len(scores)
        std = (sum((s - avg) ** 2 for s in scores) / len(scores)) ** 0.5
        if std < 8:
            issues.append(
                f"WEAK DIFFERENTIATION: Score std dev is only {std:.1f}. "
                f"If every company scores between {min(scores):.0f}-{max(scores):.0f}, "
                f"the model isn't discriminating. Widen the scoring bands or add more factors."
            )

    # Check for rating skew
    dist = report.get("rating_distribution", {})
    total = sum(dist.values())
    if total > 0:
        buy_pct = (dist.get("STRONG BUY", 0) + dist.get("BUY", 0)) / total
        if buy_pct > 0.6:
            issues.append(
                f"BULLISH BIAS: {buy_pct*100:.0f}% of companies rated BUY or better. "
                f"A sector expert who likes everything is useless. "
                f"Real conviction means saying NO to most opportunities."
            )
        sell_pct = (dist.get("SELL", 0) + dist.get("STRONG SELL", 0)) / total
        if sell_pct < 0.1:
            issues.append(
                f"INSUFFICIENT BEARISH CALLS: Only {sell_pct*100:.0f}% rated SELL. "
                f"Where are the landmines? Every sector has overvalued garbage — find it."
            )

    # Check failure rate
    failed = report.get("companies_failed", 0)
    analyzed = report.get("companies_analyzed", 0)
    total_attempted = failed + analyzed
    if total_attempted > 0 and failed / total_attempted > 0.2:
        issues.append(
            f"DATA GAPS: {failed}/{total_attempted} companies failed to analyze. "
            f"A 20%+ failure rate means blind spots. Fix data sourcing or drop broken tickers."
        )

    return issues


def critique_top_picks(report: dict) -> list:
    """Aggressively question the top 10 picks."""
    issues = []
    results = report.get("results", [])
    if len(results) < 10:
        issues.append("NOT ENOUGH DATA: Can't evaluate top picks with <10 results.")
        return issues

    top10 = results[:10]

    # Check for mega-cap bias
    mega_caps = [r for r in top10 if r.get("market_cap_b", 0) > 200]
    if len(mega_caps) > 5:
        names = ", ".join(r["ticker"] for r in mega_caps)
        issues.append(
            f"MEGA-CAP BIAS: {len(mega_caps)}/10 top picks are >$200B. "
            f"({names}) — Any idiot can recommend AAPL and MSFT. "
            f"Where are the mid-cap hidden gems with real alpha potential?"
        )

    # Check for valuation blindness
    expensive = [r for r in top10
                 if r.get("fundamentals", {}).get("pe_forward", 0) > 40]
    if len(expensive) > 3:
        names = ", ".join(f"{r['ticker']} (PE={r['fundamentals']['pe_forward']:.0f})"
                         for r in expensive)
        issues.append(
            f"VALUATION BLINDNESS: {len(expensive)} top picks trade at >40x forward PE. "
            f"({names}) — Growth doesn't justify any price. "
            f"Where's the margin of safety? What if growth decelerates?"
        )

    # Check for momentum chasing
    overbought = [r for r in top10
                  if r.get("technicals", {}).get("rsi", 0) > 70]
    if overbought:
        names = ", ".join(f"{r['ticker']} (RSI={r['technicals']['rsi']:.0f})"
                         for r in overbought)
        issues.append(
            f"MOMENTUM CHASING: {names} are overbought (RSI>70). "
            f"You're recommending at the top. What's the entry strategy? "
            f"Where's the pullback plan?"
        )

    # Check for negative growth in top picks
    declining = [r for r in top10
                 if r.get("fundamentals", {}).get("revenue_growth", 0) < 0]
    if declining:
        names = ", ".join(
            f"{r['ticker']} (rev {_pct(r['fundamentals']['revenue_growth'])})"
            for r in declining)
        issues.append(
            f"DECLINING REVENUE IN TOP PICKS: {names}. "
            f"How can a company with shrinking revenue be a 'best investment'? "
            f"Justify this or remove it."
        )

    # Check for high debt in top picks
    leveraged = [r for r in top10
                 if r.get("fundamentals", {}).get("debt_to_equity", 0) > 150]
    if leveraged:
        names = ", ".join(
            f"{r['ticker']} (D/E={r['fundamentals']['debt_to_equity']:.0f})"
            for r in leveraged)
        issues.append(
            f"EXCESSIVE LEVERAGE: {names} carry heavy debt. "
            f"In a rising rate environment, this is a ticking bomb. "
            f"Stress-test these against a 200bp rate hike."
        )

    # Check for lack of conviction spread
    top_score = top10[0]["score"]["total"]
    tenth_score = top10[9]["score"]["total"]
    if top_score - tenth_score < 5:
        issues.append(
            f"NO CLEAR WINNER: Scores range {tenth_score:.0f} to {top_score:.0f} — "
            f"only {top_score - tenth_score:.1f} point spread. "
            f"If you can't decisively rank these, your model lacks precision. "
            f"Add sector-specific factors (e.g., TAM penetration, competitive moat)."
        )

    return issues


def critique_missing_analysis(report: dict) -> list:
    """Identify what the research DIDN'T cover."""
    issues = []
    sector = report.get("sector", "")

    # Generic missing factors
    issues.append(
        "MISSING: Competitive moat analysis — no mention of switching costs, "
        "network effects, brand power, or cost advantages. "
        "Buffett would throw this in the trash."
    )
    issues.append(
        "MISSING: Management quality — no insider ownership data, "
        "capital allocation track record, or CEO tenure analysis. "
        "You're picking stocks without knowing who's driving the bus."
    )
    issues.append(
        "MISSING: Macro sensitivity — no analysis of interest rate impact, "
        "currency exposure, or recession resilience. "
        "A stock that's great in a bull market might be catastrophic in a downturn."
    )

    # Sector-specific critiques
    sector_specific = {
        "technology": [
            "MISSING: R&D efficiency — R&D spend as % of revenue and returns on innovation. "
            "A tech company burning cash on R&D with no product moat is a value trap.",
            "MISSING: Cloud/AI revenue mix — what % comes from recurring SaaS vs one-time licenses? "
            "Recurring revenue commands higher multiples for a reason.",
        ],
        "healthcare": [
            "MISSING: Pipeline analysis — no drug pipeline data, Phase III readouts, "
            "or patent cliff timeline. This is 80% of pharma/biotech valuation.",
            "MISSING: Regulatory risk — FDA approval probabilities, Medicare/Medicaid exposure, "
            "and drug pricing legislation impact.",
        ],
        "financials": [
            "MISSING: Credit quality — no NPA ratio, loan loss reserves, or stress test results. "
            "Banks look cheap until the loan book blows up.",
            "MISSING: NIM trajectory — net interest margin trend is THE driver for bank earnings. "
            "Rising/falling rates change everything.",
        ],
        "energy": [
            "MISSING: Breakeven oil price — at what $/barrel does each company break even? "
            "This is survival analysis, not optional.",
            "MISSING: Reserve replacement ratio — are they replenishing what they pump? "
            "Without this, you're valuing a melting ice cube.",
        ],
        "consumer_discretionary": [
            "MISSING: Same-store sales trends — revenue growth means nothing if it's all new stores. "
            "Organic growth is what matters.",
            "MISSING: Consumer sentiment correlation — how does each company perform "
            "when consumer confidence drops 20%?",
        ],
        "real_estate": [
            "MISSING: FFO/AFFO analysis — using P/E for REITs is amateur hour. "
            "Where's the Funds From Operations and cap rate analysis?",
            "MISSING: Occupancy rates and lease duration — the entire thesis depends on "
            "whether tenants stay and pay.",
        ],
        "utilities": [
            "MISSING: Rate case outcomes — regulated utilities live and die by rate cases. "
            "What's the allowed ROE and pending rate filings?",
            "MISSING: Renewable transition capex — who's spending wisely vs destroying "
            "shareholder value on green vanity projects?",
        ],
        "industrials": [
            "MISSING: Backlog analysis — order book depth and book-to-bill ratio. "
            "Revenue today means nothing if the pipeline is drying up.",
            "MISSING: Defense budget dependency — what % of revenue comes from government contracts?",
        ],
        "materials": [
            "MISSING: Commodity price sensitivity — a 10% move in copper/gold/lithium "
            "swings earnings 30%+. Show the elasticity.",
            "MISSING: ESG/regulatory headwinds — mining and chemicals face mounting "
            "environmental compliance costs.",
        ],
        "consumer_staples": [
            "MISSING: Private label threat — are consumers trading down to store brands? "
            "Pricing power is everything here.",
            "MISSING: Input cost trajectory — commodity inflation hits margins hard. "
            "Who can pass through costs?",
        ],
        "communication_services": [
            "MISSING: ARPU trends — average revenue per user is the lifeblood metric. "
            "User growth means nothing if monetization is falling.",
            "MISSING: Content spend ROI — who's getting subscribers per dollar spent "
            "vs burning cash on unwatched content?",
        ],
    }

    for critique in sector_specific.get(sector, []):
        issues.append(critique)

    return issues


def critique_individual_picks(report: dict) -> list:
    """Deep-dive criticism of specific companies."""
    issues = []
    results = report.get("results", [])

    for r in results[:15]:  # Critique top 15
        ticker = r["ticker"]
        fund = r.get("fundamentals", {})
        tech = r.get("technicals", {})
        score = r.get("score", {})

        problems = []

        # Negative margins in a "buy"
        margin = fund.get("profit_margin", 0)
        if margin and margin < 0 and score.get("rating") in ("BUY", "STRONG BUY"):
            problems.append(f"UNPROFITABLE yet rated {score['rating']} — this needs a path-to-profitability argument")

        # No analyst coverage
        if not fund.get("analyst_rec"):
            problems.append("No analyst consensus — under-covered stock needs extra due diligence")

        # Massive gap between price and analyst target
        target = fund.get("analyst_target", 0)
        price = r.get("price", 0)
        if price > 0 and target > 0:
            upside = (target - price) / price
            if upside < -0.15:
                problems.append(f"Analysts see {upside*100:.0f}% DOWNSIDE — why do you disagree?")

        # Technical/fundamental contradiction
        daily = tech.get("daily_rec", "")
        if daily and "SELL" in daily and score.get("rating") in ("BUY", "STRONG BUY"):
            problems.append(f"Daily technicals say SELL but you rate it BUY — resolve this contradiction")

        if problems:
            issues.append(f"{ticker}: " + "; ".join(problems))

    return issues


def run_judgment(sector_key: str = None):
    """Judge one or all sector reports."""
    if sector_key:
        sectors_to_judge = [sector_key]
    else:
        # Find all available reports
        if not DATA_DIR.exists():
            print("No sector research data found. Run sector experts first.")
            return
        sectors_to_judge = [
            f.stem.replace("_analysis", "")
            for f in DATA_DIR.glob("*_analysis.json")
        ]

    if not sectors_to_judge:
        print("No sector reports found to judge.")
        return

    all_judgments = {}

    for sector in sectors_to_judge:
        report = load_sector_report(sector)
        if not report:
            print(f"No report found for {sector}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  ADVERSARIAL JUDGE — {report.get('sector_name', sector).upper()}")
        print(f"  \"I'm not here to validate your feelings.\"")
        print(f"{'='*60}\n")

        # Run all critique modules
        methodology_issues = critique_methodology(report)
        top_pick_issues = critique_top_picks(report)
        missing_issues = critique_missing_analysis(report)
        individual_issues = critique_individual_picks(report)

        all_issues = (methodology_issues + top_pick_issues +
                      missing_issues + individual_issues)

        # Severity classification
        critical = [i for i in all_issues if any(
            w in i for w in ["FATAL", "BIAS", "BLINDNESS", "UNPROFITABLE", "DECLINING"]
        )]
        major = [i for i in all_issues if any(
            w in i for w in ["MISSING", "EXCESSIVE", "WEAK", "NO CLEAR"]
        )]
        minor = [i for i in all_issues if i not in critical and i not in major]

        # Score the research quality
        quality_score = 100
        quality_score -= len(critical) * 15
        quality_score -= len(major) * 8
        quality_score -= len(minor) * 3
        quality_score = max(0, quality_score)

        if quality_score >= 80:
            verdict = "ACCEPTABLE — but barely. Address the issues."
        elif quality_score >= 60:
            verdict = "MEDIOCRE — significant gaps need filling before this is actionable."
        elif quality_score >= 40:
            verdict = "POOR — this would lose money. Major rework required."
        else:
            verdict = "UNACCEPTABLE — start over with better methodology."

        judgment = {
            "sector": sector,
            "sector_name": report.get("sector_name", sector),
            "timestamp": datetime.now().isoformat(),
            "quality_score": quality_score,
            "verdict": verdict,
            "issues_count": {
                "critical": len(critical),
                "major": len(major),
                "minor": len(minor),
                "total": len(all_issues),
            },
            "critical_issues": critical,
            "major_issues": major,
            "minor_issues": minor,
            "improvement_demands": [
                f"[CRITICAL] {issue}" for issue in critical
            ] + [
                f"[MAJOR] {issue}" for issue in major[:5]
            ],
        }

        all_judgments[sector] = judgment

        # Print judgment
        print(f"  RESEARCH QUALITY SCORE: {quality_score}/100")
        print(f"  VERDICT: {verdict}\n")

        if critical:
            print(f"  🔴 CRITICAL ISSUES ({len(critical)}):")
            for issue in critical:
                print(f"     • {issue}")
            print()

        if major:
            print(f"  🟡 MAJOR ISSUES ({len(major)}):")
            for issue in major[:8]:
                print(f"     • {issue}")
            if len(major) > 8:
                print(f"     ... and {len(major) - 8} more")
            print()

        if minor:
            print(f"  🟢 MINOR ISSUES ({len(minor)}):")
            for issue in minor[:5]:
                print(f"     • {issue}")
            if len(minor) > 5:
                print(f"     ... and {len(minor) - 5} more")
            print()

        print(f"  DEMANDS ({len(judgment['improvement_demands'])}):")
        for i, demand in enumerate(judgment["improvement_demands"][:10], 1):
            print(f"    {i}. {demand}")

    # Save judgment report
    output_path = DATA_DIR / "judge_critique.json"
    fd, tmp_path = tempfile.mkstemp(dir=str(DATA_DIR), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(all_judgments, f, indent=2, default=str)
        os.replace(tmp_path, str(output_path))
        print(f"\n  Full critique saved: {output_path}")
    except Exception:
        os.unlink(tmp_path)
        raise

    return all_judgments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Judge Agent")
    parser.add_argument("--sector", default=None,
                        help="Judge a specific sector (default: all available)")
    args = parser.parse_args()

    run_judgment(args.sector)
