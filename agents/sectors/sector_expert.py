"""Sector Expert Agent — deep research on a single sector's top companies.

Uses TradingView technical analysis + Yahoo Finance fundamentals to
rank companies and produce investment theses.

Usage:
    python -m agents.sectors.sector_expert --sector technology
    python -m agents.sectors.sector_expert --sector energy --top 20
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.sectors.universe import SECTORS


def _safe_float(val, default=0.0):
    """Convert to float, handling None/NaN/Inf."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def _compute_rsi(prices, period=14):
    """Compute RSI from a price series."""
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _ema(prices, period):
    """Compute EMA from a price series."""
    if len(prices) < period:
        return _safe_float(prices[-1]) if prices else 0
    k = 2 / (period + 1)
    ema_val = sum(prices[:period]) / period
    for p in prices[period:]:
        ema_val = p * k + ema_val * (1 - k)
    return ema_val


def _classify_trend(price, ema20, ema50, ema200):
    """Classify trend as BUY/SELL/NEUTRAL based on EMA alignment."""
    buy_signals = 0
    sell_signals = 0
    if price > ema20 > 0:
        buy_signals += 1
    elif ema20 > 0:
        sell_signals += 1
    if price > ema50 > 0:
        buy_signals += 1
    elif ema50 > 0:
        sell_signals += 1
    if price > ema200 > 0:
        buy_signals += 1
    elif ema200 > 0:
        sell_signals += 1

    if buy_signals >= 3:
        return "STRONG_BUY"
    elif buy_signals >= 2:
        return "BUY"
    elif sell_signals >= 3:
        return "STRONG_SELL"
    elif sell_signals >= 2:
        return "SELL"
    return "NEUTRAL"


def fetch_technicals_yf(ticker: str) -> dict:
    """Fallback: compute technicals from Yahoo Finance price history."""
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < 50:
            return {"ticker": ticker, "error": "insufficient_history"}

        closes = hist["Close"].tolist()
        volumes = hist["Volume"].tolist()
        price = closes[-1]

        rsi = _compute_rsi(closes)
        ema20 = _ema(closes, 20)
        ema50 = _ema(closes, 50)
        ema200 = _ema(closes, 200) if len(closes) >= 200 else _ema(closes, len(closes))
        sma200 = sum(closes[-200:]) / min(200, len(closes))

        # MACD
        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        macd = ema12 - ema26

        # Bollinger Bands
        window = closes[-20:]
        bb_mid = sum(window) / 20
        bb_std = (sum((p - bb_mid) ** 2 for p in window) / 20) ** 0.5
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # ADX approximation (simplified)
        if len(closes) > 14:
            true_ranges = []
            for i in range(1, min(15, len(closes))):
                tr = max(
                    abs(closes[-(i)] - closes[-(i + 1)]),
                    abs(closes[-(i)] - closes[-(i + 1)]),
                )
                true_ranges.append(tr)
            atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0
        else:
            atr = 0

        # Trend classification
        daily_rec = _classify_trend(price, ema20, ema50, ema200)

        # Weekly approximation (use last 5 closes for weekly)
        weekly_closes = closes[-5:]
        weekly_rec = "BUY" if weekly_closes[-1] > weekly_closes[0] else "SELL"

        # Monthly approximation
        monthly_closes = closes[-22:] if len(closes) >= 22 else closes
        monthly_rec = "BUY" if monthly_closes[-1] > monthly_closes[0] else "SELL"

        change_pct = ((price - closes[-2]) / closes[-2] * 100) if len(closes) > 1 else 0

        return {
            "ticker": ticker,
            "exchange": "YF",
            "price": round(price, 2),
            "daily_summary": {"RECOMMENDATION": daily_rec, "BUY": 0, "SELL": 0, "NEUTRAL": 0},
            "weekly_summary": {"RECOMMENDATION": weekly_rec},
            "monthly_summary": {"RECOMMENDATION": monthly_rec},
            "rsi": round(rsi, 2),
            "macd": round(macd, 4),
            "macd_signal": 0,
            "ema_20": round(ema20, 2),
            "ema_50": round(ema50, 2),
            "ema_200": round(ema200, 2),
            "sma_200": round(sma200, 2),
            "bb_upper": round(bb_upper, 2),
            "bb_lower": round(bb_lower, 2),
            "atr": round(atr, 4),
            "adx": 20,  # simplified — no true ADX without +DI/-DI
            "stoch_k": 0,
            "stoch_d": 0,
            "cci": 0,
            "volume": _safe_float(volumes[-1]) if volumes else 0,
            "change_pct": round(change_pct, 2),
            "error": None,
            "source": "yahoo_finance",
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def fetch_technicals(ticker: str, exchange: str = "NASDAQ") -> dict:
    """Run TradingView technical analysis, fall back to Yahoo Finance."""
    try:
        from tradingview_ta import TA_Handler, Interval

        exchanges_to_try = [exchange, "NYSE", "NASDAQ", "AMEX", "NYSE ARCA"]
        for ex in exchanges_to_try:
            for attempt in range(2):
                try:
                    handler = TA_Handler(
                        symbol=ticker,
                        screener="america",
                        exchange=ex,
                        interval=Interval.INTERVAL_1_DAY,
                    )
                    analysis = handler.get_analysis()
                    indicators = analysis.indicators

                    time.sleep(1.0)

                    weekly = TA_Handler(
                        symbol=ticker, screener="america",
                        exchange=ex, interval=Interval.INTERVAL_1_WEEK,
                    ).get_analysis()

                    time.sleep(1.0)

                    monthly = TA_Handler(
                        symbol=ticker, screener="america",
                        exchange=ex, interval=Interval.INTERVAL_1_MONTH,
                    ).get_analysis()

                    return {
                        "ticker": ticker,
                        "exchange": ex,
                        "price": _safe_float(indicators.get("close")),
                        "daily_summary": analysis.summary,
                        "weekly_summary": weekly.summary,
                        "monthly_summary": monthly.summary,
                        "rsi": _safe_float(indicators.get("RSI")),
                        "macd": _safe_float(indicators.get("MACD.macd")),
                        "macd_signal": _safe_float(indicators.get("MACD.signal")),
                        "ema_20": _safe_float(indicators.get("EMA20")),
                        "ema_50": _safe_float(indicators.get("EMA50")),
                        "ema_200": _safe_float(indicators.get("EMA200")),
                        "sma_200": _safe_float(indicators.get("SMA200")),
                        "bb_upper": _safe_float(indicators.get("BB.upper")),
                        "bb_lower": _safe_float(indicators.get("BB.lower")),
                        "atr": _safe_float(indicators.get("ATR")),
                        "adx": _safe_float(indicators.get("ADX")),
                        "stoch_k": _safe_float(indicators.get("Stoch.K")),
                        "stoch_d": _safe_float(indicators.get("Stoch.D")),
                        "cci": _safe_float(indicators.get("CCI20")),
                        "volume": _safe_float(indicators.get("volume")),
                        "change_pct": _safe_float(indicators.get("change")),
                        "error": None,
                        "source": "tradingview",
                    }
                except Exception as e:
                    if "429" in str(e):
                        if attempt == 0:
                            time.sleep(5)
                            continue
                        # Rate limited on all retries — fall back to YF
                        return fetch_technicals_yf(ticker)
                    break  # non-429 error: try next exchange

        # All exchanges failed — fall back to Yahoo Finance
        return fetch_technicals_yf(ticker)
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def fetch_fundamentals(ticker: str) -> dict:
    """Fetch fundamental data from Yahoo Finance."""
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        info = stock.info or {}

        market_cap = _safe_float(info.get("marketCap"))
        enterprise_val = _safe_float(info.get("enterpriseValue"))

        return {
            "ticker": ticker,
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": market_cap,
            "market_cap_b": round(market_cap / 1e9, 2) if market_cap else 0,
            "enterprise_value_b": round(enterprise_val / 1e9, 2) if enterprise_val else 0,
            "pe_trailing": _safe_float(info.get("trailingPE")),
            "pe_forward": _safe_float(info.get("forwardPE")),
            "peg_ratio": _safe_float(info.get("pegRatio")),
            "ps_ratio": _safe_float(info.get("priceToSalesTrailing12Months")),
            "pb_ratio": _safe_float(info.get("priceToBook")),
            "ev_ebitda": _safe_float(info.get("enterpriseToEbitda")),
            "profit_margin": _safe_float(info.get("profitMargins")),
            "operating_margin": _safe_float(info.get("operatingMargins")),
            "roe": _safe_float(info.get("returnOnEquity")),
            "roa": _safe_float(info.get("returnOnAssets")),
            "revenue_growth": _safe_float(info.get("revenueGrowth")),
            "earnings_growth": _safe_float(info.get("earningsGrowth")),
            "debt_to_equity": _safe_float(info.get("debtToEquity")),
            "current_ratio": _safe_float(info.get("currentRatio")),
            "free_cashflow": _safe_float(info.get("freeCashflow")),
            "dividend_yield": _safe_float(info.get("dividendYield")),
            "beta": _safe_float(info.get("beta")),
            "52w_high": _safe_float(info.get("fiftyTwoWeekHigh")),
            "52w_low": _safe_float(info.get("fiftyTwoWeekLow")),
            "50d_avg": _safe_float(info.get("fiftyDayAverage")),
            "200d_avg": _safe_float(info.get("twoHundredDayAverage")),
            "analyst_target": _safe_float(info.get("targetMeanPrice")),
            "analyst_recommendation": info.get("recommendationKey", ""),
            "error": None,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def compute_composite_score(tech: dict, fund: dict) -> dict:
    """Score a company 0-100 based on technicals + fundamentals.

    Weights:
        Technical trend alignment:  25%
        Momentum & RSI quality:     15%
        Valuation attractiveness:   20%
        Growth metrics:             20%
        Profitability & quality:    15%
        Risk / balance sheet:        5%
    """
    scores = {}

    # --- Technical trend (25 pts) ---
    trend_score = 0
    daily = tech.get("daily_summary", {})
    weekly = tech.get("weekly_summary", {})
    monthly = tech.get("monthly_summary", {})

    for tf, weight in [(daily, 5), (weekly, 10), (monthly, 10)]:
        rec = tf.get("RECOMMENDATION", "NEUTRAL") if tf else "NEUTRAL"
        if "STRONG_BUY" in rec:
            trend_score += weight
        elif "BUY" in rec:
            trend_score += weight * 0.7
        elif "NEUTRAL" in rec:
            trend_score += weight * 0.4
        elif "SELL" in rec:
            trend_score += weight * 0.1
    scores["trend"] = round(trend_score, 1)

    # --- Momentum & RSI (15 pts) ---
    rsi = tech.get("rsi", 50)
    momentum_score = 0
    if 40 <= rsi <= 65:
        momentum_score += 8  # healthy uptrend RSI
    elif 30 <= rsi < 40:
        momentum_score += 10  # oversold bounce potential
    elif 65 < rsi <= 75:
        momentum_score += 5  # strong but extended
    else:
        momentum_score += 2  # extreme

    adx = tech.get("adx", 0)
    if adx > 25:
        momentum_score += 5  # strong trend
    elif adx > 15:
        momentum_score += 3
    else:
        momentum_score += 1
    scores["momentum"] = min(15, round(momentum_score, 1))

    # --- Valuation (20 pts) ---
    val_score = 0
    pe_fwd = fund.get("pe_forward", 0)
    if 0 < pe_fwd < 15:
        val_score += 7
    elif 15 <= pe_fwd < 25:
        val_score += 5
    elif 25 <= pe_fwd < 40:
        val_score += 3
    elif pe_fwd >= 40:
        val_score += 1

    peg = fund.get("peg_ratio", 0)
    if 0 < peg < 1:
        val_score += 7
    elif 1 <= peg < 2:
        val_score += 5
    elif 2 <= peg < 3:
        val_score += 2

    ev_ebitda = fund.get("ev_ebitda", 0)
    if 0 < ev_ebitda < 12:
        val_score += 6
    elif 12 <= ev_ebitda < 20:
        val_score += 4
    elif 20 <= ev_ebitda < 30:
        val_score += 2
    scores["valuation"] = min(20, round(val_score, 1))

    # --- Growth (20 pts) ---
    growth_score = 0
    rev_growth = fund.get("revenue_growth", 0)
    if rev_growth > 0.30:
        growth_score += 10
    elif rev_growth > 0.15:
        growth_score += 7
    elif rev_growth > 0.05:
        growth_score += 4
    elif rev_growth > 0:
        growth_score += 2

    earn_growth = fund.get("earnings_growth", 0)
    if earn_growth > 0.30:
        growth_score += 10
    elif earn_growth > 0.15:
        growth_score += 7
    elif earn_growth > 0.05:
        growth_score += 4
    elif earn_growth > 0:
        growth_score += 2
    scores["growth"] = min(20, round(growth_score, 1))

    # --- Profitability (15 pts) ---
    prof_score = 0
    margin = fund.get("profit_margin", 0)
    if margin > 0.25:
        prof_score += 5
    elif margin > 0.10:
        prof_score += 3
    elif margin > 0:
        prof_score += 1

    roe = fund.get("roe", 0)
    if roe > 0.20:
        prof_score += 5
    elif roe > 0.10:
        prof_score += 3
    elif roe > 0:
        prof_score += 1

    fcf = fund.get("free_cashflow", 0)
    if fcf > 1e9:
        prof_score += 5
    elif fcf > 1e8:
        prof_score += 3
    elif fcf > 0:
        prof_score += 1
    scores["profitability"] = min(15, round(prof_score, 1))

    # --- Risk (5 pts) ---
    risk_score = 5  # start full, deduct
    dte = fund.get("debt_to_equity", 0)
    if dte > 200:
        risk_score -= 3
    elif dte > 100:
        risk_score -= 1

    beta = fund.get("beta", 1.0)
    if beta > 2.0:
        risk_score -= 2
    elif beta > 1.5:
        risk_score -= 1
    scores["risk"] = max(0, round(risk_score, 1))

    # --- Composite ---
    total = sum(scores.values())
    scores["total"] = round(total, 1)

    # Rating
    if total >= 70:
        scores["rating"] = "STRONG BUY"
    elif total >= 55:
        scores["rating"] = "BUY"
    elif total >= 40:
        scores["rating"] = "HOLD"
    elif total >= 25:
        scores["rating"] = "SELL"
    else:
        scores["rating"] = "STRONG SELL"

    return scores


def generate_thesis(ticker: str, fund: dict, tech: dict, score: dict) -> str:
    """Generate a concise investment thesis string."""
    parts = []

    name = fund.get("name", ticker)
    rating = score.get("rating", "HOLD")
    total = score.get("total", 0)
    parts.append(f"{name} ({ticker}) — {rating} (score: {total}/100)")

    # Valuation
    pe = fund.get("pe_forward", 0)
    peg = fund.get("peg_ratio", 0)
    if pe > 0:
        parts.append(f"Valuation: Fwd P/E {pe:.1f}, PEG {peg:.2f}")

    # Growth
    rev_g = fund.get("revenue_growth", 0)
    earn_g = fund.get("earnings_growth", 0)
    if rev_g:
        parts.append(f"Growth: Rev {rev_g*100:+.1f}%, Earnings {earn_g*100:+.1f}%")

    # Profitability
    margin = fund.get("profit_margin", 0)
    roe = fund.get("roe", 0)
    parts.append(f"Quality: Margin {margin*100:.1f}%, ROE {roe*100:.1f}%")

    # Technical
    daily_rec = tech.get("daily_summary", {}).get("RECOMMENDATION", "?")
    weekly_rec = tech.get("weekly_summary", {}).get("RECOMMENDATION", "?")
    rsi = tech.get("rsi", 0)
    parts.append(f"Technicals: Daily={daily_rec}, Weekly={weekly_rec}, RSI={rsi:.0f}")

    # Upside
    price = tech.get("price", 0)
    target = fund.get("analyst_target", 0)
    if price > 0 and target > 0:
        upside = (target - price) / price * 100
        parts.append(f"Analyst target ${target:.0f} ({upside:+.1f}% upside)")

    return " | ".join(parts)


def run_sector_analysis(sector_key: str, top_n: int = 100, output_dir: str = None):
    """Run full analysis on a sector, outputting ranked results."""
    if sector_key not in SECTORS:
        print(f"Unknown sector: {sector_key}. Available: {list(SECTORS.keys())}")
        return None

    sector = SECTORS[sector_key]
    tickers = sector["tickers"][:top_n]
    total = len(tickers)

    print(f"\n{'='*60}")
    print(f"  SECTOR EXPERT: {sector['name']}")
    print(f"  {sector['description']}")
    print(f"  Analyzing {total} companies...")
    print(f"{'='*60}\n")

    results = []
    errors = []

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:3d}/{total}] {ticker:8s}", end="", flush=True)

        # Fetch data
        tech = fetch_technicals(ticker)
        if tech.get("error"):
            print(f"  ⚠ tech error: {tech['error']}")
            errors.append({"ticker": ticker, "error": f"technical: {tech['error']}"})
            time.sleep(0.3)
            continue

        fund = fetch_fundamentals(ticker)
        if fund.get("error"):
            print(f"  ⚠ fund error: {fund['error']}")
            errors.append({"ticker": ticker, "error": f"fundamental: {fund['error']}"})
            time.sleep(0.3)
            continue

        # Score & thesis
        score = compute_composite_score(tech, fund)
        thesis = generate_thesis(ticker, fund, tech, score)

        result = {
            "ticker": ticker,
            "name": fund.get("name", ticker),
            "industry": fund.get("industry", ""),
            "market_cap_b": fund.get("market_cap_b", 0),
            "price": tech.get("price", 0),
            "score": score,
            "thesis": thesis,
            "technicals": {
                "rsi": tech.get("rsi"),
                "adx": tech.get("adx"),
                "macd": tech.get("macd"),
                "daily_rec": tech.get("daily_summary", {}).get("RECOMMENDATION"),
                "weekly_rec": tech.get("weekly_summary", {}).get("RECOMMENDATION"),
                "monthly_rec": tech.get("monthly_summary", {}).get("RECOMMENDATION"),
                "ema_200": tech.get("ema_200"),
                "change_pct": tech.get("change_pct"),
            },
            "fundamentals": {
                "pe_forward": fund.get("pe_forward"),
                "peg_ratio": fund.get("peg_ratio"),
                "ev_ebitda": fund.get("ev_ebitda"),
                "revenue_growth": fund.get("revenue_growth"),
                "earnings_growth": fund.get("earnings_growth"),
                "profit_margin": fund.get("profit_margin"),
                "roe": fund.get("roe"),
                "debt_to_equity": fund.get("debt_to_equity"),
                "free_cashflow": fund.get("free_cashflow"),
                "dividend_yield": fund.get("dividend_yield"),
                "analyst_target": fund.get("analyst_target"),
                "analyst_rec": fund.get("analyst_recommendation"),
            },
        }
        results.append(result)
        rating = score["rating"]
        total_score = score["total"]
        print(f"  → {total_score:5.1f}/100  {rating}")

        # Rate limit — TradingView throttles at ~30 req/min
        time.sleep(1.5)

    # Sort by composite score descending
    results.sort(key=lambda r: r["score"]["total"], reverse=True)

    # Build report
    report = {
        "sector": sector_key,
        "sector_name": sector["name"],
        "description": sector["description"],
        "timestamp": datetime.now().isoformat(),
        "companies_analyzed": len(results),
        "companies_failed": len(errors),
        "top_picks": [r["ticker"] for r in results[:10]],
        "rating_distribution": {
            "STRONG BUY": len([r for r in results if r["score"]["rating"] == "STRONG BUY"]),
            "BUY": len([r for r in results if r["score"]["rating"] == "BUY"]),
            "HOLD": len([r for r in results if r["score"]["rating"] == "HOLD"]),
            "SELL": len([r for r in results if r["score"]["rating"] == "SELL"]),
            "STRONG SELL": len([r for r in results if r["score"]["rating"] == "STRONG SELL"]),
        },
        "results": results,
        "errors": errors,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {sector['name']} — ANALYSIS COMPLETE")
    print(f"  Analyzed: {len(results)} | Failed: {len(errors)}")
    print(f"\n  Rating Distribution:")
    for rating, count in report["rating_distribution"].items():
        bar = "█" * count
        print(f"    {rating:14s}  {count:3d}  {bar}")
    print(f"\n  TOP 10 PICKS:")
    for i, r in enumerate(results[:10], 1):
        s = r["score"]
        print(f"    {i:2d}. {r['ticker']:8s} {r['name'][:30]:30s}  "
              f"Score: {s['total']:5.1f}  {s['rating']}")
    print(f"{'='*60}\n")

    # Write output
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parents[2] / "data" / "sector_research")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{sector_key}_analysis.json")

    fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(report, f, indent=2, default=str)
        os.replace(tmp_path, output_path)
        print(f"  Report saved: {output_path}")
    except Exception:
        os.unlink(tmp_path)
        raise

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sector Expert Agent")
    parser.add_argument("--sector", required=True, help=f"Sector key: {list(SECTORS.keys())}")
    parser.add_argument("--top", type=int, default=100, help="Analyze top N companies (default: 100)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    run_sector_analysis(args.sector, args.top, args.output_dir)
