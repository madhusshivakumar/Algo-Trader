"""Alpaca News API client — fetches recent headlines for sentiment analysis."""

from datetime import datetime, timedelta
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

from config import Config
from utils.logger import log


# Default lookback window for news
DEFAULT_LOOKBACK_HOURS = 24
MAX_ARTICLES = 50


def fetch_news(symbol: str, lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
               limit: int = MAX_ARTICLES) -> list[dict]:
    """Fetch recent news articles for a symbol from Alpaca News API.

    Returns list of dicts with keys: headline, summary, source, created_at.
    Returns empty list on any error (graceful degradation).
    """
    try:
        client = NewsClient(api_key=Config.ALPACA_API_KEY, secret_key=Config.ALPACA_SECRET_KEY)
        start = datetime.now() - timedelta(hours=lookback_hours)
        request = NewsRequest(
            symbols=symbol.replace("/", ""),  # BTC/USD -> BTCUSD
            start=start,
            limit=limit,
        )
        response = client.get_news(request)

        articles = []
        for article in response.data.get("news", []):
            articles.append({
                "headline": getattr(article, "headline", ""),
                "summary": getattr(article, "summary", ""),
                "source": getattr(article, "source", ""),
                "created_at": str(getattr(article, "created_at", "")),
            })

        log.info(f"Fetched {len(articles)} articles for {symbol}")
        return articles

    except Exception as e:
        log.warning(f"News fetch failed for {symbol}: {e}")
        return []


def fetch_headlines(symbol: str, lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
                    limit: int = MAX_ARTICLES) -> list[str]:
    """Fetch just the headline strings for a symbol.

    Convenience wrapper for sentiment scoring — returns list of headline strings.
    """
    articles = fetch_news(symbol, lookback_hours, limit)
    return [a["headline"] for a in articles if a.get("headline")]
