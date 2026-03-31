"""FinBERT-based financial sentiment analyzer.

Uses ProsusAI/finbert to score financial text as positive/negative/neutral.
Lazily loads the model on first use to avoid startup cost when disabled.
"""

from utils.logger import log

# Lazy-loaded pipeline
_pipeline = None
_MODEL_NAME = "ProsusAI/finbert"


def _get_pipeline():
    """Lazy-load the FinBERT pipeline on first use."""
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline
            log.info(f"Loading FinBERT model ({_MODEL_NAME})...")
            _pipeline = pipeline(
                "sentiment-analysis",
                model=_MODEL_NAME,
                tokenizer=_MODEL_NAME,
                top_k=None,
            )
            log.success("FinBERT model loaded")
        except Exception as e:
            log.error(f"Failed to load FinBERT: {e}")
            raise
    return _pipeline


def score_text(text: str) -> float:
    """Score a single text string for financial sentiment.

    Returns float in [-1, 1]: positive = bullish, negative = bearish.
    Maps FinBERT labels: positive → +score, negative → -score, neutral → 0.
    """
    if not text or not text.strip():
        return 0.0

    pipe = _get_pipeline()
    # FinBERT returns [{"label": "positive", "score": 0.95}, ...]
    results = pipe(text[:512])[0]  # truncate to model max, get first result

    label_scores = {r["label"]: r["score"] for r in results}
    pos = label_scores.get("positive", 0)
    neg = label_scores.get("negative", 0)
    # Net sentiment: positive - negative, weighted by confidence
    return round(pos - neg, 4)


def score_headlines(headlines: list[str]) -> dict:
    """Score a list of headlines and return aggregate sentiment.

    Returns dict with:
      - sentiment_score: float [-1, 1] (mean of individual scores)
      - num_articles: int
      - positive_pct: float (fraction of positive headlines)
      - negative_pct: float (fraction of negative headlines)
    """
    if not headlines:
        return {
            "sentiment_score": 0.0,
            "num_articles": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
        }

    scores = []
    for headline in headlines:
        try:
            s = score_text(headline)
            scores.append(s)
        except Exception as e:
            log.warning(f"Failed to score headline: {e}")
            continue

    if not scores:
        return {
            "sentiment_score": 0.0,
            "num_articles": len(headlines),
            "positive_pct": 0.0,
            "negative_pct": 0.0,
        }

    avg_score = sum(scores) / len(scores)
    positive_count = sum(1 for s in scores if s > 0.1)
    negative_count = sum(1 for s in scores if s < -0.1)

    return {
        "sentiment_score": round(avg_score, 4),
        "num_articles": len(scores),
        "positive_pct": round(positive_count / len(scores), 3),
        "negative_pct": round(negative_count / len(scores), 3),
    }
