"""
Shared scoring logic: VADER + financial domain lexicon.
All sentiment sources import from here.
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Financial slang VADER doesn't know — overrides default scores
FINANCE_LEXICON = {
    # Bullish
    'moon': 3.0, 'mooning': 3.0, 'moonshot': 2.5, 'moons': 2.5,
    'bullish': 2.5, 'bull': 1.5, 'rockets': 2.0, 'rocket': 1.8,
    'diamond': 1.5, 'hodl': 1.5, 'ape': 1.0, 'squeeze': 2.0,
    'calls': 1.0, 'long': 1.0, 'buy': 1.2, 'buying': 1.2,
    'breakout': 2.0, 'ath': 2.0, 'rip': 2.5, 'skyrocket': 2.8,
    'accumulate': 1.5, 'dip': 1.0,  # "buy the dip" context
    # Bearish
    'puts': -1.5, 'shorting': -1.5, 'bearish': -2.5, 'bear': -1.5,
    'crash': -3.0, 'dump': -2.5, 'dumping': -2.5, 'sold': -0.8,
    'fud': -2.0, 'baghold': -2.5, 'bagholder': -2.5,
    'bankrupt': -3.5, 'fraud': -3.0, 'delisted': -3.5,
    'overvalued': -2.0, 'bubble': -2.0, 'correction': -1.5,
    'drill': -2.0, 'drilling': -2.0, 'tank': -2.0, 'tanking': -2.5,
    'capitulate': -2.5, 'capitulation': -2.5,
    # Zero out noise words that VADER misscores in finance context
    'yolo': 0.0, 'apes': 0.0, 'retard': 0.0, 'autist': 0.0,
    'tendies': 0.5, 'gains': 1.5, 'loss': -1.5,
}

_analyzer.lexicon.update(FINANCE_LEXICON)


def score_texts(texts: list[str]) -> float:
    """Plain average of VADER compound scores across a list of strings."""
    if not texts:
        return 0.0
    scores = [_analyzer.polarity_scores(t)['compound'] for t in texts if t and t.strip()]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def score_weighted(items: list[tuple[str, float]]) -> float:
    """Weighted average: items is list of (text, weight). Weight typically log(upvotes+2)."""
    if not items:
        return 0.0
    total_w, weighted_sum = 0.0, 0.0
    for text, weight in items:
        if text and text.strip():
            score = _analyzer.polarity_scores(text)['compound']
            weighted_sum += score * weight
            total_w += weight
    return round(weighted_sum / total_w, 3) if total_w else 0.0


def label(score: float) -> str:
    if score >= 0.05:
        return 'BULLISH'
    if score <= -0.05:
        return 'BEARISH'
    return 'NEUTRAL'


def confidence(mentions: int) -> str:
    if mentions >= 50:
        return 'HIGH'
    if mentions >= 10:
        return 'MEDIUM'
    if mentions >= 3:
        return 'LOW'
    return 'INSUFFICIENT'
