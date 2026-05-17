"""
Sentiment engine for FinanceBro — Reddit via PRAW + VADER scoring.

Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars to enable.
Without them the endpoint returns available:false gracefully.
"""
import os
import time
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

_sentiment_cache = {}
_CACHE_TTL = 600  # 10 minutes


def _score_texts(texts):
    if not texts:
        return 0.0
    scores = [_analyzer.polarity_scores(t)['compound'] for t in texts if t]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def _label(score):
    if score >= 0.05:
        return 'BULLISH'
    if score <= -0.05:
        return 'BEARISH'
    return 'NEUTRAL'


def _get_reddit():
    client_id = os.environ.get('REDDIT_CLIENT_ID')
    client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
    if not client_id or not client_secret:
        return None
    try:
        import praw
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent='financebro-sentiment/2.0',
        )
    except Exception:
        return None


_EXCHANGE_SUBS = {
    'nse': ['IndiaInvestments', 'IndianStockMarket'],
    'tsx': ['PersonalFinanceCanada', 'CanadianInvestor'],
    'default': ['wallstreetbets', 'stocks', 'investing'],
}


def get_sentiment(ticker, exchange='default'):
    """
    Search Reddit for ticker mentions in the last 24h.
    Returns sentiment dict. Cached for 10 minutes.
    """
    cache_key = f'{ticker}:{exchange}'
    if cache_key in _sentiment_cache:
        result, ts = _sentiment_cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return result

    reddit = _get_reddit()
    if not reddit:
        result = {
            'ticker': ticker,
            'available': False,
            'message': 'Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars to enable sentiment.',
        }
        _sentiment_cache[cache_key] = (result, time.time())
        return result

    subs = _EXCHANGE_SUBS.get(exchange, _EXCHANGE_SUBS['default'])
    clean = re.sub(r'\.(NS|TO|BO)$', '', ticker, flags=re.IGNORECASE)

    texts = []
    mention_count = 0

    try:
        for sub_name in subs:
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.search(clean, time_filter='day', limit=25):
                mention_count += 1
                texts.append(post.title)
                if post.selftext:
                    texts.append(post.selftext[:300])
    except Exception as e:
        print(f"Reddit error for {ticker}: {e}")
        result = {'ticker': ticker, 'available': False, 'message': str(e)}
        _sentiment_cache[cache_key] = (result, time.time())
        return result

    score = _score_texts(texts)
    result = {
        'ticker': ticker,
        'available': True,
        'mentions': mention_count,
        'sentimentScore': score,
        'sentimentLabel': _label(score),
        'subreddits': subs,
    }
    _sentiment_cache[cache_key] = (result, time.time())
    return result
