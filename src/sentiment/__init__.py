"""
FinanceBro Sentiment Package

Public API: get_sentiment(ticker, exchange) -> dict

Sources aggregated:
  - Reddit (PRAW, upvote-weighted, comment analysis, ticker disambiguation)
  - Google News (headline scrape, no API key required)

Results are cached in SQLite for 10 minutes and survive server restarts.
Background-thread fetch returns status:'pending' on first call so Flask
never blocks waiting on Reddit/Google.
"""
from datetime import datetime, timezone
from threading import Thread

from . import cache, reddit, google
from .scorer import label, confidence

_CACHE_TTL = 600  # 10 minutes

_EXCHANGE_MAP = {
    'nse': 'nse',
    'tsx': 'tsx',
}

# Sentinel stored while a fetch is in-flight
_PENDING = {'status': 'pending', 'available': False}


def _aggregate(ticker: str, exchange: str) -> dict:
    """Fetch all sources, merge, return final result."""
    sources = []

    reddit_result = reddit.fetch(ticker, exchange)
    if reddit_result:
        sources.append(reddit_result)

    google_result = google.fetch(ticker)
    if google_result:
        sources.append(google_result)

    if not sources:
        return {
            'ticker':    ticker,
            'available': False,
            'status':    'unavailable',
            'message':   'No Reddit credentials and Google scrape returned no results.',
        }

    all_scores   = [s['sentimentScore'] for s in sources]
    all_mentions = sum(s['mentions'] for s in sources)
    avg_score    = round(sum(all_scores) / len(all_scores), 3)

    top_post = next(
        (s.get('topPost') for s in sources if s.get('source') == 'reddit' and s.get('topPost')),
        None
    )

    return {
        'ticker':         ticker,
        'available':      True,
        'status':         'ready',
        'mentions':       all_mentions,
        'sentimentScore': avg_score,
        'sentimentLabel': label(avg_score),
        'confidence':     confidence(all_mentions),
        'topPost':        top_post,
        'sources':        sources,
        'scoredAt':       datetime.now(timezone.utc).isoformat(),
    }


def _fetch_and_cache(ticker: str, exchange: str, cache_key: str):
    result = _aggregate(ticker, exchange)
    cache.set(cache_key, result, ttl=_CACHE_TTL)


def get_sentiment(ticker: str, exchange: str = 'default') -> dict:
    """
    Return sentiment for ticker. Cached for 10 min.

    On the very first call for a ticker the result is fetched in a background
    thread and status:'pending' is returned immediately so the Flask route
    never blocks. The frontend re-polls after a few seconds.
    """
    exch_key  = _EXCHANGE_MAP.get(exchange, 'default')
    cache_key = f'{ticker}:{exch_key}'

    cached = cache.get(cache_key)
    if cached:
        return cached

    # Write pending sentinel so concurrent calls don't spawn duplicate threads
    cache.set(cache_key, {**_PENDING, 'ticker': ticker}, ttl=30)

    Thread(
        target=_fetch_and_cache,
        args=(ticker, exch_key, cache_key),
        daemon=True,
    ).start()

    return {**_PENDING, 'ticker': ticker}
