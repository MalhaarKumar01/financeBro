"""
Reddit sentiment source — PRAW + upvote-weighted VADER scoring.

Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars.
Returns None if credentials are absent so the caller can skip gracefully.
"""
import os
import re
import math
import time

from .scorer import score_weighted, score_texts, label as sentiment_label

_EXCHANGE_SUBS = {
    'nse': ['IndiaInvestments', 'IndianStockMarket'],
    'tsx': ['PersonalFinanceCanada', 'CanadianInvestor'],
    'default': ['wallstreetbets', 'stocks', 'investing'],
}

_SHORT_TICKER_LEN = 2  # tickers ≤ this length require $TICKER format to avoid noise


def _get_reddit():
    cid = os.environ.get('REDDIT_CLIENT_ID')
    cs  = os.environ.get('REDDIT_CLIENT_SECRET')
    if not cid or not cs:
        return None
    try:
        import praw
        return praw.Reddit(
            client_id=cid,
            client_secret=cs,
            user_agent='financebro-sentiment/2.0',
        )
    except Exception:
        return None


def _ticker_mentioned(text: str, clean: str) -> bool:
    """True only when ticker appears as standalone word or $TICKER — avoids noise matches."""
    if len(clean) <= _SHORT_TICKER_LEN:
        # Short tickers: require $ prefix (e.g. $F, $GM)
        return bool(re.search(rf'\${re.escape(clean)}\b', text, re.IGNORECASE))
    pattern = rf'(?<![a-zA-Z])(\${re.escape(clean)}|{re.escape(clean)})(?![a-zA-Z])'
    return bool(re.search(pattern, text, re.IGNORECASE))


def _with_retry(fn, retries: int = 3):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if '429' in msg or 'rate limit' in msg or 'too many' in msg:
                wait = 2 ** attempt
                print(f"[reddit] rate limit hit, retry in {wait}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Reddit API rate limit exceeded after retries")


def fetch(ticker: str, exchange: str = 'default') -> dict | None:
    """
    Search Reddit for ticker. Returns source dict or None if unavailable.
    Scoring is upvote-weighted; top 5 comments per post also analysed.
    """
    reddit = _get_reddit()
    if not reddit:
        return None

    subs  = _EXCHANGE_SUBS.get(exchange, _EXCHANGE_SUBS['default'])
    clean = re.sub(r'\.(NS|TO|BO)$', '', ticker, flags=re.IGNORECASE)

    weighted_items: list[tuple[str, float]] = []
    mention_count = 0
    top_post_title = None
    top_post_score = -1

    try:
        for sub_name in subs:
            subreddit = reddit.subreddit(sub_name)

            def _search():
                return list(subreddit.search(clean, time_filter='day', limit=30))

            posts = _with_retry(_search)

            for post in posts:
                body = post.title + ' ' + (post.selftext or '')
                if not _ticker_mentioned(body, clean):
                    continue

                mention_count += 1
                post_weight = math.log(max(post.score, 0) + 2)

                weighted_items.append((post.title, post_weight))
                if post.selftext:
                    weighted_items.append((post.selftext[:400], post_weight * 0.8))

                # Track highest-upvoted post for topPost field
                if post.score > top_post_score:
                    top_post_score  = post.score
                    top_post_title  = post.title

                # Top comments (half weight of parent post)
                try:
                    post.comments.replace_more(limit=0)
                    top_comments = sorted(
                        post.comments.list()[:20],
                        key=lambda c: getattr(c, 'score', 0),
                        reverse=True
                    )[:5]
                    for comment in top_comments:
                        body = getattr(comment, 'body', '')
                        if body and len(body) > 20:
                            comment_weight = math.log(max(getattr(comment, 'score', 0), 0) + 2) * 0.5
                            weighted_items.append((body[:300], comment_weight))
                except Exception:
                    pass  # comments optional — never block on this

    except Exception as e:
        print(f"[reddit] error for {ticker}: {e}")
        return None

    if not weighted_items:
        return None

    score = score_weighted(weighted_items)
    return {
        'source':      'reddit',
        'mentions':    mention_count,
        'sentimentScore':  score,
        'sentimentLabel':  sentiment_label(score),
        'subreddits':  subs,
        'topPost':     top_post_title,
    }
