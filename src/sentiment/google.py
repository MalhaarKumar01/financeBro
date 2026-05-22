"""
Google News sentiment source — scrapes headline text, no API key required.

Google News = institutional/media coverage signal.
Reddit = retail momentum signal.
Together they give a more complete picture.
"""
import re
import requests
from bs4 import BeautifulSoup

from .scorer import score_texts, label as sentiment_label

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
}

# Google News HTML changes over time — try multiple selectors
_HEADLINE_SELECTORS = [
    'div.BNeawe.vvjwJb',   # classic mobile SERP
    'div.BNeawe',           # broader fallback
    'h3.r',                 # desktop SERP (older)
    'a.WlydOe',             # news card title link
    '.ftSUBd span',         # another news card variant
]


def fetch(ticker: str) -> dict | None:
    """
    Scrape Google News search for ticker headlines and score with VADER.
    Returns source dict or None on failure / no results.
    """
    clean = re.sub(r'\.(NS|TO|BO)$', '', ticker, flags=re.IGNORECASE)
    query = f'{clean} stock'
    url   = f'https://www.google.com/search?q={requests.utils.quote(query)}&tbm=nws&num=10&hl=en'

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')
        headlines: list[str] = []

        for selector in _HEADLINE_SELECTORS:
            for el in soup.select(selector):
                text = el.get_text(strip=True)
                if text and len(text) > 15 and text not in headlines:
                    headlines.append(text)
            if len(headlines) >= 8:
                break

        if not headlines:
            return None

        score = score_texts(headlines[:10])
        return {
            'source':         'google_news',
            'mentions':       len(headlines),
            'sentimentScore': score,
            'sentimentLabel': sentiment_label(score),
            'headlines':      headlines[:5],
        }

    except Exception as e:
        print(f"[google] error for {ticker}: {e}")
        return None
