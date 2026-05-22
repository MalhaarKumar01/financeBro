# FINANCEBRO

A real-time stock valuation dashboard. Real-time data across 5 global exchanges — P/E, PEG, RSI, moving averages, analyst targets, sentiment, earnings calendar, sector heatmap.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.14-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## Quick Start

```bash
git clone <repo>
cd financeBro

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r src/requirements.txt

python src/api.py
# then open src/index.html in your browser
```

Optional — enable Reddit sentiment:
```bash
export REDDIT_CLIENT_ID=your_id
export REDDIT_CLIENT_SECRET=your_secret
python src/api.py
```
Get free credentials at https://www.reddit.com/prefs/apps (script app type).

---

## Exchanges

| Exchange | Securities |
|---|---|
| 🇺🇸 NYSE | 30 US stocks |
| 🇨🇦 TSX | 40 Canadian stocks |
| 🇮🇳 NSE | 5 Indian stocks |
| ₿ CRYPTO | 10 cryptocurrencies |
| 📊 ETF | 10 funds |

---

## Metrics per Stock

**Valuation** — P/E (TTM), Forward P/E, PEG ratio (vs industry avg), 5Y EPS Growth

**Technicals** — RSI(14), MA50, MA200, Golden Cross signal

**52-Week Range** — High/Low with % distance from each

**Risk + Income** — Beta (with risk label), Dividend Yield, Short Ratio

**Analyst** — Consensus rating, Mean price target, Upside %

**Balance Sheet** — Debt/Equity, Return on Equity

**Market Data** — Volume, Market Cap

**Sentiment** — Reddit (upvote-weighted, comment analysis) + Google News headlines → BULLISH / BEARISH / NEUTRAL with confidence level (HIGH / MEDIUM / LOW / INSUFFICIENT)

---

## Dashboard Features

- **Power Search** — `ticker:AAPL`, `pe<20`, `peg<1`, `rsi<30`, `beta<1`, `growth>15`, `industry:Tech`
- **Valuation filter** — ALL / UNDERVALUED / FAIR / OVERVALUED
- **Sector filter** — 13 industries
- **Sort** — Ticker, P/E, PEG, Price, Change, RSI, Analyst Upside
- **HEATMAP view** — Sector grid coloured by avg daily performance, click to filter
- **EARNINGS RADAR** — Upcoming earnings dates, EPS + revenue estimates
- **WATCHLIST tab** — Save tickers with ★ button, persisted in localStorage
- **CSV Export** — Exports current filtered + sorted view with all metrics

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `CTRL+K` | Focus search |
| `CTRL+E` | Export CSV |
| `CTRL+W` | Add first card to watchlist |
| `CTRL+1-5` | Switch exchange |
| `CTRL+R` | Refresh data |
| `ESC` | Clear search |

---

## API Endpoints

```
GET /api/health                 Health check
GET /api/stocks/<exchange>      Stocks for exchange (nyse, tsx, nse, crypto, etf)
GET /api/stocks                 All exchanges
GET /api/sentiment/<ticker>     Reddit sentiment
GET /api/earnings/<exchange>    Upcoming earnings calendar
```

Data is cached for 5 minutes. Frontend auto-refreshes every 5 minutes.

---

## Project Structure

```
financeBro/
├── .venv/                  Python environment (not committed)
├── .gitignore
├── plan.md                 Feature roadmap
├── planOfAction.md         Sentiment engine architecture deep-dive
├── README.md
└── src/
    ├── api.py              Flask API — stock data, earnings, sentiment route
    ├── sentiment/
    │   ├── __init__.py     Public get_sentiment() — background fetch, pending/ready lifecycle
    │   ├── scorer.py       VADER + 40-word financial lexicon, weighted scoring
    │   ├── reddit.py       PRAW — upvote-weighted, comment analysis, disambiguation, retry
    │   ├── google.py       Google News headline scrape (no API key)
    │   └── cache.py        SQLite persistent cache
    ├── index.html          Full dashboard (no build tools)
    ├── styles.css          Brutalist design system
    └── requirements.txt
```

---

## Dependencies

```
yfinance==1.0           Yahoo Finance data
flask==3.1.0            API server
flask-cors==5.0.0       CORS
pandas==2.3.3           RSI + MA calculations
numpy==2.2.6            Numerics
praw==7.7.1             Reddit API
vaderSentiment==3.3.2   Sentiment scoring
beautifulsoup4==4.12.3  HTML parsing
requests==2.31.0        HTTP
```

---

## Troubleshooting

**API offline** — Make sure you're using the `.venv` Python, not system Python. Run from the project root with `source .venv/bin/activate` first.

**Port conflict** — App runs on 5001 to avoid macOS AirPlay on 5000.

**Slow load** — yfinance fetches each ticker individually. First load for 30+ stocks takes ~30–60s. Subsequent loads hit the 5-min cache.

**Sentiment shows ANALYSING...** — Backend is fetching in a background thread. Badge auto-updates every 4s (up to 3 retries). If it stays loading, Reddit credentials may be missing — Google News still works without them.

**Sentiment shows NO SENTIMENT** — Both Reddit (no creds) and Google News scrape returned nothing. Check your internet connection or try a more liquid ticker.

---

## Backlog

See [plan.md](plan.md) for the full feature roadmap and [planOfAction.md](planOfAction.md) for the sentiment engine architecture, data flow diagram, and testing checklist.
