# FinanceBro — Improvement Plan

## Context

Brutalist, terminal-aesthetic stock viewer backed by Flask + yfinance. Goal: strip dead weight, add professional-grade signals (sentiment, technicals, analyst data), and keep the design untouched.

---

## ✅ Phase 0 — Purge Bloat (DONE)

- Deleted: `docker-compose.yml`, `main.py`, `finance_pipeline.py`, `.env`
- Stripped `requirements.txt` to only what's actually used
- Created `.venv` at project root: `python3 -m venv .venv && source .venv/bin/activate && pip install -r src/requirements.txt`
- Added `.gitignore`

---

## ✅ Phase 1 — Fix api.py (DONE)

- All missing values return `null` (not `0`) — frontend renders `—`
- PEG division-by-zero guard
- Cache timeout raised to 300s (was 60s)
- New fields added per stock:
  - `beta`, `dividendYield`, `shortRatio`
  - `analystTarget`, `analystRating`, `analystUpside`
  - `debtToEquity`, `roe`
  - `rsi`, `ma50`, `ma200`, `goldenCross`
  - `volume`, `marketCap` (already fetched, now exposed)
- New route: `GET /api/earnings/<exchange>` — upcoming earnings dates, EPS + revenue estimates

---

## ✅ Phase 2 — Sentiment Engine (DONE)

**File:** `src/sentiment.py`

Reddit via PRAW + VADER scoring. No ML model, no paid API for scoring.

- Requires env vars: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`
- Without them: endpoint returns `available: false` gracefully
- Subreddits: `r/wallstreetbets`, `r/stocks`, `r/investing` (US); `r/IndiaInvestments` (NSE); `r/PersonalFinanceCanada` (TSX)
- 10-min cache per ticker
- New route: `GET /api/sentiment/<ticker>`

To enable Reddit sentiment:
```bash
export REDDIT_CLIENT_ID=your_id
export REDDIT_CLIENT_SECRET=your_secret
python src/api.py
```
Get credentials free at: https://www.reddit.com/prefs/apps (script app type)

---

## ✅ Phase 3 — RSI + Moving Averages (DONE)

Computed from 6-month yfinance history inside `api.py`:
- **RSI(14)** — `<30` oversold (green), `>70` overbought (red)
- **MA50 / MA200** — with price-vs-MA direction arrows
- **Golden Cross** — MA50 > MA200 (bullish signal)

---

## ✅ Phase 4 — Frontend Upgrades (DONE)

**All changes in `src/index.html` + `src/styles.css`:**

### New tabs + views
- **WATCHLIST tab** — save tickers with ★ button or CTRL+W, persisted in localStorage
- **HEATMAP view** — sector grid coloured by avg daily change, click to filter

### New card sections
Each card now has:
- **Tech row**: RSI + label, MA50 ↑/↓, MA200 ↑/↓, GOLDEN CROSS badge
- **Sentiment badge**: BULLISH/BEARISH/NEUTRAL (lazy-loaded after render)
- **RISK + INCOME**: Beta + label, Dividend Yield, Short Ratio
- **ANALYST**: Rating text, Price Target, Upside %
- **BALANCE SHEET**: Debt/Equity, ROE
- **MARKET DATA**: Volume, Market Cap

### New controls
- **EARNINGS RADAR panel** — collapsible table below grid, fetches `/api/earnings/<exchange>`
- **CSV export** — CTRL+E or header button, exports current filtered+sorted view
- **Sort buttons added**: CHG ↑, RSI ↑/↓, UPSIDE ↓
- **Search syntax expanded**: `rsi<30`, `beta<1`

### Null safety
All `.toFixed()` calls replaced with `fmt()` helper — null renders as `—`, never `0.00`

---

## Backlog / Future Ideas

| Idea | Notes |
|---|---|
| **Options flow** | Unusual Whales API or Barchart scrape — unusual put/call activity |
| **Insider trading feed** | SEC EDGAR RSS — free, no auth |
| **Fear & Greed index** | CNN widget scrape — single number, render as terminal gauge |
| **Macro tab** | DXY, 10Y yield, VIX, gold, oil — "MACRO CONTEXT" panel |
| **Screener presets** | One-click: "Value traps", "High short squeeze", "Dividend kings" |
| **Backtester** | Given filter (e.g. `pe<15 peg<1`), show historical 1Y return |
| **Price alerts** | Browser Web Notifications API when stock crosses threshold |
| **WebSocket live prices** | Replace 5-min polling with Alpaca/Finnhub free WS feed |
| **Multi-stock compare** | Side-by-side 2-4 ticker comparison, all metrics aligned |
| **Peer group analysis** | For selected stock, auto-fetch 3 closest competitors |
| **Earnings surprise tracker** | Post-earnings: actual vs estimate, colour-code card |
| **Dark pool / institutional** | Finviz scrape or Quandl — block trade activity |
| **MoneyControl scrape** | NSE-specific headlines for sentiment (was deprioritised) |
| **Twitter/X sentiment** | v2 API free tier — supplement Reddit signal |
| **WhatsApp digest** | Markdown summary of top movers + sentiment for broadcast |

---

## File Map

```
financeBro/
├── .venv/                  ← Python environment (not committed)
├── .gitignore
├── plan.md                 ← this file
└── src/
    ├── api.py              ← Flask API + yfinance + RSI/MA + earnings route
    ├── sentiment.py        ← Reddit PRAW + VADER sentiment engine
    ├── index.html          ← Full dashboard (no external build tools)
    ├── styles.css          ← Brutalist design system
    └── requirements.txt    ← yfinance, flask, flask-cors, pandas, numpy, praw, vaderSentiment, bs4, requests
```

## Running

```bash
cd financeBro/financeBro
source .venv/bin/activate

# Optional: enable Reddit sentiment
export REDDIT_CLIENT_ID=xxx
export REDDIT_CLIENT_SECRET=yyy

python src/api.py
# Open src/index.html in browser
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/health` | Health check |
| `GET /api/stocks/<exchange>` | Stocks for exchange (nyse, tsx, nse, crypto, etf) |
| `GET /api/stocks` | All exchanges |
| `GET /api/sentiment/<ticker>` | Reddit sentiment for ticker |
| `GET /api/earnings/<exchange>` | Upcoming earnings calendar |
