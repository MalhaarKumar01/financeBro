"""
Flask API for FinanceBro - Serves stock data from yfinance
Run with: python api.py  (from within .venv)
"""
import yfinance as yf
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
import time
from sentiment import get_sentiment

app = Flask(__name__)
CORS(app)

WATCHLISTS = {
    'nyse': [
        ('AAPL', 'Apple Inc.', 'Technology'),
        ('MSFT', 'Microsoft Corp.', 'Technology'),
        ('GOOGL', 'Alphabet Inc.', 'Technology'),
        ('AMZN', 'Amazon.com Inc.', 'Consumer Cyclical'),
        ('META', 'Meta Platforms', 'Technology'),
        ('NVDA', 'NVIDIA Corp.', 'Technology'),
        ('TSLA', 'Tesla Inc.', 'Consumer Cyclical'),
        ('JPM', 'JPMorgan Chase', 'Financials'),
        ('V', 'Visa Inc.', 'Financials'),
        ('JNJ', 'Johnson & Johnson', 'Healthcare'),
        ('UNH', 'UnitedHealth Group', 'Healthcare'),
        ('XOM', 'Exxon Mobil', 'Energy'),
        ('PG', 'Procter & Gamble', 'Consumer Defensive'),
        ('MA', 'Mastercard Inc.', 'Financials'),
        ('HD', 'Home Depot', 'Consumer Cyclical'),
        ('CVX', 'Chevron Corp.', 'Energy'),
        ('MRK', 'Merck & Co.', 'Healthcare'),
        ('KO', 'Coca-Cola Co.', 'Consumer Defensive'),
        ('PEP', 'PepsiCo Inc.', 'Consumer Defensive'),
        ('BAC', 'Bank of America', 'Financials'),
        ('ABBV', 'AbbVie Inc.', 'Healthcare'),
        ('COST', 'Costco Wholesale', 'Consumer Defensive'),
        ('WMT', 'Walmart Inc.', 'Consumer Defensive'),
        ('MCD', "McDonald's Corp.", 'Consumer Cyclical'),
        ('DIS', 'Walt Disney Co.', 'Consumer Cyclical'),
        ('AMD', 'AMD Inc.', 'Technology'),
        ('INTC', 'Intel Corp.', 'Technology'),
        ('QCOM', 'Qualcomm Inc.', 'Technology'),
        ('CAT', 'Caterpillar Inc.', 'Industrials'),
        ('BA', 'Boeing Co.', 'Industrials'),
    ],
    'crypto': [
        ('BTC-USD', 'Bitcoin', 'Crypto'),
        ('ETH-USD', 'Ethereum', 'Crypto'),
        ('SOL-USD', 'Solana', 'Crypto'),
        ('XRP-USD', 'XRP', 'Crypto'),
        ('ADA-USD', 'Cardano', 'Crypto'),
        ('DOGE-USD', 'Dogecoin', 'Crypto'),
        ('DOT-USD', 'Polkadot', 'Crypto'),
        ('AVAX-USD', 'Avalanche', 'Crypto'),
        ('LINK-USD', 'Chainlink', 'Crypto'),
        ('MATIC-USD', 'Polygon', 'Crypto'),
    ],
    'etf': [
        ('SPY', 'S&P 500 ETF', 'ETF'),
        ('QQQ', 'Nasdaq 100 ETF', 'ETF'),
        ('IWM', 'Russell 2000 ETF', 'ETF'),
        ('DIA', 'Dow Jones ETF', 'ETF'),
        ('VTI', 'Total Stock Market ETF', 'ETF'),
        ('VOO', 'Vanguard S&P 500', 'ETF'),
        ('ARKK', 'ARK Innovation ETF', 'ETF'),
        ('XLF', 'Financial Select ETF', 'ETF'),
        ('XLE', 'Energy Select ETF', 'ETF'),
        ('XLK', 'Technology Select ETF', 'ETF'),
    ],
    'tsx': [
        ('SHOP.TO', 'Shopify Inc.', 'Technology'),
        ('RY.TO', 'Royal Bank of Canada', 'Financials'),
        ('TD.TO', 'Toronto-Dominion Bank', 'Financials'),
        ('ENB.TO', 'Enbridge Inc.', 'Energy'),
        ('CNR.TO', 'Canadian National Railway', 'Industrials'),
        ('BMO.TO', 'Bank of Montreal', 'Financials'),
        ('BNS.TO', 'Bank of Nova Scotia', 'Financials'),
        ('CNQ.TO', 'Canadian Natural Resources', 'Energy'),
        ('CP.TO', 'Canadian Pacific Railway', 'Industrials'),
        ('TRI.TO', 'Thomson Reuters Corp.', 'Technology'),
        ('SU.TO', 'Suncor Energy Inc.', 'Energy'),
        ('CM.TO', 'CIBC', 'Financials'),
        ('BCE.TO', 'BCE Inc.', 'Telecom'),
        ('ABX.TO', 'Barrick Gold Corp.', 'Materials'),
        ('MFC.TO', 'Manulife Financial', 'Financials'),
        ('WCN.TO', 'Waste Connections Inc.', 'Industrials'),
        ('TRP.TO', 'TC Energy Corp.', 'Energy'),
        ('NTR.TO', 'Nutrien Ltd.', 'Materials'),
        ('BAM.TO', 'Brookfield Asset Mgmt', 'Financials'),
        ('FNV.TO', 'Franco-Nevada Corp.', 'Materials'),
        ('T.TO', 'TELUS Corp.', 'Telecom'),
        ('ATD.TO', 'Alimentation Couche-Tard', 'Consumer Defensive'),
        ('QSR.TO', 'Restaurant Brands Intl', 'Consumer Cyclical'),
        ('CCL-B.TO', 'CCL Industries Inc.', 'Materials'),
        ('WPM.TO', 'Wheaton Precious Metals', 'Materials'),
        ('GWO.TO', 'Great-West Lifeco Inc.', 'Financials'),
        ('SLF.TO', 'Sun Life Financial', 'Financials'),
        ('POW.TO', 'Power Corp of Canada', 'Financials'),
        ('FSV.TO', 'FirstService Corp.', 'Industrials'),
        ('IFC.TO', 'Intact Financial Corp.', 'Financials'),
        ('DOL.TO', 'Dollarama Inc.', 'Consumer Defensive'),
        ('CSU.TO', 'Constellation Software', 'Technology'),
        ('WN.TO', 'George Weston Ltd.', 'Consumer Defensive'),
        ('L.TO', 'Loblaw Companies Ltd.', 'Consumer Defensive'),
        ('MG.TO', 'Magna International', 'Consumer Cyclical'),
        ('AEM.TO', 'Agnico Eagle Mines', 'Materials'),
        ('PPL.TO', 'Pembina Pipeline Corp.', 'Energy'),
        ('IMG.TO', 'IAMGOLD Corp.', 'Materials'),
        ('GIL.TO', 'Gildan Activewear Inc.', 'Consumer Cyclical'),
        ('SAP.TO', 'Saputo Inc.', 'Consumer Defensive'),
    ],
    'nse': [
        ('RELIANCE.NS', 'Reliance Industries', 'Energy'),
        ('LT.NS', 'Larsen & Toubro', 'Industrials'),
        ('ZENSARTECH.NS', 'Zensar Technologies', 'Technology'),
        ('BDL.NS', 'Bharat Dynamics Ltd.', 'Defence'),
        ('SUZLON.NS', 'Suzlon Energy Ltd.', 'Energy'),
    ],
}

INDUSTRY_PE = {
    'Technology': 28.5,
    'Financials': 12.5,
    'Healthcare': 22.0,
    'Energy': 10.5,
    'Consumer Defensive': 24.0,
    'Consumer Cyclical': 22.0,
    'Industrials': 18.0,
    'Materials': 15.0,
    'Utilities': 18.0,
    'Real Estate': 35.0,
    'Telecom': 15.0,
    'Defence': 25.0,
    'Crypto': None,
    'ETF': None,
}

_cache = {}
_cache_timeout = 300  # 5 minutes


def _safe(value, multiplier=1, decimals=2):
    """Return rounded float or None — never 0 for missing data."""
    if value is None or value == 0:
        return None
    try:
        result = float(value) * multiplier
        if not np.isfinite(result):
            return None
        return round(result, decimals)
    except (TypeError, ValueError):
        return None


def get_cached_data(key):
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < _cache_timeout:
            return data
    return None


def set_cached_data(key, data):
    _cache[key] = (data, time.time())


def compute_technicals(ticker_obj):
    """Compute RSI(14), MA50, MA200 from 6-month history. Returns dict or empty dict on failure."""
    try:
        hist = ticker_obj.history(period='6mo')
        if hist.empty or len(hist) < 15:
            return {}

        close = hist['Close']

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi = round(float(rsi_series.iloc[-1]), 1) if not np.isnan(rsi_series.iloc[-1]) else None

        ma50 = round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None
        ma200 = round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None

        golden_cross = bool(ma50 and ma200 and ma50 > ma200)

        return {'rsi': rsi, 'ma50': ma50, 'ma200': ma200, 'goldenCross': golden_cross}
    except Exception:
        return {}


def fetch_stock_data(symbols_info):
    symbols = [s[0] for s in symbols_info]
    symbol_map = {s[0]: {'name': s[1], 'industry': s[2]} for s in symbols_info}

    cache_key = ','.join(sorted(symbols))
    cached = get_cached_data(cache_key)
    if cached:
        return cached

    results = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
            prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose') or 0

            if not price:
                hist = ticker.history(period='2d')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    if not prev_close and len(hist) > 1:
                        prev_close = float(hist['Close'].iloc[-2])

            change = ((price - prev_close) / prev_close * 100) if prev_close and prev_close > 0 else 0

            pe_raw = info.get('trailingPE') or info.get('forwardPE')
            pe = _safe(pe_raw)
            forward_pe = _safe(info.get('forwardPE') or pe_raw)

            growth_raw = info.get('revenueGrowth') or info.get('earningsGrowth')
            growth = _safe(growth_raw, multiplier=100, decimals=1)

            # PEG: use API value first, fall back to calculated, never divide by zero
            peg_raw = info.get('pegRatio')
            if peg_raw:
                peg = _safe(peg_raw)
            elif pe and growth and growth > 0:
                peg = _safe(float(pe) / float(growth))
            else:
                peg = None

            week_52_high = _safe(info.get('fiftyTwoWeekHigh'))
            week_52_low = _safe(info.get('fiftyTwoWeekLow'))

            pct_from_high = round((price - week_52_high) / week_52_high * 100, 2) if price and week_52_high else None
            pct_from_low = round((price - week_52_low) / week_52_low * 100, 2) if price and week_52_low else None

            # New metrics
            beta = _safe(info.get('beta'))
            dividend_yield = _safe(info.get('dividendYield'), multiplier=100, decimals=2)
            short_ratio = _safe(info.get('shortRatio'))
            analyst_target = _safe(info.get('targetMeanPrice'))
            analyst_rating = _safe(info.get('recommendationMean'))
            debt_to_equity = _safe(info.get('debtToEquity'))
            roe = _safe(info.get('returnOnEquity'), multiplier=100, decimals=1)

            # Analyst upside vs current price
            analyst_upside = None
            if analyst_target and price:
                analyst_upside = round((analyst_target - price) / price * 100, 1)

            industry = symbol_map[symbol]['industry']
            industry_pe = INDUSTRY_PE.get(industry)

            technicals = compute_technicals(ticker)

            stock_data = {
                'ticker': symbol.replace('-USD', ''),
                'name': symbol_map[symbol]['name'],
                'price': round(price, 2) if price else 0,
                'change': round(change, 2),
                'pe': pe,
                'forwardPe': forward_pe,
                'peg': peg,
                'growth': growth,
                'industry': industry,
                'industryPe': industry_pe,
                'volume': info.get('volume') or info.get('regularMarketVolume') or None,
                'marketCap': info.get('marketCap') or None,
                'week52High': week_52_high,
                'week52Low': week_52_low,
                'pctFromHigh': pct_from_high,
                'pctFromLow': pct_from_low,
                # New fields
                'beta': beta,
                'dividendYield': dividend_yield,
                'shortRatio': short_ratio,
                'analystTarget': analyst_target,
                'analystRating': analyst_rating,
                'analystUpside': analyst_upside,
                'debtToEquity': debt_to_equity,
                'roe': roe,
                # Technicals
                'rsi': technicals.get('rsi'),
                'ma50': technicals.get('ma50'),
                'ma200': technicals.get('ma200'),
                'goldenCross': technicals.get('goldenCross', False),
            }

            results.append(stock_data)
            print(f"Fetched {symbol}: ${price}")

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            results.append({
                'ticker': symbol.replace('-USD', ''),
                'name': symbol_map[symbol]['name'],
                'price': 0,
                'change': 0,
                'pe': None, 'forwardPe': None, 'peg': None, 'growth': None,
                'industry': symbol_map[symbol]['industry'],
                'industryPe': INDUSTRY_PE.get(symbol_map[symbol]['industry']),
                'volume': None, 'marketCap': None,
                'week52High': None, 'week52Low': None,
                'pctFromHigh': None, 'pctFromLow': None,
                'beta': None, 'dividendYield': None, 'shortRatio': None,
                'analystTarget': None, 'analystRating': None, 'analystUpside': None,
                'debtToEquity': None, 'roe': None,
                'rsi': None, 'ma50': None, 'ma200': None, 'goldenCross': False,
            })

    results = [r for r in results if r['price'] > 0]
    set_cached_data(cache_key, results)
    return results


@app.route('/api/stocks/<exchange>')
def get_stocks(exchange):
    if exchange not in WATCHLISTS:
        return jsonify({'error': f'Unknown exchange: {exchange}'}), 404
    stocks = fetch_stock_data(WATCHLISTS[exchange])
    return jsonify({
        'exchange': exchange,
        'count': len(stocks),
        'stocks': stocks,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/stocks')
def get_all_stocks():
    all_stocks = {}
    for exchange in WATCHLISTS:
        all_stocks[exchange] = fetch_stock_data(WATCHLISTS[exchange])
    return jsonify({
        'exchanges': list(WATCHLISTS.keys()),
        'data': all_stocks,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/earnings/<exchange>')
def get_earnings(exchange):
    """Upcoming earnings dates for all tickers in an exchange."""
    if exchange not in WATCHLISTS:
        return jsonify({'error': f'Unknown exchange: {exchange}'}), 404

    results = []
    for symbol, name, _ in WATCHLISTS[exchange]:
        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if cal is None:
                continue

            # calendar can be a dict or DataFrame depending on yfinance version
            if isinstance(cal, dict):
                earnings_date = cal.get('Earnings Date')
                eps_est = cal.get('EPS Estimate')
                rev_est = cal.get('Revenue Estimate')
                if isinstance(earnings_date, list):
                    earnings_date = earnings_date[0] if earnings_date else None
            else:
                # DataFrame: index has metric names
                try:
                    earnings_date = cal.loc['Earnings Date'].iloc[0] if 'Earnings Date' in cal.index else None
                    eps_est = cal.loc['EPS Estimate'].iloc[0] if 'EPS Estimate' in cal.index else None
                    rev_est = cal.loc['Revenue Average'].iloc[0] if 'Revenue Average' in cal.index else None
                except Exception:
                    earnings_date, eps_est, rev_est = None, None, None

            if not earnings_date:
                continue

            results.append({
                'ticker': symbol.replace('-USD', ''),
                'name': name,
                'earningsDate': str(earnings_date)[:10] if earnings_date else None,
                'epsEstimate': _safe(eps_est),
                'revenueEstimate': int(rev_est) if rev_est and not np.isnan(float(rev_est)) else None,
            })
        except Exception as e:
            print(f"Earnings error {symbol}: {e}")

    results.sort(key=lambda x: x['earningsDate'] or '9999')
    return jsonify({'exchange': exchange, 'earnings': results, 'timestamp': datetime.now().isoformat()})


@app.route('/api/sentiment/<ticker>')
def get_ticker_sentiment(ticker):
    """Reddit + MoneyControl sentiment for a single ticker."""
    # Determine exchange hint from ticker suffix for subreddit routing
    exchange = 'default'
    if ticker.upper().endswith('.NS') or ticker.upper().endswith('.BO'):
        exchange = 'nse'
    elif ticker.upper().endswith('.TO'):
        exchange = 'tsx'
    result = get_sentiment(ticker.upper(), exchange)
    return jsonify(result)


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'exchanges': list(WATCHLISTS.keys()),
        'total_symbols': sum(len(v) for v in WATCHLISTS.values())
    })


@app.route('/')
def index():
    return jsonify({
        'name': 'FinanceBro API',
        'version': '2.0',
        'endpoints': {
            '/api/health': 'Health check',
            '/api/stocks': 'All stocks',
            '/api/stocks/<exchange>': 'Stocks by exchange (nyse, crypto, etf, tsx, nse)',
            '/api/sentiment/<ticker>': 'Reddit + MoneyControl sentiment',
            '/api/earnings/<exchange>': 'Upcoming earnings calendar',
        }
    })


if __name__ == '__main__':
    print("Starting FinanceBro API v2.0...")
    print("  http://localhost:5001/api/health")
    print("  http://localhost:5001/api/stocks/nyse")
    print("  http://localhost:5001/api/earnings/nyse")
    print("  http://localhost:5001/api/sentiment/AAPL")
    app.run(debug=True, port=5001)
