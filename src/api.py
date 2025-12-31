"""
Flask API for FinanceBro - Serves stock data from yfinance
Run with: python api.py
"""
import yfinance as yf
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Stock watchlists by exchange/category
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
        ('MCD', 'McDonald\'s Corp.', 'Consumer Cyclical'),
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

# Industry average P/E ratios
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

# Cache for stock data (simple in-memory cache)
_cache = {}
_cache_timeout = 60  # seconds


def get_cached_data(key):
    """Get data from cache if not expired"""
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < _cache_timeout:
            return data
    return None


def set_cached_data(key, data):
    """Set data in cache"""
    _cache[key] = (data, time.time())


def fetch_stock_data(symbols_info):
    """Fetch stock data for multiple symbols using yfinance"""
    symbols = [s[0] for s in symbols_info]
    symbol_map = {s[0]: {'name': s[1], 'industry': s[2]} for s in symbols_info}

    # Check cache first
    cache_key = ','.join(sorted(symbols))
    cached = get_cached_data(cache_key)
    if cached:
        return cached

    results = []

    try:
        # Fetch each ticker individually for better reliability in v1.0
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                # Get price data - yfinance v1.0 changed the API
                price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
                prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose') or 0

                # If we still don't have price, try getting it from history
                if not price:
                    hist = ticker.history(period='1d')
                    if not hist.empty:
                        price = float(hist['Close'].iloc[-1])
                        if not prev_close and len(hist) > 1:
                            prev_close = float(hist['Close'].iloc[-2])

                change = ((price - prev_close) / prev_close * 100) if prev_close and prev_close > 0 else 0

                # Get valuation metrics
                pe = info.get('trailingPE') or info.get('forwardPE') or 0
                forward_pe = info.get('forwardPE') or pe or 0
                peg = info.get('pegRatio') or 0

                # Get growth - use revenue growth or earnings growth
                growth = info.get('revenueGrowth', 0) or info.get('earningsGrowth', 0) or 0
                if growth:
                    growth = growth * 100  # Convert to percentage

                # If PEG is 0 but we have PE and growth, calculate it
                if peg == 0 and pe > 0 and growth > 0:
                    peg = pe / growth

                # Get 52-week high and low
                week_52_high = info.get('fiftyTwoWeekHigh') or 0
                week_52_low = info.get('fiftyTwoWeekLow') or 0

                # Calculate distance from 52-week high/low (as percentage)
                pct_from_high = 0
                pct_from_low = 0
                if price and week_52_high:
                    pct_from_high = ((price - week_52_high) / week_52_high * 100)
                if price and week_52_low:
                    pct_from_low = ((price - week_52_low) / week_52_low * 100)

                industry = symbol_map[symbol]['industry']
                industry_pe = INDUSTRY_PE.get(industry, 20.0)

                stock_data = {
                    'ticker': symbol.replace('-USD', ''),  # Clean up crypto tickers
                    'name': symbol_map[symbol]['name'],
                    'price': round(price, 2) if price else 0,
                    'change': round(change, 2),
                    'pe': round(pe, 2) if pe else 0,
                    'forwardPe': round(forward_pe, 2) if forward_pe else 0,
                    'peg': round(peg, 2) if peg else 0,
                    'growth': round(growth, 1) if growth else 0,
                    'industry': industry,
                    'industryPe': industry_pe if industry_pe else 0,
                    'volume': info.get('volume', 0) or info.get('regularMarketVolume', 0),
                    'marketCap': info.get('marketCap', 0),
                    'week52High': round(week_52_high, 2) if week_52_high else 0,
                    'week52Low': round(week_52_low, 2) if week_52_low else 0,
                    'pctFromHigh': round(pct_from_high, 2),
                    'pctFromLow': round(pct_from_low, 2),
                }

                results.append(stock_data)
                print(f"Fetched {symbol}: ${price}")

            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                # Add placeholder data
                results.append({
                    'ticker': symbol.replace('-USD', ''),
                    'name': symbol_map[symbol]['name'],
                    'price': 0,
                    'change': 0,
                    'pe': 0,
                    'forwardPe': 0,
                    'peg': 0,
                    'growth': 0,
                    'industry': symbol_map[symbol]['industry'],
                    'industryPe': INDUSTRY_PE.get(symbol_map[symbol]['industry'], 0),
                    'volume': 0,
                    'marketCap': 0,
                    'week52High': 0,
                    'week52Low': 0,
                    'pctFromHigh': 0,
                    'pctFromLow': 0,
                })

    except Exception as e:
        print(f"Error fetching batch: {e}")
        return []

    # Filter out stocks with no price data
    results = [r for r in results if r['price'] > 0]

    # Cache results
    set_cached_data(cache_key, results)

    return results


@app.route('/api/stocks/<exchange>')
def get_stocks(exchange):
    """Get stocks for a specific exchange/category"""
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
    """Get all stocks from all exchanges"""
    all_stocks = {}
    for exchange in WATCHLISTS:
        all_stocks[exchange] = fetch_stock_data(WATCHLISTS[exchange])

    return jsonify({
        'exchanges': list(WATCHLISTS.keys()),
        'data': all_stocks,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'exchanges': list(WATCHLISTS.keys()),
        'total_symbols': sum(len(v) for v in WATCHLISTS.values())
    })


@app.route('/')
def index():
    """Redirect to health check"""
    return jsonify({
        'name': 'FinanceBro API',
        'version': '1.0',
        'endpoints': {
            '/api/health': 'Health check',
            '/api/stocks': 'Get all stocks',
            '/api/stocks/<exchange>': 'Get stocks by exchange (nyse, crypto, etf, tsx, nse)',
        }
    })


if __name__ == '__main__':
    print("Starting FinanceBro API Server...")
    print("Endpoints:")
    print("  - http://localhost:5001/api/health")
    print("  - http://localhost:5001/api/stocks")
    print("  - http://localhost:5001/api/stocks/nyse")
    print("  - http://localhost:5001/api/stocks/crypto")
    print("  - http://localhost:5001/api/stocks/etf")
    app.run(debug=True, port=5001)
