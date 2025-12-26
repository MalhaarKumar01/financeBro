import os
import psycopg2
from psycopg2.extras import execute_values
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from dotenv import load_dotenv

load_dotenv()

# Database config
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_NAME = os.getenv('DB_NAME', 'finance_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
DB_PORT = os.getenv('DB_PORT', 5432)

COINGECKO_API = 'https://api.coingecko.com/api/v3'
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')
FMP_API_KEY = os.getenv('FMP_API_KEY', '')  # Financial Modeling Prep for fundamentals

# Industry average P/E ratios (2024 approximations)
INDUSTRY_PE_RATIOS = {
    'AAPL': {'industry': 'Technology', 'industry_pe': 28.5},
    'MSFT': {'industry': 'Technology', 'industry_pe': 28.5},
    'GOOGL': {'industry': 'Technology', 'industry_pe': 28.5},
    'AMZN': {'industry': 'Consumer Cyclical', 'industry_pe': 22.0},
    'META': {'industry': 'Technology', 'industry_pe': 28.5},
    'NVDA': {'industry': 'Technology', 'industry_pe': 28.5},
    'TSLA': {'industry': 'Consumer Cyclical', 'industry_pe': 22.0},
}

class FinanceDB:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        self.cur = self.conn.cursor()
    
    def init_tables(self):
        """Create tables if they don't exist"""
        # Check if price_data table exists and has the right columns
        self.cur.execute('''
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'price_data'
        ''')
        existing_columns = [row[0] for row in self.cur.fetchall()]
        
        if not existing_columns:
            # Create new table
            self.cur.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20),
                    asset_type VARCHAR(20),
                    price DECIMAL(15, 8),
                    volume DECIMAL(20, 2),
                    market_cap DECIMAL(20, 2),
                    timestamp TIMESTAMP,
                    UNIQUE(symbol, asset_type, timestamp)
                )
            ''')
        else:
            # Add missing columns if needed
            if 'volume' not in existing_columns:
                self.cur.execute('ALTER TABLE price_data ADD COLUMN volume DECIMAL(20, 2)')
                print('✅ Added volume column to price_data')
            if 'market_cap' not in existing_columns:
                self.cur.execute('ALTER TABLE price_data ADD COLUMN market_cap DECIMAL(20, 2)')
                print('✅ Added market_cap column to price_data')
        
        self.cur.execute('''
            CREATE TABLE IF NOT EXISTS fundamentals (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20),
                pe_ratio DECIMAL(10, 2),
                peg_ratio DECIMAL(10, 2),
                debt_to_equity DECIMAL(10, 2),
                roe DECIMAL(10, 4),
                free_cash_flow BIGINT,
                dividend_yield DECIMAL(10, 4),
                eps DECIMAL(10, 2),
                updated_at TIMESTAMP,
                UNIQUE(symbol, updated_at)
            )
        ''')
        
        self.cur.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20),
                asset_type VARCHAR(20),
                rsi DECIMAL(10, 4),
                macd DECIMAL(15, 8),
                macd_signal DECIMAL(15, 8),
                bollinger_upper DECIMAL(15, 8),
                bollinger_lower DECIMAL(15, 8),
                sma_20 DECIMAL(15, 8),
                sma_50 DECIMAL(15, 8),
                sma_200 DECIMAL(15, 8),
                timestamp TIMESTAMP,
                UNIQUE(symbol, asset_type, timestamp)
            )
        ''')
        
        self.cur.execute('''
            CREATE TABLE IF NOT EXISTS analysis_scores (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20),
                asset_type VARCHAR(20),
                technical_score DECIMAL(5, 2),
                fundamental_score DECIMAL(5, 2),
                overall_score DECIMAL(5, 2),
                recommendation VARCHAR(20),
                timestamp TIMESTAMP
            )
        ''')
        
        self.cur.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20),
                asset_type VARCHAR(20),
                alert_threshold DECIMAL(10, 2),
                UNIQUE(symbol, asset_type)
            )
        ''')
        
        self.cur.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20),
                alert_type VARCHAR(50),
                message TEXT,
                triggered_at TIMESTAMP,
                price DECIMAL(15, 8)
            )
        ''')
        
        self.conn.commit()
    
    def insert_prices(self, prices):
        """Insert price data with volume and market cap"""
        query = '''
            INSERT INTO price_data (symbol, asset_type, price, volume, market_cap, timestamp)
            VALUES %s
            ON CONFLICT (symbol, asset_type, timestamp) DO NOTHING
        '''
        execute_values(self.cur, query, prices)
        self.conn.commit()
    
    def insert_fundamentals(self, symbol, fundamentals):
        """Insert fundamental data"""
        self.cur.execute('''
            INSERT INTO fundamentals 
            (symbol, pe_ratio, peg_ratio, debt_to_equity, roe, free_cash_flow, 
             dividend_yield, eps, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, updated_at) DO NOTHING
        ''', (
            symbol,
            fundamentals.get('pe_ratio'),
            fundamentals.get('peg_ratio'),
            fundamentals.get('debt_to_equity'),
            fundamentals.get('roe'),
            fundamentals.get('free_cash_flow'),
            fundamentals.get('dividend_yield'),
            fundamentals.get('eps'),
            datetime.now()
        ))
        self.conn.commit()
    
    def insert_technical_indicators(self, symbol, asset_type, indicators):
        """Insert technical indicators"""
        # Convert numpy types to Python native types
        def to_native(val):
            if val is None:
                return None
            if isinstance(val, (np.integer, np.floating)):
                return float(val)
            return val
        
        self.cur.execute('''
            INSERT INTO technical_indicators 
            (symbol, asset_type, rsi, macd, macd_signal, bollinger_upper, 
             bollinger_lower, sma_20, sma_50, sma_200, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, asset_type, timestamp) DO NOTHING
        ''', (
            symbol, asset_type,
            to_native(indicators.get('rsi')),
            to_native(indicators.get('macd')),
            to_native(indicators.get('macd_signal')),
            to_native(indicators.get('bollinger_upper')),
            to_native(indicators.get('bollinger_lower')),
            to_native(indicators.get('sma_20')),
            to_native(indicators.get('sma_50')),
            to_native(indicators.get('sma_200')),
            datetime.now()
        ))
        self.conn.commit()
    
    def insert_analysis_score(self, symbol, asset_type, tech_score, fund_score, overall, rec):
        """Insert analysis score"""
        self.cur.execute('''
            INSERT INTO analysis_scores 
            (symbol, asset_type, technical_score, fundamental_score, 
             overall_score, recommendation, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (symbol, asset_type, tech_score, fund_score, overall, rec, datetime.now()))
        self.conn.commit()
    
    def insert_alert(self, symbol, alert_type, message, price):
        """Log an alert"""
        self.cur.execute('''
            INSERT INTO alerts (symbol, alert_type, message, triggered_at, price)
            VALUES (%s, %s, %s, %s, %s)
        ''', (symbol, alert_type, message, datetime.now(), price))
        self.conn.commit()
    
    def get_watchlist(self):
        """Get all watched assets"""
        self.cur.execute('SELECT symbol, asset_type FROM watchlist')
        return self.cur.fetchall()
    
    def add_to_watchlist(self, symbol, asset_type):
        """Add asset to watchlist"""
        self.cur.execute('''
            INSERT INTO watchlist (symbol, asset_type)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
        ''', (symbol, asset_type))
        self.conn.commit()
    
    def get_price_history(self, symbol, days=30):
        """Get historical price data with volume"""
        self.cur.execute('''
            SELECT timestamp, price, volume FROM price_data
            WHERE symbol = %s AND timestamp > NOW() - INTERVAL '%s days'
            ORDER BY timestamp ASC
        ''', (symbol, days))
        return self.cur.fetchall()
    
    def get_latest_price(self, symbol):
        """Get most recent price"""
        self.cur.execute('''
            SELECT price, timestamp FROM price_data
            WHERE symbol = %s
            ORDER BY timestamp DESC LIMIT 1
        ''', (symbol,))
        return self.cur.fetchone()
    
    def close(self):
        self.cur.close()
        self.conn.close()

class DataFetcher:
    @staticmethod
    def get_crypto_price(crypto_ids):
        url = f'{COINGECKO_API}/simple/price'
        params = {
            'ids': ','.join(crypto_ids),
            'vs_currencies': 'usd',
            'include_market_cap': 'true',
            'include_24hr_vol': 'true'
        }
        backoff = 2
        for _ in range(5):
            try:
                resp = requests.get(url, params=params, timeout=5, headers={"User-Agent": "financeBro/1.0"})
                if resp.status_code == 429:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print(f'Error fetching {crypto_ids}: {e}')
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
        return None

    
    @staticmethod
    def get_crypto_history(crypto_id, days=90):
        """Get historical crypto data"""
        try:
            url = f'{COINGECKO_API}/coins/{crypto_id}/market_chart'
            params = {'vs_currency': 'usd', 'days': days}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f'Error fetching history for {crypto_id}: {e}')
            return None
    
    @staticmethod
    def get_stock_price(symbol):
        """Get stock price from Alpha Vantage"""
        if not ALPHA_VANTAGE_KEY:
            return None
        
        try:
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': ALPHA_VANTAGE_KEY
            }
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']
                return {
                    'price': float(quote.get('05. price', 0)),
                    'volume': float(quote.get('06. volume', 0))
                }
            return None
        except Exception as e:
            print(f'Error fetching {symbol}: {e}')
            return None
    
    @staticmethod
    def get_stock_fundamentals(symbol):
        """Get fundamental data (example using FMP API - free tier available)"""
        if not FMP_API_KEY:
            return None
        
        try:
            # Key metrics endpoint
            url = f'https://financialmodelingprep.com/api/v3/key-metrics/{symbol}'
            params = {'apikey': FMP_API_KEY}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if data and len(data) > 0:
                metrics = data[0]
                return {
                    'pe_ratio': metrics.get('peRatio'),
                    'peg_ratio': metrics.get('pegRatio'),
                    'debt_to_equity': metrics.get('debtToEquity'),
                    'roe': metrics.get('roe'),
                    'free_cash_flow': metrics.get('freeCashFlow'),
                    'dividend_yield': metrics.get('dividendYield'),
                    'eps': metrics.get('netIncomePerShare')
                }
            return None
        except Exception as e:
            print(f'Error fetching fundamentals for {symbol}: {e}')
            return None

class TechnicalAnalysis:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return None
        
        df = pd.DataFrame(prices, columns=['timestamp', 'price', 'volume'])
        df['price'] = pd.to_numeric(df['price'])
        
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """MACD indicator"""
        if len(prices) < slow + signal:
            return None, None
        
        df = pd.DataFrame(prices, columns=['timestamp', 'price', 'volume'])
        df['price'] = pd.to_numeric(df['price'])
        
        ema_fast = df['price'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['price'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        
        return macd.iloc[-1], macd_signal.iloc[-1]
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        """Bollinger Bands"""
        if len(prices) < period:
            return None, None, None
        
        df = pd.DataFrame(prices, columns=['timestamp', 'price', 'volume'])
        df['price'] = pd.to_numeric(df['price'])
        
        sma = df['price'].rolling(window=period).mean()
        std = df['price'].rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]
    
    @staticmethod
    def calculate_sma(prices, period):
        """Simple Moving Average"""
        if len(prices) < period:
            return None
        
        df = pd.DataFrame(prices, columns=['timestamp', 'price', 'volume'])
        df['price'] = pd.to_numeric(df['price'])
        return df['price'].rolling(window=period).mean().iloc[-1]
    
    @staticmethod
    def calculate_volatility(prices, period=30):
        """Price volatility (standard deviation of returns)"""
        if len(prices) < period:
            return None
        
        df = pd.DataFrame(prices, columns=['timestamp', 'price', 'volume'])
        df['price'] = pd.to_numeric(df['price'])
        returns = df['price'].pct_change()
        return returns.std() * np.sqrt(252)  # Annualized

class ScoringSystem:
    """Score assets based on technical and fundamental analysis"""
    
    @staticmethod
    def score_technical(indicators):
        """Score based on technical indicators (0-100)"""
        score = 0
        max_score = 0
        
        # RSI scoring (0-20 points)
        if indicators.get('rsi') is not None:
            rsi = indicators['rsi']
            if 40 <= rsi <= 60:
                score += 20  # Neutral is good
            elif 30 <= rsi < 40 or 60 < rsi <= 70:
                score += 15
            elif rsi < 30:
                score += 10  # Oversold - potential buy
            else:
                score += 5   # Overbought - caution
            max_score += 20
        
        # MACD scoring (0-20 points)
        if indicators.get('macd') is not None and indicators.get('macd_signal') is not None:
            macd_diff = indicators['macd'] - indicators['macd_signal']
            if macd_diff > 0:
                score += 20  # Bullish crossover
            else:
                score += 10  # Bearish
            max_score += 20
        
        # Moving Average trend (0-30 points)
        sma_20 = indicators.get('sma_20')
        sma_50 = indicators.get('sma_50')
        sma_200 = indicators.get('sma_200')
        
        if all([sma_20, sma_50, sma_200]):
            if sma_20 > sma_50 > sma_200:
                score += 30  # Strong uptrend
            elif sma_20 > sma_50:
                score += 20
            elif sma_20 > sma_200:
                score += 15
            else:
                score += 5
            max_score += 30
        
        # Bollinger Bands position (0-15 points)
        price = indicators.get('current_price')
        bb_upper = indicators.get('bollinger_upper')
        bb_lower = indicators.get('bollinger_lower')
        
        if all([price, bb_upper, bb_lower]):
            bb_range = bb_upper - bb_lower
            position = (price - bb_lower) / bb_range if bb_range > 0 else 0.5
            
            if 0.3 <= position <= 0.7:
                score += 15  # Middle of bands
            elif position < 0.3:
                score += 10  # Near lower band - potential buy
            else:
                score += 5   # Near upper band - caution
            max_score += 15
        
        # Volume trend (0-15 points)
        if indicators.get('volume_trend') == 'increasing':
            score += 15
        elif indicators.get('volume_trend') == 'stable':
            score += 10
        else:
            score += 5
        max_score += 15
        
        return (score / max_score * 100) if max_score > 0 else 50
    
    @staticmethod
    def score_fundamental(fundamentals):
        """Score based on fundamental metrics (0-100)"""
        if not fundamentals:
            return None
        
        score = 0
        max_score = 0
        
        # P/E Ratio (0-25 points)
        pe = fundamentals.get('pe_ratio')
        if pe is not None and pe > 0:
            if 10 <= pe <= 20:
                score += 25  # Reasonable valuation
            elif 5 <= pe < 10 or 20 < pe <= 30:
                score += 20
            elif pe < 5:
                score += 15  # Might be undervalued or troubled
            else:
                score += 5   # Overvalued
            max_score += 25
        
        # PEG Ratio (0-20 points)
        peg = fundamentals.get('peg_ratio')
        if peg is not None and peg > 0:
            if peg < 1:
                score += 20  # Undervalued relative to growth
            elif 1 <= peg <= 2:
                score += 15
            else:
                score += 5
            max_score += 20
        
        # Debt to Equity (0-20 points)
        dte = fundamentals.get('debt_to_equity')
        if dte is not None:
            if dte < 0.5:
                score += 20  # Low debt
            elif 0.5 <= dte <= 1.0:
                score += 15
            elif 1.0 < dte <= 2.0:
                score += 10
            else:
                score += 5   # High debt
            max_score += 20
        
        # ROE (0-20 points)
        roe = fundamentals.get('roe')
        if roe is not None:
            if roe > 0.15:
                score += 20  # Strong returns
            elif 0.10 <= roe <= 0.15:
                score += 15
            elif 0.05 <= roe < 0.10:
                score += 10
            else:
                score += 5
            max_score += 20
        
        # Dividend Yield (0-15 points)
        div_yield = fundamentals.get('dividend_yield')
        if div_yield is not None:
            if 0.02 <= div_yield <= 0.06:
                score += 15  # Healthy dividend
            elif 0.01 <= div_yield < 0.02 or 0.06 < div_yield <= 0.10:
                score += 10
            else:
                score += 5
            max_score += 15
        
        return (score / max_score * 100) if max_score > 0 else None
    
    @staticmethod
    def get_recommendation(tech_score, fund_score):
        """Generate buy/hold/sell recommendation"""
        if fund_score is None:
            overall = tech_score
        else:
            overall = (tech_score * 0.6 + fund_score * 0.4)
        
        if overall >= 75:
            return overall, "STRONG_BUY"
        elif overall >= 60:
            return overall, "BUY"
        elif overall >= 45:
            return overall, "HOLD"
        elif overall >= 30:
            return overall, "SELL"
        else:
            return overall, "STRONG_SELL"

def analyze_asset(db, symbol, asset_type, history):
    """Complete analysis of an asset"""
    if len(history) < 20:
        print(f'  ⚠️  Insufficient data for {symbol}')
        return
    
    print(f'\n  📊 Analyzing {symbol}...')
    
    # Calculate technical indicators
    rsi = TechnicalAnalysis.calculate_rsi(history)
    macd, macd_signal = TechnicalAnalysis.calculate_macd(history)
    bb_upper, bb_mid, bb_lower = TechnicalAnalysis.calculate_bollinger_bands(history)
    sma_20 = TechnicalAnalysis.calculate_sma(history, 20)
    sma_50 = TechnicalAnalysis.calculate_sma(history, 50) if len(history) >= 50 else None
    sma_200 = TechnicalAnalysis.calculate_sma(history, 200) if len(history) >= 200 else None
    volatility = TechnicalAnalysis.calculate_volatility(history)
    
    current_price = float(history[-1][1])
    
    # Volume trend
    if len(history) >= 10:
        recent_vol = np.mean([float(h[2]) for h in history[-5:] if h[2]])
        older_vol = np.mean([float(h[2]) for h in history[-10:-5] if h[2]])
        volume_trend = 'increasing' if recent_vol > older_vol * 1.1 else 'stable' if recent_vol > older_vol * 0.9 else 'decreasing'
    else:
        volume_trend = 'unknown'
    
    # Store technical indicators
    indicators = {
        'rsi': rsi,
        'macd': macd,
        'macd_signal': macd_signal,
        'bollinger_upper': bb_upper,
        'bollinger_lower': bb_lower,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'sma_200': sma_200,
        'current_price': current_price,
        'volume_trend': volume_trend
    }
    
    db.insert_technical_indicators(symbol, asset_type, indicators)
    
    # Get fundamentals for stocks
    fundamentals = None
    if asset_type == 'stock':
        fundamentals = DataFetcher.get_stock_fundamentals(symbol)
        if fundamentals:
            db.insert_fundamentals(symbol, fundamentals)
    
    # Calculate scores
    tech_score = ScoringSystem.score_technical(indicators)
    fund_score = ScoringSystem.score_fundamental(fundamentals) if fundamentals else None
    overall_score, recommendation = ScoringSystem.get_recommendation(tech_score, fund_score)
    
    # Store scores
    db.insert_analysis_score(symbol, asset_type, tech_score, fund_score, overall_score, recommendation)
    
    # Print analysis
    sma_20_str = f'{sma_20:.2f}' if sma_20 is not None else 'N/A'
    sma_50_str = f'{sma_50:.2f}' if sma_50 is not None else 'N/A'
    sma_200_str = f'{sma_200:.2f}' if sma_200 is not None else 'N/A'
    
    print(f'    RSI: {rsi:.2f}' if rsi else '    RSI: N/A')
    print(f'    MACD: {macd:.4f} | Signal: {macd_signal:.4f}' if macd else '    MACD: N/A')
    print(f'    Price: ${current_price:.2f} | BB: ${bb_lower:.2f} - ${bb_upper:.2f}' if bb_lower else f'    Price: ${current_price:.2f}')
    print(f'    SMA(20/50/200): {sma_20_str} / {sma_50_str} / {sma_200_str}')
    print(f'    Volatility: {volatility:.2%}' if volatility else '    Volatility: N/A')
    print(f'    Volume Trend: {volume_trend}')
    
    if fundamentals:
        print(f'    P/E: {fundamentals.get("pe_ratio"):.2f}' if fundamentals.get("pe_ratio") else '    P/E: N/A')
        print(f'    Debt/Equity: {fundamentals.get("debt_to_equity"):.2f}' if fundamentals.get("debt_to_equity") else '    Debt/Equity: N/A')
    
    print(f'    📈 Technical Score: {tech_score:.1f}/100')
    if fund_score:
        print(f'    📊 Fundamental Score: {fund_score:.1f}/100')
    print(f'    ⭐ Overall Score: {overall_score:.1f}/100')
    print(f'    💡 Recommendation: {recommendation}')

def backfill_historical_data(db, symbol, crypto_id, asset_type):
    """Backfill historical data for crypto"""
    print(f'  📚 Backfilling historical data for {symbol}...')
    
    if asset_type == 'crypto':
        history = DataFetcher.get_crypto_history(crypto_id, days=90)
        if history and 'prices' in history:
            prices_to_insert = []
            for timestamp_ms, price in history['prices']:
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                # Get volume if available
                volume = 0
                if 'total_volumes' in history:
                    for vol_ts, vol in history['total_volumes']:
                        if vol_ts == timestamp_ms:
                            volume = vol
                            break
                
                prices_to_insert.append((
                    symbol, asset_type, price, volume, None, timestamp
                ))
            
            if prices_to_insert:
                db.insert_prices(prices_to_insert)
                print(f'    ✅ Inserted {len(prices_to_insert)} historical records')
                return True
    return False

def run_pipeline(backfill=False):
    """Main pipeline"""
    db = FinanceDB()
    db.init_tables()
    
    # Watchlist: (symbol, crypto_id or 'stock', asset_type)
    watchlist = [
        ('BTC', 'bitcoin', 'crypto'),
        ('ETH', 'ethereum', 'crypto'),
        ('SOL', 'solana', 'crypto'),
        ('AAPL', 'stock', 'stock'),
        ('MSFT', 'stock', 'stock'),
        ('GOOGL', 'stock', 'stock'),
    ]
    
    print(f'[{datetime.now()}] 🚀 Starting FinanceBro Analysis Pipeline...\n')
    
    # Add to watchlist
    for symbol, _, asset_type in watchlist:
        db.add_to_watchlist(symbol, asset_type)
    
    # Backfill historical data if requested
    if backfill:
        print('📚 Backfilling historical data...')
        for symbol, identifier, asset_type in watchlist:
            if asset_type == 'crypto':
                backfill_historical_data(db, symbol, identifier, asset_type)
        print()
    
    # Fetch and store current prices
    print('📥 Fetching current prices...')
    for symbol, identifier, asset_type in watchlist:
        if asset_type == 'crypto':
            data = DataFetcher.get_crypto_price(identifier)
            if data and identifier in data:
                price = data[identifier]['usd']
                volume = data[identifier].get('usd_24h_vol', 0)
                market_cap = data[identifier].get('usd_market_cap', 0)
                print(f'  {symbol}: ${price:.2f} | Vol: ${volume:,.0f}')
                db.insert_prices([(symbol, asset_type, price, volume, market_cap, datetime.now())])
            time.sleep(1)  # Rate limit protection
        
        elif asset_type == 'stock' and ALPHA_VANTAGE_KEY:
            data = DataFetcher.get_stock_price(symbol)
            if data:
                price = data['price']
                volume = data.get('volume', 0)
                print(f'  {symbol}: ${price:.2f} | Vol: {volume:,.0f}')
                db.insert_prices([(symbol, asset_type, price, volume, None, datetime.now())])
            time.sleep(1)  # Rate limit protection
    
    # Perform analysis
    print('\n' + '='*60)
    print('📊 RUNNING TECHNICAL & FUNDAMENTAL ANALYSIS')
    print('='*60)
    
    for symbol, _, asset_type in watchlist:
        history = db.get_price_history(symbol, days=90)
        if history:
            analyze_asset(db, symbol, asset_type, history)
    
    # Generate summary report
    print('\n' + '='*60 + "khujli")
    print('📋 SUMMARY REPORT')
    print('='*60)
    
    db.cur.execute('''
        SELECT symbol, asset_type, overall_score, recommendation, timestamp
        FROM analysis_scores
        WHERE timestamp > NOW() - INTERVAL '1 hour'
        ORDER BY overall_score DESC
    ''')
    results = db.cur.fetchall()
    
    if results:
        print('\n🏆 Top Recommendations:')
        for symbol, asset_type, score, rec, timestamp in results:
            emoji = '🚀' if 'BUY' in rec else '⚠️' if rec == 'HOLD' else '📉'
            print(f'  {emoji} {symbol} ({asset_type}): {score:.1f}/100 - {rec}')
    
    db.close()
    print(f'\n[{datetime.now()}] ✅ Pipeline complete!')

if __name__ == '__main__':
    import sys
    
    # Check if backfill flag is provided
    backfill = '--backfill' in sys.argv
    
    if backfill:
        print('🔄 Running with historical data backfill...\n')
    
    run_pipeline(backfill=backfill)
