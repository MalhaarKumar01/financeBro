# FINANCEBRO 💰

A real-time stock valuation terminal that fetches live market data from Yahoo Finance and provides P/E ratio and PEG ratio analysis across multiple global exchanges.

![Version](https://img.shields.io/badge/version-5.0-blue)
![Python](https://img.shields.io/badge/python-3.13-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## 🌟 Features

### Multi-Exchange Support
Track stocks across 5 major exchanges with **95 total securities**:
- 🇺🇸 **NYSE** - 30 US stocks (Apple, Microsoft, Google, Amazon, etc.)
- 🇨🇦 **TSX** - 40 Canadian stocks (Shopify, Royal Bank, TD Bank, etc.)
- 🇮🇳 **NSE** - 5 Indian stocks (Reliance, L&T, Zensar, Bharat Dynamics, Suzlon)
- ₿ **CRYPTO** - 10 cryptocurrencies (Bitcoin, Ethereum, Solana, etc.)
- 📊 **ETF** - 10 ETFs (SPY, QQQ, ARKK, etc.)

### Real-Time Data & Analysis
- **Live Price Data** - Real-time stock prices via yfinance v1.0
- **P/E Ratio Analysis** - Compare trailing and forward P/E ratios
- **PEG Ratio Valuation** - Identify undervalued, fair value, and overvalued stocks
- **Industry Comparison** - Compare against industry average P/E ratios
- **Growth Metrics** - 5-year EPS growth rates
- **Auto-Refresh** - Data updates every 60 seconds

### Advanced Search & Filtering
**Power Search Syntax:**
```
ticker:AAPL              # Search by ticker symbol
pe<20                    # P/E ratio less than 20
pe>10                    # P/E ratio greater than 10
peg<1                    # PEG ratio less than 1 (undervalued)
growth>15                # Growth rate greater than 15%
industry:Tech            # Filter by industry
```

**Quick Filters:**
- All Stocks
- Undervalued (PEG < 1)
- Fair Value (PEG 1-2)
- Overvalued (PEG > 2)

**Sector Filters:**
- Technology
- Financials
- Healthcare
- Energy
- Industrials
- Consumer Cyclical
- Consumer Defensive
- Materials
- Utilities
- Telecom
- Defence
- Real Estate

### Sorting Options
- Sort by Ticker (A-Z)
- Sort by P/E Ratio (Low to High / High to Low)
- Sort by PEG Ratio (Low to High / High to Low)
- Sort by Price (Low to High / High to Low)
- Sort by Daily Change (Top Gainers)

### Keyboard Shortcuts (Power User Mode)
- `Ctrl/Cmd + K` - Focus search bar
- `Ctrl/Cmd + 1` - Switch to NYSE
- `Ctrl/Cmd + 2` - Switch to TSX
- `Ctrl/Cmd + 3` - Switch to NSE
- `Ctrl/Cmd + 4` - Switch to Crypto
- `Ctrl/Cmd + 5` - Switch to ETF
- `Ctrl/Cmd + R` - Refresh data
- `ESC` - Clear search

### Currency Support
- **USD ($)** - NYSE, Crypto, ETF
- **CAD (C$)** - TSX (Toronto Stock Exchange)
- **INR (₹)** - NSE (National Stock Exchange of India)

## 📋 Prerequisites

- Python 3.13+
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
cd /Users/malhaarkayy/Desktop/financeBro
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies
```bash
cd financeBro/src
pip install -r requirements.txt
```

### 4. Start the API Server
```bash
python main.py
```

The server will start on `http://localhost:5001`

### 5. Open the Frontend
Open `financeBro/src/index.html` in your web browser.

## 📁 Project Structure

```
financeBro/
├── .venv/                  # Virtual environment (git ignored)
├── financeBro/
│   └── src/
│       ├── main.py         # Flask app entry point
│       ├── api.py          # Flask API & stock data fetching
│       ├── index.html      # Frontend UI
│       ├── styles.css      # UI styling (if exists)
│       └── requirements.txt # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔧 Core Components

### `main.py`
Entry point for the Flask application. Imports and runs the Flask app from `api.py`.

**Usage:**
```bash
python main.py
```

### `api.py`
Flask REST API that fetches stock data from Yahoo Finance and serves it to the frontend.

**Key Features:**
- **Stock Watchlists** - Curated lists for each exchange
- **Industry P/E Ratios** - Reference values for 14 industries
- **In-Memory Caching** - 60-second cache to reduce API calls
- **Error Handling** - Graceful fallbacks for failed API calls

**API Endpoints:**
```
GET /                        # API info
GET /api/health             # Health check
GET /api/stocks             # All stocks (all exchanges)
GET /api/stocks/nyse        # NYSE stocks only
GET /api/stocks/tsx         # TSX stocks only
GET /api/stocks/nse         # NSE stocks only
GET /api/stocks/crypto      # Cryptocurrency prices
GET /api/stocks/etf         # ETF prices
```

**Data Structure:**
```json
{
  "exchange": "nyse",
  "count": 30,
  "timestamp": "2025-12-29T13:45:34.658349",
  "stocks": [
    {
      "ticker": "AAPL",
      "name": "Apple Inc.",
      "price": 273.95,
      "change": 0.20,
      "pe": 31.45,
      "forwardPe": 29.82,
      "peg": 1.85,
      "growth": 12.5,
      "industry": "Technology",
      "industryPe": 28.5,
      "volume": 45678900,
      "marketCap": 4234567890000
    }
  ]
}
```

### `index.html`
Single-page application (SPA) frontend with vanilla JavaScript.

**Features:**
- Responsive grid layout (12 stocks per page)
- Real-time data visualization
- Advanced search parser
- Keyboard shortcuts
- Auto-refresh functionality
- Pagination
- Mobile-friendly design

## 📊 Stock Watchlists

### NYSE (30 Stocks)
Technology giants, major financials, healthcare leaders, energy companies, and consumer brands.

**Notable Stocks:**
- AAPL (Apple), MSFT (Microsoft), GOOGL (Alphabet)
- NVDA (NVIDIA), TSLA (Tesla), META (Meta)
- JPM (JPMorgan), V (Visa), MA (Mastercard)
- JNJ (Johnson & Johnson), UNH (UnitedHealth)

### TSX (40 Stocks)
Major Canadian companies across all sectors.

**Notable Stocks:**
- SHOP.TO (Shopify) - E-commerce platform
- RY.TO, TD.TO, BMO.TO - Big 5 Canadian Banks
- ENB.TO, CNQ.TO - Energy infrastructure
- CNR.TO, CP.TO - Railway transportation
- ABX.TO, NTR.TO - Mining and materials

### NSE (5 Stocks)
Selected Indian market leaders in defense and energy sectors.

**Stocks:**
- RELIANCE.NS - Reliance Industries (Energy conglomerate)
- LT.NS - Larsen & Toubro (Engineering & construction)
- ZENSARTECH.NS - Zensar Technologies (IT services)
- BDL.NS - Bharat Dynamics (Defense)
- SUZLON.NS - Suzlon Energy (Renewable energy)

### Crypto (10 Coins)
Major cryptocurrencies by market cap.

**Coins:**
BTC, ETH, SOL, XRP, ADA, DOGE, DOT, AVAX, LINK, MATIC

### ETF (10 Funds)
Popular index and sector ETFs.

**Funds:**
SPY, QQQ, IWM, DIA, VTI, VOO, ARKK, XLF, XLE, XLK

## 🧮 Valuation Methodology

### PEG Ratio Classification
- **Undervalued**: PEG < 1.0 (Green highlight)
- **Fair Value**: PEG 1.0 - 2.0 (Yellow highlight)
- **Overvalued**: PEG > 2.0 (Red highlight)

### Industry Average P/E Ratios
```
Technology:         28.5x
Financials:         12.5x
Healthcare:         22.0x
Energy:             10.5x
Consumer Defensive: 24.0x
Consumer Cyclical:  22.0x
Industrials:        18.0x
Materials:          15.0x
Defence:            25.0x
Utilities:          18.0x
Real Estate:        35.0x
Telecom:            15.0x
```

## 🔒 Security Notes

### Port Configuration
The app runs on **port 5001** (not 5000) to avoid conflicts with macOS AirPlay service.

### API Rate Limiting
- Yahoo Finance API has rate limits
- Built-in 60-second cache reduces API calls
- Fetches each ticker individually for reliability

### Environment Variables
If you need to configure any sensitive data, create a `.env` file (already in `.gitignore`):
```
# Example .env (not needed for basic usage)
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
```

## 🐛 Troubleshooting

### Issue: "403 Forbidden" Error
**Solution:** Port 5000 is used by macOS AirPlay. The app now uses port 5001.

### Issue: No stock data showing
**Solutions:**
1. Check if Flask server is running: `http://localhost:5001/api/health`
2. Check terminal for error messages
3. Verify internet connection
4. Clear browser cache and refresh

### Issue: yfinance not working
**Solution:** Update to yfinance v1.0:
```bash
pip install --upgrade yfinance
```

### Issue: Python cache files everywhere
**Solution:** These are normal `.pyc` files in `__pycache__/`. They're in `.gitignore` and safe to ignore.

## 📦 Dependencies

```
psycopg2-binary==2.9.11      # PostgreSQL adapter (if needed)
pandas==2.3.3                # Data manipulation
requests==2.31.0             # HTTP library
python-dotenv==1.0.0         # Environment variables
pandas-ta==0.4.71b0          # Technical analysis (future use)
yfinance==1.0                # Yahoo Finance API
numpy==2.2.6                 # Numerical computing
flask==3.1.0                 # Web framework
flask-cors==5.0.0            # CORS support
```

## 🔄 How It Works

1. **Backend (Flask)**
   - `main.py` starts the Flask server
   - `api.py` fetches stock data from yfinance
   - Data is cached for 60 seconds
   - REST API serves JSON to frontend

2. **Frontend (Vanilla JS)**
   - `index.html` fetches data from API
   - Renders stock cards with metrics
   - Handles search, filtering, sorting
   - Auto-refreshes every 60 seconds

3. **Data Flow**
   ```
   Yahoo Finance API → yfinance → Flask API → Frontend → User
   ```

## 🎨 UI Features

- **Terminal Aesthetic** - Monospace fonts, tech-inspired design
- **Color-Coded Valuations** - Visual P/E and PEG indicators
- **Responsive Grid** - Adapts to screen size
- **Live Status Indicator** - API connection status
- **Real-Time Clock** - Current timestamp display
- **Pagination** - Navigate large datasets easily

## 🚦 API Health Check

Check if the server is running:
```bash
curl http://localhost:5001/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-29T13:45:09.924012",
  "exchanges": ["nyse", "crypto", "etf", "tsx", "nse"],
  "total_symbols": 95
}
```

## 🔮 Future Enhancements

- [ ] Add historical price charts
- [ ] Include dividend yield data
- [ ] Add more international exchanges (LSE, HKEX, etc.)
- [ ] Implement user watchlists
- [ ] Add stock comparison tool
- [ ] Export to CSV/Excel
- [ ] Dark/light theme toggle
- [ ] Mobile app version
- [ ] Real-time WebSocket updates
- [ ] Machine learning price predictions

## 📝 License

MIT License - Feel free to use this project for personal or commercial purposes.

## 🤝 Contributing

This is a personal project, but suggestions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📞 Support

For issues or questions:
- Check the Troubleshooting section
- Review the code comments in `api.py` and `index.html`
- Ensure all dependencies are installed correctly

## 🙏 Acknowledgments

- **yfinance** - Yahoo Finance API wrapper
- **Flask** - Python web framework
- **Pandas** - Data manipulation library

---

**Built with ❤️ for stock market enthusiasts**

*Last Updated: December 29, 2025*
