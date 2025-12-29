"""
FinanceBro - Simple stock data viewer
Just fetches stocks from yfinance and serves them via Flask API to index.html
"""
from api import app

if __name__ == '__main__':
    print("Starting FinanceBro...")
    print("Open http://localhost:5001 in your browser")
    print("Or view index.html which connects to the API")
    app.run(debug=True, port=5001)
