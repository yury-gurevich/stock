"""
Data Fetcher — Yahoo Finance wrapper for global stock data.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Optional


def fetch_stock_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker.
    """
    stock = yf.Ticker(ticker)

    if start:
        df = stock.history(start=start, end=end or datetime.now().strftime("%Y-%m-%d"), interval=interval)
    else:
        df = stock.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    df.index = df.index.tz_localize(None)
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def fetch_stock_info(ticker: str) -> dict:
    """Fetch metadata/info for a ticker."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "symbol": info.get("symbol", ticker),
        "name": info.get("longName") or info.get("shortName", ticker),
        "currency": info.get("currency", "USD"),
        "exchange": info.get("exchange", ""),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "market_cap": info.get("marketCap"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "previous_close": info.get("previousClose"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
    }


def search_tickers(query: str) -> list:
    """Search for tickers matching a query string."""
    try:
        import requests
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.json()
        return [
            {
                "symbol": q.get("symbol"),
                "name": q.get("longname") or q.get("shortname", ""),
                "exchange": q.get("exchange", ""),
                "type": q.get("quoteType", ""),
            }
            for q in data.get("quotes", [])
        ]
    except Exception:
        return []
