"""
QuantDesk — Stock Analysis Platform API
FastAPI backend tying all modules together.
"""
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data_fetcher import fetch_stock_data, fetch_stock_info, search_tickers
from indicators import compute_all_indicators
from monte_carlo import monte_carlo_simulation
from signals import generate_composite_signal
from portfolio import Portfolio
from backtester import backtest_strategy, STRATEGIES

app = FastAPI(title="QuantDesk", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory portfolio store
portfolios = {"default": Portfolio(100000.0)}


# ── Pydantic Models ───────────────────────────────────────────────────
class TradeRequest(BaseModel):
    ticker: str
    shares: int
    price: float
    portfolio_id: str = "default"


class BacktestRequest(BaseModel):
    ticker: str
    strategy: str = "combined"
    period: str = "2y"
    initial_cash: float = 100000.0
    position_size_pct: float = 0.25
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10


# ── Helpers ───────────────────────────────────────────────────────────
def clean_for_json(obj):
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else v
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def df_to_records(df):
    """Convert DataFrame to JSON-safe list of dicts."""
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if k == "date":
                r[k] = str(v)
            elif isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
                r[k] = None
    return records


# ── API Endpoints ─────────────────────────────────────────────────────

@app.get("/api/search")
async def api_search(q: str):
    """Search for stock tickers."""
    results = search_tickers(q)
    return {"results": results}


@app.get("/api/stock/{ticker}")
async def api_stock_info(ticker: str):
    """Get stock metadata."""
    try:
        info = fetch_stock_info(ticker)
        return clean_for_json(info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stock/{ticker}/data")
async def api_stock_data(ticker: str, period: str = "1y", interval: str = "1d"):
    """Fetch historical OHLCV data."""
    try:
        df = fetch_stock_data(ticker, period=period, interval=interval)
        return {"ticker": ticker, "count": len(df), "data": df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stock/{ticker}/analysis")
async def api_analysis(ticker: str, period: str = "1y"):
    """Full analysis: indicators + signals."""
    try:
        df = fetch_stock_data(ticker, period=period)
        df = compute_all_indicators(df)
        signals = generate_composite_signal(df)
        chart_data = df_to_records(df.tail(min(len(df), 300)))

        latest = df.iloc[-1]
        latest_info = {
            "close": float(latest["close"]),
            "rsi": float(latest.get("rsi_14", 0)) if pd.notna(latest.get("rsi_14")) else None,
            "macd": float(latest.get("macd", 0)) if pd.notna(latest.get("macd")) else None,
            "sma_20": float(latest.get("sma_20", 0)) if pd.notna(latest.get("sma_20")) else None,
            "sma_50": float(latest.get("sma_50", 0)) if pd.notna(latest.get("sma_50")) else None,
        }

        return clean_for_json({"ticker": ticker, "signals": signals, "chart_data": chart_data, "latest": latest_info})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stock/{ticker}/monte-carlo")
async def api_monte_carlo(ticker: str, days: int = 30, simulations: int = 1000, period: str = "1y"):
    """Run Monte Carlo simulation."""
    try:
        df = fetch_stock_data(ticker, period=period)
        result = monte_carlo_simulation(df["close"], days_ahead=days, num_simulations=simulations)
        return clean_for_json({"ticker": ticker, **result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stock/{ticker}/predict")
async def api_predict(ticker: str, period: str = "2y", horizon: int = 5, epochs: int = 30):
    """Run ML prediction."""
    try:
        from ml_model import train_and_predict
        df = fetch_stock_data(ticker, period=period)
        df = compute_all_indicators(df)
        result = train_and_predict(df, forecast_horizon=horizon, epochs=epochs)
        return clean_for_json({"ticker": ticker, **result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stock/{ticker}/full-signal")
async def api_full_signal(ticker: str, period: str = "1y", run_ml: bool = False, run_mc: bool = True):
    """Comprehensive signal using all available methods."""
    try:
        df = fetch_stock_data(ticker, period=period)
        df = compute_all_indicators(df)

        ml_signal = None
        mc_signal = None

        if run_mc:
            mc = monte_carlo_simulation(df["close"], days_ahead=30, num_simulations=500)
            mc_signal = mc["final_distribution"]

        if run_ml:
            from ml_model import train_and_predict
            df_2y = fetch_stock_data(ticker, period="2y")
            df_2y = compute_all_indicators(df_2y)
            ml_result = train_and_predict(df_2y)
            ml_signal = ml_result["forecast"]

        signals = generate_composite_signal(df, ml_signal=ml_signal, mc_signal=mc_signal)
        return clean_for_json({"ticker": ticker, "signals": signals})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Portfolio Endpoints ───────────────────────────────────────────────

@app.post("/api/portfolio/buy")
async def api_portfolio_buy(req: TradeRequest):
    if req.portfolio_id not in portfolios:
        portfolios[req.portfolio_id] = Portfolio()
    result = portfolios[req.portfolio_id].buy(req.ticker.upper(), req.shares, req.price)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/portfolio/sell")
async def api_portfolio_sell(req: TradeRequest):
    if req.portfolio_id not in portfolios:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    result = portfolios[req.portfolio_id].sell(req.ticker.upper(), req.shares, req.price)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/portfolio/{portfolio_id}")
async def api_portfolio_summary(portfolio_id: str):
    if portfolio_id not in portfolios:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    port = portfolios[portfolio_id]
    current_prices = {}
    for ticker in port.holdings:
        try:
            info = fetch_stock_info(ticker)
            current_prices[ticker] = info.get("current_price") or info.get("previous_close", 0)
        except Exception:
            current_prices[ticker] = port.holdings[ticker]["avg_cost"]
    summary = port.get_summary(current_prices)
    summary["transactions"] = port.transactions[-50:]
    return clean_for_json(summary)


@app.post("/api/portfolio/reset")
async def api_portfolio_reset(initial_cash: float = 100000.0):
    portfolios["default"] = Portfolio(initial_cash)
    return {"message": "Portfolio reset", "initial_cash": initial_cash}


# ── Backtesting ───────────────────────────────────────────────────────

@app.get("/api/strategies")
async def api_strategies():
    return {"strategies": STRATEGIES}


@app.post("/api/backtest")
async def api_backtest(req: BacktestRequest):
    try:
        df = fetch_stock_data(req.ticker, period=req.period)
        result = backtest_strategy(
            df, strategy=req.strategy, initial_cash=req.initial_cash,
            position_size_pct=req.position_size_pct,
            stop_loss_pct=req.stop_loss_pct, take_profit_pct=req.take_profit_pct,
        )
        return clean_for_json({"ticker": req.ticker, **result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Serve Frontend ───────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

@app.get("/")
async def serve_frontend():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "QuantDesk API running. Frontend not found at " + FRONTEND_DIR}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
