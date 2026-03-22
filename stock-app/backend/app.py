"""
QuantDesk — Stock Analysis Platform API
FastAPI backend tying all modules together.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data_fetcher import fetch_stock_data, fetch_stock_info, search_tickers
from indicators import compute_all_indicators
from ml_model import train_and_predict
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
    """Request body for buy/sell trade operations."""

    ticker: str
    shares: int
    price: Optional[float] = None
    portfolio_id: str = "default"


class BacktestRequest(BaseModel):
    """Request body for backtesting configuration."""

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
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(obj, (np.integer,)):
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
async def api_stock_info(ticker: str, period: str = "1y"):
    """Get stock metadata and historical data combined."""
    try:
        info = fetch_stock_info(ticker)
        df = fetch_stock_data(ticker, period=period)
        meta = {
            "name": info.get("name", ticker),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("market_cap"),
            "pe_ratio": info.get("pe_ratio") or info.get("previous_close"),
            "current_price": info.get("current_price"),
        }
        return clean_for_json({
            "ticker": info.get("symbol", ticker),
            "meta": meta,
            "data": df_to_records(df),
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/stock/{ticker}/data")
async def api_stock_data(ticker: str, period: str = "1y", interval: str = "1d"):
    """Fetch historical OHLCV data."""
    try:
        df = fetch_stock_data(ticker, period=period, interval=interval)
        return {"ticker": ticker, "count": len(df), "data": df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


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
            "rsi": (float(latest.get("rsi_14", 0))
                    if pd.notna(latest.get("rsi_14")) else None),
            "macd": (float(latest.get("macd", 0))
                     if pd.notna(latest.get("macd")) else None),
            "sma_20": (float(latest.get("sma_20", 0))
                       if pd.notna(latest.get("sma_20")) else None),
            "sma_50": (float(latest.get("sma_50", 0))
                       if pd.notna(latest.get("sma_50")) else None),
        }

        return clean_for_json({
            "ticker": ticker,
            "signals": signals,
            "chart_data": chart_data,
            "latest": latest_info,
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/stock/{ticker}/monte-carlo")
async def api_monte_carlo(
    ticker: str, days: int = 30,
    simulations: int = 1000, period: str = "1y",
):
    """Run Monte Carlo simulation."""
    try:
        df = fetch_stock_data(ticker, period=period)
        result = monte_carlo_simulation(
            df["close"], days_ahead=days,
            num_simulations=simulations,
        )
        return clean_for_json({"ticker": ticker, **result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/stock/{ticker}/predict")
async def api_predict(ticker: str, period: str = "2y", horizon: int = 5, epochs: int = 30):
    """Run ML prediction."""
    try:
        df = fetch_stock_data(ticker, period=period)
        df = compute_all_indicators(df)
        result = train_and_predict(df, forecast_horizon=horizon, epochs=epochs)
        return clean_for_json({"ticker": ticker, **result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/stock/{ticker}/full-signal")
async def api_full_signal(
    ticker: str, period: str = "1y",
    run_ml: bool = False, run_mc: bool = True,
):
    """Comprehensive signal using all available methods."""
    try:
        df = fetch_stock_data(ticker, period=period)
        df = compute_all_indicators(df)

        ml_signal = None
        mc_signal = None

        if run_mc:
            mc = monte_carlo_simulation(
                df["close"], days_ahead=30,
                num_simulations=500,
            )
            mc_signal = mc["final_distribution"]

        if run_ml:
            df_2y = fetch_stock_data(ticker, period="2y")
            df_2y = compute_all_indicators(df_2y)
            ml_result = train_and_predict(df_2y)
            ml_signal = ml_result["forecast"]

        signals = generate_composite_signal(
            df, ml_signal=ml_signal, mc_signal=mc_signal,
        )
        return clean_for_json({"ticker": ticker, "signals": signals})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# ── Portfolio Endpoints ───────────────────────────────────────────────

@app.post("/api/portfolio/buy")
async def api_portfolio_buy(req: TradeRequest):
    """Buy shares and add to portfolio."""
    if req.portfolio_id not in portfolios:
        portfolios[req.portfolio_id] = Portfolio()
    price = req.price
    if price is None:
        info = fetch_stock_info(req.ticker)
        price = info.get("current_price") or info.get("previous_close", 0)
    result = portfolios[req.portfolio_id].buy(req.ticker.upper(), req.shares, price)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    result["price"] = price
    return result


@app.post("/api/portfolio/sell")
async def api_portfolio_sell(req: TradeRequest):
    """Sell shares from portfolio."""
    if req.portfolio_id not in portfolios:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    price = req.price
    if price is None:
        info = fetch_stock_info(req.ticker)
        price = info.get("current_price") or info.get("previous_close", 0)
    result = portfolios[req.portfolio_id].sell(req.ticker.upper(), req.shares, price)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    result["price"] = price
    return result


@app.get("/api/portfolio/{portfolio_id}")
async def api_portfolio_summary(portfolio_id: str):
    """Get portfolio summary with current market prices."""
    if portfolio_id not in portfolios:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    port = portfolios[portfolio_id]
    current_prices = {}
    for ticker in port.holdings:
        try:
            info = fetch_stock_info(ticker)
            current_prices[ticker] = (
                info.get("current_price")
                or info.get("previous_close", 0)
            )
        except (OSError, ValueError, KeyError):
            current_prices[ticker] = port.holdings[ticker]["avg_cost"]
    summary = port.get_summary(current_prices)
    # Map field names for frontend compatibility
    summary["total_portfolio_value"] = summary["total_value"]
    for h in summary.get("holdings", []):
        h["avg_price"] = h.pop("avg_cost", 0)
    summary["recent_transactions"] = [
        {
            "type": tx.get("action", ""),
            "ticker": tx.get("ticker", ""),
            "shares": tx.get("shares", 0),
            "price": tx.get("price", 0),
            "total": tx.get("total", 0),
            "date": tx.get("timestamp", ""),
        }
        for tx in port.transactions[-50:]
    ]
    return clean_for_json(summary)


@app.post("/api/portfolio/reset")
async def api_portfolio_reset(initial_cash: float = 100000.0):
    """Reset default portfolio to initial state."""
    portfolios["default"] = Portfolio(initial_cash)
    return {"message": "Portfolio reset", "initial_cash": initial_cash}


# ── Backtesting ───────────────────────────────────────────────────────

@app.get("/api/strategies")
async def api_strategies():
    """List available backtesting strategies."""
    return {"strategies": STRATEGIES}


@app.post("/api/backtest")
async def api_backtest(req: BacktestRequest):
    """Run a backtest with the given configuration."""
    try:
        df = fetch_stock_data(req.ticker, period=req.period)
        result = backtest_strategy(
            df, strategy=req.strategy, initial_cash=req.initial_cash,
            position_size_pct=req.position_size_pct,
            stop_loss_pct=req.stop_loss_pct, take_profit_pct=req.take_profit_pct,
        )
        return clean_for_json({"ticker": req.ticker, **result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# ── Frontend-compatible Routes ────────────────────────────────────────

@app.get("/api/indicators/{ticker}")
async def api_indicators(ticker: str, period: str = "1y"):
    """Indicators + signals in the format the frontend expects."""
    try:
        df = fetch_stock_data(ticker, period=period)
        df = compute_all_indicators(df)
        signals = generate_composite_signal(df)
        chart_data = df_to_records(df.tail(min(len(df), 300)))

        latest = df.iloc[-1]
        rsi_val = (float(latest.get("rsi_14", 0))
                   if pd.notna(latest.get("rsi_14")) else None)
        has_smas = (pd.notna(latest.get("sma_20"))
                    and pd.notna(latest.get("sma_50")))
        ma_trend = ("bullish"
                    if has_smas and latest["sma_20"] > latest["sma_50"]
                    else "bearish")
        close = float(latest["close"])
        bb_upper = (float(latest.get("bb_upper", 0))
                    if pd.notna(latest.get("bb_upper")) else None)
        bb_lower = (float(latest.get("bb_lower", 0))
                    if pd.notna(latest.get("bb_lower")) else None)

        bb_position = "middle"
        if bb_upper and bb_lower:
            if close >= bb_upper:
                bb_position = "above_upper"
            elif close <= bb_lower:
                bb_position = "below_lower"

        # Rename columns for frontend compatibility
        for rec in chart_data:
            if "rsi_14" in rec:
                rec["rsi"] = rec.pop("rsi_14")
            if "macd_signal" in rec:
                rec["macd_signal_line"] = rec.pop("macd_signal")

        summary = {
            "action": signals["overall_signal"],
            "composite_score": signals["composite_score"],
            "rsi": rsi_val,
            "ma_trend": ma_trend,
            "bb_position": bb_position,
        }

        return clean_for_json({"data": chart_data, "summary": summary})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/montecarlo/{ticker}")
async def api_montecarlo_compat(
    ticker: str, period: str = "1y",
    forecast_days: int = 30, simulations: int = 500,
):
    """Monte Carlo simulation in frontend-expected format."""
    try:
        df = fetch_stock_data(ticker, period=period)
        result = monte_carlo_simulation(
            df["close"], days_ahead=forecast_days,
            num_simulations=simulations,
        )

        last_price = result["last_price"]
        final_dist = result["final_distribution"]
        stats_raw = result["stats"]

        # Build forecast array with dates
        last_date = pd.Timestamp(df.iloc[-1]["date"])
        forecast = []
        for i in range(forecast_days + 1):
            d = last_date + pd.Timedelta(days=i)
            forecast.append({
                "date": d.strftime("%Y-%m-%d"),
                "p5": stats_raw["p5"][i],
                "p25": stats_raw["p25"][i],
                "median": stats_raw["p50"][i],
                "p75": stats_raw["p75"][i],
                "p95": stats_raw["p95"][i],
            })

        stats = {
            "num_simulations": result["num_simulations"],
            "forecast_days": forecast_days,
            "current_price": last_price,
            "expected_return_pct": round(final_dist["expected_return_pct"], 2),
            "prob_profit": round(final_dist["prob_profit"] * 100, 1),
            "var_5_pct": round(
                (stats_raw["p5"][-1] - last_price)
                / last_price * 100, 2,
            ),
            "annualized_volatility": float(
                np.std(np.log(
                    np.array(stats_raw["p50"][1:])
                    / np.array(stats_raw["p50"][:-1])
                )) * np.sqrt(252)
            ),
            "best_case_95": stats_raw["p95"][-1],
            "expected_final_price": final_dist["mean"],
            "worst_case_5": stats_raw["p5"][-1],
        }

        return clean_for_json({"stats": stats, "forecast": forecast})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/ml/{ticker}")
async def api_ml_compat(ticker: str, period: str = "2y", forecast_days: int = 30):
    """ML prediction in frontend-expected format."""
    try:
        df = fetch_stock_data(ticker, period=period)
        df = compute_all_indicators(df)
        result = train_and_predict(df, forecast_horizon=forecast_days, epochs=30)
        return clean_for_json({"ticker": ticker, **result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/backtest/{ticker}")
async def api_backtest_compat(
    ticker: str, period: str = "2y", strategy: str = "combined",
    initial_capital: float = 100000.0,
):
    """Backtest via GET in frontend-expected format."""
    try:
        df = fetch_stock_data(ticker, period=period)
        result = backtest_strategy(
            df, strategy=strategy,
            initial_cash=initial_capital,
        )
        perf = result["performance"]

        # Build portfolio_history from equity_curve
        portfolio_history = []
        for entry in result.get("equity_curve", []):
            portfolio_history.append({
                "date": entry.get("date", ""),
                "portfolio_value": entry.get("equity", entry.get("portfolio_value", 0)),
                "price": entry.get("price", 0),
            })

        return clean_for_json({
            "ticker": ticker,
            "strategy": strategy,
            "initial_capital": initial_capital,
            "final_value": perf["final_equity"],
            "total_return_pct": perf["total_return_pct"],
            "buy_hold_return_pct": perf["buy_hold_return_pct"],
            "alpha": perf["alpha_pct"],
            "sharpe_ratio": perf["sharpe_ratio"],
            "max_drawdown_pct": perf["max_drawdown_pct"],
            "win_rate_pct": perf["win_rate_pct"],
            "total_trades": perf["total_trades"],
            "trades": result.get("trades", []),
            "portfolio_history": portfolio_history,
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/portfolio")
async def api_portfolio_default():
    """Get default portfolio summary."""
    return await api_portfolio_summary("default")


# ── Serve Frontend ───────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

@app.get("/")
async def serve_frontend():
    """Serve the frontend index.html."""
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "QuantDesk API running. Frontend not found at " + FRONTEND_DIR}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
