# QuantDesk — Stock Analysis Platform

Full-stack trading analysis platform with multiple prediction methods, portfolio tracking, and strategy backtesting.

## Quick Start

```bash
chmod +x run.sh
./run.sh
```

Then open **http://localhost:8000** in your browser.

## Features

### Prediction Methods
- **Moving Averages** — SMA (20/50/200), EMA (12/26), Golden Cross / Death Cross
- **RSI / MACD / Bollinger Bands** — Full indicator suite with signal generation
- **Monte Carlo Simulation** — GBM price forecasting with confidence intervals
- **LSTM Neural Network** — Deep learning prediction (falls back to Ridge if TensorFlow unavailable)

### Core Capabilities
- **Composite Buy/Sell/Hold Signals** — Weighted scoring across all methods
- **Portfolio Tracker** — Virtual portfolio with P&L, unrealized gains, transaction history
- **Strategy Backtester** — 5 strategies with equity curves, Sharpe ratio, max drawdown
- **Interactive Dashboard** — Charts with overlaid indicators, signal meters, trade logs

### Data
- **Yahoo Finance** — Any global exchange (US, ASX, TSE, LSE, etc.)

## Project Structure

```
stock-app/
├── backend/
│   ├── app.py              # FastAPI REST API (20 endpoints)
│   ├── data_fetcher.py     # Yahoo Finance data
│   ├── indicators.py       # SMA, EMA, RSI, MACD, Bollinger
│   ├── monte_carlo.py      # Monte Carlo simulation
│   ├── ml_model.py         # LSTM / Ridge prediction
│   ├── signals.py          # Composite signal generation
│   ├── portfolio.py        # Portfolio tracker
│   ├── backtester.py       # Strategy backtesting engine
│   └── requirements.txt
├── frontend/
│   └── index.html          # React SPA dashboard
├── run.sh                  # One-click startup
└── README.md
```

## API Endpoints

### Core Routes

| Endpoint | Method | Description |
|---|---|---|
| `/api/search?q=` | GET | Search tickers |
| `/api/stock/{ticker}` | GET | Stock metadata + historical data |
| `/api/stock/{ticker}/data` | GET | Historical OHLCV |
| `/api/stock/{ticker}/analysis` | GET | Indicators + signals |
| `/api/stock/{ticker}/monte-carlo` | GET | Monte Carlo sim |
| `/api/stock/{ticker}/predict` | GET | LSTM prediction |
| `/api/stock/{ticker}/full-signal` | GET | All-methods signal |

### Frontend-Compatible Routes

| Endpoint | Method | Description |
|---|---|---|
| `/api/indicators/{ticker}` | GET | Indicators + signals (flat format) |
| `/api/montecarlo/{ticker}` | GET | Monte Carlo with date-stamped forecast |
| `/api/ml/{ticker}` | GET | ML prediction (flat format) |
| `/api/backtest/{ticker}` | GET | Backtest via GET with portfolio history |

### Portfolio

| Endpoint | Method | Description |
|---|---|---|
| `/api/portfolio` | GET | Default portfolio summary |
| `/api/portfolio/{id}` | GET | Portfolio summary by ID |
| `/api/portfolio/buy` | POST | Buy shares (price auto-fetched if omitted) |
| `/api/portfolio/sell` | POST | Sell shares (price auto-fetched if omitted) |
| `/api/portfolio/reset` | POST | Reset portfolio |

### Backtesting

| Endpoint | Method | Description |
|---|---|---|
| `/api/backtest` | POST | Run strategy backtest |
| `/api/strategies` | GET | List strategies |

## Requirements

- Python 3.10+
- Internet connection (Yahoo Finance)
- Conda environment recommended (see `requirements.txt`)

## Disclaimer

Educational/research tool only. Not financial advice.
