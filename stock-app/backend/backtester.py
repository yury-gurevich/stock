"""Backtester — Test trading strategies on historical data."""
import pandas as pd
import numpy as np
from indicators import compute_all_indicators
from signals import generate_ma_signals, generate_rsi_signals, generate_macd_signals, generate_bb_signals


STRATEGIES = {
    "ma_crossover": "Moving Average Crossover (EMA12/EMA26)",
    "rsi_reversal": "RSI Reversal (Buy <30, Sell >70)",
    "macd_crossover": "MACD Signal Line Crossover",
    "bollinger_bounce": "Bollinger Band Bounce",
    "combined": "Combined Strategy (majority vote)",
}


def _get_signal(window, strategy):
    """Get trading signal for a given strategy from a data window."""
    if strategy == "ma_crossover":
        s = generate_ma_signals(window)
        return "BUY" if s["score"] >= 2 else "SELL" if s["score"] <= -2 else "HOLD"
    elif strategy == "rsi_reversal":
        s = generate_rsi_signals(window)
        return "BUY" if s["score"] >= 2 else "SELL" if s["score"] <= -2 else "HOLD"
    elif strategy == "macd_crossover":
        s = generate_macd_signals(window)
        return "BUY" if s["score"] >= 2 else "SELL" if s["score"] <= -2 else "HOLD"
    elif strategy == "bollinger_bounce":
        s = generate_bb_signals(window)
        return "BUY" if s["score"] >= 2 else "SELL" if s["score"] <= -2 else "HOLD"
    elif strategy == "combined":
        total = sum([
            generate_ma_signals(window)["score"],
            generate_rsi_signals(window)["score"],
            generate_macd_signals(window)["score"],
            generate_bb_signals(window)["score"],
        ])
        return "BUY" if total >= 3 else "SELL" if total <= -3 else "HOLD"
    return "HOLD"


def backtest_strategy(
    df,
    strategy="combined",
    initial_cash=100000.0,
    position_size_pct=0.25,
    stop_loss_pct=0.05,
    take_profit_pct=0.10,
):
    """
    Run a backtest on historical data with a given strategy.
    Returns trade log, equity curve, and performance metrics.
    """
    df = compute_all_indicators(df)
    required = ["sma_50", "sma_200", "rsi_14", "macd", "bb_upper"]
    valid = df.dropna(subset=[c for c in required if c in df.columns])
    if len(valid) < 10:
        raise ValueError("Insufficient data for backtesting after indicator computation")
    df = valid.reset_index(drop=True)

    cash = initial_cash
    shares = 0
    entry_price = 0.0
    trades = []
    equity_curve = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        window = df.iloc[max(0, i - 5):i + 1]
        price = float(row["close"])

        signal = _get_signal(window, strategy)

        # Stop loss / take profit
        if shares > 0:
            change = (price - entry_price) / entry_price
            if change <= -stop_loss_pct or change >= take_profit_pct:
                signal = "SELL"

        if signal == "BUY" and shares == 0:
            buy_amount = cash * position_size_pct
            shares = int(buy_amount / price)
            if shares > 0:
                cost = shares * price
                cash -= cost
                entry_price = price
                trades.append({"date": str(row["date"]), "action": "BUY", "price": price, "shares": shares, "value": cost})

        elif signal == "SELL" and shares > 0:
            revenue = shares * price
            pnl = revenue - (shares * entry_price)
            pnl_pct = (price - entry_price) / entry_price * 100
            cash += revenue
            trades.append({"date": str(row["date"]), "action": "SELL", "price": price, "shares": shares,
                           "value": revenue, "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2)})
            shares = 0
            entry_price = 0.0

        equity = cash + (shares * price)
        equity_curve.append({"date": str(row["date"]), "equity": round(equity, 2), "price": price})

    # Close out remaining position
    if shares > 0:
        cash += shares * float(df.iloc[-1]["close"])
        shares = 0

    final_equity = cash
    total_return = (final_equity - initial_cash) / initial_cash * 100

    sell_trades = [t for t in trades if t["action"] == "SELL"]
    winners = [t for t in sell_trades if t.get("pnl", 0) > 0]
    losers = [t for t in sell_trades if t.get("pnl", 0) <= 0]
    win_rate = len(winners) / len(sell_trades) * 100 if sell_trades else 0
    avg_win = float(np.mean([t["pnl"] for t in winners])) if winners else 0
    avg_loss = float(np.mean([t["pnl"] for t in losers])) if losers else 0

    # Max drawdown
    equities = [e["equity"] for e in equity_curve]
    peak = equities[0] if equities else initial_cash
    max_dd = 0
    for eq in equities:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd: max_dd = dd

    # Sharpe
    if len(equities) > 1:
        rets = pd.Series(equities).pct_change().dropna()
        sharpe = float((rets.mean() / rets.std()) * np.sqrt(252)) if rets.std() > 0 else 0
    else:
        sharpe = 0

    bh_return = (float(df.iloc[-1]["close"]) - float(df.iloc[0]["close"])) / float(df.iloc[0]["close"]) * 100

    # Downsample equity curve for frontend
    step = max(1, len(equity_curve) // 200)

    return {
        "strategy": strategy,
        "strategy_name": STRATEGIES.get(strategy, strategy),
        "period": {"start": str(df.iloc[0]["date"]), "end": str(df.iloc[-1]["date"]), "trading_days": len(df)},
        "performance": {
            "initial_cash": initial_cash,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return, 2),
            "buy_hold_return_pct": round(bh_return, 2),
            "alpha_pct": round(total_return - bh_return, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "total_trades": len(sell_trades),
            "win_rate_pct": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
        },
        "trades": trades,
        "equity_curve": equity_curve[::step],
    }
