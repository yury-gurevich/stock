"""Portfolio Tracker — In-memory portfolio management with P&L tracking."""
from datetime import datetime


class Portfolio:
    def __init__(self, initial_cash=100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {}
        self.transactions = []

    def buy(self, ticker, shares, price, timestamp=None):
        cost = shares * price
        if cost > self.cash:
            return {"error": f"Insufficient cash. Need ${cost:.2f}, have ${self.cash:.2f}"}
        self.cash -= cost
        if ticker in self.holdings:
            h = self.holdings[ticker]
            total = h["shares"] + shares
            h["avg_cost"] = (h["avg_cost"] * h["shares"] + cost) / total
            h["shares"] = total
        else:
            self.holdings[ticker] = {"shares": shares, "avg_cost": price}
        tx = {"action": "BUY", "ticker": ticker, "shares": shares, "price": price,
              "total": cost, "timestamp": timestamp or datetime.now().isoformat()}
        self.transactions.append(tx)
        return tx

    def sell(self, ticker, shares, price, timestamp=None):
        if ticker not in self.holdings or self.holdings[ticker]["shares"] < shares:
            avail = self.holdings.get(ticker, {}).get("shares", 0)
            return {"error": f"Insufficient shares. Have {avail}, want to sell {shares}"}
        revenue = shares * price
        self.cash += revenue
        h = self.holdings[ticker]
        pnl = (price - h["avg_cost"]) * shares
        h["shares"] -= shares
        if h["shares"] == 0:
            del self.holdings[ticker]
        tx = {"action": "SELL", "ticker": ticker, "shares": shares, "price": price,
              "total": revenue, "pnl": pnl, "timestamp": timestamp or datetime.now().isoformat()}
        self.transactions.append(tx)
        return tx

    def get_summary(self, current_prices):
        holdings_detail = []
        total_market_value = 0
        total_unrealized_pnl = 0
        for ticker, h in self.holdings.items():
            current = current_prices.get(ticker, h["avg_cost"])
            mv = h["shares"] * current
            unrealized = (current - h["avg_cost"]) * h["shares"]
            pnl_pct = ((current - h["avg_cost"]) / h["avg_cost"]) * 100 if h["avg_cost"] else 0
            holdings_detail.append({
                "ticker": ticker, "shares": h["shares"],
                "avg_cost": round(h["avg_cost"], 2), "current_price": round(current, 2),
                "market_value": round(mv, 2), "unrealized_pnl": round(unrealized, 2),
                "pnl_pct": round(pnl_pct, 2),
            })
            total_market_value += mv
            total_unrealized_pnl += unrealized

        total_value = self.cash + total_market_value
        realized_pnl = sum(tx.get("pnl", 0) for tx in self.transactions if tx["action"] == "SELL")
        return {
            "cash": round(self.cash, 2),
            "holdings_value": round(total_market_value, 2),
            "total_value": round(total_value, 2),
            "initial_cash": self.initial_cash,
            "total_return": round((total_value - self.initial_cash) / self.initial_cash * 100, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(total_unrealized_pnl, 2),
            "holdings": holdings_detail,
        }
