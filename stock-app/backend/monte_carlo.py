"""Monte Carlo Simulation — Price path forecasting with GBM."""
import numpy as np
import pandas as pd


def monte_carlo_simulation(prices, days_ahead=30, num_simulations=1000, seed=None):
    """
    Run Monte Carlo simulation using Geometric Brownian Motion.
    Returns simulation results, statistics and percentiles.
    """
    if seed is not None:
        np.random.seed(seed)

    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()
    last_price = float(prices.iloc[-1])

    dt = 1
    random_walks = np.random.normal(
        loc=(mu - 0.5 * sigma**2) * dt,
        scale=sigma * np.sqrt(dt),
        size=(num_simulations, days_ahead),
    )

    price_paths = last_price * np.exp(np.cumsum(random_walks, axis=1))
    start_col = np.full((num_simulations, 1), last_price)
    price_paths = np.hstack([start_col, price_paths])

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    stats = {}
    for p in percentiles:
        stats[f"p{p}"] = np.percentile(price_paths, p, axis=0).tolist()
    stats["mean"] = np.mean(price_paths, axis=0).tolist()

    final_prices = price_paths[:, -1]
    prob_profit = float(np.mean(final_prices > last_price))
    expected_return = float((np.mean(final_prices) - last_price) / last_price * 100)

    return {
        "last_price": last_price,
        "days_ahead": days_ahead,
        "num_simulations": num_simulations,
        "stats": stats,
        "final_distribution": {
            "mean": float(np.mean(final_prices)),
            "median": float(np.median(final_prices)),
            "min": float(np.min(final_prices)),
            "max": float(np.max(final_prices)),
            "prob_profit": prob_profit,
            "expected_return_pct": expected_return,
        },
        "sample_paths": price_paths[:min(50, num_simulations)].tolist(),
    }
