"""
Technical Indicators — SMA, EMA, RSI, MACD, Bollinger Bands.
"""
import pandas as pd
import numpy as np


def sma(series, window):
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=1).mean()


def ema(series, span):
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal_period=9):
    """MACD indicator with signal and histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def bollinger_bands(series, window=20, num_std=2.0):
    """Bollinger Bands."""
    middle = sma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle
    percent_b = (series - lower) / (upper - lower)
    return {
        "upper": upper, "middle": middle, "lower": lower,
        "bandwidth": bandwidth, "percent_b": percent_b,
    }


def compute_all_indicators(df):
    """Compute all technical indicators and append as columns."""
    out = df.copy()
    close = out["close"]

    out["sma_20"] = sma(close, 20)
    out["sma_50"] = sma(close, 50)
    out["sma_200"] = sma(close, 200)
    out["ema_12"] = ema(close, 12)
    out["ema_26"] = ema(close, 26)

    out["rsi_14"] = rsi(close, 14)

    macd_data = macd(close)
    out["macd"] = macd_data["macd"]
    out["macd_signal"] = macd_data["signal"]
    out["macd_histogram"] = macd_data["histogram"]

    bb = bollinger_bands(close)
    out["bb_upper"] = bb["upper"]
    out["bb_middle"] = bb["middle"]
    out["bb_lower"] = bb["lower"]
    out["bb_bandwidth"] = bb["bandwidth"]
    out["bb_percent_b"] = bb["percent_b"]

    return out
