"""
Signal Generator — Combines technical indicators into Buy/Sell/Hold signals.
"""
import pandas as pd
import numpy as np


def generate_ma_signals(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    signals = []
    score = 0

    if "sma_50" in df.columns and "sma_200" in df.columns:
        if latest["sma_50"] > latest["sma_200"]:
            if prev["sma_50"] <= prev["sma_200"]:
                signals.append("Golden Cross (SMA50 crossed above SMA200) — strong BUY")
                score += 3
            else:
                signals.append("SMA50 above SMA200 — bullish trend")
                score += 1
        else:
            if prev["sma_50"] >= prev["sma_200"]:
                signals.append("Death Cross (SMA50 crossed below SMA200) — strong SELL")
                score -= 3
            else:
                signals.append("SMA50 below SMA200 — bearish trend")
                score -= 1

    if "sma_20" in df.columns:
        if latest["close"] > latest["sma_20"]:
            signals.append("Price above SMA20 — short-term bullish")
            score += 1
        else:
            signals.append("Price below SMA20 — short-term bearish")
            score -= 1

    if "ema_12" in df.columns and "ema_26" in df.columns:
        if latest["ema_12"] > latest["ema_26"] and prev["ema_12"] <= prev["ema_26"]:
            signals.append("EMA12 crossed above EMA26 — BUY signal")
            score += 2
        elif latest["ema_12"] < latest["ema_26"] and prev["ema_12"] >= prev["ema_26"]:
            signals.append("EMA12 crossed below EMA26 — SELL signal")
            score -= 2

    return {"signals": signals, "score": score, "method": "Moving Averages"}


def generate_rsi_signals(df):
    signals = []
    score = 0
    if "rsi_14" not in df.columns:
        return {"signals": [], "score": 0, "method": "RSI"}

    rsi_val = float(df["rsi_14"].iloc[-1])
    if rsi_val < 30:
        signals.append(f"RSI={rsi_val:.1f} — oversold, potential BUY")
        score += 2
    elif rsi_val < 40:
        signals.append(f"RSI={rsi_val:.1f} — approaching oversold")
        score += 1
    elif rsi_val > 70:
        signals.append(f"RSI={rsi_val:.1f} — overbought, potential SELL")
        score -= 2
    elif rsi_val > 60:
        signals.append(f"RSI={rsi_val:.1f} — approaching overbought")
        score -= 1
    else:
        signals.append(f"RSI={rsi_val:.1f} — neutral range")

    return {"signals": signals, "score": score, "method": "RSI"}


def generate_macd_signals(df):
    signals = []
    score = 0
    if "macd" not in df.columns:
        return {"signals": [], "score": 0, "method": "MACD"}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    if latest["macd"] > latest["macd_signal"] and prev["macd"] <= prev["macd_signal"]:
        signals.append("MACD crossed above signal — bullish crossover BUY")
        score += 2
    elif latest["macd"] < latest["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
        signals.append("MACD crossed below signal — bearish crossover SELL")
        score -= 2

    hist = latest["macd_histogram"]
    if hist > 0:
        signals.append(f"MACD histogram positive ({hist:.4f}) — bullish momentum")
        score += 1
    else:
        signals.append(f"MACD histogram negative ({hist:.4f}) — bearish momentum")
        score -= 1

    if latest["macd"] > 0 and prev["macd"] <= 0:
        signals.append("MACD crossed above zero line — trend turning bullish")
        score += 1
    elif latest["macd"] < 0 and prev["macd"] >= 0:
        signals.append("MACD crossed below zero line — trend turning bearish")
        score -= 1

    return {"signals": signals, "score": score, "method": "MACD"}


def generate_bb_signals(df):
    signals = []
    score = 0
    if "bb_upper" not in df.columns:
        return {"signals": [], "score": 0, "method": "Bollinger Bands"}

    latest = df.iloc[-1]
    close = latest["close"]

    if close >= latest["bb_upper"]:
        signals.append("Price at/above upper Bollinger Band — overbought, potential reversal SELL")
        score -= 2
    elif close <= latest["bb_lower"]:
        signals.append("Price at/below lower Bollinger Band — oversold, potential reversal BUY")
        score += 2
    else:
        pct_b = latest.get("bb_percent_b", 0.5)
        if pct_b > 0.8:
            signals.append(f"%B={pct_b:.2f} — near upper band, watch for reversal")
            score -= 1
        elif pct_b < 0.2:
            signals.append(f"%B={pct_b:.2f} — near lower band, watch for bounce")
            score += 1
        else:
            signals.append(f"%B={pct_b:.2f} — mid-band, neutral")

    bw = latest.get("bb_bandwidth", None)
    if bw is not None and bw < 0.05:
        signals.append(f"Bollinger squeeze (BW={bw:.4f}) — expect volatility breakout")

    return {"signals": signals, "score": score, "method": "Bollinger Bands"}


def generate_composite_signal(df, ml_signal=None, mc_signal=None):
    """Generate a composite signal from all methods."""
    ma = generate_ma_signals(df)
    rsi_s = generate_rsi_signals(df)
    macd_s = generate_macd_signals(df)
    bb = generate_bb_signals(df)

    methods = [ma, rsi_s, macd_s, bb]

    if ml_signal:
        ml_score = 0
        if ml_signal.get("signal") == "BUY":
            ml_score = 2
        elif ml_signal.get("signal") == "SELL":
            ml_score = -2
        methods.append({
            "signals": [f"LSTM predicts {ml_signal['signal']} ({ml_signal.get('predicted_change_pct', 0):.1f}% change)"],
            "score": ml_score,
            "method": "LSTM Neural Network",
        })

    if mc_signal:
        mc_score = 0
        prob = mc_signal.get("prob_profit", 0.5)
        if prob > 0.6:
            mc_score = 2
        elif prob < 0.4:
            mc_score = -2
        methods.append({
            "signals": [f"Monte Carlo: {prob*100:.0f}% probability of profit"],
            "score": mc_score,
            "method": "Monte Carlo Simulation",
        })

    total_score = sum(m["score"] for m in methods)
    max_possible = sum(abs(m["score"]) for m in methods) or 1
    normalized = (total_score / max_possible) * 100 if max_possible > 0 else 0

    if normalized > 25:
        overall = "BUY"
    elif normalized < -25:
        overall = "SELL"
    else:
        overall = "HOLD"

    confidence = min(50 + abs(normalized) * 0.5, 95)

    return {
        "overall_signal": overall,
        "composite_score": round(normalized, 1),
        "confidence": round(confidence, 1),
        "methods": methods,
        "summary": f"{overall} signal with {confidence:.0f}% confidence (score: {normalized:.1f}/100)",
    }
