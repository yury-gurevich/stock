"""ML Model — LSTM or fallback Ridge regression for price prediction."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


def train_and_predict(df, lookback=60, forecast_horizon=5, epochs=30, batch_size=32):
    """
    Full pipeline: prepare data, train model, generate predictions.
    Tries LSTM (TensorFlow); falls back to Ridge regression if unavailable.
    """
    feature_cols = [c for c in ["close", "volume", "sma_20", "sma_50", "ema_12", "ema_26",
                                 "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower"]
                    if c in df.columns]
    if "close" not in feature_cols:
        raise ValueError("DataFrame must contain 'close' column")

    data = df[feature_cols].dropna().values
    if len(data) < lookback + 50:
        raise ValueError(f"Insufficient data: need {lookback + 50} rows, got {len(data)}")

    target_idx = feature_cols.index("close")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # Build sequences
    X, y = [], []
    for i in range(lookback, len(scaled) - forecast_horizon + 1):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i + forecast_horizon - 1, target_idx])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Try LSTM, fall back to Ridge
    use_lstm = False
    try:
        import tensorflow as tf
        tf.random.set_seed(42)
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
        y_pred = model.predict(X_test, verbose=0).flatten()
        future_pred = model.predict(X[-1:], verbose=0).flatten()
        use_lstm = True
        model_name = "LSTM Neural Network"
    except ImportError:
        from sklearn.linear_model import Ridge
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        model = Ridge(alpha=1.0)
        model.fit(X_train_flat, y_train)
        y_pred = model.predict(X_test_flat)
        future_pred = model.predict(X[-1:].reshape(1, -1))
        model_name = "Ridge Regression (TensorFlow unavailable)"

    # Inverse transform
    def inv(vals):
        dummy = np.zeros((len(vals), len(feature_cols)))
        dummy[:, target_idx] = vals
        return scaler.inverse_transform(dummy)[:, target_idx]

    y_test_inv = inv(y_test)
    y_pred_inv = inv(y_pred)
    future_price = float(inv(future_pred)[0])
    current_price = float(df["close"].dropna().iloc[-1])

    # Metrics
    mae = float(np.mean(np.abs(y_test_inv - y_pred_inv)))
    rmse = float(np.sqrt(np.mean((y_test_inv - y_pred_inv) ** 2)))
    mape = float(np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100)
    if len(y_test_inv) > 1:
        dir_acc = float(np.mean(np.sign(np.diff(y_test_inv)) == np.sign(np.diff(y_pred_inv))) * 100)
    else:
        dir_acc = 0.0

    change_pct = (future_price - current_price) / current_price * 100
    if change_pct > 2:
        signal = "BUY"
    elif change_pct < -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "model_name": model_name,
        "metrics": {"mae": mae, "rmse": rmse, "mape": mape, "direction_accuracy": dir_acc},
        "test_predictions": {"actual": y_test_inv.tolist(), "predicted": y_pred_inv.tolist()},
        "forecast": {
            "current_price": current_price, "predicted_price": future_price,
            "predicted_change_pct": change_pct, "horizon_days": forecast_horizon,
            "signal": signal, "confidence": min(50 + abs(change_pct) * 10, 95),
        },
    }
