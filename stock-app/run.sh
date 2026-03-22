#!/bin/bash
# QuantDesk — Stock Analysis Platform
# Usage: chmod +x run.sh && ./run.sh

echo "╔══════════════════════════════════════════╗"
echo "║   QuantDesk — Stock Analysis Platform    ║"
echo "╚══════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

echo "→ Installing Python dependencies..."
pip install fastapi uvicorn yfinance pandas numpy scikit-learn scipy pydantic httpx --break-system-packages -q 2>/dev/null || \
pip install fastapi uvicorn yfinance pandas numpy scikit-learn scipy pydantic httpx -q

echo "→ Installing TensorFlow for LSTM (may take a moment)..."
pip install tensorflow --break-system-packages -q 2>/dev/null || {
    echo "  ⚠ TensorFlow unavailable — ML will use Ridge regression fallback"
}

echo ""
echo "→ Starting server on http://localhost:8000"
echo "→ Press Ctrl+C to stop"
echo ""

cd backend
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
