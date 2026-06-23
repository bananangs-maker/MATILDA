"""
app.py
Flask 서버. 대시보드(차트) + 시세/지표 엔드포인트를 같은 출처로 서빙(=CORS 없음).

로컬:   python app.py            -> http://localhost:8000
배포:   gunicorn app:app --bind 0.0.0.0:$PORT   (Render가 자동 실행)

데이터: 공개 소스(Yahoo Finance -> Stooq), API 키 불필요.
MOCK:  환경변수 USE_MOCK=true 이면 합성 데이터.
"""
import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_from_directory
from dotenv import load_dotenv

from indicators import compute_signals, compute_series

load_dotenv()
USE_MOCK = os.environ.get("USE_MOCK", "false").lower() == "true"

app = Flask(__name__, static_folder=None)
HERE = os.path.dirname(os.path.abspath(__file__))


def mock_ohlcv(ticker: str, n: int = 500) -> pd.DataFrame:
    seed = sum(ord(c) for c in ticker)
    rng = np.random.default_rng(seed)
    drift = np.linspace(0, 0.9 if ticker == "TQQQ" else 0.6, n)
    noise = np.cumsum(rng.normal(0, 0.022, n))
    close = pd.Series(30 * np.exp(drift * 0.4 + noise))
    high = close * (1 + np.abs(rng.normal(0, 0.012, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.012, n)))
    op = close.shift(1).fillna(close.iloc[0])
    vol = rng.integers(1_000_000, 8_000_000, n).astype(float)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "open": op, "high": high,
                         "low": low, "close": close, "volume": vol})


import time
_CACHE = {}          # ticker -> (df, source, fetched_at)
_TTL = 600           # 10분 캐시 (무료 API 호출 절약 + 콜드스타트 후 빠른 응답)


def _load(ticker):
    if USE_MOCK:
        return mock_ohlcv(ticker), "MOCK (합성 데이터)"
    hit = _CACHE.get(ticker)
    if hit and time.time() - hit[2] < _TTL:
        return hit[0], hit[1] + " · 캐시"
    from public_data import daily_ohlcv
    df, source = daily_ohlcv(ticker, yrange="2y")
    _CACHE[ticker] = (df, source, time.time())
    return df, source


@app.route("/")
def index():
    return send_from_directory(HERE, "dashboard.html")


@app.route("/chart/<ticker>")
def chart(ticker):
    ticker = ticker.upper()
    if ticker not in ("TQQQ", "SOXL"):
        return jsonify({"error": f"지원하지 않는 종목: {ticker}"}), 400
    try:
        df, source = _load(ticker)
        payload = compute_series(df)
        payload["signals"] = compute_signals(df)
        import backtest as bt
        payload["markers"] = bt.markers_from_signals(df, bt.generate_signals(df))
        payload["meta"] = {
            "ticker": ticker, "source": source,
            "asof": str(df["date"].iloc[-1]),
            "last": round(float(df["close"].iloc[-1]), 2),
            "bars": int(len(df)),
        }
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500


@app.route("/backtest/<ticker>")
def backtest_route(ticker):
    ticker = ticker.upper()
    if ticker not in ("TQQQ", "SOXL"):
        return jsonify({"error": f"지원하지 않는 종목: {ticker}"}), 400
    from flask import request
    try:
        trend_ma = int(request.args.get("trend_ma", 200))
    except ValueError:
        trend_ma = 200
    trend_ma = max(20, min(trend_ma, 250))
    try:
        df, source = _load(ticker)
        import backtest as bt
        res = bt.run(df, trend_ma=trend_ma)
        res["meta"] = {"ticker": ticker, "source": source}
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500


@app.route("/sentiment")
def sentiment_route():
    try:
        from sentiment import cnn_sentiment
        return jsonify(cnn_sentiment())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"ok": True, "mock": USE_MOCK})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"  데이터: {'MOCK' if USE_MOCK else 'Yahoo->Stooq'}  ·  http://localhost:{port}")
    app.run(host="0.0.0.0", port=port)
