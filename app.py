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
from flask import Flask, jsonify, send_from_directory, request
from dotenv import load_dotenv

from indicators import compute_signals, compute_series

load_dotenv()
USE_MOCK = os.environ.get("USE_MOCK", "false").lower() == "true"
SUPA_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPA_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "") or os.environ.get("SUPABASE_KEY", "")


def _supa_enabled():
    return bool(SUPA_URL and SUPA_KEY)

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
        try:
            vix = float(request.args.get("vix")) if request.args.get("vix") else None
        except ValueError:
            vix = None
        import engine as eng, patterns as pat, summary as summ
        payload["engine"] = eng.compute_engine(df, vix=vix)
        payload["summary"] = summ.compute_summary(df)
        cross_sigs, cross_marks = pat.detect_ma_cross(df)
        payload["ma_cross"] = cross_sigs
        payload["cross_markers"] = cross_marks
        payload["candles_patterns"] = pat.detect_candles(df)
        payload["meta"] = {
            "ticker": ticker, "source": source,
            "asof": str(df["date"].iloc[-1]),
            "last": round(float(df["close"].iloc[-1]), 2),
            "bars": int(len(df)),
        }
        return jsonify(payload)
    except Exception as e:
        import traceback; traceback.print_exc()
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


@app.route("/validate/<ticker>")
def validate_route(ticker):
    ticker = ticker.upper()
    if ticker not in ("TQQQ", "SOXL"):
        return jsonify({"error": f"지원하지 않는 종목: {ticker}"}), 400
    try:
        cost = float(request.args.get("cost", 5))
    except ValueError:
        cost = 5.0
    try:
        df, source = _load(ticker)
        import backtest_engine as be
        res = be.analyze(df, cost_bps=cost)
        res["meta"] = {"ticker": ticker, "source": source}
        return jsonify(res)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/drawings/<ticker>", methods=["GET", "POST"])
def drawings(ticker):
    ticker = ticker.upper()
    if not _supa_enabled():
        # Supabase 미설정 -> 클라이언트가 localStorage 사용
        return jsonify({"configured": False, "drawings": []})
    import requests as rq
    headers = {"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}",
               "Content-Type": "application/json"}
    base = f"{SUPA_URL}/rest/v1/chart_drawings"
    try:
        if request.method == "GET":
            r = rq.get(base, params={"ticker": f"eq.{ticker}", "select": "drawings"},
                       headers=headers, timeout=10)
            r.raise_for_status()
            rows = r.json()
            return jsonify({"configured": True,
                            "drawings": rows[0]["drawings"] if rows else []})
        body = request.get_json(force=True) or {}
        payload = [{"ticker": ticker, "drawings": body.get("drawings", [])}]
        r = rq.post(base, json=payload,
                    headers={**headers, "Prefer": "resolution=merge-duplicates"}, timeout=10)
        r.raise_for_status()
        return jsonify({"configured": True, "ok": True})
    except Exception as e:
        return jsonify({"configured": True, "error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"ok": True, "mock": USE_MOCK})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"  데이터: {'MOCK' if USE_MOCK else 'Yahoo->Stooq'}  ·  http://localhost:{port}")
    app.run(host="0.0.0.0", port=port)
