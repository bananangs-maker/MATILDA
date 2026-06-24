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


def mock_ohlcv(ticker: str, n: int = 500, interval: str = "1day") -> pd.DataFrame:
    seed = sum(ord(c) for c in ticker)
    rng = np.random.default_rng(seed)
    freq = {"1day": "B", "1week": "W-FRI", "1month": "ME"}.get(interval, "B")
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq=freq).strftime("%Y-%m-%d")
    n = len(dates)   # date_range 가 freq 앵커로 ±1 될 수 있어 실제 길이에 맞춤
    drift = np.linspace(0, 0.9 if ticker == "TQQQ" else 0.6, n)
    noise = np.cumsum(rng.normal(0, 0.022, n))
    close = pd.Series(30 * np.exp(drift * 0.4 + noise))
    high = close * (1 + np.abs(rng.normal(0, 0.012, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.012, n)))
    op = close.shift(1).fillna(close.iloc[0])
    vol = rng.integers(1_000_000, 8_000_000, n).astype(float)
    return pd.DataFrame({"date": dates, "open": op, "high": high,
                         "low": low, "close": close, "volume": vol})


import time
_CACHE = {}          # ticker -> (df, source, fetched_at)
_TTL = 600           # 10분 캐시 (무료 API 호출 절약 + 콜드스타트 후 빠른 응답)
_SENT = {"data": None, "ts": 0.0}   # 시장심리(매크로) 캐시
_FNGH = {"data": None, "ts": 0.0}   # 공포탐욕 과거 시계열 캐시
_FNGF = {"data": None, "ts": 0.0}   # 공포탐욕 전체(overview+7지표) 캐시
_MKT = {"data": None, "ts": 0.0}    # 환율·유가·금·BTC 시세 캐시 (3분)


def _sentiment_cached():
    """엔진 매크로 입력용 시장심리(VIX·공포탐욕) — 10분 캐시. 실패 시 None."""
    if USE_MOCK:
        return {"fng": 50, "vix_raw": 18.0, "vix_comp": {"score": 18.0}}
    if _SENT["data"] is not None and time.time() - _SENT["ts"] < _TTL:
        return _SENT["data"]
    try:
        from sentiment import cnn_sentiment
        d = cnn_sentiment()
        _SENT["data"] = d; _SENT["ts"] = time.time()
        return d
    except Exception:
        return _SENT["data"]   # 직전 캐시라도 있으면 사용, 없으면 None


def _load(ticker, interval="1day"):
    if USE_MOCK:
        n = {"1day": 500, "1week": 400, "1month": 300}.get(interval, 500)
        return mock_ohlcv(ticker, n, interval), "MOCK (합성 데이터)"
    ckey = (ticker, interval)
    hit = _CACHE.get(ckey)
    if hit and time.time() - hit[2] < _TTL:
        return hit[0], hit[1] + " · 캐시"
    from public_data import daily_ohlcv
    df, source = daily_ohlcv(ticker, yrange="2y", interval=interval)
    _CACHE[ckey] = (df, source, time.time())
    return df, source


@app.route("/")
def index():
    return send_from_directory(HERE, "dashboard.html")


@app.route("/chart/<ticker>")
def chart(ticker):
    ticker = ticker.upper()
    from public_data import valid_ticker
    if not valid_ticker(ticker):
        return jsonify({"error": f"잘못된 티커: {ticker}"}), 400
    try:
        interval = request.args.get("interval", "1day")
        if interval not in ("1day", "1week", "1month"):
            interval = "1day"
        df, source = _load(ticker, interval)
        payload = compute_series(df)
        payload["signals"] = compute_signals(df)
        import patterns as _pat0
        payload["markers"] = _pat0.detect_macd_markers(df)
        # 매크로(VIX·공포탐욕): 쿼리 오버라이드 우선, 없으면 서버 캐시(시장심리) 사용
        def _argf(name):
            v = request.args.get(name)
            try:
                return float(v) if v not in (None, "") else None
            except ValueError:
                return None
        vix = _argf("vix"); fng = _argf("fng")
        if vix is None and fng is None:
            sent = _sentiment_cached()
            if sent:
                if sent.get("vix_raw") is not None:
                    vix = sent["vix_raw"]
                elif sent.get("vix_comp", {}).get("score") is not None:
                    vix = sent["vix_comp"]["score"]
                if sent.get("fng") is not None:
                    fng = sent["fng"]
        import engine as eng, patterns as pat, summary as summ
        payload["engine"] = eng.compute_engine(df, vix=vix, fng=fng)
        payload["summary"] = summ.compute_summary(df)
        cross_sigs, cross_marks = pat.detect_ma_cross(df)
        payload["ma_cross"] = cross_sigs
        payload["cross_markers"] = cross_marks
        payload["candles_patterns"] = pat.detect_candles(df)
        try:
            import chart_patterns as cpat
            payload["chart_patterns"] = cpat.detect(df)
        except Exception as ce:
            payload["chart_patterns"] = []
        try:
            import fundamentals as fund
            payload["fundamentals"] = fund.metrics(ticker, df)
        except Exception as fe:
            payload["fundamentals"] = {"error": str(fe)}
        payload["meta"] = {
            "ticker": ticker, "source": source,
            "asof": str(df["date"].iloc[-1]),
            "last": round(float(df["close"].iloc[-1]), 2),
            "bars": int(len(df)),
            "interval": interval,
            "engine_daily_only": interval != "1day",
        }
        return jsonify(payload)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "ticker": ticker}), 500


@app.route("/quant/<ticker>")
def quant_route(ticker):
    ticker = ticker.upper()
    from public_data import valid_ticker
    if not valid_ticker(ticker):
        return jsonify({"error": f"잘못된 티커: {ticker}"}), 400
    try:
        cost = float(request.args.get("cost", 5))
    except ValueError:
        cost = 5.0
    try:
        expense = float(request.args.get("expense", 0.95)) / 100.0   # % → 분수
    except ValueError:
        expense = 0.0095
    try:
        df, source = _load(ticker, "1day")   # 백테스트는 항상 일봉
        import quant as Q
        res = Q.analyze(df, cost_bps=cost, expense=expense)
        res.pop("ret", None)  # 직렬화 불가 Series 제거(내부용)
        res["meta"] = {"ticker": ticker, "source": source}
        return jsonify(res)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "ticker": ticker}), 500


@app.route("/mtf/<ticker>")
def mtf_route(ticker):
    ticker = ticker.upper()
    from public_data import valid_ticker
    if not valid_ticker(ticker):
        return jsonify({"error": f"잘못된 티커: {ticker}"}), 400
    try:
        import engine as eng
        out = []
        for iv, lab in (("1day", "일봉"), ("1week", "주봉"), ("1month", "월봉")):
            try:
                df, _ = _load(ticker, iv)
                e = eng.compute_engine(df)   # 매크로 없이 가격 기준
                m = e.get("metrics", {})
                out.append({"tf": lab, "interval": iv, "regime": e["regime"],
                            "size": e["size"], "risk": e["risk"], "risk_label": e["risk_label"],
                            "above200": m.get("above200"), "rvol": m.get("rvol"), "adx": m.get("adx"),
                            "ret1": m.get("ret1")})
            except Exception as ie:
                out.append({"tf": lab, "interval": iv, "error": str(ie)})
        return jsonify({"ticker": ticker, "tfs": out})
    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500


@app.route("/search")
def search_route():
    q = (request.args.get("q") or "").strip()
    if len(q) < 1:
        return jsonify({"results": []})
    if USE_MOCK:
        demo = [{"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "type": "주식"},
                {"symbol": "NVDA", "name": "NVIDIA Corp", "exchange": "NASDAQ", "type": "주식"},
                {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "exchange": "NYSE", "type": "ETF"}]
        ql = q.upper()
        return jsonify({"results": [d for d in demo if ql in d["symbol"] or ql in d["name"].upper()]})
    try:
        from public_data import symbol_search
        return jsonify({"results": symbol_search(q)})
    except Exception as e:
        return jsonify({"results": [], "error": str(e)})


@app.route("/watchlist", methods=["GET", "POST"])
def watchlist():
    if not _supa_enabled():
        return jsonify({"configured": False, "tickers": []})
    import requests as rq
    headers = {"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}",
               "Content-Type": "application/json"}
    base = f"{SUPA_URL}/rest/v1/watchlist"
    try:
        if request.method == "GET":
            r = rq.get(base, params={"id": "eq.me", "select": "tickers"}, headers=headers, timeout=10)
            r.raise_for_status()
            rows = r.json()
            return jsonify({"configured": True, "tickers": rows[0]["tickers"] if rows else []})
        body = request.get_json(force=True) or {}
        payload = [{"id": "me", "tickers": body.get("tickers", [])}]
        r = rq.post(base, json=payload,
                    headers={**headers, "Prefer": "resolution=merge-duplicates"}, timeout=10)
        r.raise_for_status()
        return jsonify({"configured": True, "ok": True})
    except Exception as e:
        return jsonify({"configured": True, "error": str(e)}), 500


@app.route("/seasonality/<ticker>")
def seasonality_route(ticker):
    ticker = ticker.upper()
    from public_data import valid_ticker
    if not valid_ticker(ticker):
        return jsonify({"error": f"잘못된 티커: {ticker}"}), 400
    try:
        df, source = _load(ticker)
        import seasonality as se
        return jsonify(se.seasonality(df))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/tickers")
def tickers_route():
    if USE_MOCK:
        import random
        r = random.Random(int(time.time() // 180))
        demo = [("USD/KRW", "원/달러", "₩", 1387.5, 1), ("BTC/USD", "비트코인", "$", 64210, 0),
                ("XAU/USD", "금", "$", 3012.4, 1), ("WTI", "WTI 유가", "$", 71.83, 2)]
        return jsonify({"tickers": [{"symbol": s, "label": l, "unit": u,
                        "price": round(p, d), "pct": round(r.uniform(-2.2, 2.2), 2)} for s, l, u, p, d in demo]})
    global _MKT
    try:
        if _MKT["data"] is not None and time.time() - _MKT["ts"] < 180:
            return jsonify({"tickers": _MKT["data"]})
        from public_data import market_quotes
        q = market_quotes(); _MKT["data"] = q; _MKT["ts"] = time.time()
        return jsonify({"tickers": q})
    except Exception as e:
        if _MKT["data"]:
            return jsonify({"tickers": _MKT["data"]})
        return jsonify({"tickers": [], "error": str(e)})


@app.route("/fng_full")
def fng_full_route():
    if USE_MOCK:
        import math, datetime, random
        base = datetime.date.today()
        def mkser(seed, days=90):
            r = random.Random(seed); out = []
            for i in range(days, 0, -1):
                d = (base - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                out.append({"time": d, "value": round(max(2, min(98, 50 + 30 * math.sin(i / 15.0 + seed) + r.uniform(-6, 6))), 1)})
            return out
        names = [("시장 모멘텀", "S&P500 vs 125일 이동평균"), ("주가 강도", "52주 신고가 vs 신저가"),
                 ("주가 폭(breadth)", "상승·하락 거래량"), ("풋·콜 옵션", "5일 풋/콜 비율"),
                 ("시장 변동성", "VIX"), ("정크본드 수요", "고수익채 스프레드"), ("안전자산 수요", "주식 vs 국채")]
        rate = lambda v: "extreme greed" if v >= 75 else "greed" if v >= 55 else "neutral" if v >= 45 else "fear" if v >= 25 else "extreme fear"
        inds = [{"key": "k%d" % i, "name": n, "desc": de, "score": mkser(i)[-1]["value"],
                 "rating": rate(mkser(i)[-1]["value"])} for i, (n, de) in enumerate(names)]
        ts = mkser(99, 365)
        return jsonify({"overview": {"score": ts[-1]["value"], "rating": rate(ts[-1]["value"]),
                        "prev_close": ts[-2]["value"], "prev_week": ts[-6]["value"],
                        "prev_month": ts[-31]["value"], "prev_year": ts[0]["value"]},
                        "timeline": ts, "indicators": inds})
    global _FNGF
    try:
        if _FNGF["data"] is not None and time.time() - _FNGF["ts"] < _TTL:
            return jsonify(_FNGF["data"])
        from sentiment import fng_full
        d = fng_full(); _FNGF["data"] = d; _FNGF["ts"] = time.time()
        return jsonify(d)
    except Exception as e:
        if _FNGF["data"]:
            return jsonify(_FNGF["data"])
        return jsonify({"error": str(e)}), 500


@app.route("/fng_history")
def fng_history_route():
    if USE_MOCK:
        import math, datetime
        base = datetime.date.today()
        out = []
        for i in range(180, 0, -1):
            d = (base - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            v = 50 + 28 * math.sin(i / 18.0) + 6 * math.sin(i / 3.0)
            out.append({"time": d, "value": round(max(2, min(98, v)), 1)})
        return jsonify({"series": out})
    global _FNGH
    try:
        if _FNGH["data"] is not None and time.time() - _FNGH["ts"] < _TTL:
            return jsonify({"series": _FNGH["data"]})
        from sentiment import fng_history
        s = fng_history()
        _FNGH["data"] = s; _FNGH["ts"] = time.time()
        return jsonify({"series": s})
    except Exception as e:
        if _FNGH["data"]:
            return jsonify({"series": _FNGH["data"]})
        return jsonify({"series": [], "error": str(e)})


@app.route("/sentiment")
def sentiment_route():
    try:
        from sentiment import cnn_sentiment
        return jsonify(cnn_sentiment())
    except Exception as e:
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
