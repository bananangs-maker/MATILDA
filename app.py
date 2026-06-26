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
_TTL = 3600          # 시장심리(sentiment) 캐시 1시간
_DATA_TTL_CHART = 3600       # 3년 차트: 1시간(장중 현재가 신선도 유지)
_DATA_TTL_BT = 12 * 3600     # 10년 백테스트: 12시간(역사 분석은 당일 봉 신선도 불필요 → API 한도 대폭 절약)
_SENT = {"data": None, "ts": 0.0}   # 시장심리(매크로) 캐시
_FNGH = {"data": None, "ts": 0.0}   # 공포탐욕 과거 시계열 캐시
_FNGF = {"data": None, "ts": 0.0}   # 공포탐욕 전체(overview+7지표) 캐시
_MKT = {"data": None, "ts": 0.0}    # 환율·유가·금·BTC 시세 캐시 (30분 — Twelve Data 크레딧 절약)


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


def _load(ticker, interval="1day", long=False):
    if USE_MOCK:
        n = {"1day": 5200 if long else 780, "1week": 400, "1month": 300}.get(interval, 780)
        return mock_ohlcv(ticker, n, interval), "MOCK (합성 데이터)"
    ckey = (ticker, interval, long)
    ttl = _DATA_TTL_BT if long else _DATA_TTL_CHART
    hit = _CACHE.get(ckey)
    if hit and time.time() - hit[2] < ttl:
        return hit[0], hit[1] + " · 캐시"
    from public_data import daily_ohlcv
    yrange = "10y" if long else "3y"   # 백테스트=10년(클라우드에서 안정적). 2000~ 깊은 역사는 CSV 업로드로.
    try:
        df, source = daily_ohlcv(ticker, yrange=yrange, interval=interval)
        _CACHE[ckey] = (df, source, time.time())
        return df, source
    except Exception:
        # 모든 소스 실패(429 등) → 만료된 캐시라도 있으면 사용(완전 실패 방지)
        if hit:
            return hit[0], hit[1] + " · 캐시(만료·한도소진으로 갱신실패, 직전 데이터 사용)"
        raise


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
        note = None
        try:
            df, source = _load(ticker, "1day", long=True)   # 백테스트는 10년
        except Exception as e_long:
            # 10년 요청이 한도/차단(429)이면, 차트가 받아둔 3년 데이터로라도 분석(완전 실패 방지)
            df, source = _load(ticker, "1day", long=False)
            note = "10년 데이터 차단(API 한도) → 3년 구간으로 분석. 200일선 워밍업 탓 유효구간이 짧으니 참고용."
        import quant as Q
        # 검증 종료일 컷오프(비정상 구간 제외용). end=YYYY-MM-DD 이전 데이터만 사용.
        end = (request.args.get("end") or "").strip()
        if end:
            before = len(df)
            df = df[df["date"] <= end].reset_index(drop=True)
            if len(df) < 260:
                return jsonify({"error": f"종료일 {end} 이전 데이터가 부족합니다({len(df)}봉). 더 늦은 날짜를 쓰세요."}), 400
            note = (note + " · " if note else "") + f"검증 종료일 {end} 적용({before}→{len(df)}봉, 이후 구간 제외)"
        res = Q.analyze(df, cost_bps=cost, expense=expense)
        res.pop("ret", None)  # 직렬화 불가 Series 제거(내부용)
        res["meta"] = {"ticker": ticker, "source": source, "bars": len(df)}
        if end:
            res["meta"]["end"] = end
        if note:
            res["meta"]["note"] = note
        return jsonify(res)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "ticker": ticker}), 500


@app.route("/quant_csv", methods=["POST"])
def quant_csv_route():
    """업로드한 CSV(Date,Open,High,Low,Close,Volume)로 백테스트 — 무료 클라우드가 못 받는
    장기 히스토리(2000~ 닷컴·금융위기)를 로컬에서 받아 올려 검증하기 위함."""
    import io as _io
    try:
        cost = float(request.form.get("cost", 5))
    except (ValueError, TypeError):
        cost = 5.0
    try:
        expense = float(request.form.get("expense", 0.95)) / 100.0
    except (ValueError, TypeError):
        expense = 0.0095
    label = (request.form.get("label") or "업로드").upper()[:12]
    f = request.files.get("file")
    if f is None:
        return jsonify({"error": "CSV 파일이 없습니다"}), 400
    try:
        raw = f.read().decode("utf-8", errors="replace")
        df = pd.read_csv(_io.StringIO(raw))
        # 컬럼명 표준화 (대소문자·한글·약어 허용)
        cmap = {}
        for c in df.columns:
            lc = str(c).strip().lower()
            if lc in ("date", "날짜", "time", "timestamp"): cmap[c] = "date"
            elif lc in ("open", "시가"): cmap[c] = "open"
            elif lc in ("high", "고가"): cmap[c] = "high"
            elif lc in ("low", "저가"): cmap[c] = "low"
            elif lc in ("close", "close/last", "adj close", "adjclose", "종가", "last"): cmap[c] = "close"
            elif lc in ("volume", "vol", "거래량"): cmap[c] = "volume"
        df = df.rename(columns=cmap)
        need = {"date", "open", "high", "low", "close"}
        if not need.issubset(df.columns):
            return jsonify({"error": f"필요 컬럼 누락. 필요: Date,Open,High,Low,Close,Volume / 받은: {list(df.columns)}"}), 400
        if "volume" not in df.columns:
            df["volume"] = 0
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for col in ("open", "high", "low", "close", "volume"):
            # Nasdaq 등은 가격에 '$', 천단위 ',' 가 붙어 옴 → 제거 후 숫자화
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[$,]", "", regex=True).str.strip(),
                errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        df = df[["date", "open", "high", "low", "close", "volume"]]
        if len(df) < 250:
            return jsonify({"error": f"데이터가 너무 짧습니다({len(df)}행). 200일선+검증엔 최소 250행 필요."}), 400
        import quant as Q
        res = Q.analyze(df, cost_bps=cost, expense=expense)
        res.pop("ret", None)
        res["meta"] = {"ticker": label, "source": "업로드 CSV", "bars": len(df),
                       "start": df["date"].iloc[0], "end": df["date"].iloc[-1]}
        return jsonify(res)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"CSV 처리 실패: {e}"}), 500


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
        if _MKT["data"] is not None and time.time() - _MKT["ts"] < 1800:   # 30분 캐시(시세 바는 참고용 → 크레딧 절약)
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
