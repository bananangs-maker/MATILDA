"""
public_data.py
공개 시세에서 미국 ETF 일봉을 받아온다.
- 1순위: Twelve Data (무료 키 필요, 클라우드 IP에서도 안정적)  <-- Render 배포용
- 2순위: Stooq CSV (키 없음, 로컬에선 잘 됨)
반환: date(YYYY-MM-DD), open, high, low, close, volume / 오래된->최신.

Twelve Data 무료 키: https://twelvedata.com 가입 후 발급 (800회/일).
환경변수 TWELVEDATA_API_KEY 로 주입.
"""
import os
import io
import re
import time
import requests
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
PINNED = ("TQQQ", "SOXL")          # 기본 고정 종목
TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,7}$")


def valid_ticker(t: str) -> bool:
    return bool(TICKER_RE.match((t or "").upper()))


def _parse_twelvedata(j: dict) -> pd.DataFrame:
    if isinstance(j, dict) and j.get("status") == "error":
        raise RuntimeError("Twelve Data: " + str(j.get("message")))
    vals = j.get("values") if isinstance(j, dict) else None
    if not vals:
        raise RuntimeError("Twelve Data: 빈 응답")
    df = pd.DataFrame(vals).rename(columns={"datetime": "date"})
    if "volume" not in df.columns:
        df["volume"] = 0
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = df["volume"].fillna(0)
    df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    return df.sort_values("date").reset_index(drop=True)  # 오래된 -> 최신


def _from_twelvedata(ticker: str, outputsize: int = 500, interval: str = "1day") -> pd.DataFrame:
    key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY 미설정 (Render 환경변수에 추가하세요)")
    params = {"symbol": ticker, "interval": interval, "outputsize": outputsize, "apikey": key}
    delays = [0, 3]   # 1차 즉시 + 1회 짧은 재시도. 한도면 빠르게 실패하고 폴백으로 넘김
    last = None
    for d in delays:
        if d:
            time.sleep(d)
        r = requests.get("https://api.twelvedata.com/time_series",
                         params=params, headers=UA, timeout=12)
        body_429 = False
        if r.status_code == 200:
            try:
                jj = r.json()
                body_429 = isinstance(jj, dict) and jj.get("code") == 429
            except Exception:
                jj = None
            if not body_429:
                return _parse_twelvedata(jj if jj is not None else r.json())
        if r.status_code == 429 or body_429:
            last = "429 한도 초과"
            continue
        r.raise_for_status()
        return _parse_twelvedata(r.json())
    raise RuntimeError(f"Twelve Data {last or '요청 실패'}")


def _from_yahoo(ticker: str, interval: str = "1day", yrange: str = None) -> pd.DataFrame:
    """야후 파이낸스 chart v8 (키 불필요 · 일/주/월 지원 · 폴백 주력). 헤더 보강 + 429 재시도."""
    iv = {"1day": "1d", "1week": "1wk", "1month": "1mo"}.get(interval, "1d")
    rng = yrange or {"1day": "3y", "1week": "10y", "1month": "max"}.get(interval, "3y")
    hdr = {"User-Agent": UA["User-Agent"],
           "Accept": "application/json,text/plain,*/*",
           "Accept-Language": "en-US,en;q=0.9",
           "Referer": "https://finance.yahoo.com/"}
    last = None
    for attempt in range(2):                       # 429면 한 번 더(호스트 교차)
        for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
            try:
                r = requests.get(f"https://{host}/v8/finance/chart/{ticker}",
                                 params={"range": rng, "interval": iv,
                                         "includePrePost": "false"},
                                 headers=hdr, timeout=12)
                if r.status_code == 429:
                    last = "HTTP 429"; continue
                if r.status_code != 200:
                    last = f"HTTP {r.status_code}"; continue
                j = r.json() or {}
                ch = (j.get("chart") or {})
                if ch.get("error"):
                    last = str(ch["error"]); continue
                res = (ch.get("result") or [None])[0]
                if not res or not res.get("timestamp"):
                    last = "빈 응답"; continue
                ts = res["timestamp"]
                q = (res.get("indicators", {}).get("quote") or [{}])[0]
                adj = (res.get("indicators", {}).get("adjclose") or [{}])
                adjc = adj[0].get("adjclose") if adj and adj[0] else None
                op, hi, lo, cl, vo = q.get("open"), q.get("high"), q.get("low"), q.get("close"), q.get("volume")
                rows = []
                for i, t in enumerate(ts):
                    c = (adjc[i] if adjc else None)
                    c = c if c is not None else (cl[i] if cl else None)
                    if c is None:
                        continue
                    rows.append({"date": pd.to_datetime(t, unit="s").strftime("%Y-%m-%d"),
                                 "open": (op[i] if op and op[i] is not None else c),
                                 "high": (hi[i] if hi and hi[i] is not None else c),
                                 "low": (lo[i] if lo and lo[i] is not None else c),
                                 "close": c,
                                 "volume": (vo[i] if vo and vo[i] is not None else 0)})
                if not rows:
                    last = "유효 행 없음"; continue
                df = pd.DataFrame(rows).dropna(subset=["close"])
                return df.sort_values("date").reset_index(drop=True)
            except Exception as e:
                last = str(e); continue
        if last == "HTTP 429":
            time.sleep(5)                          # 한도면 잠깐 쉬고 1회 재시도
    raise RuntimeError(f"Yahoo: {last or '실패'}")


def _from_stooq(ticker: str) -> pd.DataFrame:
    """Stooq 일봉 CSV — 전체 히스토리. 차단(HTML) 감지 시 1회 재시도."""
    hdr = {"User-Agent": UA["User-Agent"],
           "Accept": "text/csv,text/plain,*/*",
           "Accept-Language": "en-US,en;q=0.9",
           "Referer": "https://stooq.com/"}
    last = None
    for attempt in range(2):
        try:
            r = requests.get(f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d",
                             headers=hdr, timeout=12)
            txt = r.text or ""
            if r.status_code != 200 or txt.lstrip()[:1] == "<" or "Date" not in txt[:200]:
                last = "차단/예상치 못한 응답"; time.sleep(2); continue
            df = pd.read_csv(io.StringIO(txt))
            df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                    "Low": "low", "Close": "close", "Volume": "volume"})
            if "date" not in df.columns:
                last = "헤더 없음"; continue
            df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
            return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            last = str(e); time.sleep(1)
    raise RuntimeError(f"Stooq: {last or '실패'}")


MARKET_TICKERS = [
    ("USD/KRW", "원/달러", "₩", 1),
    ("BTC/USD", "비트코인", "$", 0),
    ("XAU/USD", "금", "$", 1),
    ("WTI", "WTI 유가", "$", 2),
]


def market_quotes() -> list:
    """환율·비트코인·금·유가 실시간 시세 (Twelve Data /quote 배치). 종목별 graceful 실패."""
    key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY 미설정")
    syms = ",".join(t[0] for t in MARKET_TICKERS)
    r = requests.get("https://api.twelvedata.com/quote",
                     params={"symbol": syms, "apikey": key}, headers=UA, timeout=12)
    r.raise_for_status()
    j = r.json() or {}
    out = []
    for sym, label, unit, dec in MARKET_TICKERS:
        o = j.get(sym) if (isinstance(j, dict) and sym in j) else (j if j.get("symbol") == sym else None)
        try:
            if not o or o.get("status") == "error" or o.get("close") in (None, ""):
                out.append({"symbol": sym, "label": label, "unit": unit, "price": None})
                continue
            price = float(o["close"]); pct = float(o.get("percent_change") or 0)
            out.append({"symbol": sym, "label": label, "unit": unit,
                        "price": round(price, dec), "pct": round(pct, 2)})
        except Exception:
            out.append({"symbol": sym, "label": label, "unit": unit, "price": None})
    return out


def daily_ohlcv(ticker: str, yrange: str = "3y", interval: str = "1day") -> tuple[pd.DataFrame, str]:
    ticker = ticker.upper()
    if not valid_ticker(ticker):
        raise ValueError(f"잘못된 티커 형식: {ticker}")
    errors = []
    # 일봉 outputsize: yrange에 맞춰. 주봉/월봉은 충분히 크게(200기간 MA 등)
    if interval == "1day":
        osize = {"2y": 520, "3y": 780, "5y": 1300, "10y": 2600, "max": 5000}.get(yrange, 780)
    else:
        osize = 600 if interval == "1week" else 360
    sources = [("Yahoo", lambda: _from_yahoo(ticker, interval, yrange)),
               ("Twelve Data", lambda: _from_twelvedata(ticker, osize, interval))]
    if interval == "1day":
        sources.append(("Stooq", lambda: _from_stooq(ticker)))  # 마지막 폴백(일봉만, 전체 히스토리)
    for name, fn in sources:
        try:
            df = fn()
            if len(df) > 60:
                return df, name
            errors.append(f"{name}: 데이터 부족({len(df)})")
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError("공개 소스 실패 — " + " / ".join(errors))


def symbol_search(query: str, limit: int = 12) -> list:
    """미국 주식·ETF 종목 검색 (Twelve Data symbol_search)."""
    key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY 미설정")
    r = requests.get("https://api.twelvedata.com/symbol_search",
                     params={"symbol": query, "outputsize": 30, "apikey": key},
                     headers=UA, timeout=10)
    r.raise_for_status()
    data = (r.json() or {}).get("data", []) or []
    out, seen = [], set()
    for d in data:
        if d.get("country") != "United States":
            continue
        typ = (d.get("instrument_type") or d.get("type") or "")
        if not any(k in typ for k in ("Common Stock", "ETF", "Stock")):
            continue
        sym = (d.get("symbol") or "").upper()
        if not valid_ticker(sym) or sym in seen:
            continue
        seen.add(sym)
        out.append({"symbol": sym, "name": d.get("instrument_name", ""),
                    "exchange": d.get("exchange", ""), "type": "ETF" if "ETF" in typ else "주식"})
        if len(out) >= limit:
            break
    return out


if __name__ == "__main__":
    try:
        df, src = daily_ohlcv("TQQQ")
        print(src, len(df)); print(df.tail(3))
    except Exception as e:
        print("실패:", e)
