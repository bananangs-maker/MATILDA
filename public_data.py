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
    delays = [0, 7, 14]   # 429(분당 한도) 시 점증 백오프 재시도
    last = None
    for i, d in enumerate(delays):
        if d:
            time.sleep(d)
        r = requests.get("https://api.twelvedata.com/time_series",
                         params=params, headers=UA, timeout=12)
        # Twelve Data는 본문 JSON code=429 로도 한도를 알림
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
            last = "429 한도 초과 (재시도)"
            continue
        r.raise_for_status()
        return _parse_twelvedata(r.json())
    raise RuntimeError(f"Twelve Data {last or '요청 실패'} — 잠시 후 자동 재시도되며, 일봉은 Stooq로 폴백됩니다")


def _from_stooq(ticker: str) -> pd.DataFrame:
    r = requests.get(f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d",
                     headers=UA, timeout=12)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Volume": "volume"})
    if "date" not in df.columns:
        raise RuntimeError("Stooq: 예상치 못한 응답(차단 가능)")
    df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    return df.sort_values("date").reset_index(drop=True)


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


def daily_ohlcv(ticker: str, yrange: str = "2y", interval: str = "1day") -> tuple[pd.DataFrame, str]:
    ticker = ticker.upper()
    if not valid_ticker(ticker):
        raise ValueError(f"잘못된 티커 형식: {ticker}")
    errors = []
    # 주봉/월봉은 outputsize를 키워 충분한 히스토리 확보 (200기간 MA 등)
    osize = 500 if interval == "1day" else (600 if interval == "1week" else 360)
    sources = [("Twelve Data", lambda: _from_twelvedata(ticker, osize, interval))]
    if interval == "1day":
        sources.append(("Stooq", lambda: _from_stooq(ticker)))  # 폴백은 일봉만
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
