"""
public_data.py
공개 시세 소스에서 미국 ETF 일봉을 받아온다. API 키 불필요.
- 1순위: Yahoo Finance chart API (키 없음, 서버 호출이라 CORS 무관)
- 2순위: Stooq CSV (키 없음)
반환: date(YYYY-MM-DD), open, high, low, close, volume / 오래된->최신 정렬.

15분 지연 등은 무관 — 일봉 기반 지표라 전일 종가까지면 충분.
"""
import io
import datetime as dt
import requests
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# Yahoo/Stooq 심볼은 TQQQ/SOXL 그대로 사용 가능
SUPPORTED = ("TQQQ", "SOXL")


def _from_yahoo(ticker: str, yrange: str = "2y") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    r = requests.get(url, params={"range": yrange, "interval": "1d"},
                     headers=UA, timeout=12)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(ts):
        c = q["close"][i]
        if c is None:
            continue
        rows.append({
            "date": dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"),
            "open": q["open"][i], "high": q["high"][i],
            "low": q["low"][i], "close": c, "volume": q["volume"][i] or 0,
        })
    df = pd.DataFrame(rows).dropna(subset=["close"])
    return df.sort_values("date").reset_index(drop=True)


def _from_stooq(ticker: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    r = requests.get(url, headers=UA, timeout=12)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Volume": "volume"})
    df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    return df.sort_values("date").reset_index(drop=True)


def daily_ohlcv(ticker: str, yrange: str = "2y") -> tuple[pd.DataFrame, str]:
    """(DataFrame, source) 반환. Yahoo -> Stooq 순으로 시도."""
    ticker = ticker.upper()
    if ticker not in SUPPORTED:
        raise ValueError(f"지원하지 않는 종목: {ticker}")
    errors = []
    for name, fn in (("Yahoo Finance", lambda: _from_yahoo(ticker, yrange)),
                     ("Stooq", lambda: _from_stooq(ticker))):
        try:
            df = fn()
            if len(df) > 60:
                return df, name
            errors.append(f"{name}: 데이터 부족({len(df)}봉)")
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError("공개 소스 모두 실패 — " + " / ".join(errors))


if __name__ == "__main__":
    try:
        df, src = daily_ohlcv("TQQQ")
        print(src, len(df), "봉")
        print(df.tail(3))
    except Exception as e:
        print("실패:", e)
