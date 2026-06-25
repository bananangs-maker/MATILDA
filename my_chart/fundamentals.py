"""
fundamentals.py — 종목 정보/재무 (선별 표시)
 - 공통(모든 종목, 가격만으로): 일일변동·연율변동성·52주 고저/위치·기간수익률·1년 MDD
 - 레버리지 ETF: 기초자산 대비 추적 베타 + 변동성 부식률(decay)
 - 개별주: Twelve Data /statistics 에서 PER/PBR/시총/배당 (무료 제공 시에만)
"""
import os
import numpy as np
import pandas as pd
import requests
import indicators as I

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}

# 레버리지 ETF -> (기초 1x ETF, 배수)
LEVERAGED = {
    "TQQQ": ("QQQ", 3), "SQQQ": ("QQQ", -3),
    "SOXL": ("SOXX", 3), "SOXS": ("SOXX", -3),
    "SPXL": ("SPY", 3), "SPXS": ("SPY", -3), "UPRO": ("SPY", 3), "SPXU": ("SPY", -3),
    "TNA": ("IWM", 3), "TZA": ("IWM", -3),
    "TECL": ("XLK", 3), "TECS": ("XLK", -3),
    "UDOW": ("DIA", 3), "SDOW": ("DIA", -3),
    "LABU": ("XBI", 3), "LABD": ("XBI", -3),
    "TMF": ("TLT", 3), "TMV": ("TLT", -3),
}


def _last(s):
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def price_metrics(df: pd.DataFrame) -> dict:
    close = df["close"]
    c = float(close.iloc[-1])
    dvol = float(close.pct_change().iloc[-1] * 100) if len(close) > 1 else 0.0
    annv = _last(I.realized_vol(close, 20))
    tail = close.tail(252)
    hi52, lo52 = float(tail.max()), float(tail.min())
    pos = (c - lo52) / (hi52 - lo52) * 100 if hi52 > lo52 else 50.0
    rng_ret = lambda d: round((c / float(close.iloc[-d]) - 1) * 100, 1) if len(close) > d else None
    eq = tail / tail.cummax()
    mdd = float((eq - 1).min() * 100)
    return {
        "price": round(c, 2), "day_chg": round(dvol, 2),
        "ann_vol": round(annv, 1) if annv == annv else None,
        "hi52": round(hi52, 2), "lo52": round(lo52, 2), "pos52": round(pos, 0),
        "ret_1m": rng_ret(21), "ret_3m": rng_ret(63), "ret_1y": rng_ret(252),
        "mdd_1y": round(mdd, 1),
    }


def leverage_metrics(ticker: str, df: pd.DataFrame) -> dict:
    """레버리지 ETF의 추적 베타 + 부식률(기초 배수 대비 실제 성과 괴리)."""
    if ticker not in LEVERAGED:
        return None
    und, lev = LEVERAGED[ticker]
    out = {"underlying": und, "leverage": lev}
    try:
        from public_data import daily_ohlcv
        udf, _ = daily_ohlcv(und)
        a = df[["date", "close"]].rename(columns={"close": "etf"})
        b = udf[["date", "close"]].rename(columns={"close": "und"})
        m = pd.merge(a, b, on="date").tail(252).reset_index(drop=True)
        if len(m) < 60:
            return out
        er = m["etf"].pct_change().dropna()
        ur = m["und"].pct_change().dropna()
        beta = float(np.cov(er, ur)[0, 1] / np.var(ur)) if np.var(ur) > 0 else None
        etf_cum = m["etf"].iloc[-1] / m["etf"].iloc[0] - 1
        und_cum = m["und"].iloc[-1] / m["und"].iloc[0] - 1
        expected = lev * und_cum
        decay = etf_cum - expected           # 음수 = 기대 대비 부식
        out.update({
            "beta": round(float(beta), 2) if beta is not None else None,
            "etf_ret": round(float(etf_cum) * 100, 1), "und_ret": round(float(und_cum) * 100, 1),
            "expected_ret": round(float(expected) * 100, 1), "decay": round(float(decay) * 100, 1),
        })
    except Exception as e:
        out["note"] = "기초자산 데이터 실패"
    return out


def stock_stats(ticker: str) -> dict:
    """Twelve Data /statistics — 무료 제공 시 PER/PBR/시총/배당."""
    key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    if not key:
        return None
    try:
        r = requests.get("https://api.twelvedata.com/statistics",
                         params={"symbol": ticker, "apikey": key}, headers=UA, timeout=10)
        j = r.json()
        st = (j or {}).get("statistics")
        if not st:
            return None
        val = st.get("valuations_metrics", {}) or {}
        div = st.get("dividends_and_splits", {}) or {}
        fin = st.get("financials", {}) or {}
        out = {
            "pe": val.get("trailing_pe") or val.get("forward_pe"),
            "pb": val.get("price_to_book_mrq"),
            "ps": val.get("price_to_sales_ttm"),
            "mktcap": val.get("market_capitalization"),
            "div_yield": div.get("forward_annual_dividend_yield"),
            "profit_margin": (fin.get("profit_margin") if isinstance(fin, dict) else None),
        }
        if all(v is None for v in out.values()):
            return None
        return out
    except Exception:
        return None


def metrics(ticker: str, df: pd.DataFrame) -> dict:
    ticker = ticker.upper()
    lev = leverage_metrics(ticker, df)
    stats = None if lev else stock_stats(ticker)   # 레버리지 ETF엔 재무 생략
    return {"ticker": ticker, "price": price_metrics(df),
            "leverage": lev, "stats": stats,
            "kind": "leveraged_etf" if lev else ("stock" if stats else "other")}
