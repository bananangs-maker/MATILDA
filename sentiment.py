"""
sentiment.py
CNN Fear & Greed graphdata 엔드포인트에서 시장 심리를 한 번에 수집한다 (무료, 키 없음).
- 공포·탐욕 지수(0~100, headline)  : 거시 패널 자동 입력에 사용
- 풋콜 옵션 컴포넌트(0~100, rating) : 시장 심리 표시
- VIX 변동성 컴포넌트(0~100, rating): 시장 심리 표시
(선택) 원시 VIX 값은 Twelve Data에 키가 있고 지원될 때만.

주의: CNN의 비공식 엔드포인트라 변경/차단 가능. 실패 시 거시 패널은 수동 입력으로 동작.
"""
import os
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "application/json"}


def _comp(j, key):
    o = j.get(key, {}) or {}
    s = o.get("score")
    return ({"score": round(float(s), 1) if s is not None else None,
             "rating": o.get("rating")})


def cnn_sentiment() -> dict:
    r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                     headers=UA, timeout=12)
    r.raise_for_status()
    j = r.json()
    fg = j.get("fear_and_greed", {}) or {}
    out = {
        "fng": round(float(fg.get("score", 0))) if fg.get("score") is not None else None,
        "fng_rating": fg.get("rating"),
        "putcall": _comp(j, "put_call_options"),
        "vix_comp": _comp(j, "market_volatility_vix"),
        "source": "CNN",
    }
    # 원시 VIX (선택): Twelve Data 키가 있으면 시도
    out["vix_raw"] = _twelvedata_vix()
    return out


def _twelvedata_vix():
    key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    if not key:
        return None
    try:
        r = requests.get("https://api.twelvedata.com/quote",
                         params={"symbol": "VIX", "apikey": key}, timeout=10)
        j = r.json()
        c = j.get("close")
        return round(float(c), 1) if c else None
    except Exception:
        return None


if __name__ == "__main__":
    try:
        print(cnn_sentiment())
    except Exception as e:
        print("실패:", e)
