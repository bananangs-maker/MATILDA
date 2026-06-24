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


def _series(j, key, days=180):
    import datetime as _dt
    o = j.get(key, {}) or {}
    data = o.get("data") if isinstance(o, dict) else None
    out = []
    for p in (data or []):
        try:
            x = p.get("x") if isinstance(p, dict) else p[0]
            y = p.get("y") if isinstance(p, dict) else p[1]
            if x is None or y is None:
                continue
            ts = float(x) / 1000.0 if float(x) > 1e11 else float(x)
            out.append({"time": _dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),
                        "value": round(float(y), 1)})
        except Exception:
            continue
    dd = {}
    for o2 in out:
        dd[o2["time"]] = o2["value"]
    s = [{"time": k, "value": v} for k, v in sorted(dd.items())]
    return s[-days:] if days else s


# CNN 7개 구성 지표 (key, 한글명, 설명)
FNG_INDICATORS = [
    ("market_momentum_sp125", "시장 모멘텀", "S&P500 vs 125일 이동평균"),
    ("stock_price_strength", "주가 강도", "52주 신고가 vs 신저가 종목 수"),
    ("stock_price_breadth", "주가 폭(breadth)", "상승·하락 거래량 (McClellan)"),
    ("put_call_options", "풋·콜 옵션", "5일 풋/콜 비율"),
    ("market_volatility_vix", "시장 변동성", "VIX (변동성 지수)"),
    ("junk_bond_demand", "정크본드 수요", "고수익채 vs 우량채 스프레드"),
    ("safe_haven_demand", "안전자산 수요", "주식 vs 국채 20일 수익률 차"),
]


def fng_full(days: int = 180) -> dict:
    """CNN 공포탐욕: OVERVIEW + TIMELINE + 7개 구성 지표 (각 score·rating·미니 시계열)."""
    r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                     headers=UA, timeout=12)
    r.raise_for_status()
    j = r.json()
    fg = j.get("fear_and_greed", {}) or {}
    overview = {
        "score": round(float(fg.get("score")), 1) if fg.get("score") is not None else None,
        "rating": fg.get("rating"),
        "prev_close": fg.get("previous_close"),
        "prev_week": fg.get("previous_1_week"),
        "prev_month": fg.get("previous_1_month"),
        "prev_year": fg.get("previous_1_year"),
    }
    indicators = []
    for key, name, desc in FNG_INDICATORS:
        o = j.get(key, {}) or {}
        sc = o.get("score")
        indicators.append({
            "key": key, "name": name, "desc": desc,
            "score": round(float(sc), 1) if sc is not None else None,
            "rating": o.get("rating"),
            "spark": _series(j, key, 90),
        })
    return {"overview": overview, "timeline": _series(j, "fear_and_greed_historical", days),
            "indicators": indicators}


def fng_history(days: int = 240) -> list:
    """CNN 공포탐욕 과거 시계열 → [{time:'YYYY-MM-DD', value:int}] (오래된→최신)."""
    import datetime as _dt
    r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                     headers=UA, timeout=12)
    r.raise_for_status()
    j = r.json()
    fgh = j.get("fear_and_greed_historical", {}) or {}
    data = fgh.get("data") if isinstance(fgh, dict) else (fgh if isinstance(fgh, list) else None)
    out = []
    for p in (data or []):
        try:
            x = p.get("x") if isinstance(p, dict) else p[0]
            y = p.get("y") if isinstance(p, dict) else p[1]
            if x is None or y is None:
                continue
            ts = float(x) / 1000.0 if float(x) > 1e11 else float(x)
            d = _dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            out.append({"time": d, "value": round(float(y), 1)})
        except Exception:
            continue
    # 날짜 중복 제거(마지막 값 유지) + 정렬
    dedup = {}
    for o in out:
        dedup[o["time"]] = o["value"]
    series = [{"time": k, "value": v} for k, v in sorted(dedup.items())]
    return series[-days:] if days else series


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
