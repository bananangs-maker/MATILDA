"""
engine.py  — 마틸다 신호/리스크 엔진 (Phase 2)

철학(레버리지 ETF에 통계적으로 견고한 접근):
  1) 추세추종 게이트  : 200일선 위에서만 본격 노출 (장기 추세 방향에만 베팅)
  2) 변동성 타겟팅    : 실현변동성이 높을수록 비중을 연속적으로 축소
  3) 연속 리스크 점수 : 이격·낙폭·급락·자금흐름·VIX를 가중 합산(0~100)해 추가 감산
→ 0/100 하드컷이 아니라 0~80% 사이에서 '연속적으로' 권장 비중을 산출.
"""
import numpy as np
import pandas as pd
import indicators as I

TARGET_VOL = 35.0   # 포지션 목표 연율 변동성(%) — 보수적
MAXCAP = 0.80       # 최대 권장 비중


def _last(s):
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def _clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def compute_engine(df: pd.DataFrame, vix=None) -> dict:
    close = df["close"]
    c = float(close.iloc[-1])
    sma20 = _last(close.rolling(20).mean())
    sma50 = _last(close.rolling(50).mean())
    sma60 = _last(close.rolling(60).mean())
    sma120 = _last(close.rolling(120).mean())
    sma200 = _last(close.rolling(200).mean())
    adx_, pdi, mdi = I.adx(df)
    adxv, pdiv, mdiv = _last(adx_), _last(pdi), _last(mdi)
    rvol = _last(I.realized_vol(close, 20))
    cmfv = _last(I.cmf(df))
    ret1 = float(close.pct_change().iloc[-1] * 100) if len(close) > 1 else 0.0
    hi20 = _last(close.rolling(20).max())
    dd20 = (c / hi20 - 1) * 100 if hi20 else 0.0
    ext20 = ((c - sma20) / sma20 * 100) if sma20 == sma20 and sma20 else 0.0
    ext60 = ((c - sma60) / sma60 * 100) if sma60 == sma60 and sma60 else 0.0
    ext120 = ((c - sma120) / sma120 * 100) if sma120 == sma120 and sma120 else 0.0

    # ---- 추세 레짐 (장기 게이트) ----
    above200 = (sma200 == sma200) and c > sma200
    align_up = (sma50 == sma50) and (sma200 == sma200) and sma50 > sma200
    if above200 and align_up:
        regime, trend_factor, rko = "상승추세", 1.0, False
    elif above200:
        regime, trend_factor, rko = "추세약화", 0.55, False
    elif sma200 == sma200:
        regime, trend_factor, rko = "하락추세", 0.12, True
    else:
        regime, trend_factor, rko = "판단보류", 0.40, False
    ranging = (adxv == adxv) and adxv < 20
    if ranging:
        trend_factor *= 0.6

    # ---- 변동성 타겟팅 ----
    vt = (TARGET_VOL / rvol) if (rvol == rvol and rvol > 0) else 0.5
    vt = _clamp(vt, 0.15, 1.0)

    # ---- 리스크 점수 (0~100) ----
    comp = {
        "변동성": _clamp((rvol - 40) / 120) if rvol == rvol else 0.0,
        "추세": 1.0 if regime == "하락추세" else 0.5 if regime in ("추세약화", "판단보류")
                else (0.3 if ranging else 0.0),
        "이격": _clamp((abs(ext20) - 10) / 20),
        "낙폭": _clamp((-dd20 - 5) / 20),
        "급락": _clamp((-ret1 - 3) / 10),
        "자금흐름": _clamp(-cmfv * 2) if cmfv == cmfv else 0.0,
        "VIX": _clamp((vix - 15) / 25) if vix else 0.0,
    }
    W = {"변동성": 0.22, "추세": 0.26, "이격": 0.12, "낙폭": 0.14,
         "급락": 0.12, "자금흐름": 0.06, "VIX": 0.08}
    risk = round(sum(comp[k] * W[k] for k in W) * 100)
    rlabel = "위험" if risk >= 60 else "경계" if risk >= 35 else "양호"

    # ---- 비중(연속): 급성 위험만 추가 감산(중복 방지) ----
    acute = (comp["이격"] * 0.25 + comp["낙폭"] * 0.30 + comp["급락"] * 0.25
             + comp["자금흐름"] * 0.10 + comp["VIX"] * 0.10)
    size = MAXCAP * trend_factor * vt * (1 - 0.7 * acute)
    size = round(_clamp(size, 0, MAXCAP) * 100)
    if rko and risk >= 60:
        size = min(size, 5)

    rationale = []
    rationale.append("추세 레짐: %s (200일선 %s)" % (regime, "위" if above200 else "아래"))
    rationale.append("실현변동성 %.0f%% → 변동성타겟 비중계수 %.2f" % (rvol if rvol == rvol else 0, vt))
    if ranging:
        rationale.append("ADX %.0f (<20) 횡보 — 추세추종 비중 축소" % (adxv if adxv == adxv else 0))
    if acute > 0.15:
        big = sorted(((comp[k], k) for k in ("이격", "낙폭", "급락", "자금흐름", "VIX")), reverse=True)
        rationale.append("급성 리스크: " + ", ".join("%s" % k for v, k in big if v > 0.2))

    return {
        "risk": risk, "risk_label": rlabel, "components": {k: round(comp[k] * 100) for k in comp},
        "weights": {k: W[k] for k in W},
        "regime": regime, "size": int(size), "maxcap": int(MAXCAP * 100),
        "trend_factor": round(trend_factor, 2), "vol_factor": round(vt, 2),
        "metrics": {
            "rvol": round(rvol, 1) if rvol == rvol else None,
            "adx": round(adxv, 1) if adxv == adxv else None,
            "pdi": round(pdiv, 1) if pdiv == pdiv else None,
            "mdi": round(mdiv, 1) if mdiv == mdiv else None,
            "cmf": round(cmfv, 3) if cmfv == cmfv else None,
            "ext20": round(ext20, 1), "ext60": round(ext60, 1), "ext120": round(ext120, 1),
            "dd20": round(dd20, 1), "ret1": round(ret1, 1),
            "above200": bool(above200),
        },
        "rationale": rationale,
    }
