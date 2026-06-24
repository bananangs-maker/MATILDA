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


def size_series(df: pd.DataFrame, target_vol: float = TARGET_VOL,
                maxcap: float = MAXCAP) -> pd.Series:
    """엔진 권장 비중(0~maxcap, 분수)을 전 기간에 대해 벡터화로 계산.
    각 t의 값은 t까지의 데이터(rolling)만 사용 → 미래참조 없음. 백테스트용."""
    close = df["close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    adx_, _, _ = I.adx(df)
    rvol = I.realized_vol(close, 20)
    cmfv = I.cmf(df, 20)
    ret1 = close.pct_change() * 100
    dd20 = (close / close.rolling(20).max() - 1) * 100
    ext20 = (close - sma20) / sma20 * 100
    above200 = close > sma200
    align_up = sma50 > sma200
    ranging = adx_ < 20

    tf = pd.Series(np.select([above200 & align_up, above200 & ~align_up, ~above200],
                             [1.0, 0.55, 0.12], default=0.4), index=close.index, dtype=float)
    tf = tf * np.where(ranging.fillna(False), 0.6, 1.0)
    vt = (target_vol / rvol).clip(0.15, 1.0).fillna(0.5)
    cl = lambda x: x.clip(0, 1)
    acute = (cl((ext20.abs() - 10) / 20) * 0.25 + cl((-dd20 - 5) / 20) * 0.30
             + cl((-ret1 - 3) / 10) * 0.25 + cl(-cmfv * 2).fillna(0) * 0.10)
    size = (maxcap * tf * vt * (1 - 0.7 * acute)).clip(0, maxcap)
    size[sma200.isna()] = np.nan          # 200봉 워밍업 전엔 거래 안 함
    return size


def compute_engine(df: pd.DataFrame, vix=None, fng=None) -> dict:
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

    # ---- 매크로 (VIX + 공포탐욕) : 시장 전반 리스크 레짐 ----
    macro_avail = (vix is not None) or (fng is not None)
    vix_risk = _clamp((vix - 15) / 25) if vix is not None else 0.0      # 0@15 ~ 1@40
    if fng is not None:
        greed_risk = _clamp((fng - 70) / 30)        # 과열(탐욕) 0@70 ~ 1@100
        fear_risk = _clamp((30 - fng) / 30)         # 패닉(공포) 0@30 ~ 1@0
        fng_risk = max(greed_risk, 0.6 * fear_risk)  # 탐욕을 더 무겁게(자만이 낙폭 선행)
    else:
        fng_risk = 0.0
    if vix is not None and fng is not None:
        macro_risk = _clamp(0.6 * vix_risk + 0.4 * fng_risk)
    elif vix is not None:
        macro_risk = vix_risk
    elif fng is not None:
        macro_risk = fng_risk
    else:
        macro_risk = 0.0

    # ---- 리스크 점수 (0~100) ----
    comp = {
        "변동성": _clamp((rvol - 40) / 120) if rvol == rvol else 0.0,
        "추세": 1.0 if regime == "하락추세" else 0.5 if regime in ("추세약화", "판단보류")
                else (0.3 if ranging else 0.0),
        "이격": _clamp((abs(ext20) - 10) / 20),
        "낙폭": _clamp((-dd20 - 5) / 20),
        "급락": _clamp((-ret1 - 3) / 10),
        "자금흐름": _clamp(-cmfv * 2) if cmfv == cmfv else 0.0,
        "매크로": macro_risk,
    }
    W = {"변동성": 0.20, "추세": 0.24, "이격": 0.10, "낙폭": 0.12,
         "급락": 0.10, "자금흐름": 0.06, "매크로": 0.18}
    risk = round(sum(comp[k] * W[k] for k in W) * 100)

    # ---- 비중(연속): 급성 위험 추가 감산 (매크로 소프트 반영) ----
    acute = (comp["이격"] * 0.22 + comp["낙폭"] * 0.26 + comp["급락"] * 0.22
             + comp["자금흐름"] * 0.10 + comp["매크로"] * 0.20)
    size = MAXCAP * trend_factor * vt * (1 - 0.7 * acute)
    size = round(_clamp(size, 0, MAXCAP) * 100)

    # ---- 하드 게이트: 극단 매크로/하락추세는 강제 축소 ----
    hard = None
    if rko and risk >= 60:
        size = min(size, 5); hard = "하락추세 + 고위험"
    if vix is not None and vix >= 45:
        size = min(size, 5); hard = "VIX 패닉(≥45)"
    elif (vix is not None and vix >= 32) or (fng is not None and fng >= 90):
        size = min(size, 15); hard = hard or "매크로 극단(VIX≥32 또는 극단적 탐욕)"
    elif (fng is not None and fng <= 8):
        size = min(size, 22); hard = hard or "극단적 공포(패닉)"
    if hard:
        risk = max(risk, 65)
    rlabel = "위험" if risk >= 60 else "경계" if risk >= 35 else "양호"

    rationale = []
    rationale.append("추세 레짐: %s (200일선 %s)" % (regime, "위" if above200 else "아래"))
    rationale.append("실현변동성 %.0f%% → 변동성타겟 비중계수 %.2f" % (rvol if rvol == rvol else 0, vt))
    if ranging:
        rationale.append("ADX %.0f (<20) 횡보 — 추세추종 비중 축소" % (adxv if adxv == adxv else 0))
    if macro_avail:
        mp = []
        if vix is not None: mp.append("VIX %.0f" % vix)
        if fng is not None: mp.append("공포탐욕 %d" % fng)
        rationale.append("매크로: " + ", ".join(mp) + (" → " + hard if hard else ""))
    else:
        rationale.append("매크로 미반영(데이터 없음) — 새로고침으로 재시도")
    if acute > 0.15:
        big = sorted(((comp[k], k) for k in ("이격", "낙폭", "급락", "자금흐름", "매크로")), reverse=True)
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
        "macro": {
            "vix": round(float(vix), 1) if vix is not None else None,
            "fng": int(fng) if fng is not None else None,
            "risk": round(macro_risk * 100),
            "status": "ok" if macro_avail else "missing",
            "hard": hard,
        },
        "rationale": rationale,
    }
