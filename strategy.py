"""
strategy.py — 마틸다 단일 전략 코어 (라이브 대시보드 = 백테스트가 '같은 코드'를 쓴다)

설계 원칙
  1) 지표는 여기서 한 번만 계산한다 (engine/backtest 중복 제거, DRY).
  2) 포지션 사이징(exposure_core)은 '가격/거래량만' 쓰는 완전 벡터화 함수다.
     - 각 시점 t의 값은 t까지의 rolling 데이터만 사용 → 미래참조(look-ahead) 없음.
     - 이 함수가 백테스트와 라이브 스냅샷 양쪽에서 동일하게 호출된다.
  3) 매크로(VIX·공포탐욕)는 '코어'가 아니라 라이브 전용 안전 오버레이로 분리한다.
     - 무료로 신뢰할 만한 과거 VIX/공포탐욕 시계열이 없어 백테스트엔 넣지 않는다(정직성).
     - 따라서 '백테스트로 검증되는 대상 = 코어(추세+변동성+급성리스크)'이고,
       라이브는 그 위에 매크로 하드게이트를 덧씌운다. 이 경계를 명확히 한다.
  4) 모든 임계/계수는 PARAMS 로 외부화한다. 하드코딩 상수를 없앤다.
     - 단, "최적 파라미터"는 존재하지 않는다. quant.py 의 워크포워드로
       '특정 값에 민감하지 않은지(robustness)'를 검사하는 용도로만 쓴다.
"""
import numpy as np
import pandas as pd
import indicators as I

# ─────────────────────────────────────────────────────────────
# 파라미터 (전부 외부화). RANGES 는 워크포워드 견고성 검사용 탐색 격자.
# ─────────────────────────────────────────────────────────────
PARAMS = {
    "target_vol": 35.0,   # 포지션 목표 연율변동성(%)
    "maxcap": 0.80,       # 최대 권장 비중
    "vt_floor": 0.15,     # 변동성타겟 계수 하한
    "vt_cap": 1.00,       # 상한
    # 추세 레짐별 기본 비중계수
    "tf_strong": 1.00,    # 200일선 위 + 50>200 정배열
    "tf_weak": 0.55,      # 200일선 위, 정배열 아님
    "tf_down": 0.12,      # 200일선 아래
    "tf_unknown": 0.40,   # 200일선 미형성(워밍업)
    "adx_range": 20.0,    # ADX 이 값 미만이면 횡보 → 비중 축소
    "tf_range_mult": 0.60,
    # 급성 리스크(가격 기반) 정규화 임계 (start, span)
    "ext_lo": 10.0, "ext_span": 20.0,    # |20MA 이격|
    "dd_lo": 5.0,  "dd_span": 20.0,      # 20일 고점 낙폭
    "crash_lo": 3.0, "crash_span": 10.0,  # 1일 급락
    "cmf_k": 2.0,                         # 자금유출 민감도
    "acute_w_ext": 0.22, "acute_w_dd": 0.26, "acute_w_crash": 0.22, "acute_w_cmf": 0.10,
    "acute_strength": 0.70,               # 급성리스크가 비중을 깎는 강도
    # 리스크 점수(0~100, 표시용) 가중치
    "rw_vol": 0.20, "rw_trend": 0.24, "rw_ext": 0.10, "rw_dd": 0.12,
    "rw_crash": 0.10, "rw_cmf": 0.06, "rw_macro": 0.18,
    "vol_risk_lo": 40.0, "vol_risk_span": 120.0,
}

# 워크포워드에서 견고성을 볼 핵심 파라미터만 (전부 돌리면 과적합·계산폭발)
RANGES = {
    "target_vol": [25, 30, 35, 40, 45, 50],
    "acute_strength": [0.4, 0.55, 0.7, 0.85, 1.0],
    "adx_range": [15, 20, 25],
}


def _clamp01(s):
    return s.clip(0.0, 1.0)


def indikit(df: pd.DataFrame) -> dict:
    """전략에 필요한 rolling 지표를 '한 번만' 계산해 dict로 반환 (중복 제거)."""
    close = df["close"]
    adx_, pdi, mdi = I.adx(df)
    return {
        "close": close,
        "sma20": close.rolling(20).mean(),
        "sma50": close.rolling(50).mean(),
        "sma60": close.rolling(60).mean(),
        "sma120": close.rolling(120).mean(),
        "sma200": close.rolling(200).mean(),
        "adx": adx_, "pdi": pdi, "mdi": mdi,
        "rvol": I.realized_vol(close, 20),
        "cmf": I.cmf(df, 20),
        "ret1": close.pct_change() * 100,
        "dd20": (close / close.rolling(20).max() - 1) * 100,
    }


def _components(k: dict, p: dict) -> dict:
    """리스크/급성 컴포넌트(0~1 시계열). exposure 와 risk score 가 공유."""
    close, sma20 = k["close"], k["sma20"]
    ext20 = (close - sma20) / sma20 * 100
    comp = {
        "ext": _clamp01((ext20.abs() - p["ext_lo"]) / p["ext_span"]),
        "dd": _clamp01((-k["dd20"] - p["dd_lo"]) / p["dd_span"]),
        "crash": _clamp01((-k["ret1"] - p["crash_lo"]) / p["crash_span"]),
        "cmf": _clamp01(-k["cmf"] * p["cmf_k"]).fillna(0.0),
        "vol": _clamp01((k["rvol"] - p["vol_risk_lo"]) / p["vol_risk_span"]).fillna(0.0),
    }
    comp["ext20_raw"] = ext20
    return comp


def regime_series(k: dict, p: dict):
    """추세 레짐 계수(시계열) + 횡보 여부. 코어 사이징의 추세 게이트."""
    close, sma50, sma200 = k["close"], k["sma50"], k["sma200"]
    above200 = close > sma200
    align_up = sma50 > sma200
    tf = pd.Series(np.select(
        [above200 & align_up, above200 & ~align_up, ~above200],
        [p["tf_strong"], p["tf_weak"], p["tf_down"]], default=p["tf_unknown"]),
        index=close.index, dtype=float)
    ranging = (k["adx"] < p["adx_range"]).fillna(False)
    tf = tf * np.where(ranging, p["tf_range_mult"], 1.0)
    tf[sma200.isna()] = p["tf_unknown"]
    return tf, ranging, above200, align_up


def exposure_core(df: pd.DataFrame, p: dict = None, k: dict = None) -> pd.Series:
    """
    코어 포지션 비중(0~maxcap, 분수) 시계열 — 가격/거래량만 사용, 미래참조 없음.
    ★ 라이브 스냅샷과 백테스트가 모두 이 함수를 호출한다 (단일 진실원). ★
    k(indikit 결과)를 넘기면 재계산을 생략한다 — 격자 탐색(워크포워드/견고성)에서 지표는 파라미터와
    무관하므로 한 번만 계산해 재사용하면 연산이 수십 배 빨라진다.
    """
    p = {**PARAMS, **(p or {})}
    if k is None:
        k = indikit(df)
    comp = _components(k, p)
    tf, ranging, _, _ = regime_series(k, p)
    vt = (p["target_vol"] / k["rvol"]).clip(p["vt_floor"], p["vt_cap"]).fillna(0.5)
    acute = (comp["ext"] * p["acute_w_ext"] + comp["dd"] * p["acute_w_dd"]
             + comp["crash"] * p["acute_w_crash"] + comp["cmf"] * p["acute_w_cmf"])
    size = (p["maxcap"] * tf * vt * (1 - p["acute_strength"] * acute)).clip(0, p["maxcap"])
    size[k["sma200"].isna()] = np.nan   # 200봉 워밍업 전엔 미거래
    return size


# ─────────────────────────────────────────────────────────────
# 매크로 오버레이 (라이브 전용) — 백테스트엔 적용하지 않음(과거 데이터 부재)
# ─────────────────────────────────────────────────────────────
def macro_risk_value(vix, fng) -> float:
    if vix is None and fng is None:
        return 0.0
    vix_risk = max(0.0, min(1.0, (vix - 15) / 25)) if vix is not None else 0.0
    if fng is not None:
        greed = max(0.0, min(1.0, (fng - 70) / 30))
        fear = max(0.0, min(1.0, (30 - fng) / 30))
        fng_risk = max(greed, 0.6 * fear)
    else:
        fng_risk = 0.0
    if vix is not None and fng is not None:
        return max(0.0, min(1.0, 0.6 * vix_risk + 0.4 * fng_risk))
    return vix_risk if vix is not None else fng_risk


def macro_overlay(size_pct: int, risk: int, vix, fng):
    """라이브 last-bar 비중에 매크로 소프트 감산 + 하드게이트 적용."""
    p = PARAMS
    mr = macro_risk_value(vix, fng)
    # 소프트: 매크로 위험을 급성리스크처럼 추가 감산
    size = size_pct * (1 - 0.20 * mr)
    hard = None
    if vix is not None and vix >= 45:
        size = min(size, 5); hard = "VIX 패닉(≥45)"
    elif (vix is not None and vix >= 32) or (fng is not None and fng >= 90):
        size = min(size, 15); hard = "매크로 극단(VIX≥32 또는 극단적 탐욕)"
    elif fng is not None and fng <= 8:
        size = min(size, 22); hard = "극단적 공포(패닉)"
    if hard:
        risk = max(risk, 65)
    return int(round(size)), int(risk), round(mr * 100), hard
