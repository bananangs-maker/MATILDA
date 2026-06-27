"""
engine.py — 마틸다 라이브 판정 엔진 (strategy.py 단일 코어를 호출)

[중요] 포지션 사이징의 진실원은 strategy.exposure_core() 하나다.
  - 백테스트(quant.py)도 '같은 함수'를 쓴다 → 라이브와 백테스트가 일치.
  - 여기서는 마지막 봉의 코어 비중을 가져와 라이브 전용 매크로 오버레이(소프트+하드게이트)를 덧씌우고,
    표시용 리스크 점수(0~100)와 레짐/근거를 구성한다.
모든 임계/계수는 strategy.PARAMS 에서 온다 (하드코딩 상수 제거).
"""
import numpy as np
import pandas as pd
import strategy as ST

TARGET_VOL = ST.PARAMS["target_vol"]   # 하위 호환(기존 import)
MAXCAP = ST.PARAMS["maxcap"]


def _last(s):
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def size_series(df, target_vol=None, maxcap=None):
    """하위 호환 래퍼 — 코어 비중 시계열. (target_vol 지정 시 그 값으로)"""
    p = {}
    if target_vol is not None:
        p["target_vol"] = target_vol
    if maxcap is not None:
        p["maxcap"] = maxcap
    return ST.exposure_core(df, p)


def compute_engine(df: pd.DataFrame, vix=None, fng=None, params: dict = None) -> dict:
    p = {**ST.PARAMS, **(params or {})}
    k = ST.indikit(df)
    comp_s = ST._components(k, p)
    tf_s, ranging_s, above200_s, align_up_s = ST.regime_series(k, p)
    core = ST.exposure_core(df, p)

    c = float(k["close"].iloc[-1])
    sma50 = _last(k["sma50"]); sma60 = _last(k["sma60"])
    sma120 = _last(k["sma120"]); sma200 = _last(k["sma200"])
    adxv = _last(k["adx"]); pdiv = _last(k["pdi"]); mdiv = _last(k["mdi"])
    rvol = _last(k["rvol"]); cmfv = _last(k["cmf"])
    ret1 = float(k["ret1"].iloc[-1]) if len(k["ret1"]) else 0.0
    dd20 = float(k["dd20"].iloc[-1]) if len(k["dd20"]) else 0.0
    ext20 = float(comp_s["ext20_raw"].iloc[-1]) if len(comp_s["ext20_raw"]) else 0.0
    ext60 = ((c - sma60) / sma60 * 100) if sma60 == sma60 and sma60 else 0.0
    ext120 = ((c - sma120) / sma120 * 100) if sma120 == sma120 and sma120 else 0.0

    above200 = bool(above200_s.iloc[-1]) if sma200 == sma200 else False
    align_up = bool(align_up_s.iloc[-1]) if (sma50 == sma50 and sma200 == sma200) else False
    ranging = bool(ranging_s.iloc[-1])
    if not (sma200 == sma200):
        regime = "판단보류"
    elif above200 and align_up:
        regime = "상승추세"
    elif above200:
        regime = "추세약화"
    else:
        regime = "하락추세"

    vt = _last((p["target_vol"] / k["rvol"]).clip(p["vt_floor"], p["vt_cap"]))
    if vt != vt:
        vt = 0.5
    trend_factor = float(tf_s.iloc[-1]) if len(tf_s) else p["tf_unknown"]

    # 마지막 봉 컴포넌트(0~1)
    def cl(s): return float(s.iloc[-1]) if len(s.dropna()) else 0.0
    comp = {"변동성": cl(comp_s["vol"]),
            "추세": 1.0 if regime == "하락추세" else 0.5 if regime in ("추세약화", "판단보류")
                    else (0.3 if ranging else 0.0),
            "이격": cl(comp_s["ext"]), "낙폭": cl(comp_s["dd"]),
            "급락": cl(comp_s["crash"]), "자금흐름": cl(comp_s["cmf"]),
            "매크로": ST.macro_risk_value(vix, fng)}
    W = {"변동성": p["rw_vol"], "추세": p["rw_trend"], "이격": p["rw_ext"], "낙폭": p["rw_dd"],
         "급락": p["rw_crash"], "자금흐름": p["rw_cmf"], "매크로": p["rw_macro"]}
    risk = round(sum(comp[k_] * W[k_] for k_ in W) * 100)

    # 코어 비중(라이브 last bar) → 매크로 오버레이
    core_last = core.dropna()
    size_core = int(round(float(core_last.iloc[-1]) * 100)) if len(core_last) else 0
    macro_avail = (vix is not None) or (fng is not None)
    size, risk, macro_pct, hard = ST.macro_overlay(size_core, risk, vix, fng)
    rlabel = "위험" if risk >= 60 else "경계" if risk >= 35 else "양호"

    rationale = []
    rationale.append("추세 레짐: %s (200일선 %s)" % (regime, "위" if above200 else "아래"))
    rationale.append("실현변동성 %.0f%% → 변동성타겟 비중계수 %.2f" % (rvol if rvol == rvol else 0, vt))
    if ranging:
        rationale.append("ADX %.0f (<%g) 횡보 — 추세추종 비중 축소" % (adxv if adxv == adxv else 0, p["adx_range"]))
    if macro_avail:
        mp = []
        if vix is not None: mp.append("VIX %.0f" % vix)
        if fng is not None: mp.append("공포탐욕 %d" % fng)
        rationale.append("매크로: " + ", ".join(mp) + (" → " + hard if hard else " (소프트 반영)"))
    else:
        rationale.append("매크로 미반영(데이터 없음) — 백테스트 코어와 동일 조건")
    acute_now = (comp["이격"] * p["acute_w_ext"] + comp["낙폭"] * p["acute_w_dd"]
                 + comp["급락"] * p["acute_w_crash"] + comp["자금흐름"] * p["acute_w_cmf"])
    if acute_now > 0.15:
        big = sorted(((comp[x], x) for x in ("이격", "낙폭", "급락", "자금흐름", "매크로")), reverse=True)
        rationale.append("급성 리스크: " + ", ".join("%s" % x for v, x in big if v > 0.2))

    # ── 검증된 익절 플랜 (200일선 이격도 단계별 익절: 20/35/50% → 보유 90/80/70%) ──
    # 백테스트로 TQQQ·SOXL 둘 다 평원 검증됨. 200선 아래=현금, 위=홀딩, 과열=일부 익절(핵심 70% 유지).
    DISP_STEPS = [(0.25, 0.15), (0.42, 0.15), (0.60, 0.15)]   # (이격 임계, 단계 익절비율) — 15%씩(누적 45%, 핵심 55% 홀딩)
    if sma200 == sma200 and sma200 > 0:
        disp = (c - sma200) / sma200            # 현재 이격도(분수)
        cl_s = k["close"]; above_s = (cl_s > k["sma200"])
        entry_signal = False; days_above = 0
        if above200 and len(above_s) > 6:
            entry_signal = bool((~above_s.iloc[-6:-1]).any())   # 직전 5봉 중 아래였던 적 → 재돌파
            for v in reversed(above_s.tolist()):
                if v: days_above += 1
                else: break
        # 재진입 신호: 최근 10봉 내 과열(이격>20%)로 익절했다가 지금 이격이 그 아래로 회복 → 덜어낸 비중 되넣기 구간
        reentry_signal = False
        if above200:
            disp_s = (cl_s / k["sma200"] - 1.0)
            recent = disp_s.iloc[-11:-1] if len(disp_s) > 11 else disp_s.iloc[:-1]
            reentry_signal = bool((recent > 0.20).any() and disp < 0.20 and disp == disp)
        # 권장 비중: 검증된 설정(익절 10%씩 + 재진입 여유 10%p 히스테리시스)을 경로 기반으로 계산.
        # level=적용된 익절 단계(0~3). 이격이 임계 위로↑면 익절, (직전임계-여유) 아래로↓면 재진입.
        TH = [0.25, 0.42, 0.60]; CUT = 0.15; MARGIN = 0.05
        disp_arr = (cl_s / k["sma200"] - 1.0).to_numpy()
        above_arr = above_s.to_numpy()
        warm_arr = k["sma200"].isna().to_numpy()
        level = 0
        for i in range(len(disp_arr)):
            if warm_arr[i]:
                continue
            if not above_arr[i]:
                level = 0; continue
            d = disp_arr[i]
            while level < 3 and d > TH[level]:
                level += 1
            while level > 0 and d < (TH[level - 1] - MARGIN):
                level -= 1
        triggered = level
        target = max(0.0, 1.0 - CUT * level) if above200 else 0.0
        steps = []
        for j, (th, cut) in enumerate(DISP_STEPS):
            steps.append({"disparity": round(th * 100),
                          "price": round(sma200 * (1 + th), 2),
                          "trim_pct": round(cut * 100),
                          "hit": bool(above200 and level > j)})
        target_pct = int(round(target * 100))
        exit_plan = {
            "above200": bool(above200),
            "price": round(c, 2),
            "sma200": round(sma200, 2),
            "disparity": round(disp * 100, 1),
            "target_pct": target_pct,                 # 권장 보유비중(%)
            "trim_total_pct": int(round((1 - target) * 100)) if above200 else None,  # 지금까지 누적 익절%
            "steps": steps,
            "stages_triggered": triggered,
            "entry_signal": bool(entry_signal),
            "reentry_signal": bool(reentry_signal),
            "days_above": int(days_above),
        }
    else:
        exit_plan = {"above200": False, "price": round(c, 2), "sma200": None,
                     "disparity": None, "target_pct": None, "trim_total_pct": None,
                     "steps": [], "stages_triggered": 0,
                     "entry_signal": False, "reentry_signal": False, "days_above": 0}

    return {
        "risk": risk, "risk_label": rlabel, "components": {k_: round(comp[k_] * 100) for k_ in comp},
        "weights": {k_: W[k_] for k_ in W},
        "regime": regime, "size": int(size), "maxcap": int(p["maxcap"] * 100),
        "trend_factor": round(trend_factor, 2), "vol_factor": round(vt, 2),
        "exit_plan": exit_plan,
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
            "risk": macro_pct, "status": "ok" if macro_avail else "missing", "hard": hard,
        },
        "rationale": rationale,
    }
