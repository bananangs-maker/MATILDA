"""
quant.py — 퀀트 백테스트/리서치 레이어

핵심 원칙
  • 백테스트 대상 = strategy.exposure_core (라이브 엔진이 쓰는 '바로 그' 코어). 별도 전략 아님.
  • 체결: 종가[t]에서 결정 → 익일 시가[t+1] 체결(T+1). 같은 봉 내 미래참조 차단.
  • 비용: 거래비용(bps, 스프레드+수수료) + 레버리지 ETF 운용보수(연 0.9% 일할).
  • 검증: 롤링 워크포워드(인샘플 최적화 → 아웃오브샘플 적용)로 과적합 점검.
           견고성 히트맵(파라미터 평원) + 몬테카를로(블록 부트스트랩).

정직한 한계 (반드시 인지)
  • 매크로(VIX·공포탐욕)는 백테스트에 없음 — 신뢰할 무료 과거 시계열 부재. 코어=가격only.
  • 양도소득세(연 단위·실현기준)는 모델링하지 않음(일별 수익률 백테스트와 성격이 다름).
  • 무료 데이터: 레버리지 ETF는 2010년 전후부터, 배당·분할 조정이 불완전할 수 있어 약간 낙관 편향.
  • '최적 파라미터'를 찾는 게 목적이 아니다 → 넓은 평원(robustness)과 OOS 유지가 목적.
"""
import numpy as np
import pandas as pd
import strategy as ST
import indicators as I

ANN = 252


def _exec_returns(df, e, cost_bps=5.0, expense=0.0095):
    """주어진 목표비중 시계열 e[t](종가[t] 결정)를 T+1 시가 체결로 실행 → 일수익률/포지션/회전율.
    워밍업(NaN) 구간은 NaN 유지 → 통계에서 제외."""
    o = df["open"].to_numpy(dtype=float)
    e = np.asarray(e, dtype=float)
    e_prev = np.concatenate([[np.nan], e[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        oo = np.concatenate([o[1:] / o[:-1] - 1.0, [np.nan]])
    pos = e_prev
    gross = pos * oo
    prev = np.concatenate([[np.nan], pos[:-1]])
    turn = np.where(np.isnan(pos), np.nan, np.abs(pos - np.nan_to_num(prev)))
    cost = turn * (cost_bps / 1e4) + pos * (expense / ANN)
    strat = gross - cost
    return (pd.Series(strat, index=df.index), pd.Series(pos, index=df.index),
            pd.Series(turn, index=df.index), pd.Series(oo, index=df.index))


def _strat_returns(df, params, cost_bps=5.0, expense=0.0095):
    """전략 코어(exposure_core)를 T+1 실행. + 동일 활성구간 매수후보유(BH)."""
    e = ST.exposure_core(df, params).to_numpy(dtype=float)
    strat, pos, turn, oo = _exec_returns(df, e, cost_bps, expense)
    bh = pd.Series(np.where(np.isnan(pos.to_numpy()), np.nan, oo.to_numpy()), index=df.index)
    return strat, bh, pos, turn


def _metrics(r: pd.Series, pos: pd.Series = None, turn=None) -> dict:
    r = r.dropna()                       # 워밍업/마지막 NaN 제외 → 활성 구간만
    if len(r) == 0:
        return {"total": 0, "cagr": 0, "vol": 0, "sharpe": 0, "sortino": 0, "mdd": 0, "calmar": 0}
    eq = (1 + r).cumprod()
    n = len(r)
    total = float(eq.iloc[-1] - 1) * 100
    cagr = float(eq.iloc[-1] ** (ANN / max(1, n)) - 1) * 100 if eq.iloc[-1] > 0 else -100.0
    vol = float(r.std() * np.sqrt(ANN) * 100)
    sharpe = float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0
    downside = r[r < 0]
    sortino = float(r.mean() / downside.std() * np.sqrt(ANN)) if len(downside) > 1 and downside.std() > 0 else 0.0
    mdd = float(((eq / eq.cummax()) - 1).min() * 100)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    worst = float(r.min() * 100)
    cvar5 = float(r[r <= r.quantile(0.05)].mean() * 100) if n >= 20 else worst
    ddc = (eq / eq.cummax() - 1)
    ulcer = float(np.sqrt((ddc ** 2).mean()) * 100)
    out = {"total": round(total, 1), "cagr": round(cagr, 1), "vol": round(vol, 1),
           "sharpe": round(sharpe, 2), "sortino": round(sortino, 2),
           "mdd": round(mdd, 1), "calmar": round(calmar, 2),
           "worst_day": round(worst, 1), "cvar5": round(cvar5, 1), "ulcer": round(ulcer, 1)}
    if pos is not None:
        pp = pos.dropna()
        out["avg_expo"] = round(float(pp.mean()) * 100, 1) if len(pp) else 0.0
        out["time_in"] = round(float((pp > 0.01).mean()) * 100, 1) if len(pp) else 0.0
    if turn is not None:
        tv = turn.to_numpy() if hasattr(turn, "to_numpy") else turn
        out["turnover_yr"] = round(float(np.nansum(tv)) / max(1, n) * ANN, 1)
    return out


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(ANN)) if len(r) > 1 and r.std() > 0 else 0.0


def backtest(df, params=None, cost_bps=5.0, expense=0.0095):
    df = df.reset_index(drop=True)
    p = {**ST.PARAMS, **(params or {})}
    strat, bh, pos, turn = _strat_returns(df, p, cost_bps, expense)
    st = _metrics(strat, pos, turn)
    bh_st = _metrics(bh)
    st["bh_total"] = bh_st["total"]; st["bh_mdd"] = bh_st["mdd"]; st["bh_cagr"] = bh_st["cagr"]
    dates = df["date"].astype(str).tolist()
    sa = strat.dropna()                      # 활성 구간만(워밍업 0-수익 선두 제거)
    ba = bh.reindex(sa.index)
    eq = (1 + sa.fillna(0)).cumprod(); bhe = (1 + ba.fillna(0)).cumprod()
    pidx = list(sa.index)                    # df 정수 인덱스(활성 봉 위치)
    m = len(sa); step = max(1, m // 400)
    equity = [{"time": dates[pidx[i]], "value": round(float(eq.iloc[i]), 4)} for i in range(0, m, step)]
    bhcurve = [{"time": dates[pidx[i]], "value": round(float(bhe.iloc[i]), 4)} for i in range(0, m, step)]
    st["period"] = f"{dates[pidx[0]]} ~ {dates[pidx[-1]]}" if m else "—"
    st["bars"] = m
    st["cost_bps"] = cost_bps; st["expense"] = expense
    return {"stats": st, "equity": equity, "bh": bhcurve, "ret": strat}


# ── 롤링 워크포워드: 인샘플 최적화 → 아웃오브샘플 적용 (과적합 점검) ──
def walk_forward(df, n_folds=4, cost_bps=5.0, expense=0.0095):
    df = df.reset_index(drop=True)
    N = len(df)
    if N < 300:
        return {"error": "데이터 부족(워크포워드 최소 300봉)"}
    grid = [(tv, ac) for tv in ST.RANGES["target_vol"] for ac in ST.RANGES["acute_strength"]]
    fold = N // (n_folds + 1)   # 첫 fold는 학습 전용, 이후 각 구간을 OOS로
    rows, oos_all = [], []
    for f in range(n_folds):
        is_end = fold * (f + 1)
        oos_end = min(fold * (f + 2), N)
        is_df = df.iloc[:is_end]
        oos_df = df.iloc[max(0, is_end - 220):oos_end]   # 지표 워밍업 위해 약간 겹쳐 로드
        # 인샘플 최적화 (Sharpe 최대) — 지표는 파라미터 무관하므로 1회만 계산해 재사용
        k_is = ST.indikit(is_df)
        best, best_sh = None, -1e9
        for tv, ac in grid:
            e = ST.exposure_core(is_df, {"target_vol": tv, "acute_strength": ac}, k=k_is).to_numpy(dtype=float)
            r, _, _, _ = _exec_returns(is_df, e, cost_bps, expense)
            sh = _sharpe(r)
            if sh > best_sh:
                best_sh, best = sh, (tv, ac)
        # 아웃오브샘플 적용 (겹친 워밍업 구간 제거)
        r_oos_full, _, _, _ = _strat_returns(oos_df, {"target_vol": best[0], "acute_strength": best[1]}, cost_bps, expense)
        cut = len(oos_df) - (oos_end - is_end)
        r_oos = r_oos_full.iloc[cut:] if cut > 0 else r_oos_full
        oos_all.append(r_oos)
        rows.append({"fold": f + 1,
                     "is_period": f"{df['date'].iloc[0]}~{df['date'].iloc[is_end-1]}",
                     "oos_period": f"{df['date'].iloc[is_end]}~{df['date'].iloc[oos_end-1]}",
                     "best_tv": best[0], "best_acute": best[1],
                     "is_sharpe": round(best_sh, 2), "oos_sharpe": round(_sharpe(r_oos), 2),
                     "oos_ret": round((np.prod(1 + r_oos.fillna(0)) - 1) * 100, 1)})
    oos = pd.concat(oos_all)
    oos_sh = _sharpe(oos)
    is_avg = float(np.mean([r["is_sharpe"] for r in rows]))
    verdict = ("견고 (OOS가 IS의 절반 이상 유지)" if oos_sh > 0 and oos_sh >= is_avg * 0.5
               else "주의 (아웃오브샘플에서 성과 약화 — 과적합 가능)")
    return {"rows": rows, "oos_sharpe": round(oos_sh, 2), "is_avg_sharpe": round(is_avg, 2),
            "oos_total": round((np.prod(1 + oos.fillna(0)) - 1) * 100, 1),
            "n_folds": n_folds, "verdict": verdict}


# ── 견고성 히트맵: target_vol × acute_strength 그리드의 Sharpe (평원 확인) ──
def robustness(df, cost_bps=5.0, expense=0.0095):
    df = df.reset_index(drop=True)
    tvs = ST.RANGES["target_vol"]; acs = ST.RANGES["acute_strength"]
    k_all = ST.indikit(df)                      # 지표 1회 계산 후 격자 전체 재사용
    z_sh, z_cal = [], []
    for ac in acs:
        rs, rc = [], []
        for tv in tvs:
            e = ST.exposure_core(df, {"target_vol": tv, "acute_strength": ac}, k=k_all).to_numpy(dtype=float)
            r, _, _, _ = _exec_returns(df, e, cost_bps, expense)
            m = _metrics(r)
            rs.append(round(m["sharpe"], 2)); rc.append(round(m["calmar"], 2))
        z_sh.append(rs); z_cal.append(rc)
    flat = [v for row in z_sh for v in row]
    flatc = [v for row in z_cal for v in row]
    return {"x_label": "target_vol", "x": tvs, "y_label": "acute_strength", "y": acs,
            "z": z_sh, "max": max(flat), "min": min(flat), "spread": round(max(flat) - min(flat), 2),
            "z_calmar": z_cal, "max_c": max(flatc), "min_c": min(flatc)}


# ── 몬테카를로: 블록 부트스트랩으로 최대낙폭/최종수익 분포 ──
def monte_carlo(strat_ret: pd.Series, n_sims=600, block=5, seed=7):
    rng = np.random.default_rng(seed)
    r = strat_ret.dropna().to_numpy()
    N = len(r)
    if N < block * 5:
        return {"error": "데이터 부족"}
    finals, mdds = [], []
    nblocks = N // block + 1
    for _ in range(n_sims):
        starts = rng.integers(0, N - block, size=nblocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:N]
        s = r[idx]
        eq = np.cumprod(1 + s)
        finals.append(eq[-1] - 1)
        mdds.append(((eq / np.maximum.accumulate(eq)) - 1).min())
    finals = np.array(finals) * 100; mdds = np.array(mdds) * 100
    pct = lambda a, q: round(float(np.percentile(a, q)), 1)
    hist, edges = np.histogram(mdds, bins=10)
    return {"final": {"p5": pct(finals, 5), "p50": pct(finals, 50), "p95": pct(finals, 95)},
            "mdd": {"p5": pct(mdds, 5), "p50": pct(mdds, 50), "worst": round(float(mdds.min()), 1)},
            "mdd_hist": {"counts": hist.tolist(), "edges": [round(float(e), 1) for e in edges]},
            "n_sims": n_sims, "block": block}


def _baseline_exposures(df, p):
    """복잡한 엔진의 가치를 검증하기 위한 단순 대안 비중들. 모두 전략과 같은 활성구간(200일선 형성 후)."""
    k = ST.indikit(df)
    close, sma200, rvol = k["close"], k["sma200"], k["rvol"]
    warm = sma200.isna()
    maxcap = p["maxcap"]
    ma = pd.Series(np.where(close > sma200, 1.0, 0.0), index=df.index)   # 200일선 위=풀투자, 아래=현금
    vt = (p["target_vol"] / rvol).clip(p["vt_floor"], p["vt_cap"]).clip(upper=maxcap)  # 변동성타겟만
    fixed = pd.Series(maxcap, index=df.index)                           # 상수 비중(=maxcap)
    for s in (ma, vt, fixed):
        s[warm] = np.nan
    return {"200MA 필터": ma, "변동성타겟": vt, "고정비중": fixed}


def baselines(df, p, cost_bps=5.0, expense=0.0095):
    out = {}
    for name, e in _baseline_exposures(df, p).items():
        sr, pos, turn, _ = _exec_returns(df, e, cost_bps, expense)
        out[name] = _metrics(sr, pos, turn)
    return out


def regime_perf(df, p, cost_bps=5.0, expense=0.0095):
    """레짐(추세/횡보, 200일선 위/아래)별 조건부 성과 — 언제 작동/실패하는지."""
    strat, _, _, _ = _strat_returns(df, p, cost_bps, expense)
    k = ST.indikit(df)
    above = (k["close"] > k["sma200"]).shift(1).fillna(False).astype(bool)   # 보유 포지션은 직전 봉 레짐
    ranging = (k["adx"] < p["adx_range"]).shift(1).fillna(False).astype(bool)

    def seg(mask):
        rr = strat[mask].dropna()
        if len(rr) < 10:
            return None
        eq = (1 + rr).cumprod()
        return {"n": int(len(rr)), "total": round(float(eq.iloc[-1] - 1) * 100, 1),
                "sharpe": round(_sharpe(rr), 2),
                "mdd": round(float((eq / eq.cummax() - 1).min() * 100), 1)}
    return {"above200": seg(above), "below200": seg(~above),
            "trending": seg(~ranging), "ranging": seg(ranging)}


def rolling_series(df, p, cost_bps=5.0, expense=0.0095, win=63):
    """롤링 샤프(63일) + 진행 드로다운 — 차트용 다운샘플."""
    strat, _, _, _ = _strat_returns(df, p, cost_bps, expense)
    sa = strat.dropna()
    if len(sa) < win + 10:
        return {"points": []}
    eq = (1 + sa).cumprod()
    dd = (eq / eq.cummax() - 1) * 100
    rs = (sa.rolling(win).mean() / sa.rolling(win).std() * np.sqrt(ANN))
    dates = df["date"].astype(str).to_numpy()
    idx = list(sa.index)
    step = max(1, len(sa) // 400)
    pts = []
    for i in range(0, len(sa), step):
        v = rs.iloc[i]
        pts.append({"time": dates[idx[i]], "dd": round(float(dd.iloc[i]), 1),
                    "sharpe": (round(float(v), 2) if v == v else None)})
    return {"points": pts, "win": win}


def entry_research(df, cost_bps=5.0, expense=0.0095):
    """[분할 진입 연구] 200MA 위=홀딩 골격은 고정. '진입 속도(한 방 vs 분할)'와 '재돌파 확인일'만 변형.
    가설: 분할/확인은 휩쏘(가짜 돌파) 손실을 줄인다. 단 추세장 상승은 약간 놓친다 → 칼마/MDD/회전율로 판정."""
    df = df.reset_index(drop=True)
    k = ST.indikit(df)
    close, sma200 = k["close"], k["sma200"]
    warm = sma200.isna().to_numpy()
    above = (close > sma200).to_numpy()
    N = len(df)

    def in_signal(confirm):
        if confirm <= 1:
            return above.copy()
        s = pd.Series(above.astype(float)).rolling(confirm).sum().to_numpy()
        return (s >= confirm)          # confirm일 연속 200선 위

    def ramp_target(insig, ramp_days):
        e = np.empty(N); cur = 0.0
        for t in range(N):
            if warm[t]:
                e[t] = np.nan; cur = 0.0; continue
            if insig[t]:
                cur = min(1.0, cur + 1.0 / ramp_days)   # 보유 중 매일 목표까지 분할 증액
            else:
                cur = 0.0                               # 200선 아래 = 현금
            e[t] = cur
        return e

    variants = {
        "즉시(200MA)": ramp_target(in_signal(1), 1),
        "분할5일": ramp_target(in_signal(1), 5),
        "분할10일": ramp_target(in_signal(1), 10),
        "분할20일": ramp_target(in_signal(1), 20),
        "확인3일+분할10": ramp_target(in_signal(3), 10),
        "확인5일+분할10": ramp_target(in_signal(5), 10),
    }
    out = {}
    for nm, e in variants.items():
        sr, pos, turn, _ = _exec_returns(df, e, cost_bps, expense)
        out[nm] = _metrics(sr, pos, turn)

    # ── 휩쏘 진단: 200선 교차/년 + (분할10일 − 즉시) 칼마 차이 ──
    act = ~warm
    av = above[act]
    crossings = int(np.sum(av[1:] != av[:-1])) if av.size > 1 else 0
    years = max(0.1, av.size / 252.0)
    cross_yr = round(crossings / years, 1)
    ci = out["즉시(200MA)"]["calmar"]; cs = out["분할10일"]["calmar"]
    delta = round(cs - ci, 2)
    rec = "분할 유리" if delta >= 0.05 else ("즉시 유리" if delta <= -0.05 else "차이 미미")
    diag = {"cross_yr": cross_yr, "calmar_즉시": ci, "calmar_분할10": cs,
            "delta": delta, "recommend": rec,
            "mdd_즉시": out["즉시(200MA)"]["mdd"], "mdd_분할10": out["분할10일"]["mdd"]}
    return out, diag


def exit_research(df, cost_bps=5.0, expense=0.0095):
    """[과열 익절 견고성 검증] 이격익절의 임계값 세트를 흔들어 칼마가 평원인지(강건) 스파이크인지(과적합) 확인.
    여러 임계값 세트에서 모두 홀딩(200MA)을 이기면 진짜 효과. TQQQ·SOXL 둘 다 봐야 함."""
    df = df.reset_index(drop=True)
    k = ST.indikit(df)
    close, sma200 = k["close"], k["sma200"]
    warm = sma200.isna().to_numpy()
    above = (close > sma200).to_numpy()
    disp = (close / sma200 - 1.0).to_numpy()
    rsi = I.rsi(close, 14).to_numpy()
    idx = df.index

    def E(arr):
        s = pd.Series(arr, index=idx, dtype=float)
        s[pd.Series(warm, index=idx)] = np.nan
        return s

    def trim_by(x, t1, t2, t3):   # 3단계 익절: t1↑→0.9, t2↑→0.8, t3↑→0.7
        return np.where(x > t3, 0.7, np.where(x > t2, 0.8, np.where(x > t1, 0.9, 1.0)))

    # 이격익절 임계값 세트들(공격적→보수적). 평원이면 강건.
    disp_sets = {
        "홀딩(200MA)": None,
        "이격 10/22/35": (0.10, 0.22, 0.35),
        "이격 15/30/45": (0.15, 0.30, 0.45),   # 기존 채택 후보
        "이격 20/35/50": (0.20, 0.35, 0.50),
        "이격 25/42/60": (0.25, 0.42, 0.60),
        "RSI 70/78/85": "rsi",                  # 비교용(RSI 기준)
    }
    out = {}
    for nm, st in disp_sets.items():
        if st is None:
            e = np.where(above, 1.0, 0.0)
        elif st == "rsi":
            e = np.where(above, trim_by(rsi, 70, 78, 85), 0.0)
        else:
            e = np.where(above, trim_by(disp, *st), 0.0)
        sr, pos, turn, _ = _exec_returns(df, E(e).to_numpy(dtype=float), cost_bps, expense)
        out[nm] = _metrics(sr, pos, turn)
    return out


def exit_reentry_research(df, cost_bps=5.0, expense=0.0095):
    """[익절 비율·재진입 검증] (A) 단계당 익절 비율(5/10/15/20%) 중 칼마 최적은?
    (B) 이격 회복 시 재진입을 '같은 임계'(휩쏘 위험) vs '히스테리시스(여유)' 중 뭐가 나은가.
    임계는 20/35/50 고정. TQQQ·SOXL 둘 다 봐야 함. 회전율(turnover)로 휩쏘 비용도 확인."""
    df = df.reset_index(drop=True)
    k = ST.indikit(df)
    close, sma200 = k["close"], k["sma200"]
    warm = sma200.isna().to_numpy()
    above = (close > sma200).to_numpy()
    disp = (close / sma200 - 1.0).to_numpy()
    idx = df.index
    TH = (0.20, 0.35, 0.50)

    def E(arr):
        s = pd.Series(arr, index=idx, dtype=float)
        s[pd.Series(warm, index=idx)] = np.nan
        return s

    def exposure(disp_arr, above_arr, cut, margin):
        """단계 익절+재진입(히스테리시스). cut=단계당 익절비율, margin=재진입 여유(이격 %p).
        level=적용된 익절 단계 수(0~3). 이격이 TH[level] 위로↑면 익절(level+1),
        TH[level-1]-margin 아래로↓면 재진입(level-1)."""
        n = len(disp_arr); expo = np.zeros(n); level = 0
        for i in range(n):
            d = disp_arr[i]
            if not above_arr[i]:
                expo[i] = 0.0; level = 0; continue
            # 익절: 다음 임계 위로 올라가면 단계 상승
            while level < 3 and d > TH[level]:
                level += 1
            # 재진입: 직전 임계 - margin 아래로 내려오면 단계 하강
            while level > 0 and d < (TH[level - 1] - margin):
                level -= 1
            expo[i] = max(0.0, 1.0 - cut * level)
        return expo

    out = {"base": {}, "trim_sweep": {}, "reentry_sweep": {}}
    # 기준: 홀딩(200MA)
    sr, pos, turn, _ = _exec_returns(df, E(np.where(above, 1.0, 0.0)).to_numpy(float), cost_bps, expense)
    out["base"] = _metrics(sr, pos, turn)
    # (A) 익절 비율 sweep — 25/33%까지 넓혀 '끝없이 좋아지는지(과적합)' vs '정점 후 꺾이는지(진짜)' 확인
    #     33%씩×3단계 = 사실상 과열 시 전량청산. 단조증가면 신호의 본질은 '과열=전량탈출'이라는 뜻.
    for cut in (0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.33):
        e = exposure(disp, above, cut, 0.0)
        sr, pos, turn, _ = _exec_returns(df, E(e).to_numpy(float), cost_bps, expense)
        m = _metrics(sr, pos, turn); _t = turn.dropna(); m["turnover"] = round(float(_t.sum()) / max(1, len(_t)) * ANN, 1)
        out["trim_sweep"][f"{int(cut*100)}%씩"] = m
    # (B) 재진입 여유(히스테리시스) sweep (익절=10% 고정)
    for margin in (0.0, 0.05, 0.10):
        e = exposure(disp, above, 0.10, margin)
        sr, pos, turn, _ = _exec_returns(df, E(e).to_numpy(float), cost_bps, expense)
        m = _metrics(sr, pos, turn); _t = turn.dropna(); m["turnover"] = round(float(_t.sum()) / max(1, len(_t)) * ANN, 1)
        lbl = "같은 임계(여유0)" if margin == 0 else f"여유 {int(margin*100)}%p"
        out["reentry_sweep"][lbl] = m
    # 신뢰도 진단: 익절이 실제로 '몇 번의 과열 사건'에 기대는가 (표본 두께)
    d_above20 = (disp > 0.20) & above & ~warm
    episodes = int(((d_above20) & ~(pd.Series(d_above20).shift(1).fillna(False).to_numpy())).sum())  # 20% 상향 돌파 횟수
    days20 = int(d_above20.sum())
    days35 = int(((disp > 0.35) & above & ~warm).sum())
    days50 = int(((disp > 0.50) & above & ~warm).sum())
    yrs = max(0.1, int((~warm).sum()) / 252)
    out["reliability"] = {"episodes": episodes, "ep_per_yr": round(episodes / yrs, 1),
                          "days20": days20, "days35": days35, "days50": days50,
                          "total_bars": int((~warm).sum())}
    return out


def bear_research(df, cost_bps=5.0, expense=0.0095):
    """[대세하락 검증] 역배열(50<200) 장기하락에서 '200MA 재돌파 재진입'이 휩쏘로 손해인지,
    '정배열/200기울기 필터로 관망'이 나은지 검증. 전체 히스토리(2000~ 약세장 포함) 필요.
    핵심: 역배열 구간만 떼어 각 규칙의 성과를 비교."""
    df = df.reset_index(drop=True)
    k = ST.indikit(df)
    close, sma50, sma200 = k["close"], k["sma50"], k["sma200"]
    warm = sma200.isna().to_numpy()
    above = (close > sma200).to_numpy()
    align = (sma50 > sma200).to_numpy()                       # 정배열
    rising = (sma200 > sma200.shift(20)).to_numpy()           # 200선 우상향(20일 기준)
    bear = (sma50 < sma200).to_numpy() & ~warm                # 역배열 구간
    idx = df.index

    def E(arr):
        s = pd.Series(arr, index=idx, dtype=float)
        s[pd.Series(warm, index=idx)] = np.nan
        return s

    variants = {
        "200MA 단독": E(np.where(above, 1.0, 0.0)),
        "정배열 필터": E(np.where(above & align, 1.0, 0.0)),
        "200기울기 필터": E(np.where(above & rising, 1.0, 0.0)),
        "정배열+기울기": E(np.where(above & align & rising, 1.0, 0.0)),
    }
    out = {}
    bear_perf = {}
    for nm, e in variants.items():
        sr, pos, turn, _ = _exec_returns(df, e.to_numpy(dtype=float), cost_bps, expense)
        out[nm] = _metrics(sr, pos, turn)
        # 역배열 구간만의 성과(보유 포지션은 직전봉 레짐 기준)
        bmask = pd.Series(bear, index=idx).shift(1).fillna(False).astype(bool).to_numpy()
        rb = sr.to_numpy()[bmask]
        rb = rb[~np.isnan(rb)]
        if len(rb) > 5:
            eqb = np.cumprod(1 + rb)
            mdd_b = float((eqb / np.maximum.accumulate(eqb) - 1).min() * 100)
            bear_perf[nm] = {"ret": round((eqb[-1] - 1) * 100, 1), "mdd": round(mdd_b, 1), "days": int(len(rb))}
        else:
            bear_perf[nm] = {"ret": None, "mdd": None, "days": int(len(rb))}

    # 진단: 역배열 비중, 역배열 중 200선 교차 횟수(휩쏘)
    bear_days = int(bear.sum())
    total_days = int((~warm).sum())
    cross_in_bear = 0
    av = above
    for i in range(1, len(av)):
        if bear[i] and av[i] != av[i - 1]:
            cross_in_bear += 1
    bear_yrs = max(0.1, bear_days / 252.0)
    diag = {"bear_days": bear_days, "bear_pct": round(bear_days / max(1, total_days) * 100, 1),
            "cross_in_bear_yr": round(cross_in_bear / bear_yrs, 1),
            "period_start": str(df["date"].iloc[0]) if "date" in df.columns else None,
            "period_end": str(df["date"].iloc[-1]) if "date" in df.columns else None}
    return {"variants": out, "bear": bear_perf, "diag": diag}


def breakdown_research(df):
    """[하향돌파 후 이격 분포] 200선 하향돌파 후, 가격이 200선 대비 '얼마나 더 깊이' 빠졌다가
    회복하는지의 분포. 패닉 기준(X% 이격)의 데이터 근거. 보통 -A%에서 반등, 드물게 -B%까지."""
    df = df.reset_index(drop=True)
    k = ST.indikit(df)
    close = k["close"].to_numpy()
    sma200 = k["sma200"].to_numpy()
    warm = np.isnan(sma200)
    above = close > sma200
    dates = df["date"].tolist() if "date" in df.columns else list(range(len(df)))

    episodes = []
    i = 0
    n = len(df)
    while i < n:
        if warm[i] or above[i]:
            i += 1; continue
        # 하향돌파 시작 (직전봉이 위였거나 워밍업 직후)
        start = i
        trough = 0.0
        while i < n and not warm[i] and not above[i]:
            d = (close[i] / sma200[i] - 1.0) * 100
            if d < trough:
                trough = d
            i += 1
        recovered = (i < n and above[i])   # 200선 위로 회복하며 종료했나
        episodes.append({"start": str(dates[start]), "trough": float(round(trough, 1)),
                         "duration": int(i - start), "recovered": bool(recovered)})

    troughs = sorted([e["trough"] for e in episodes])   # 음수, 오름차순(깊은 게 앞)
    durs = sorted([e["duration"] for e in episodes])
    def pct(arr, q):
        if not arr: return None
        idx = min(len(arr) - 1, int(round(q / 100 * (len(arr) - 1))))
        return float(arr[idx])
    # trough는 음수라 '깊이' 백분위: 90%깊이 = 하위10%(가장 깊은 쪽)
    summary = {
        "n": len(episodes),
        "median": pct(troughs, 50),
        "p75_depth": pct(troughs, 25),   # 더 깊은 25% 경계
        "p90_depth": pct(troughs, 10),   # 더 깊은 10% 경계
        "worst": float(troughs[0]) if troughs else None,
        "dur_median": pct(durs, 50),
        "dur_max": int(durs[-1]) if durs else None,
    }
    deepest = sorted(episodes, key=lambda e: e["trough"])[:12]   # 가장 깊었던 사건들

    # 이격구간별 반등 통계: 각 밴드(200선 대비 음수 이격)에서 역사적으로 어땠나.
    # reached=그 깊이까지 간 사건 수, held=거기서 멈춤(더 깊이 안 감), 반등률=held/reached,
    # 회복률=200선 위로 돌아온 비율, 평균회복일=그 깊이까지 간 사건의 평균 지속.
    band_levels = [-10, -15, -20, -30, -45]
    bands = []
    for j, B in enumerate(band_levels):
        deeper = band_levels[j + 1] if j + 1 < len(band_levels) else None
        reached_eps = [e for e in episodes if e["trough"] <= B]
        reached = len(reached_eps)
        if deeper is not None:
            deeper_cnt = sum(1 for e in episodes if e["trough"] <= deeper)
            held = reached - deeper_cnt          # 이 밴드~다음 밴드 사이에서 멈춤(반등)
        else:
            deeper_cnt = 0
            held = reached                       # 최심 밴드: 도달분 전체를 '여기 근처서 멈춤'으로
        recovered_cnt = sum(1 for e in reached_eps if e["recovered"])
        avg_days = (round(float(np.mean([e["duration"] for e in reached_eps])), 0)
                    if reached_eps else None)
        bands.append({
            "band": B, "reached": reached,
            "bounce_rate": round(held / reached * 100, 0) if reached else None,   # 여기서 멈출 확률
            "deeper_rate": round(deeper_cnt / reached * 100, 0) if reached else None,  # 더 깊이 갈 확률
            "recover_rate": round(recovered_cnt / reached * 100, 0) if reached else None,
            "avg_recover_days": avg_days,
        })

    return {"summary": summary, "episodes": deepest, "bands": bands,
            "total_breaks": len(episodes)}


def dipentry_research(df, cost_bps=5.0, expense=0.0095):
    """[200선 아래 분할진입 검증] 준호 가설: 200선 하향돌파 후 깊은 이격 구간에서 분할 매수하면
    200선 상향돌파 시 전량진입보다 나은가. 비교: 전량(기준) vs 분할 A/B/C(이격 단계 다르게).
    분할 규칙: 200선 위=100% 보유. 200선 아래=각 이격 단계 도달 시 1/3씩 누적 진입(평단 낮추기),
    재상향돌파하면 100%. 깊은 약세장 표본이 적으니 신뢰도(하향돌파 횟수)도 함께 봄."""
    df = df.reset_index(drop=True)
    k = ST.indikit(df)
    close, sma200 = k["close"], k["sma200"]
    warm = sma200.isna().to_numpy()
    above = (close > sma200).to_numpy()
    disp = (close / sma200 - 1.0).to_numpy()
    idx = df.index

    def E(arr):
        s = pd.Series(arr, index=idx, dtype=float)
        s[pd.Series(warm, index=idx)] = np.nan
        return s

    def full_expo():
        """기준: 200선 위 100%, 아래 0% (현재 엔진)."""
        return np.where(above, 1.0, 0.0)

    def dip_expo(levels):
        """200선 위=100%. 아래로 내려가며 levels(음수 이격 %, 깊을수록)에서 1/3씩 진입.
        한번 잡은 비중은 더 깊어지면 추가(누적), 200선 위로 회복하면 100%로."""
        n = len(disp); expo = np.zeros(n); held = 0.0
        L = [l / 100.0 for l in levels]
        for i in range(n):
            if warm[i]:
                expo[i] = 0.0; continue
            if above[i]:
                held = 1.0
            else:
                d = disp[i]
                # 도달한 단계 수(깊을수록 더 많이 진입). 누적이라 더 깊어질 때만 증가.
                reached = sum(1 for l in L if d <= l)
                target = reached / 3.0
                if target > held:      # 더 깊어졌으면 추가 매수(누적)
                    held = target
                # 반등해도 200선 아래선 줄이지 않음(분할 매수 후 보유)
            expo[i] = held
        return expo

    variants = {
        "전량(200선 돌파)": full_expo(),
        "분할 A (-10/-20/-30)": dip_expo([-10, -20, -30]),
        "분할 B (-15/-30/-45)": dip_expo([-15, -30, -45]),
        "분할 C (-20/-35/-50)": dip_expo([-20, -35, -50]),
    }
    out = {}
    for nm, e in variants.items():
        sr, pos, turn, _ = _exec_returns(df, E(e).to_numpy(float), cost_bps, expense)
        out[nm] = _metrics(sr, pos, turn)

    # 신뢰도: 하향돌파 횟수 + 각 깊이 도달 횟수(분할 단계가 실제 몇 번 발동했나)
    n_break = 0
    for i in range(1, len(above)):
        if not warm[i] and above[i - 1] and not above[i]:
            n_break += 1
    reach = {"−15%이상": 0, "−30%이상": 0, "−45%이상": 0}
    troughs = []
    i = 0
    while i < len(df):
        if warm[i] or above[i]:
            i += 1; continue
        tr = 0.0
        while i < len(df) and not warm[i] and not above[i]:
            if disp[i] < tr: tr = disp[i]
            i += 1
        troughs.append(tr * 100)
    for t in troughs:
        if t <= -15: reach["−15%이상"] += 1
        if t <= -30: reach["−30%이상"] += 1
        if t <= -45: reach["−45%이상"] += 1
    diag = {"n_break": n_break, "reach": reach,
            "median_trough": round(float(np.median(troughs)), 1) if troughs else None,
            "worst_trough": round(float(min(troughs)), 1) if troughs else None}
    return {"variants": out, "diag": diag}


def trim_regime_compare(df, cost_bps=5.0, expense=0.0095, split="2026-01-01"):
    """[익절 비율 — 강세장 vs 정상장] 같은 익절 비율을 2026 포함(강세장)과 2026 제외(정상장)에서
    각각 칼마 계산. '15%가 강세장 이득 > 정상장 손해인가'를 한눈에. 견고한 타협점 탐색."""
    df = df.reset_index(drop=True)
    k = ST.indikit(df)
    TH = (0.20, 0.35, 0.50)

    def sweep(sub):
        sub = sub.reset_index(drop=True)
        kk = ST.indikit(sub)
        close, sma200 = kk["close"], kk["sma200"]
        warm = sma200.isna().to_numpy()
        above = (close > sma200).to_numpy()
        disp = (close / sma200 - 1.0).to_numpy()
        idx = sub.index

        def E(arr):
            s = pd.Series(arr, index=idx, dtype=float)
            s[pd.Series(warm, index=idx)] = np.nan
            return s

        def expo(cut):
            n = len(disp); out = np.zeros(n); level = 0
            for i in range(n):
                if not above[i]:
                    out[i] = 0.0; level = 0; continue
                while level < 3 and disp[i] > TH[level]:
                    level += 1
                while level > 0 and disp[i] < TH[level - 1]:
                    level -= 1
                out[i] = max(0.0, 1.0 - cut * level)
            return out
        res = {}
        for cut in (0.10, 0.15, 0.18, 0.20, 0.25, 0.33):
            sr, pos, turn, _ = _exec_returns(sub, E(expo(cut)).to_numpy(float), cost_bps, expense)
            m = _metrics(sr, pos, turn)
            res[f"{int(cut*100)}%"] = {"calmar": m["calmar"], "total": m["total"], "mdd": m["mdd"]}
        return res

    full = sweep(df)
    normal = sweep(df[df["date"] < split]) if (df["date"] >= split).any() else None
    # 각 국면 최적 비율
    def best(d):
        return max(d.items(), key=lambda kv: kv[1]["calmar"])[0] if d else None
    return {"ratios": ["10%", "15%", "18%", "20%", "25%", "33%"],
            "full": full, "normal": normal, "split": split,
            "best_full": best(full), "best_normal": best(normal),
            "has_2026": normal is not None}


def analyze(df, cost_bps=5.0, expense=0.0095):
    p = {**ST.PARAMS}
    base = backtest(df, None, cost_bps, expense)
    wf = walk_forward(df, cost_bps=cost_bps, expense=expense)
    rob = robustness(df, cost_bps, expense)
    mc = monte_carlo(base["ret"])
    bl = baselines(df, p, cost_bps, expense)
    rp = regime_perf(df, p, cost_bps, expense)
    roll = rolling_series(df, p, cost_bps, expense)
    mar, entry_diag = entry_research(df, cost_bps, expense)
    exitr = exit_research(df, cost_bps, expense)
    reentry = exit_reentry_research(df, cost_bps, expense)
    bearr = bear_research(df, cost_bps, expense)
    breakdown = breakdown_research(df)
    dipentry = dipentry_research(df, cost_bps, expense)
    trim_regime = trim_regime_compare(df, cost_bps, expense)
    return {"stats": base["stats"], "equity": base["equity"], "bh": base["bh"],
            "walk_forward": wf, "robustness": rob, "monte_carlo": mc,
            "baselines": bl, "regime_perf": rp, "rolling": roll, "ma_research": mar,
            "entry_diag": entry_diag, "exit_research": exitr, "reentry_research": reentry,
            "bear_research": bearr, "breakdown_research": breakdown, "dipentry_research": dipentry,
            "trim_regime": trim_regime,
            "cost_bps": cost_bps, "expense": expense}
