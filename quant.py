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
    """비교 대안 비중. '200MA engine'은 라이브 엔진(engine.py)과 동일: 200선 위 보유 + 과열 15%씩 익절
    (이격 25/42/60단계, 재진입 여유 5%p). 'exposure_core'(20-parameter engine)는 메인 stats로 별도 표시."""
    k = ST.indikit(df)
    close, sma200, rvol = k["close"], k["sma200"], k["rvol"]
    warm = sma200.isna()
    maxcap = p["maxcap"]
    above = (close > sma200).to_numpy()
    disp = (close / sma200 - 1.0).to_numpy()
    warm_a = warm.to_numpy()
    # 역배열 기울기전환용: 역배열(50<200) & 200선 하락이면 관망(현금)
    bear_a = (k["sma50"] < sma200).to_numpy()
    rising_a = (sma200 > sma200.shift(20)).to_numpy()
    watch_a = bear_a & ~rising_a   # 역배열+하락 = 관망

    # 200MA engine = 라이브 코어 (200선 + 단계 익절 15% + 재진입 여유 5%p, 경로의존) + 역배열 관망
    TH = (0.25, 0.42, 0.60); CUT = 0.15; MARG = 0.05
    eng = np.zeros(len(close)); level = 0
    for i in range(len(close)):
        if warm_a[i] or not above[i] or watch_a[i]:
            eng[i] = 0.0; level = 0; continue
        d = disp[i]
        while level < 3 and d > TH[level]:
            level += 1
        while level > 0 and d < (TH[level - 1] - MARG):
            level -= 1
        eng[i] = max(0.0, 1.0 - CUT * level)
    eng200 = pd.Series(eng, index=df.index)

    ma = pd.Series(np.where(above, 1.0, 0.0), index=df.index)            # 순수 200선(참고)
    vt = (p["target_vol"] / rvol).clip(p["vt_floor"], p["vt_cap"]).clip(upper=maxcap)
    fixed = pd.Series(maxcap, index=df.index)
    for s in (eng200, ma, vt, fixed):
        s[warm] = np.nan
    return {"200MA engine": eng200, "200MA 단순": ma, "변동성타겟": vt, "고정비중": fixed}


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
    TH = (0.25, 0.42, 0.60)

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
        "역배열시 기울기전환": E(np.where(bear, np.where(above & rising, 1.0, 0.0), np.where(above, 1.0, 0.0))),
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


def below200_response_research(df, cost_bps=5.0, expense=0.0095):
    """[200선 이하 분할 대응 검증] CAGR 엔진 + 패닉그리드 가중을 하나로.
    200선 이탈 후 행동을 5개 시나리오로 비교 — 청산방식 × 받는방식.
    A: 칼마(전량청산, 안받음) / B: 청산+균등받기 / C: 청산+가중받기
    D: 홀딩+균등물타기 / E: 홀딩+가중물타기. 단계 -15/-30/-45%. 홀딩(D·E)은 손절선 스윕.
    위(200선 상향)·과열익절은 전 시나리오 동일(칼마와 같음) — 차이는 오직 200선 이하 대응.
    ※ 깊은 이격 표본 적음(과적합 위험) → 단순 가중(3:2:1), 4종목·칼마 우선으로 판정."""
    df = df.reset_index(drop=True)
    k = ST.indikit(df)
    close, sma200 = k["close"], k["sma200"]
    warm = sma200.isna().to_numpy()
    above = (close > sma200).to_numpy()
    disp = (close / sma200 - 1.0).to_numpy()
    idx = df.index; n = len(df)
    # 과열 익절(전 시나리오 공통, 칼마와 동일): 200선 위에서의 비중
    TH = (0.25, 0.42, 0.60); CUT = 0.15
    def above_expo(i, level_box):
        d = disp[i]; lv = level_box[0]
        while lv < 3 and d > TH[lv]: lv += 1
        while lv > 0 and d < (TH[lv-1] - 0.05): lv -= 1
        level_box[0] = lv
        return max(0.0, 1.0 - CUT*lv)
    STEPS = (-0.15, -0.30, -0.45)       # 분할 받기 단계
    W_EQ = (1/3, 1/3, 1/3)              # 균등
    W_WT = (1/6, 2/6, 3/6)             # 가중(깊을수록↑, 3:2:1 → -45%에 3)

    def E(arr):
        s = pd.Series(arr, index=idx, dtype=float); s[pd.Series(warm, index=idx)] = np.nan
        return s
    def run(arr):
        sr, pos, turn, _ = _exec_returns(df, E(arr).to_numpy(float), cost_bps, expense)
        m = _metrics(sr, pos, turn); m.update(_trade_stats(sr, pos)); return m

    def build(mode, weights=None, stop=None):
        """mode: 'calmar'|'liq_buy'|'hold_buy'.
        calmar: 200선 아래 전량 현금(0).
        liq_buy: 200선 이탈 시 청산(0), 깊은 이격 단계에서 누적 분할매수(현금→매수), cap 1.0.
        hold_buy: 200선 아래도 핵심 0.55 유지(홀딩), 깊은 단계서 추가매수 cap 1.0, 손절선 이하 0."""
        expo = np.zeros(n); lvbox = [0]
        for i in range(n):
            if warm[i]:
                expo[i] = 0.0; continue
            if above[i]:
                expo[i] = above_expo(i, lvbox); continue
            lvbox[0] = 0  # 200선 아래로 내려오면 익절단계 리셋
            d = disp[i]
            if mode == 'calmar':
                expo[i] = 0.0
            elif mode == 'liq_buy':
                base = 0.0
                for s_i, th in enumerate(STEPS):
                    if d <= th: base += weights[s_i]
                expo[i] = min(1.0, base)
            elif mode == 'hold_buy':
                if stop is not None and d <= stop:
                    expo[i] = 0.0
                else:
                    add = 0.0
                    for s_i, th in enumerate(STEPS):
                        if d <= th: add += weights[s_i]
                    expo[i] = min(1.0, 0.55 + add)   # 핵심 55% 홀딩 + 깊은 단계 추가매수
        return expo

    out = {"scenarios": {}, "stops": ["-40%", "-50%", "-60%", "없음"]}
    out["scenarios"]["A_칼마"] = run(build('calmar'))
    out["scenarios"]["B_청산균등"] = run(build('liq_buy', W_EQ))
    out["scenarios"]["C_청산가중"] = run(build('liq_buy', W_WT))
    # D·E: 손절선 스윕
    STOPS = {"-40%": -0.40, "-50%": -0.50, "-60%": -0.60, "없음": -0.99}
    out["D_홀딩균등"] = {}; out["E_홀딩가중"] = {}
    for label, sv in STOPS.items():
        out["D_홀딩균등"][label] = run(build('hold_buy', W_EQ, sv))
        out["E_홀딩가중"][label] = run(build('hold_buy', W_WT, sv))
    return out


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
    TH = (0.25, 0.42, 0.60)

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


def bear_history_research(df, cost_bps=5.0, expense=0.0095, lev_fee=0.0095):
    """[장기 약세장 검증] 기초지수(예: QQQ, 1999~)에 200MA+이격 엔진을 적용해 닷컴·2008 등
    진짜 약세장에서 작동하는지 본다. 1배(지수 그대로) + ×3합성(일일수익률×3, 변동성 감쇠는
    복리로 자동 반영, 운용보수 차감) 둘 다 계산. 신호는 1배 지수 기준(레버리지 ETF 전략 표준).
    레버리지 ETF가 그 시절 없었으므로 합성은 '있었다면' 추정치."""
    df = df.reset_index(drop=True)
    k = ST.indikit(df)
    close = k["close"].to_numpy(float)
    sma200 = k["sma200"].to_numpy(float)
    warm = np.isnan(sma200)
    above = close > sma200
    disp = (close / sma200 - 1.0)
    dates = pd.to_datetime(df["date"])
    dret = np.zeros(len(close))
    dret[1:] = close[1:] / close[:-1] - 1.0
    # 역배열 관망(50<200 & 200하락)
    watch = (k["sma50"].to_numpy(float) < sma200) & ~(k["sma200"] > k["sma200"].shift(20)).to_numpy()

    # 두 엔진 노출 계산
    # (1) 200MA engine = 라이브 코어 (200선 + 15%익절 + 5%p재진입 + 역배열 관망)
    TH = (0.25, 0.42, 0.60); CUT = 0.15; MARG = 0.05
    e200 = np.zeros(len(close)); level = 0
    for i in range(len(close)):
        if warm[i] or not above[i] or watch[i]:
            e200[i] = 0.0; level = 0; continue
        d = disp[i]
        while level < 3 and d > TH[level]:
            level += 1
        while level > 0 and d < (TH[level - 1] - MARG):
            level -= 1
        e200[i] = max(0.0, 1.0 - CUT * level)
    # (2) 20-parameter engine = exposure_core (변동성타겟·급성리스크)
    try:
        ecore = ST.exposure_core(df, {**ST.PARAMS}, k).to_numpy(float)
        ecore = np.where(np.isnan(ecore), 0.0, ecore)
    except Exception:
        ecore = e200.copy()

    cost = cost_bps / 10000.0
    fee_d = expense / 252.0
    levfee_d = lev_fee / 252.0

    def series_metrics(daily_ret):
        eq = np.cumprod(1.0 + daily_ret)
        peak = np.maximum.accumulate(eq)
        mdd = float((eq / peak - 1.0).min() * 100)
        yrs = len(daily_ret) / 252.0
        cagr = float((eq[-1] ** (1 / yrs) - 1.0) * 100) if yrs > 0 and eq[-1] > 0 else None
        total = float((eq[-1] - 1.0) * 100)
        calmar = round(cagr / abs(mdd), 2) if (cagr and mdd < 0) else None
        return {"cagr": round(cagr, 1) if cagr is not None else None,
                "total": round(total, 1), "mdd": round(mdd, 1), "calmar": calmar}

    dret3 = 3.0 * dret
    bh_1x = dret.copy()
    bh_3x = dret3 - (fee_d + levfee_d)

    def engine_series(expo):
        pos = np.zeros(len(close)); pos[1:] = expo[:-1]   # 신호 다음날 반영
        turn = np.abs(np.diff(pos, prepend=0.0))
        s1 = pos * dret - turn * cost - (pos > 0) * fee_d
        s3 = pos * dret3 - turn * cost - (pos > 0) * (fee_d + levfee_d)
        return s1, s3

    eng_defs = [("200MA engine", e200), ("20-parameter engine", ecore)]
    engines = {}
    eng_arrays = {}
    for nm, ex in eng_defs:
        s1, s3 = engine_series(ex)
        engines[nm] = {"strat_1x": series_metrics(s1), "strat_3x": series_metrics(s3)}
        eng_arrays[nm] = (s1, s3)

    out = {
        "engines": engines,
        "bh_1x": series_metrics(bh_1x), "bh_3x": series_metrics(bh_3x),
        "start": str(df["date"].iloc[0]), "end": str(df["date"].iloc[-1]),
        "bars": len(df),
    }
    # 약세장 구간별 (지수가 그 기간을 포함할 때만)
    windows = [("닷컴버블", "2000-03-01", "2002-10-31"),
               ("금융위기", "2007-10-01", "2009-03-31"),
               ("코로나", "2020-02-15", "2020-04-30"),
               ("2022 약세장", "2022-01-01", "2022-12-31")]
    wins = []
    dser = dates
    for nm, s, e in windows:
        mask = (dser >= s) & (dser <= e)
        if mask.sum() < 5:
            continue
        idx = np.where(mask.to_numpy())[0]
        sl = slice(idx[0], idx[-1] + 1)
        def wret(arr):
            return round(float((np.cumprod(1.0 + arr[sl])[-1] - 1.0) * 100), 1)
        def wmdd(arr):
            eq = np.cumprod(1.0 + arr[sl]); pk = np.maximum.accumulate(eq)
            return round(float((eq / pk - 1.0).min() * 100), 1)
        w = {"name": nm, "from": s, "to": e,
             "bh_3x": wret(bh_3x), "bh_3x_mdd": wmdd(bh_3x), "eng": {}}
        for enm, _ex in eng_defs:
            _s1, _s3 = eng_arrays[enm]
            w["eng"][enm] = {"ret_3x": wret(_s3), "mdd_3x": wmdd(_s3)}
        wins.append(w)
    out["windows"] = wins
    return out


def cross_pretrade_research(df, cost_bps=5.0, expense=0.0095):
    """[200선 부근 크로스 선매매 격자검증] 준호 가설 + 세분화.
    주가가 200일선 ±Z% 부근일 때, 크로스 신호 방향으로 돌파를 선반영하면 단순 200MA보다 나은가.
    축A 신호: MACD만 / 이평크로스만(20/60·50/200·5/20) / 둘다(교집합).
    축B 존: 1.0~3.5% 촘촘히.
    각 칸에 칼마·수익(기본대비) + 선매매 적중률. 평원(이웃 Z도 함께 좋은가)으로 과적합 점검.
    ※ 한 종목·한 칸만 좋은 건 과적합 — 4종목 일관 + 넓은 평원이라야 의미."""
    df = df.reset_index(drop=True)
    close = df["close"]
    sma200 = close.rolling(200).mean()
    warm = sma200.isna().to_numpy()
    above = (close > sma200).to_numpy()
    disp = (close / sma200 - 1.0).to_numpy()
    idx = df.index
    n = len(df)
    LB, K = 5, 10
    ZONES = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035]

    def cross_dir(up_bool, dn_bool):
        """크로스 이벤트(상향/하향)를 최근 LB봉 이내 유지되는 방향(+1/-1/0)으로."""
        up = np.asarray(up_bool); dn = np.asarray(dn_bool)
        r = np.zeros(n); ld, li = 0, -999
        for i in range(n):
            if up[i]:
                ld, li = 1, i
            elif dn[i]:
                ld, li = -1, i
            r[i] = ld if (i - li) <= LB else 0
        return r

    # MACD 크로스
    mline, msig, _ = I.macd(close)
    macd_dir = cross_dir((mline > msig) & (mline.shift(1) <= msig.shift(1)),
                         (mline < msig) & (mline.shift(1) >= msig.shift(1)))
    # 이평 크로스 (기간쌍별)
    ma_pairs = {"20/60": (20, 60), "50/200": (50, 200), "5/20": (5, 20)}
    ma_dirs = {}
    for nm, (f, s) in ma_pairs.items():
        fa = close.rolling(f).mean(); sa = close.rolling(s).mean()
        ma_dirs[nm] = cross_dir((fa > sa) & (fa.shift(1) <= sa.shift(1)),
                                (fa < sa) & (fa.shift(1) >= sa.shift(1)))

    def E(arr):
        ser = pd.Series(arr, index=idx, dtype=float)
        ser[pd.Series(warm, index=idx)] = np.nan
        return ser

    def run(e):
        strat, pos, turn, _ = _exec_returns(df, e, cost_bps, expense)
        return _metrics(strat, pos, turn)

    e_base = np.where(above, 1.0, 0.0)
    base_m = run(E(e_base))

    def variant(direction, Z):
        """방향신호 + 존Z로 선매매 비중배열 + 적중률."""
        zone = np.abs(disp) <= Z
        e = e_base.copy(); hits = 0; trades = 0
        for i in range(n):
            if warm[i] or not zone[i] or direction[i] == 0:
                continue
            nv = 1.0 if direction[i] > 0 else 0.0
            if nv != e_base[i]:
                trades += 1
                j = min(n - 1, i + K)
                if (direction[i] > 0 and above[j]) or (direction[i] < 0 and not above[j]):
                    hits += 1
            e[i] = nv
        m = run(E(e))
        m["trades"] = trades
        m["hit_rate"] = round(hits / trades * 100, 1) if trades else None
        m["d_calmar"] = round(m["calmar"] - base_m["calmar"], 2)
        m["d_total"] = round(m["total"] - base_m["total"], 1)
        return m

    # 신호 종류별 방향배열
    signals = {"MACD": macd_dir}
    for nm in ma_pairs:
        signals["이평 " + nm] = ma_dirs[nm]
    # 둘다(교집합): MACD와 (주력)20/60이 같은 방향일 때만
    both = np.where((macd_dir != 0) & (macd_dir == ma_dirs["20/60"]), macd_dir, 0.0)
    signals["MACD+이평20/60"] = both

    grid = {}
    for snm, dirarr in signals.items():
        row = {}
        for Z in ZONES:
            row["±%.1f%%" % (Z * 100)] = variant(dirarr, Z)
        # 평원 점검: d_calmar>0 인 존이 연속으로 몇 개인가(최대 연속)
        flags = [row[k]["d_calmar"] > 0.02 for k in row]
        best = 0; cur = 0
        for f in flags:
            cur = cur + 1 if f else 0
            best = max(best, cur)
        grid[snm] = {"cells": row, "plateau": best, "n_zones": len(ZONES)}
    return {"base": base_m, "zones": ["±%.1f%%" % (z * 100) for z in ZONES], "grid": grid}


def _div_sign_series(close, ind, lookback=40, win=3):
    """as-of-time 다이버전스 부호(+1 강세/-1 약세/0). 피벗은 win봉 뒤 확정 → 미래참조 차단.
    패널 divergence()와 동일 판정(강세 우선), 단 각 시점에서 그때까지 '확정된' 피벗만 사용."""
    cv = np.asarray(close, float); iv = np.asarray(ind, float)
    n = len(cv)
    his, los = [], []  # (known_at, pivot_idx)
    for i in range(win, n - win):
        seg = cv[i - win:i + win + 1]
        if cv[i] == seg.max():
            his.append((i + win, i))
        if cv[i] == seg.min():
            los.append((i + win, i))
    out = np.zeros(n)
    hi_ptr = lo_ptr = 0
    known_hi, known_lo = [], []
    for t in range(n):
        while hi_ptr < len(his) and his[hi_ptr][0] <= t:
            known_hi.append(his[hi_ptr][1]); hi_ptr += 1
        while lo_ptr < len(los) and los[lo_ptr][0] <= t:
            known_lo.append(los[lo_ptr][1]); lo_ptr += 1
        sign = 0
        ll = [p for p in known_lo if p > t - lookback]
        if len(ll) >= 2:
            a, b = ll[-2], ll[-1]
            if cv[b] < cv[a] and iv[b] > iv[a]:
                sign = 1
        if sign == 0:
            hh = [p for p in known_hi if p > t - lookback]
            if len(hh) >= 2:
                a, b = hh[-2], hh[-1]
                if cv[b] > cv[a] and iv[b] < iv[a]:
                    sign = -1
        out[t] = sign
    return out


def _trade_stats(strat, pos):
    """포지션 시계열에서 거래당 통계: 거래수, 승률, 거래당 기대값(%), time-in-market."""
    pos = np.nan_to_num(np.asarray(pos, float), nan=0.0)
    r = np.nan_to_num(np.asarray(strat, float), nan=0.0)
    tim = float(np.mean(pos > 0)) * 100 if len(pos) else 0.0
    trades = []; in_pos = False; eq = 1.0
    for i in range(len(pos)):
        if pos[i] > 0 and not in_pos:
            in_pos = True; eq = 1.0
        if in_pos:
            eq *= (1.0 + (r[i] if i < len(r) else 0.0))
        if (pos[i] == 0 or i == len(pos) - 1) and in_pos:
            in_pos = False; trades.append(eq - 1.0)
    nt = len(trades)
    wins = [x for x in trades if x > 0]
    win = (len(wins) / nt * 100) if nt else None
    exp = (float(np.mean(trades)) * 100) if nt else None
    return {"trades": nt, "win_rate": round(win, 1) if win is not None else None,
            "expectancy": round(exp, 2) if exp is not None else None,
            "time_in_market": round(tim, 1)}


def divergence_pretrade_research(df, cost_bps=5.0, expense=0.0095):
    """[다이버전스 선매매 검증] 다른 세션 설계 + 평원/4종목/단순화 방어.
    A: 200MA 단독(기준) / B: 다이버전스 단독 / C: 200MA+다이버전스 선행트리거 / D: Buy&Hold.
    score = Σ(부호×가중치) — 강세 다이버전스 +(매수), 약세 -(청산). 임계 S를 스윕(평원 점검).
    단순화 대조: 5지표(중복 많음) vs 2축(RSI=모멘텀·OBV=거래량만). 미래참조 차단(피벗 win봉 지연).
    ※ 5지표가 같은 종가 피벗을 써서 정보중복 큼 → 과적합/무의미 기각 가능성 높음(검증용)."""
    df = df.reset_index(drop=True)
    close = df["close"]
    sma200 = close.rolling(200).mean()
    warm = sma200.isna().to_numpy()
    above = (close > sma200).to_numpy()
    disp = (close / sma200 - 1.0).to_numpy()
    idx = df.index; n = len(df)
    # 5개 지표 부호 시계열 (패널과 동일 지표)
    rsi14 = I.rsi(close, 14)
    mline, _, _ = I.macd(close)
    _, stoch_d = I.stoch(df)
    mfi14 = I.mfi(df, 14)
    obv_s = I.obv(df)
    sgn = {
        "RSI": _div_sign_series(close, rsi14), "MACD": _div_sign_series(close, mline),
        "STOCH": _div_sign_series(close, stoch_d), "MFI": _div_sign_series(close, mfi14),
        "OBV": _div_sign_series(close, obv_s),
    }
    W = {"RSI": 1.0, "MACD": 1.0, "STOCH": 0.8, "MFI": 0.7, "OBV": 0.7}
    score5 = sum(W[k] * sgn[k] for k in W)               # 5지표 가중합 (max 4.2)
    score2 = 1.0 * sgn["RSI"] + 1.0 * sgn["OBV"]         # 2축: 모멘텀(RSI)+거래량(OBV) (max 2.0)

    def E(arr):
        s = pd.Series(arr, index=idx, dtype=float); s[pd.Series(warm, index=idx)] = np.nan
        return s

    def run(e):
        strat, pos, turn, _ = _exec_returns(df, e, cost_bps, expense)
        m = _metrics(strat, pos, turn); m.update(_trade_stats(strat, pos)); return m

    # 기준 A / 벤치 D
    A = run(E(np.where(above, 1.0, 0.0)))
    D = run(E(np.ones(n)))

    def stratB(score, S):
        """다이버전스 단독: 강세 score>=S 매수, 약세 score<=-S 청산, 그 외 유지."""
        e = np.zeros(n); cur = 0.0
        for i in range(n):
            if score[i] >= S: cur = 1.0
            elif score[i] <= -S: cur = 0.0
            e[i] = cur
        return e

    def stratC(score, S, Z=0.03, K=10):
        """200MA + 선행: 200선 ±Z% 부근에서 강세면 선진입, 약세면 선청산. 그 외 200MA.
        헛진입(whipsaw): 선진입했으나 K봉 내 실제 200상향 미확정 비율/손익."""
        base = np.where(above, 1.0, 0.0); e = base.copy()
        pre_up = 0; pre_up_fail = 0
        for i in range(n):
            if warm[i] or abs(disp[i]) > Z: continue
            if score[i] >= S:
                if base[i] == 0.0:  # 선진입(아직 200 아래인데 미리 매수)
                    pre_up += 1
                    j = min(n - 1, i + K)
                    if not above[j]: pre_up_fail += 1
                e[i] = 1.0
            elif score[i] <= -S:
                e[i] = 0.0
        wf = round(pre_up_fail / pre_up * 100, 1) if pre_up else None
        return e, pre_up, wf

    SCORES5 = [0.7, 0.9, 1.4, 2.0, 2.8]
    SCORES2 = [1.0, 2.0]   # 2축은 max 2.0
    out = {"A": A, "D": D, "scores5": SCORES5, "scores2": SCORES2,
           "B5": {}, "B2": {}, "C5": {}, "C2": {}}

    def delta(m):
        m = dict(m); m["d_calmar"] = round(m["calmar"] - A["calmar"], 2)
        m["d_total"] = round(m["total"] - A["total"], 1); return m

    for S in SCORES5:
        out["B5"]["%.1f" % S] = delta(run(E(stratB(score5, S))))
        e, pu, wf = stratC(score5, S); m = delta(run(E(e))); m["pre_up"] = pu; m["whipsaw"] = wf
        out["C5"]["%.1f" % S] = m
    for S in SCORES2:
        out["B2"]["%.1f" % S] = delta(run(E(stratB(score2, S))))
        e, pu, wf = stratC(score2, S); m = delta(run(E(e))); m["pre_up"] = pu; m["whipsaw"] = wf
        out["C2"]["%.1f" % S] = m

    def plateau(d):
        ks = sorted(d.keys(), key=float)
        flags = [d[k]["d_calmar"] > 0.02 for k in ks]
        best = cur = 0
        for f in flags:
            cur = cur + 1 if f else 0; best = max(best, cur)
        return best
    out["plateau"] = {"B5": plateau(out["B5"]), "C5": plateau(out["C5"]),
                      "B2": plateau(out["B2"]), "C2": plateau(out["C2"])}
    out["n5"] = len(SCORES5); out["n2"] = len(SCORES2)
    return out


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
    bear_hist = bear_history_research(df, cost_bps, expense)
    cross_pre = cross_pretrade_research(df, cost_bps, expense)
    div_pre = divergence_pretrade_research(df, cost_bps, expense)
    below200 = below200_response_research(df, cost_bps, expense)
    return {"stats": base["stats"], "equity": base["equity"], "bh": base["bh"],
            "walk_forward": wf, "robustness": rob, "monte_carlo": mc,
            "baselines": bl, "regime_perf": rp, "rolling": roll, "ma_research": mar,
            "entry_diag": entry_diag, "exit_research": exitr, "reentry_research": reentry,
            "bear_research": bearr, "breakdown_research": breakdown, "dipentry_research": dipentry,
            "trim_regime": trim_regime, "bear_history": bear_hist, "cross_pretrade": cross_pre,
            "divergence_pretrade": div_pre, "below200_response": below200,
            "cost_bps": cost_bps, "expense": expense}
