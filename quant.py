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

ANN = 252


def _strat_returns(df, params, cost_bps=5.0, expense=0.0095):
    """T+1 시가 체결 기준 전략 일수익률 시계열 + 포지션/벤치마크 반환."""
    o = df["open"].to_numpy(dtype=float)
    e = ST.exposure_core(df, params).to_numpy(dtype=float)   # e[t]=종가[t]에서 결정된 목표비중
    N = len(df)
    e_prev = np.concatenate([[np.nan], e[:-1]])              # 시가[t]에 체결되는 비중 = e[t-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        oo = np.concatenate([o[1:] / o[:-1] - 1.0, [np.nan]])  # 시가[t]->시가[t+1] 수익(index t)
    pos = np.nan_to_num(e_prev)
    gross = pos * np.nan_to_num(oo)
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))     # 비중 변화량(회전율)
    cost = turn * (cost_bps / 1e4) + pos * (expense / ANN)   # 거래비용 + 운용보수 일할
    strat = gross - cost
    bh = np.nan_to_num(oo)                                   # 매수후보유(시가 풀투자)
    # 마지막 봉은 oo가 NaN → 0 처리됨
    return pd.Series(strat, index=df.index), pd.Series(bh, index=df.index), pd.Series(pos, index=df.index), turn


def _metrics(r: pd.Series, pos: pd.Series = None, turn=None) -> dict:
    r = r.fillna(0.0)
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
    out = {"total": round(total, 1), "cagr": round(cagr, 1), "vol": round(vol, 1),
           "sharpe": round(sharpe, 2), "sortino": round(sortino, 2),
           "mdd": round(mdd, 1), "calmar": round(calmar, 2)}
    if pos is not None:
        out["avg_expo"] = round(float(pos.mean()) * 100, 1)
        out["time_in"] = round(float((pos > 0.01).mean()) * 100, 1)
    if turn is not None:
        out["turnover_yr"] = round(float(np.nansum(turn)) / max(1, n) * ANN, 1)
    return out


def _sharpe(r: pd.Series) -> float:
    r = r.fillna(0.0)
    return float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0


def backtest(df, params=None, cost_bps=5.0, expense=0.0095):
    df = df.reset_index(drop=True)
    p = {**ST.PARAMS, **(params or {})}
    strat, bh, pos, turn = _strat_returns(df, p, cost_bps, expense)
    st = _metrics(strat, pos, turn)
    bh_st = _metrics(bh)
    st["bh_total"] = bh_st["total"]; st["bh_mdd"] = bh_st["mdd"]; st["bh_cagr"] = bh_st["cagr"]
    dates = df["date"].astype(str).tolist()
    eq = (1 + strat.fillna(0)).cumprod(); bhe = (1 + bh.fillna(0)).cumprod()
    step = max(1, len(df) // 400)
    equity = [{"time": dates[i], "value": round(float(eq.iloc[i]), 4)} for i in range(0, len(df), step)]
    bhcurve = [{"time": dates[i], "value": round(float(bhe.iloc[i]), 4)} for i in range(0, len(df), step)]
    st["period"] = f"{dates[0]} ~ {dates[-1]}"; st["bars"] = len(df)
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
        # 인샘플 최적화 (Sharpe 최대)
        best, best_sh = None, -1e9
        for tv, ac in grid:
            r, _, _, _ = _strat_returns(is_df, {"target_vol": tv, "acute_strength": ac}, cost_bps, expense)
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
    grid = []
    for ac in acs:
        row = []
        for tv in tvs:
            r, _, _, _ = _strat_returns(df, {"target_vol": tv, "acute_strength": ac}, cost_bps, expense)
            row.append(round(_sharpe(r), 2))
        grid.append(row)
    flat = [v for row in grid for v in row]
    return {"x_label": "target_vol", "x": tvs, "y_label": "acute_strength", "y": acs,
            "z": grid, "max": max(flat), "min": min(flat),
            "spread": round(max(flat) - min(flat), 2)}


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


def analyze(df, cost_bps=5.0, expense=0.0095):
    base = backtest(df, None, cost_bps, expense)
    wf = walk_forward(df, cost_bps=cost_bps, expense=expense)
    rob = robustness(df, cost_bps, expense)
    mc = monte_carlo(base["ret"])
    return {"stats": base["stats"], "equity": base["equity"], "bh": base["bh"],
            "walk_forward": wf, "robustness": rob, "monte_carlo": mc,
            "cost_bps": cost_bps, "expense": expense}
