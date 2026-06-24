"""
backtest_engine.py — Phase 3 전략 검증
엔진(연속 비중)을 과거에 적용해 검증한다.
 - 거래비용/슬리피지 반영
 - 워크포워드: 인샘플에서 파라미터 선택 → 아웃오브샘플 성과로 과적합 점검
 - 몬테카를로: 블록 부트스트랩으로 최대낙폭/최종수익 분포 추정
"""
import numpy as np
import pandas as pd
from engine import size_series, TARGET_VOL, MAXCAP

ANN = 252


def _metrics(strat_ret: pd.Series, eq: pd.Series, n: int) -> dict:
    mdd = float(((eq / eq.cummax()) - 1).min() * 100)
    vol = float(strat_ret.std() * np.sqrt(ANN) * 100)
    sharpe = float(strat_ret.mean() / strat_ret.std() * np.sqrt(ANN)) if strat_ret.std() > 0 else 0.0
    cagr = float((eq.iloc[-1] ** (ANN / max(1, n)) - 1) * 100) if eq.iloc[-1] > 0 else -100.0
    return {"total": round((eq.iloc[-1] - 1) * 100, 1), "cagr": round(cagr, 1),
            "vol": round(vol, 1), "sharpe": round(sharpe, 2), "mdd": round(mdd, 1)}


def _strat_returns(df, target_vol, cost_bps):
    close = df["close"].reset_index(drop=True)
    ret = close.pct_change().fillna(0)
    expo = (size_series(df, target_vol).reset_index(drop=True).shift(1)).fillna(0)
    turn = expo.diff().abs().fillna(expo.abs())
    cost = turn * (cost_bps / 10000.0)
    return expo * ret - cost, ret, expo


def run_engine(df, target_vol=TARGET_VOL, cost_bps=5.0):
    df = df.reset_index(drop=True)
    dates = df["date"].astype(str).tolist()
    sr, ret, expo = _strat_returns(df, target_vol, cost_bps)
    eq = (1 + sr).cumprod()
    bh = (1 + ret).cumprod()
    n = len(df)
    st = _metrics(sr, eq, n)
    st["bh_total"] = round((bh.iloc[-1] - 1) * 100, 1)
    bh_mdd = float(((bh / bh.cummax()) - 1).min() * 100)
    st["bh_mdd"] = round(bh_mdd, 1)
    st["avg_expo"] = round(float(expo.mean()) * 100, 1)
    st["time_in"] = round(float((expo > 0.01).mean()) * 100, 1)
    st["period"] = f"{dates[0]} ~ {dates[-1]}"
    st["target_vol"] = target_vol
    step = max(1, n // 360)
    equity = [{"time": dates[i], "value": round(float(eq.iloc[i]), 4)} for i in range(0, n, step)]
    bhcurve = [{"time": dates[i], "value": round(float(bh.iloc[i]), 4)} for i in range(0, n, step)]
    return {"stats": st, "equity": equity, "bh": bhcurve, "ret": sr}


def walk_forward(df, grid=(25, 35, 45, 55), split=0.6, cost_bps=5.0):
    n = len(df); k = int(n * split)
    rows = []
    for tv in grid:
        sr, _, _ = _strat_returns(df, tv, cost_bps)
        is_sr, oos_sr = sr.iloc[:k], sr.iloc[k:]
        sh = lambda s: float(s.mean() / s.std() * np.sqrt(ANN)) if s.std() > 0 else 0.0
        rows.append({"target_vol": tv, "is_sharpe": round(sh(is_sr), 2),
                     "oos_sharpe": round(sh(oos_sr), 2),
                     "is_ret": round((np.prod(1 + is_sr) - 1) * 100, 1),
                     "oos_ret": round((np.prod(1 + oos_sr) - 1) * 100, 1)})
    best = max(rows, key=lambda r: r["is_sharpe"])
    return {"rows": rows, "best": best,
            "split_date": df["date"].astype(str).iloc[k],
            "verdict": "견고(과적합 낮음)" if best["oos_sharpe"] >= best["is_sharpe"] * 0.5 and best["oos_sharpe"] > 0
                       else "주의(아웃오브샘플 약화)"}


def monte_carlo(strat_ret: pd.Series, n_sims=600, block=5, seed=7):
    rng = np.random.default_rng(seed)
    r = strat_ret.dropna().values
    N = len(r)
    if N < block * 3:
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
    pct = lambda a, p: round(float(np.percentile(a, p)), 1)
    # MDD 히스토그램(표시용)
    hist, edges = np.histogram(mdds, bins=10)
    return {
        "final": {"p5": pct(finals, 5), "p50": pct(finals, 50), "p95": pct(finals, 95)},
        "mdd": {"p5": pct(mdds, 5), "p50": pct(mdds, 50), "worst": round(float(mdds.min()), 1)},
        "mdd_hist": {"counts": hist.tolist(),
                     "edges": [round(float(e), 1) for e in edges]},
        "n_sims": n_sims,
    }


def analyze(df, cost_bps=5.0):
    base = run_engine(df, TARGET_VOL, cost_bps)
    wf = walk_forward(df, cost_bps=cost_bps)
    mc = monte_carlo(base["ret"])
    out = {"stats": base["stats"], "equity": base["equity"], "bh": base["bh"],
           "walk_forward": wf, "monte_carlo": mc, "cost_bps": cost_bps}
    return out
