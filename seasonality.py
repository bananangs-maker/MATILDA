"""
seasonality.py — 역사적 계절성 (이미지 3)
각 연도별로 '연초 대비 누적수익률(%)' 곡선을 만들어 겹쳐 본다.
일봉을 (연, 월) 평균/말일 기준으로 집계해 1~12월 12포인트 곡선을 연도마다 생성.
+ 전 연도 평균 곡선.
"""
import numpy as np
import pandas as pd


def seasonality(df: pd.DataFrame, max_years: int = 8) -> dict:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    # 각 (연,월)의 마지막 종가
    monthly = d.groupby(["year", "month"])["close"].last().reset_index()

    years = sorted(monthly["year"].unique())
    if len(years) > max_years:
        years = years[-max_years:]

    series = []
    all_curves = []   # 평균 계산용 (월별 수익률 배열)
    for y in years:
        ym = monthly[monthly["year"] == y].sort_values("month")
        if len(ym) < 2:
            continue
        base = ym["close"].iloc[0]
        # 1~12월 포인트(해당 월 데이터 있을 때만), 연초 대비 %
        pts = []
        cur = {int(r.month): float(r.close) for r in ym.itertuples()}
        for m in range(1, 13):
            if m in cur:
                pts.append({"m": m, "v": round((cur[m] / base - 1) * 100, 1)})
        if pts:
            series.append({"year": int(y), "points": pts,
                           "ytd": pts[-1]["v"], "current": bool(int(y) == int(years[-1]))})
            arr = {p["m"]: p["v"] for p in pts}
            all_curves.append(arr)

    # 평균 곡선 (월별 평균)
    avg = []
    for m in range(1, 13):
        vals = [c[m] for c in all_curves if m in c]
        if vals:
            avg.append({"m": m, "v": round(float(np.mean(vals)), 1)})

    # 월별 평균 수익률 통계(베스트/워스트 달)
    month_ret = {m: [] for m in range(1, 13)}
    for c in all_curves:
        prev = 0.0
        for m in range(1, 13):
            if m in c:
                month_ret[m].append(c[m] - prev)
                prev = c[m]
    month_stats = []
    for m in range(1, 13):
        if month_ret[m]:
            month_stats.append({"m": m, "avg": round(float(np.mean(month_ret[m])), 2),
                                "win": round(float(np.mean([1 if x > 0 else 0 for x in month_ret[m]]) * 100), 0)})

    return {"series": series, "avg": avg, "month_stats": month_stats, "years": [int(y) for y in years]}
