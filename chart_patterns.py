"""
chart_patterns.py
가격 차트 패턴(머리어깨형·더블탑/바텀·삼각수렴 등)을 피벗(스윙 고/저점) 기반으로 탐지한다.

[솔직한 한계]
- 가격 패턴 자동탐지는 본질적으로 오탐이 많다. 여기서 반환하는 false_pct(오탐 가능성)는
  통계적으로 검증된 확률이 아니라 '형태 완성도/대칭성/확정 여부'에서 역산한 추정치다.
  값이 낮을수록 형태가 깔끔하다는 의미일 뿐, 수익을 보장하지 않는다. 보조 참고용.
"""
import numpy as np
import pandas as pd

# 패턴별 기본 신뢰도(낮게 잡음 — 가격 패턴은 노이즈가 큼)
BASE_CONF = {
    "Head & Shoulders": 0.42, "Inverse Head & Shoulders": 0.42,
    "Double Top": 0.46, "Double Bottom": 0.46,
    "Ascending Triangle": 0.40, "Descending Triangle": 0.40,
}
NAME_KO = {
    "Head & Shoulders": "머리어깨형", "Inverse Head & Shoulders": "역머리어깨형",
    "Double Top": "더블탑", "Double Bottom": "더블바텀",
    "Ascending Triangle": "상승삼각수렴", "Descending Triangle": "하락삼각수렴",
}


def _pivots(arr, order=5, kind="high"):
    """국소 고점/저점 인덱스 목록."""
    n = len(arr)
    out = []
    for i in range(order, n - order):
        win = arr[i - order:i + order + 1]
        if kind == "high":
            if arr[i] == win.max() and arr[i] > arr[i - 1]:
                out.append(i)
        else:
            if arr[i] == win.min() and arr[i] < arr[i - 1]:
                out.append(i)
    return out


def _conf_to_false(name, quality, confirmed):
    base = BASE_CONF.get(name, 0.42)
    conf = base * (0.5 + 0.5 * max(0.0, min(1.0, quality)))
    if confirmed:
        conf += 0.10
    conf = max(0.15, min(0.85, conf))
    return int(round((1 - conf) * 100))


def detect(df: pd.DataFrame, order=5, lookback=170, tol=0.035):
    """최근 lookback 봉에서 가격 패턴 탐지 → 리스트(최신순, 최대 5개)."""
    if df is None or len(df) < 40:
        return []
    d = df.tail(lookback).reset_index(drop=True)
    dates = d["date"].astype(str).tolist()
    high = d["high"].to_numpy(dtype=float)
    low = d["low"].to_numpy(dtype=float)
    close = d["close"].to_numpy(dtype=float)
    n = len(d)
    ph = _pivots(high, order, "high")
    pl = _pivots(low, order, "low")
    found = []

    def avg(a, b):
        return (a + b) / 2.0

    # ---------- Double Top: 비슷한 두 고점 + 사이 저점(넥라인) ----------
    for a, b in zip(ph, ph[1:]):
        if b - a < order:
            continue
        h1, h2 = high[a], high[b]
        if abs(h1 - h2) / avg(h1, h2) > tol:
            continue
        troughs = [j for j in pl if a < j < b]
        if not troughs:
            continue
        tr = min(troughs, key=lambda j: low[j])
        neck = low[tr]
        if neck >= min(h1, h2):
            continue
        confirmed = bool(np.any(close[b:] < neck))
        sym = 1 - abs(h1 - h2) / avg(h1, h2) / tol  # 0~1
        depth = (avg(h1, h2) - neck) / avg(h1, h2)
        quality = 0.6 * sym + 0.4 * min(1.0, depth / 0.08)
        found.append({
            "name": "Double Top", "name_ko": NAME_KO["Double Top"], "dir": "bear",
            "false_pct": _conf_to_false("Double Top", quality, confirmed),
            "confirmed": confirmed, "end_idx": b,
            "lines": [
                [{"time": dates[a], "value": round(h1, 2)}, {"time": dates[b], "value": round(h2, 2)}],
                [{"time": dates[a], "value": round(neck, 2)}, {"time": dates[min(n - 1, b + order)], "value": round(neck, 2)}],
            ],
            "marker": {"time": dates[b], "position": "aboveBar"},
        })

    # ---------- Double Bottom ----------
    for a, b in zip(pl, pl[1:]):
        if b - a < order:
            continue
        l1, l2 = low[a], low[b]
        if abs(l1 - l2) / avg(l1, l2) > tol:
            continue
        peaks = [j for j in ph if a < j < b]
        if not peaks:
            continue
        pk = max(peaks, key=lambda j: high[j])
        neck = high[pk]
        if neck <= max(l1, l2):
            continue
        confirmed = bool(np.any(close[b:] > neck))
        sym = 1 - abs(l1 - l2) / avg(l1, l2) / tol
        depth = (neck - avg(l1, l2)) / avg(l1, l2)
        quality = 0.6 * sym + 0.4 * min(1.0, depth / 0.08)
        found.append({
            "name": "Double Bottom", "name_ko": NAME_KO["Double Bottom"], "dir": "bull",
            "false_pct": _conf_to_false("Double Bottom", quality, confirmed),
            "confirmed": confirmed, "end_idx": b,
            "lines": [
                [{"time": dates[a], "value": round(l1, 2)}, {"time": dates[b], "value": round(l2, 2)}],
                [{"time": dates[a], "value": round(neck, 2)}, {"time": dates[min(n - 1, b + order)], "value": round(neck, 2)}],
            ],
            "marker": {"time": dates[b], "position": "belowBar"},
        })

    # ---------- Head & Shoulders: 고점 3개(가운데가 최고) ----------
    for i in range(len(ph) - 2):
        a, b, c = ph[i], ph[i + 1], ph[i + 2]
        ha, hb, hc = high[a], high[b], high[c]
        if not (hb > ha and hb > hc):
            continue
        if abs(ha - hc) / avg(ha, hc) > tol * 1.6:
            continue
        tr = [j for j in pl if a < j < c]
        if len(tr) < 2:
            continue
        t1 = min([j for j in tr if j < b], default=None, key=lambda j: low[j]) if any(j < b for j in tr) else None
        t2 = min([j for j in tr if j > b], default=None, key=lambda j: low[j]) if any(j > b for j in tr) else None
        if t1 is None or t2 is None:
            continue
        neck = avg(low[t1], low[t2])
        confirmed = bool(np.any(close[c:] < neck))
        sym = 1 - abs(ha - hc) / avg(ha, hc) / (tol * 1.6)
        head_prom = (hb - avg(ha, hc)) / hb
        quality = 0.6 * sym + 0.4 * min(1.0, head_prom / 0.05)
        found.append({
            "name": "Head & Shoulders", "name_ko": NAME_KO["Head & Shoulders"], "dir": "bear",
            "false_pct": _conf_to_false("Head & Shoulders", quality, confirmed),
            "confirmed": confirmed, "end_idx": c,
            "lines": [
                [{"time": dates[a], "value": round(ha, 2)}, {"time": dates[b], "value": round(hb, 2)}, {"time": dates[c], "value": round(hc, 2)}],
                [{"time": dates[t1], "value": round(low[t1], 2)}, {"time": dates[t2], "value": round(low[t2], 2)}],
            ],
            "marker": {"time": dates[b], "position": "aboveBar"},
        })

    # ---------- Inverse Head & Shoulders ----------
    for i in range(len(pl) - 2):
        a, b, c = pl[i], pl[i + 1], pl[i + 2]
        la, lb, lc = low[a], low[b], low[c]
        if not (lb < la and lb < lc):
            continue
        if abs(la - lc) / avg(la, lc) > tol * 1.6:
            continue
        pk = [j for j in ph if a < j < c]
        p1 = max([j for j in pk if j < b], default=None, key=lambda j: high[j]) if any(j < b for j in pk) else None
        p2 = max([j for j in pk if j > b], default=None, key=lambda j: high[j]) if any(j > b for j in pk) else None
        if p1 is None or p2 is None:
            continue
        neck = avg(high[p1], high[p2])
        confirmed = bool(np.any(close[c:] > neck))
        sym = 1 - abs(la - lc) / avg(la, lc) / (tol * 1.6)
        head_prom = (avg(la, lc) - lb) / avg(la, lc)
        quality = 0.6 * sym + 0.4 * min(1.0, head_prom / 0.05)
        found.append({
            "name": "Inverse Head & Shoulders", "name_ko": NAME_KO["Inverse Head & Shoulders"], "dir": "bull",
            "false_pct": _conf_to_false("Inverse Head & Shoulders", quality, confirmed),
            "confirmed": confirmed, "end_idx": c,
            "lines": [
                [{"time": dates[a], "value": round(la, 2)}, {"time": dates[b], "value": round(lb, 2)}, {"time": dates[c], "value": round(lc, 2)}],
                [{"time": dates[p1], "value": round(high[p1], 2)}, {"time": dates[p2], "value": round(high[p2], 2)}],
            ],
            "marker": {"time": dates[b], "position": "belowBar"},
        })

    # ---------- Triangles: 최근 고점 3+, 저점 3+의 기울기 ----------
    def _slope(idx, vals):
        if len(idx) < 3:
            return None, None
        x = np.array(idx[-3:], dtype=float)
        y = np.array([vals[j] for j in idx[-3:]], dtype=float)
        m = np.polyfit(x, y, 1)[0]
        rel = m * (x[-1] - x[0]) / max(1e-9, y.mean())  # 구간 변화율
        return m, rel

    mh, relh = _slope(ph, high)
    ml, rell = _slope(pl, low)
    if relh is not None and rell is not None:
        flat = 0.02
        last = max(ph[-1] if ph else 0, pl[-1] if pl else 0)
        if abs(relh) < flat and rell > flat:        # 상단 평평 + 하단 상승
            hi = high[ph[-1]]
            found.append({
                "name": "Ascending Triangle", "name_ko": NAME_KO["Ascending Triangle"], "dir": "bull",
                "false_pct": _conf_to_false("Ascending Triangle", 0.5, bool(np.any(close[last:] > hi))),
                "confirmed": bool(np.any(close[last:] > hi)), "end_idx": last,
                "lines": [
                    [{"time": dates[ph[-3]], "value": round(high[ph[-3]], 2)}, {"time": dates[ph[-1]], "value": round(high[ph[-1]], 2)}],
                    [{"time": dates[pl[-3]], "value": round(low[pl[-3]], 2)}, {"time": dates[pl[-1]], "value": round(low[pl[-1]], 2)}],
                ],
                "marker": {"time": dates[last], "position": "belowBar"},
            })
        elif abs(rell) < flat and relh < -flat:      # 하단 평평 + 상단 하락
            lo = low[pl[-1]]
            found.append({
                "name": "Descending Triangle", "name_ko": NAME_KO["Descending Triangle"], "dir": "bear",
                "false_pct": _conf_to_false("Descending Triangle", 0.5, bool(np.any(close[last:] < lo))),
                "confirmed": bool(np.any(close[last:] < lo)), "end_idx": last,
                "lines": [
                    [{"time": dates[ph[-3]], "value": round(high[ph[-3]], 2)}, {"time": dates[ph[-1]], "value": round(high[ph[-1]], 2)}],
                    [{"time": dates[pl[-3]], "value": round(low[pl[-3]], 2)}, {"time": dates[pl[-1]], "value": round(low[pl[-1]], 2)}],
                ],
                "marker": {"time": dates[last], "position": "aboveBar"},
            })

    # 최신순 정렬 + 영역 중복 제거 + 최대 5개
    found.sort(key=lambda p: p["end_idx"], reverse=True)
    out, used = [], []
    for p in found:
        if any(abs(p["end_idx"] - u) < order for u in used):
            continue
        used.append(p["end_idx"])
        p.pop("end_idx", None)
        out.append(p)
        if len(out) >= 5:
            break
    return out
