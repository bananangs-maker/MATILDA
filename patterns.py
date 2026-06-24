"""
patterns.py
(1) 이평선 크로스 중요도: 이평 조합별로 골든/데드 크로스를 탐지하고 중요도(제1~제3 신호)로 표기.
(2) 캔들 패턴: 반전형 캔들 패턴을 탐지하고 신뢰도(%)와 '오탐 주의'를 함께 반환 (보조 지표).

둘 다 오탐이 있을 수 있으므로 confidence(신뢰도)를 함께 제공한다.
"""
import numpy as np
import pandas as pd

UP = "#3DD68C"
DN = "#FF5A4D"

# (fast, slow, tier) — tier 1 = 가장 강력
MA_PAIRS = [(50, 200, 1), (20, 60, 2), (5, 20, 3)]


def detect_ma_cross(df: pd.DataFrame, lookback: int = 25):
    """최근 lookback 봉 안의 이평 크로스를 중요도순으로 반환 + 차트 마커."""
    close = df["close"].reset_index(drop=True)
    dates = df["date"].astype(str).tolist()
    sma200 = close.rolling(200).mean()
    signals, markers = [], []
    for fast, slow, tier in MA_PAIRS:
        f = close.rolling(fast).mean()
        s = close.rolling(slow).mean()
        diff = (f - s)
        sign = np.sign(diff)
        cross_idx = None
        kind = None
        # 최근에서 과거로 크로스 탐색
        for i in range(len(close) - 1, max(0, len(close) - lookback) - 1, -1):
            if i < slow or np.isnan(sign.iloc[i]) or np.isnan(sign.iloc[i - 1]):
                continue
            if sign.iloc[i] != sign.iloc[i - 1] and sign.iloc[i] != 0:
                cross_idx = i
                kind = "golden" if sign.iloc[i] > 0 else "dead"
                break
        # 현재 정렬 상태(크로스 없어도 보고)
        cur_state = None
        if not np.isnan(sign.iloc[-1]):
            cur_state = "above" if sign.iloc[-1] > 0 else "below"
        if cross_idx is None:
            continue
        bars_ago = (len(close) - 1) - cross_idx
        # 200일선 추세와 정렬?
        aligned = True
        if not np.isnan(sma200.iloc[cross_idx]):
            up_regime = bool(close.iloc[cross_idx] > sma200.iloc[cross_idx])
            aligned = (kind == "golden" and up_regime) or (kind == "dead" and not up_regime)
        aligned = bool(aligned)
        base_conf = {1: 80, 2: 65, 3: 50}[tier]
        conf = base_conf + (8 if aligned else -15) - min(bars_ago, 12)
        conf = max(20, min(95, conf))
        side = "buy" if kind == "golden" else "sell"
        name = ("제%d 매수신호" % tier) if side == "buy" else ("제%d 매도신호" % tier)
        signals.append({
            "pair": "%d/%d" % (fast, slow), "tier": tier, "kind": kind, "side": side,
            "name": name, "bars_ago": bars_ago, "date": dates[cross_idx],
            "price": round(float(close.iloc[cross_idx]), 2),
            "aligned": aligned, "confidence": int(conf), "cur_state": cur_state,
            "label": "%s (%s %s)" % (name, "%d/%d" % (fast, slow),
                                     "골든크로스" if kind == "golden" else "데드크로스"),
        })
        markers.append({
            "time": dates[cross_idx], "position": "belowBar" if side == "buy" else "aboveBar",
            "color": UP if side == "buy" else DN,
            "shape": "arrowUp" if side == "buy" else "arrowDown",
            "text": "%d" % tier,
        })
    signals.sort(key=lambda x: x["tier"])
    return signals, markers


# ---------- 캔들 패턴 ----------
def _ctx(close, i):
    """추세 맥락: i 시점이 상승/하락/중립 추세인지 (SMA10 기울기·위치)."""
    if i < 12:
        return "flat"
    sma = close.iloc[:i + 1].rolling(10).mean()
    if np.isnan(sma.iloc[i]) or np.isnan(sma.iloc[i - 3]):
        return "flat"
    slope = sma.iloc[i] - sma.iloc[i - 3]
    if close.iloc[i] > sma.iloc[i] and slope > 0:
        return "up"
    if close.iloc[i] < sma.iloc[i] and slope < 0:
        return "down"
    return "flat"


# (base reliability) — 캔들 패턴은 본질적으로 노이즈가 큼
BASE_REL = {
    "샛별": 70, "석별": 70, "적삼병": 68, "흑삼병": 68,
    "상승장악": 65, "하락장악": 65, "관통형": 60, "흑운형": 60,
    "망치": 55, "유성": 55, "역망치": 50, "교수형": 52,
    "상승잉태": 50, "하락잉태": 50, "장대양봉": 60, "장대음봉": 60, "도지": 40,
}


def detect_candles(df: pd.DataFrame, scan: int = 12):
    """최근 scan 봉에서 캔들 패턴 탐지. 신뢰도(%)와 오탐 주의 포함."""
    o, h, l, c = (df[k].reset_index(drop=True) for k in ("open", "high", "low", "close"))
    close = c
    dates = df["date"].astype(str).tolist()
    n = len(df)
    out = []

    def add(name, kind, i, clean, ctx_ok):
        rel = BASE_REL.get(name, 50)
        conf = rel * (0.7 + 0.4 * clean) * (1.1 if ctx_ok else 0.85)
        conf = int(max(20, min(95, conf)))
        out.append({"name": name, "kind": kind, "date": dates[i],
                    "bars_ago": (n - 1) - i, "confidence": conf})

    start = max(2, n - scan)
    for i in range(start, n):
        oi, hi, li, ci = float(o[i]), float(h[i]), float(l[i]), float(c[i])
        body = abs(ci - oi); rng = hi - li
        if rng <= 0:
            continue
        upper = hi - max(oi, ci); lower = min(oi, ci) - li
        bull = ci > oi; ctx = _ctx(close, i)
        bodyR = body / rng

        # 단봉
        if bodyR <= 0.1:
            add("도지", "neutral", i, 1 - bodyR * 10, True)
        if lower >= 2 * body and upper <= body * 0.7 and bodyR <= 0.4:
            if ctx == "down":
                add("망치", "bull", i, min(1, lower / (2 * body + 1e-9)), True)
            elif ctx == "up":
                add("교수형", "bear", i, min(1, lower / (2 * body + 1e-9)), True)
        if upper >= 2 * body and lower <= body * 0.7 and bodyR <= 0.4:
            if ctx == "up":
                add("유성", "bear", i, min(1, upper / (2 * body + 1e-9)), True)
            elif ctx == "down":
                add("역망치", "bull", i, min(1, upper / (2 * body + 1e-9)), True)
        if bodyR >= 0.9:
            add("장대양봉" if bull else "장대음봉", "bull" if bull else "bear", i, bodyR,
                (ctx == "down") if bull else (ctx == "up"))

        # 2봉
        po, pc = float(o[i - 1]), float(c[i - 1])
        pbody = abs(pc - po); pbull = pc > po
        if pbody > 0 and body > 0:
            if (not pbull) and bull and ci >= po and oi <= pc and body > pbody:
                add("상승장악", "bull", i, min(1, body / (pbody + 1e-9) - 0.5), ctx == "down")
            if pbull and (not bull) and ci <= po and oi >= pc and body > pbody:
                add("하락장악", "bear", i, min(1, body / (pbody + 1e-9) - 0.5), ctx == "up")
            if (not pbull) and bull and oi < pc and ci > (po + pc) / 2 and ci < po:
                add("관통형", "bull", i, (ci - (po + pc) / 2) / (pbody + 1e-9), ctx == "down")
            if pbull and (not bull) and oi > pc and ci < (po + pc) / 2 and ci > po:
                add("흑운형", "bear", i, ((po + pc) / 2 - ci) / (pbody + 1e-9), ctx == "up")
            if pbody > body * 1.6 and max(oi, ci) <= max(po, pc) and min(oi, ci) >= min(po, pc):
                if (not pbull) and ctx == "down":
                    add("상승잉태", "bull", i, 0.8, True)
                elif pbull and ctx == "up":
                    add("하락잉태", "bear", i, 0.8, True)

        # 3봉
        if i >= 2:
            o1, c1 = float(o[i - 2]), float(c[i - 2])
            b1 = abs(c1 - o1); bull1 = c1 > o1
            mid2 = body
            if bull1 is False and mid2 < b1 * 0.5 and bull and ci > (o1 + c1) / 2:
                add("샛별", "bull", i, 0.85, ctx in ("down", "flat"))
            if bull1 and mid2 < b1 * 0.5 and (not bull) and ci < (o1 + c1) / 2:
                add("석별", "bear", i, 0.85, ctx in ("up", "flat"))
            # 적삼병 / 흑삼병
            if bull and (float(c[i - 1]) > o[i - 1]) and (float(c[i - 2]) > o[i - 2]) \
               and ci > c[i - 1] > c[i - 2]:
                add("적삼병", "bull", i, 0.8, True)
            if (not bull) and (float(c[i - 1]) < o[i - 1]) and (float(c[i - 2]) < o[i - 2]) \
               and ci < c[i - 1] < c[i - 2]:
                add("흑삼병", "bear", i, 0.8, True)

    # 최신순, 중복 정리(같은 봉 같은 패턴 1회)
    seen = set(); uniq = []
    for p in sorted(out, key=lambda x: x["bars_ago"]):
        key = (p["name"], p["date"])
        if key in seen:
            continue
        seen.add(key); uniq.append(p)
    return uniq[:8]
