"""
summary.py — 트레이딩뷰식 '기술 요약' (이미지 1·2)
오실레이터/이동평균을 각각 매수·매도·중립으로 투표시키고 종합 등급(스트롱셀~스트롱바이)을 산출.
※ 트레이딩뷰의 정확한 룰은 비공개라 통상적 규칙으로 근사한 것임.
"""
import numpy as np
import pandas as pd
import indicators as I


def _l(s, k=0):
    s = s.dropna()
    return float(s.iloc[-1 - k]) if len(s) > k else float("nan")


def _v(sig):  # 'buy'/'sell'/'neutral' -> 점수
    return {"buy": 1, "sell": -1, "neutral": 0}[sig]


def compute_summary(df: pd.DataFrame) -> dict:
    close = df["close"]
    c = float(close.iloc[-1])
    osc = []

    rsi = I.rsi(close); r = _l(rsi); rp = _l(rsi, 1)
    osc.append(("상대강도지수 (14)", r, "buy" if r < 30 and r > rp else "sell" if r > 70 and r < rp else "neutral"))

    k, d = I.stoch(df); kk, dd = _l(k), _l(d)
    osc.append(("스토캐스틱 %K (14,3,3)", kk, "buy" if kk < 20 and kk > dd else "sell" if kk > 80 and kk < dd else "neutral"))

    cci = I.cci(df); cc = _l(cci); ccp = _l(cci, 1)
    osc.append(("CCI (20)", cc, "buy" if cc < -100 and cc > ccp else "sell" if cc > 100 and cc < ccp else "neutral"))

    adx_, pdi, mdi = I.adx(df); a, pq, mq = _l(adx_), _l(pdi), _l(mdi)
    osc.append(("ADX (14)", a, "neutral" if a < 20 else "buy" if pq > mq else "sell"))

    # Awesome Oscillator
    med = (df["high"] + df["low"]) / 2
    ao = med.rolling(5).mean() - med.rolling(34).mean(); av, ap = _l(ao), _l(ao, 1)
    osc.append(("어썸 오실레이터", av, "buy" if av > 0 and av > ap else "sell" if av < 0 and av < ap else "neutral"))

    mom = close.diff(10); mv, mp = _l(mom), _l(mom, 1)
    osc.append(("모멘텀 (10)", mv, "buy" if mv > mp else "sell" if mv < mp else "neutral"))

    line, sigl, _ = I.macd(close); ml, sl = _l(line), _l(sigl)
    osc.append(("MACD (12,26)", ml, "buy" if ml > sl else "sell" if ml < sl else "neutral"))

    # Stoch RSI
    rmin = rsi.rolling(14).min(); rmax = rsi.rolling(14).max()
    srsi = 100 * (rsi - rmin) / (rmax - rmin).replace(0, np.nan); sv = _l(srsi)
    osc.append(("스토캐스틱 RSI", sv, "buy" if sv < 20 else "sell" if sv > 80 else "neutral"))

    wr = I.williams_r(df); w = _l(wr); wp = _l(wr, 1)
    osc.append(("윌리엄스 %R (14)", w, "buy" if w < -80 and w > wp else "sell" if w > -20 and w < wp else "neutral"))

    # Bull Bear Power
    ema13 = close.ewm(span=13, adjust=False).mean()
    bbp = (df["high"] - ema13) + (df["low"] - ema13); bb = _l(bbp); bbpv = _l(bbp, 1)
    osc.append(("불 베어 파워", bb, "buy" if bb < 0 and bb > bbpv else "sell" if bb > 0 and bb < bbpv else "neutral"))

    # Ultimate Oscillator (7,14,28)
    bp = close - pd.concat([df["low"], close.shift()], axis=1).min(axis=1)
    tr = pd.concat([df["high"], close.shift()], axis=1).max(axis=1) - pd.concat([df["low"], close.shift()], axis=1).min(axis=1)
    avg = lambda p: bp.rolling(p).sum() / tr.rolling(p).sum().replace(0, np.nan)
    uo = 100 * (4 * avg(7) + 2 * avg(14) + avg(28)) / 7; uv = _l(uo)
    osc.append(("얼티미트 오실레이터", uv, "buy" if uv < 30 else "sell" if uv > 70 else "neutral"))

    # ---- 이동평균 ----
    mas = []
    for p in (10, 20, 30, 50, 100, 200):
        e = _l(close.ewm(span=p, adjust=False).mean())
        s = _l(close.rolling(p).mean())
        mas.append(("EMA (%d)" % p, e, "buy" if c > e else "sell" if c < e else "neutral"))
        mas.append(("SMA (%d)" % p, s, "buy" if c > s else "sell" if c < s else "neutral"))
    conv, base, sa, sb = I.ichimoku(df); bj = _l(base)
    mas.append(("일목 기준선 (26)", bj, "buy" if c > bj else "sell" if c < bj else "neutral"))
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwma = (tp * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum(); vw = _l(vwma)
    mas.append(("VWMA (20)", vw, "buy" if c > vw else "sell" if c < vw else "neutral"))
    # Hull MA(9)
    wma = lambda x, p: x.rolling(p).apply(lambda v: np.dot(v, np.arange(1, p + 1)) / (p * (p + 1) / 2), raw=True)
    hull = wma(2 * wma(close, 4) - wma(close, 9), 3); hv = _l(hull)
    mas.append(("헐 이동평균 (9)", hv, "buy" if c > hv else "sell" if c < hv else "neutral"))

    def tally(rows):
        b = sum(1 for _, _, s in rows if s == "buy")
        se = sum(1 for _, _, s in rows if s == "sell")
        ne = sum(1 for _, _, s in rows if s == "neutral")
        score = (b - se) / max(1, (b + se + ne))
        return {"buy": b, "sell": se, "neutral": ne, "score": round(score, 3)}

    to = tally(osc); tm = tally(mas); tall = tally(osc + mas)

    def rate(score):
        return ("스트롱 바이" if score >= 0.5 else "바이" if score >= 0.1
                else "스트롱 셀" if score <= -0.5 else "셀" if score <= -0.1 else "뉴트럴")

    fmt = lambda rows: [{"name": n, "value": (round(v, 2) if v == v else None), "signal": s} for n, v, s in rows]
    return {
        "oscillators": fmt(osc), "mas": fmt(mas),
        "osc_count": to, "ma_count": tm, "all_count": tall,
        "summary": {"score": tall["score"], "rating": rate(tall["score"])},
        "osc_summary": {"score": to["score"], "rating": rate(to["score"])},
        "ma_summary": {"score": tm["score"], "rating": rate(tm["score"])},
    }
