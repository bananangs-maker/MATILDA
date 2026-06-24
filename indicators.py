"""
indicators.py
일봉 OHLCV(DataFrame)를 받아 대시보드 입력값으로 쓸 지표/버킷을 계산한다.
컬럼 규약: date, open, high, low, close, volume  (오래된 날짜 -> 최신 날짜 순으로 정렬되어 있어야 함)

여기서 계산하는 것(가격 기반): MA180 위치, 멀티이평(10/20/60) 정·역배열,
장기크로스(WMA240), MACD 상태, RSI(14), MFI(14), CCI(20),
볼린저(20,2) 위치/폭, 파라볼릭 SAR 방향, (근사) 다이버전스.

계산하지 않는 것(대시보드에서 수동 유지): VIX, 공포·탐욕, 추세선, 피보나치.
"""
import numpy as np
import pandas as pd


# ---------- 개별 지표 ----------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing (alpha = 1/period)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(period).sum()
    neg_sum = neg.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - 100 / (1 + mfr)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad)


def macd(close: pd.Series, fast=12, slow=26, sig=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    signal = line.ewm(span=sig, adjust=False).mean()
    hist = line - signal
    return line, signal, hist


def bollinger(close: pd.Series, period=20, k=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    pct_b = (close - lower) / (upper - lower)
    width = (upper - lower) / mid
    return mid, upper, lower, pct_b, width


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def parabolic_sar(df: pd.DataFrame, af_step=0.02, af_max=0.2) -> pd.Series:
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    sar = np.zeros(n)
    if n < 2:
        return pd.Series(sar, index=df.index)
    # 초기값
    up = True
    af = af_step
    ep = high[0]
    sar[0] = low[0]
    for i in range(1, n):
        prev = sar[i - 1]
        sar[i] = prev + af * (ep - prev)
        if up:
            sar[i] = min(sar[i], low[i - 1], low[max(i - 2, 0)])
            if high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)
            if low[i] < sar[i]:           # 추세 전환
                up = False
                sar[i] = ep
                ep = low[i]
                af = af_step
        else:
            sar[i] = max(sar[i], high[i - 1], high[max(i - 2, 0)])
            if low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)
            if high[i] > sar[i]:
                up = True
                sar[i] = ep
                ep = high[i]
                af = af_step
    return pd.Series(sar, index=df.index)


# ---------- 다이버전스 (근사) ----------
def _swings(series: pd.Series, win: int = 3):
    """local min/max index 위치 반환 (단순 window 방식)."""
    vals = series.values
    highs, lows = [], []
    for i in range(win, len(vals) - win):
        seg = vals[i - win:i + win + 1]
        if vals[i] == seg.max():
            highs.append(i)
        if vals[i] == seg.min():
            lows.append(i)
    return highs, lows


def divergence(close: pd.Series, rsi_series: pd.Series, lookback: int = 40) -> str:
    if len(close) < lookback + 5:
        return "none"
    c = close.iloc[-lookback:].reset_index(drop=True)
    r = rsi_series.iloc[-lookback:].reset_index(drop=True)
    _, lows = _swings(c)
    highs, _ = _swings(c)
    # 강세 다이버전스: 가격 저점 낮아지는데 RSI 저점 높아짐
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if c[b] < c[a] and r[b] > r[a]:
            return "bull"
    # 약세 다이버전스: 가격 고점 높아지는데 RSI 고점 낮아짐
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if c[b] > c[a] and r[b] < r[a]:
            return "bear"
    return "none"


# ---------- 버킷/상태 매핑 (대시보드 입력값 형식) ----------
def _safe_last(series: pd.Series, default=np.nan):
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else default


def compute_signals(df: pd.DataFrame) -> dict:
    """OHLCV DataFrame -> 대시보드 state 형식 dict."""
    df = df.copy().reset_index(drop=True)
    close = df["close"]

    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    sma60 = close.rolling(60).mean()
    sma180 = close.rolling(180).mean()
    wma240 = wma(close, 240)

    rsi14 = rsi(close, 14)
    mfi14 = mfi(df, 14)
    cci20 = cci(df, 20)
    mline, msig, mhist = macd(close)
    _, _, _, pct_b, width = bollinger(close)
    sar = parabolic_sar(df)

    last = len(df) - 1
    c_last = float(close.iloc[last])

    # --- MA180 위치 ---
    s180 = _safe_last(sma180)
    if np.isnan(s180):
        ma180 = "unknown"
    elif c_last > s180 * 1.005:
        ma180 = "above"
    elif c_last < s180 * 0.995:
        ma180 = "below"
    else:
        ma180 = "at"

    # --- 멀티이평 10/20/60 ---
    a, b, cc = _safe_last(sma10), _safe_last(sma20), _safe_last(sma60)
    if not any(np.isnan(x) for x in (a, b, cc)):
        if a > b > cc:
            multima = "bull"
        elif a < b < cc:
            multima = "bear"
        else:
            multima = "mixed"
    else:
        multima = "unknown"

    # --- 장기 크로스 (WMA240 기울기 + 가격 위치) ---
    w_now = _safe_last(wma240)
    ltcross = "unknown"
    wseries = wma240.dropna()
    if len(wseries) > 20 and not np.isnan(w_now):
        w_prev = float(wseries.iloc[-21])  # 약 20거래일 전
        rising = w_now > w_prev
        if c_last > w_now and rising:
            ltcross = "bull"
        elif c_last < w_now and not rising:
            ltcross = "bear"
        else:
            ltcross = "neutral"

    # --- MACD 상태 ---
    ml, sg = _safe_last(mline), _safe_last(msig)
    h_now = _safe_last(mhist)
    hist_clean = mhist.dropna()
    h_prev = float(hist_clean.iloc[-2]) if len(hist_clean) > 1 else h_now
    if ml > sg and h_now >= h_prev:
        macd_state = "golden"
    elif ml < sg and h_now <= h_prev:
        macd_state = "death"
    else:
        macd_state = "weak"

    # --- 볼린저 위치 ---
    pb = _safe_last(pct_b)
    if np.isnan(pb):
        bbpos = "mid"
    elif pb > 1.0:
        bbpos = "above"
    elif pb > 0.6:
        bbpos = "upper"
    elif pb >= 0.4:
        bbpos = "mid"
    elif pb >= 0.0:
        bbpos = "lower"
    else:
        bbpos = "below"

    # --- 볼린저 폭 (최근 120일 분위) ---
    w_last = _safe_last(width)
    wclean = width.dropna().iloc[-120:]
    if len(wclean) > 20 and not np.isnan(w_last):
        pctile = (wclean < w_last).mean()
        bbwidth = "squeeze" if pctile < 0.2 else "wide" if pctile > 0.8 else "normal"
    else:
        bbwidth = "normal"

    # --- SAR 방향 ---
    sar_last = _safe_last(sar)
    sar_dir = "long" if c_last > sar_last else "short"

    # --- 다이버전스 ---
    diver = divergence(close, rsi14)

    # --- 매수/매도 압력 프록시 (최근 20일 상승거래량 비중) ---
    win = 20
    chg = close.diff()
    upv = df["volume"].where(chg > 0, 0).rolling(win).sum()
    dnv = df["volume"].where(chg < 0, 0).rolling(win).sum()
    tot = (upv + dnv)
    press = _safe_last((upv / tot.replace(0, np.nan)) * 100)
    pressure = round(press, 0) if not np.isnan(press) else 50

    # --- 가격 기반 리스크 (급락 / 낙폭 / 이격 / 장대음봉 / 더블탑) ---
    ret1 = float(close.pct_change().iloc[-1] * 100) if len(close) > 1 else 0.0
    roll_hi20 = _safe_last(close.rolling(20).max())
    dd20 = ((c_last / roll_hi20) - 1) * 100 if roll_hi20 else 0.0
    sma20v = _safe_last(sma20)
    ext20 = ((c_last - sma20v) / sma20v * 100) if sma20v else 0.0
    rngs = (df["high"] - df["low"])
    atr = rngs.rolling(14).mean()
    big_down = 0
    for i in range(max(0, len(df) - 5), len(df)):
        rng_i = float(df["high"].iloc[i] - df["low"].iloc[i])
        body = float(df["open"].iloc[i] - df["close"].iloc[i])  # 양수=음봉
        a = atr.iloc[i]
        if body > 0 and rng_i > 0 and not np.isnan(a) and rng_i > a * 1.6 and body > rng_i * 0.55:
            big_down += 1
    # 더블탑 휴리스틱: 최근 60봉 내 비슷한 두 고점 + 그 사이 골 + 현재 둘째 고점 아래
    dtop = False
    if len(close) >= 50:
        seg = close.iloc[-60:].reset_index(drop=True)
        hs, _ = _swings(seg, 3)
        if len(hs) >= 2:
            a, b = hs[-2], hs[-1]
            pa, pb = seg[a], seg[b]
            trough = seg[a:b].min() if b > a else pa
            if abs(pa - pb) / max(pa, pb) < 0.05 and (min(pa, pb) - trough) / min(pa, pb) > 0.04 \
               and c_last < pb:
                dtop = True

    return {
        "ma180": ma180,
        "ltcross": ltcross,
        "macd": macd_state,
        "multima": multima,
        "sar": sar_dir,
        "bbpos": bbpos,
        "bbwidth": bbwidth,
        "rsi": round(_safe_last(rsi14), 1),
        "mfi": round(_safe_last(mfi14), 1),
        "cci": round(_safe_last(cci20), 0),
        "diver": diver,
        "pressure": pressure,
        "ret1": round(ret1, 1),
        "dd20": round(dd20, 1),
        "ext20": round(ext20, 1),
        "bigdown": int(big_down),
        "dtop": bool(dtop),
        # 참고용 원시값
        "_price": round(c_last, 2),
        "_pctB": round(pb, 3) if not np.isnan(pb) else None,
        "_bars": int(len(df)),
    }


# ---------- 차트용 풀 시리즈 ----------
def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _series(dates, values, color_fn=None):
    """lightweight-charts용 [{time, value(, color)}] (NaN 제외)."""
    out = []
    for d, v in zip(dates, values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        item = {"time": d, "value": round(float(v), 4)}
        if color_fn:
            item["color"] = color_fn(v)
        out.append(item)
    return out


def compute_series(df: pd.DataFrame) -> dict:
    """OHLCV -> 차트 렌더용 시리즈 묶음. date는 'YYYY-MM-DD' 가정."""
    df = df.copy().reset_index(drop=True)
    dates = df["date"].astype(str).tolist()
    close = df["close"]

    mid, up, low, _, _ = bollinger(close)
    mline, msig, mhist = macd(close)
    sar = parabolic_sar(df)
    rsi14 = rsi(close, 14)

    # 캔들
    candles = [{"time": dates[i], "open": round(float(df["open"][i]), 4),
                "high": round(float(df["high"][i]), 4),
                "low": round(float(df["low"][i]), 4),
                "close": round(float(df["close"][i]), 4)}
               for i in range(len(df))]
    # 거래량 (상승=초록, 하락=빨강)
    vol = []
    for i in range(len(df)):
        c_up = df["close"][i] >= df["open"][i]
        vol.append({"time": dates[i], "value": float(df["volume"][i]),
                    "color": "rgba(61,214,140,.45)" if c_up else "rgba(255,90,77,.45)"})

    hist_color = lambda v: ("#3DD68C" if v >= 0 else "#FF5A4D")
    return {
        "candles": candles,
        "volume": vol,
        "bb_upper": _series(dates, up),
        "bb_mid": _series(dates, mid),
        "bb_lower": _series(dates, low),
        "ema5": _series(dates, ema(close, 5)),
        "ema20": _series(dates, ema(close, 20)),
        "sma60": _series(dates, close.rolling(60).mean()),
        "sma180": _series(dates, close.rolling(180).mean()),
        "sar": _series(dates, sar),
        "rsi": _series(dates, rsi14),
        "mfi": _series(dates, mfi(df, 14)),
        "cci": _series(dates, cci(df, 20)),
        "macd_line": _series(dates, mline),
        "macd_signal": _series(dates, msig),
        "macd_hist": _series(dates, mhist, hist_color),
        "obv": _series(dates, _obv(df)),
    }


def _obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (매수/매도 압력 프록시)."""
    sign = np.sign(df["close"].diff().fillna(0))
    return (sign * df["volume"]).fillna(0).cumsum()
