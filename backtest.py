"""
backtest.py
과거 일봉으로 규칙 기반 매매를 검증한다 (롱 온리, 신호 발생 시 전량 진입/청산).

기본 규칙 (사용자 도구 기반):
- 진입(BUY): MACD 골든크로스 AND 종가 > 추세선(SMA, 기본 200) — 추세 위에서만 진입
- 청산(SELL): MACD 데드크로스

같은 신호가 차트의 매수/매도 화살표 마커로도 표시된다.
"""
import numpy as np
import pandas as pd
from indicators import macd

UP = "#26C281"
DN = "#F23645"


def generate_signals(df: pd.DataFrame, trend_ma: int = 200):
    """[(index, 'buy'|'sell')] 반환."""
    close = df["close"].reset_index(drop=True)
    line, sig, _ = macd(close)
    sma = close.rolling(trend_ma).mean()
    cross_up = (line > sig) & (line.shift(1) <= sig.shift(1))
    cross_dn = (line < sig) & (line.shift(1) >= sig.shift(1))
    sigs, pos = [], False
    for i in range(len(df)):
        trend_ok = (not np.isnan(sma.iloc[i])) and close.iloc[i] > sma.iloc[i]
        if (not pos) and cross_up.iloc[i] and trend_ok:
            sigs.append((i, "buy")); pos = True
        elif pos and cross_dn.iloc[i]:
            sigs.append((i, "sell")); pos = False
    return sigs


def markers_from_signals(df, sigs):
    dates = df["date"].astype(str).tolist()
    out = []
    for i, t in sigs:
        if t == "buy":
            out.append({"time": dates[i], "position": "belowBar", "color": UP,
                        "shape": "arrowUp", "text": "B"})
        else:
            out.append({"time": dates[i], "position": "aboveBar", "color": DN,
                        "shape": "arrowDown", "text": "S"})
    return out


def run(df: pd.DataFrame, trend_ma: int = 200, start_cash: float = 10000.0) -> dict:
    df = df.reset_index(drop=True)
    close = df["close"]
    dates = df["date"].astype(str).tolist()
    sigs = generate_signals(df, trend_ma)
    sig_map = {i: t for i, t in sigs}

    cash, shares, entry = start_cash, 0.0, None
    trades, equity = [], []
    for i in range(len(df)):
        px = float(close.iloc[i])
        t = sig_map.get(i)
        if t == "buy" and shares == 0:
            shares = cash / px; cash = 0.0; entry = (dates[i], px)
        elif t == "sell" and shares > 0:
            cash = shares * px
            trades.append({"entry_date": entry[0], "entry": round(entry[1], 2),
                           "exit_date": dates[i], "exit": round(px, 2),
                           "ret": round((px / entry[1] - 1) * 100, 2)})
            shares = 0.0; entry = None
        equity.append({"time": dates[i], "value": round(cash + shares * px, 2)})

    final = cash + shares * float(close.iloc[-1])
    bh_ret = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100
    total_ret = (final / start_cash - 1) * 100
    wins = [t for t in trades if t["ret"] > 0]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
    avg_win = float(np.mean([t["ret"] for t in wins])) if wins else 0.0
    losses = [t for t in trades if t["ret"] <= 0]
    avg_loss = float(np.mean([t["ret"] for t in losses])) if losses else 0.0
    eq = pd.Series([e["value"] for e in equity])
    max_dd = float(((eq / eq.cummax()) - 1).min() * 100) if len(eq) else 0.0

    return {
        "equity": equity,
        "trades": trades[-50:],          # 최근 50건만 표로
        "markers": markers_from_signals(df, sigs),
        "stats": {
            "total_return": round(total_ret, 1),
            "bh_return": round(bh_ret, 1),
            "win_rate": round(win_rate, 1),
            "n_trades": len(trades),
            "max_dd": round(max_dd, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "final": round(final, 0),
            "start": int(start_cash),
            "trend_ma": trend_ma,
            "period": f"{dates[0]} ~ {dates[-1]}",
        },
    }
