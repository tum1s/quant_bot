"""Intraday RSI + Bollinger alert bot. Headless, free to run 24/7.

Env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (never hardcode these)
Optional: SYMBOL=BTCUSDT INTERVAL=15m HORIZON=5 FEE_PCT=0.1

Usage:
  python quant_bot.py          # loop forever (VM / home server)
  python quant_bot.py --once   # single scan (GitHub Actions cron)
"""
import os, sys, time
import numpy as np
import pandas as pd
import requests

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "15m")
HORIZON = int(os.getenv("HORIZON", 5))        # bars to hold in stats
FEE_PCT = float(os.getenv("FEE_PCT", 0.1))    # per side, %
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")
SECS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}[INTERVAL]
API = "https://api.binance.com/api/v3/klines"  # public, no key, real-time


def fetch(limit=500):
    r = requests.get(API, params={"symbol": SYMBOL, "interval": INTERVAL, "limit": limit}, timeout=10)
    r.raise_for_status()
    cols = ["t", "open", "high", "low", "close", "vol", "ct"]
    df = pd.DataFrame([row[:7] for row in r.json()], columns=cols)
    df[["open", "high", "low", "close", "vol"]] = df[["open", "high", "low", "close", "vol"]].astype(float)
    df.index = pd.to_datetime(df["t"], unit="ms")
    # drop the still-forming candle: signals on it repaint
    return df[df["ct"] < time.time() * 1000]


def indicators(df, rsi_n=14, bb_n=20, atr_n=14):
    c = df["close"]
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / rsi_n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / rsi_n, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    ma, sd = c.rolling(bb_n).mean(), c.rolling(bb_n).std(ddof=0)
    df["bb_up"], df["bb_lo"] = ma + 2 * sd, ma - 2 * sd
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - c.shift()).abs(),
                    (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / atr_n, adjust=False).mean()
    df["sig"] = 0
    df.loc[(df["rsi"] < 30) & (c <= df["bb_lo"]), "sig"] = 1
    df.loc[(df["rsi"] > 70) & (c >= df["bb_up"]), "sig"] = -1
    return df


def backtest_stats(df, h=HORIZON):
    """Vectorized: fixed-horizon return per signal, net of fees."""
    fwd = df["close"].shift(-h)
    ret = df["sig"] * (fwd - df["close"]) / df["close"] * 100 - 2 * FEE_PCT
    r = ret[(df["sig"] != 0) & fwd.notna()]
    if r.empty:
        return dict(n=0, win=0.0, pf=0.0, avg=0.0)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=len(r), win=100 * (r > 0).mean(),
                pf=gp / gl if gl > 0 else float("inf"), avg=r.mean())


def divergence(df, k=3):
    """Causal divergence: pivots confirmed k bars after they form (no lookahead)."""
    w = 2 * k + 1
    lo_piv = df["low"] == df["low"].rolling(w, center=True).min()
    hi_piv = df["high"] == df["high"].rolling(w, center=True).max()
    n = len(df)
    last = n - 1 - k  # newest pivot we can confirm right now
    out = None
    for piv, col, rcol, bull in ((lo_piv, "low", "rsi", True), (hi_piv, "high", "rsi", False)):
        idx = np.flatnonzero(piv.values[: last + 1])
        if len(idx) < 2 or idx[-1] != last:
            continue  # only fire when the pivot was just confirmed
        a, b = idx[-2], idx[-1]
        p, r = df[col].values, df[rcol].values
        if bull and p[b] < p[a] and r[b] > r[a]:
            out = "BULLISH_DIV"
        if not bull and p[b] > p[a] and r[b] < r[a]:
            out = "BEARISH_DIV"
    return out


def analyze():
    df = indicators(fetch())
    last = df.iloc[-1]
    stats = backtest_stats(df)
    div = divergence(df)
    action = {1: "BUY", -1: "SELL"}.get(int(last["sig"]))
    return df, last, stats, div, action


def notify(text):
    if not (TOKEN and CHAT):
        print(text); return
    try:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          json={"chat_id": CHAT, "text": text}, timeout=10)
        if r.status_code != 200:
            print("Telegram error:", r.text)
    except requests.RequestException as e:
        print("Telegram error:", e)


def build_message(last, stats, div, action):
    lines = [f"{'🟢' if action == 'BUY' else '🔴' if action == 'SELL' else '🔷'} "
             f"{action or div} {SYMBOL} [{INTERVAL}]",
             f"Price: {last['close']:,.2f}  RSI: {last['rsi']:.1f}"]
    if action:
        sgn = 1 if action == "BUY" else -1
        lines += [f"Stop (1.5 ATR): {last['close'] - sgn * 1.5 * last['atr']:,.2f}",
                  f"Target (2 ATR): {last['close'] + sgn * 2 * last['atr']:,.2f}"]
    if div and action:
        lines.append(f"Confluence: {div}")
    lines.append(f"Backtest ({stats['n']} sigs, {HORIZON}-bar hold, after fees): "
                 f"win {stats['win']:.0f}% | PF {stats['pf']:.2f} | avg {stats['avg']:+.2f}%")
    lines.append(f"Candle close: {last.name:%Y-%m-%d %H:%M} UTC")
    return "\n".join(lines)


def scan(last_alert_ts):
    df, last, stats, div, action = analyze()
    if (action or div) and last.name != last_alert_ts:  # dedupe per candle
        notify(build_message(last, stats, div, action))
        return last.name
    return last_alert_ts


def main():
    if "--once" in sys.argv:
        df, last, stats, div, action = analyze()
        fresh = (time.time() * 1000 - last["ct"]) < SECS * 1000 * 1.5  # no state needed
        if (action or div) and fresh:
            notify(build_message(last, stats, div, action))
        return
    last_alert, backoff = None, 5
    while True:
        try:
            last_alert = scan(last_alert)
            backoff = 5
            # sleep until the next candle closes (+2s)
            time.sleep(SECS - (time.time() % SECS) + 2)
        except Exception as e:
            print("scan error:", e)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)


if __name__ == "__main__":
    main()
