"""Phase-4 live signal computation — the daily-facing half of the bot.

The research harness (adaptive.py) runs over a parquet feature store; the
paper bot needs the SAME signal math computed from live inputs instead:

  mom   close[t-21] / close[t-252] - 1          (Alpaca daily bars, iex feed)
  rev   -(close[t] / close[t-21] - 1)           (Alpaca daily bars)
  vq    latest composite XBRL value/quality rank (snapshot parquet — the
        fundamentals move quarterly, refreshed by the refresh_snapshot job)
  news  latest EW FinBERT score per symbol      (snapshot parquet from the
        Kaggle daily_sentiment push, if present)

Model signals (gbdt/ev) are intentionally NOT computable live — they need
the monthly-retrain store pipeline. If a pinned config lists a signal this
module can't produce, the bot drops it and renormalises, logging loudly.

Weighting: 'equal' = 1/k each; 'ewic' = EW rank-IC weights exactly like
adaptive.ew_ic_weights, computed on bars-derived signals (mom/rev) plus
any snapshot signal that carries a dated history — currently none, so
snapshot signals take equal share under ewic (documented limitation).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from quant_research import adaptive as ad

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
BAR_LOOKBACK_D = 270         # enough for 252d momentum + buffer
SPY_LOOKBACK_D = 560         # regime needs 504d vol lookback


def log(msg: str) -> None:
    print(f"[live {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Alpaca market data (free iex feed — no subscription)


def to_alpaca(sym: str) -> str:
    """canon ticker (BRK-B) -> Alpaca ticker (BRK.B)."""
    return sym.replace("-", ".")


def alpaca_bars(key: str, secret: str, symbols: list[str],
                start: pd.Timestamp, end: pd.Timestamp,
                timeframe: str = "1Day", feed: str = "iex",
                pause: float = 0.35, batch: int = 100) -> pd.DataFrame:
    """Daily closes for `symbols` between start/end -> (symbol, date) frame.

    Paginates the multi-symbol bars endpoint, batching `batch` tickers per
    request (the endpoint caps the symbol list). Missing rows -> NaN."""
    hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    frames = []
    for i in range(0, len(symbols), batch):
        chunk = ",".join(to_alpaca(s) for s in symbols[i:i + batch])
        token = None
        while True:
            params = {"symbols": chunk, "timeframe": timeframe,
                      "start": start.strftime("%Y-%m-%d"),
                      "end": end.strftime("%Y-%m-%d"),
                      "limit": 10000, "feed": feed,
                      "adjustment": "all", "sort": "asc"}
            if token:
                params["page_token"] = token
            r = requests.get(BARS_URL, headers=hdr, params=params,
                             timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2.0)
                r = requests.get(BARS_URL, headers=hdr, params=params,
                                 timeout=60)
            r.raise_for_status()
            d = r.json()
            bars = d.get("bars", {})
            rows = [{"symbol": s, "date": b["t"], "close": b["c"]}
                    for s, bl in bars.items() for b in bl]
            if rows:
                frames.append(pd.DataFrame(rows))
            token = d.get("next_page_token")
            if not token:
                break
            time.sleep(pause)
    if not frames:
        return pd.DataFrame(columns=["close"])
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    wide = df.pivot(index="date", columns="symbol", values="close") \
             .sort_index()
    # Alpaca uses dots in class shares (BRK.B); canon uses dashes (BRK-B)
    wide.columns = [str(c).replace(".", "-") for c in wide.columns]
    return wide


def spy_closes(key: str, secret: str, end: pd.Timestamp,
               days: int = SPY_LOOKBACK_D) -> pd.Series:
    wide = alpaca_bars(key, secret, ["SPY"],
                       end - pd.Timedelta(days=days * 1.6), end)
    if "SPY" not in wide.columns:
        raise RuntimeError("no SPY bars returned")
    return wide["SPY"].dropna()


# ---------------------------------------------------------------------------
# signal math (same formulas as adaptive.static_signals / phase4_build)


def momentum(close: pd.Series, day: pd.Timestamp) -> float:
    """close[t-21]/close[t-252]-1 using closes strictly <= day."""
    px = close.loc[:day].dropna()
    if len(px) < 253:
        return np.nan
    return float(px.iloc[-22] / px.iloc[-253] - 1.0)


def reversal(close: pd.Series, day: pd.Timestamp) -> float:
    """-(close[t]/close[t-21]-1) — negative one-month return."""
    px = close.loc[:day].dropna()
    if len(px) < 22:
        return np.nan
    return -float(px.iloc[-1] / px.iloc[-22] - 1.0)


def reversal5(close: pd.Series, day: pd.Timestamp) -> float:
    """-(close[t]/close[t-5]-1) — negative one-week return (phase 6)."""
    px = close.loc[:day].dropna()
    if len(px) < 6:
        return np.nan
    return -float(px.iloc[-1] / px.iloc[-6] - 1.0)


def momentum6(close: pd.Series, day: pd.Timestamp) -> float:
    """close[t-21]/close[t-126]-1 — 6-month momentum, skip last month."""
    px = close.loc[:day].dropna()
    if len(px) < 127:
        return np.nan
    return float(px.iloc[-22] / px.iloc[-127] - 1.0)


def lowvol(close: pd.Series, day: pd.Timestamp) -> float:
    """-std(daily returns, 20) — low realised-volatility tilt."""
    px = close.loc[:day].dropna()
    if len(px) < 21:
        return np.nan
    return -float(px.pct_change().iloc[-20:].std())


def lowbeta(close: pd.Series, mkt: pd.Series, day: pd.Timestamp) -> float:
    """-cov(r, r_mkt)/var(r_mkt) over 60d — market = equal-weight universe."""
    px = close.loc[:day].dropna()
    if len(px) < 61:
        return np.nan
    r = px.pct_change().iloc[-60:]
    rm = mkt.loc[r.index]
    v = float(rm.var())
    return -float(r.cov(rm) / v) if v > 0 else np.nan


_BARS_FNS = {"mom": momentum, "rev": reversal, "srev5": reversal5,
             "mom6": momentum6, "lowvol": lowvol}


def fwd_return(close: pd.Series, day: pd.Timestamp,
               skip: int = 1, horizon: int = 1) -> float:
    """close[day+skip+horizon]/close[day+skip]-1 — same 2-day shifted label
    as the research pipeline (Ref($close,-2)/Ref($close,-1)-1)."""
    px = close.loc[day:].dropna()
    if len(px) <= skip + horizon:
        return np.nan
    return float(px.iloc[skip + horizon] / px.iloc[skip] - 1.0)


def bars_signals(closes: pd.DataFrame, days: pd.DatetimeIndex,
                 names: list[str]) -> dict[str, pd.Series]:
    """bars-derived signals per (date, symbol) — point-in-time via .loc[:day].

    Known names: mom, rev, srev5, mom6, lowvol, beta (equal-weight-universe
    market proxy). Snapshot-sourced signals (vq/news) are handled
    elsewhere."""
    out: dict[str, pd.Series] = {}
    mkt = closes.pct_change().mean(axis=1)
    for sig in names:
        if sig not in _BARS_FNS and sig != "beta":
            continue
        recs = {}
        for d in days:
            if sig == "beta":
                recs[d] = {s: lowbeta(closes[s], mkt, d)
                           for s in closes.columns}
            else:
                recs[d] = {s: _BARS_FNS[sig](closes[s], d)
                           for s in closes.columns}
        s_ = pd.DataFrame(recs).T.stack()
        s_.index.names = ["datetime", "instrument"]
        out[sig] = s_
    return out


def labels_from_bars(closes: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """Forward label for every (day, symbol) — used by ewic IC history."""
    recs = {}
    for d in days:
        recs[d] = {s: fwd_return(closes[s], d) for s in closes.columns}
    s_ = pd.DataFrame(recs).T.stack()
    s_.index.names = ["datetime", "instrument"]
    return s_


def snapshot_signal(path: Path, name: str, stale_days: int = 45,
                    asof_day: pd.Timestamp | None = None
                    ) -> tuple[pd.Series, str]:
    """Latest per-symbol score from a snapshot parquet -> (series, note).

    Staleness is measured against `asof_day` (the signal day) — wall-clock
    'now' would make backtests and --date runs look artificially stale.
    Snapshot schemas: columns [symbol, score, asof] — asof optional."""
    note = ""
    if not path.exists():
        return pd.Series(dtype=float, name=name), f"{name}: snapshot missing"
    snap = pd.read_parquet(path)
    if "asof" in snap.columns and not snap.empty:
        asof = pd.to_datetime(snap["asof"]).max()
        ref = pd.Timestamp(asof_day) if asof_day is not None \
            else pd.Timestamp.now().normalize()
        age = (ref - asof).days
        note = f"{name}: snapshot asof {asof.date()} (age {age}d)"
        if age > stale_days:
            return pd.Series(dtype=float, name=name), \
                f"{note} — STALE >{stale_days}d, dropped"
    s = pd.Series(snap["score"].values, index=snap["symbol"], name=name)
    return s.groupby(level=0).mean(), note


# ---------------------------------------------------------------------------
# regime + weights + final picks


def regime_state(spy: pd.Series, cfg: dict,
                 day: pd.Timestamp) -> tuple[float, str]:
    """(exposure, reason) for `day` using the predeclared guard math."""
    exp = ad.regime_exposure(spy, sma=cfg["sma"], vol_win=cfg["vol_win"],
                             vol_lookback=cfg["vol_lookback"],
                             vol_q=cfg["vol_q"], action=cfg["action"])
    e = float(exp.loc[:day].iloc[-1]) if len(exp.loc[:day]) else 1.0
    spy_l = spy.loc[:day]
    ma = spy_l.rolling(cfg["sma"], min_periods=cfg["sma"] // 2).mean()
    below = bool(len(ma.dropna()) and spy_l.iloc[-1] < ma.iloc[-1])
    rv = spy_l.pct_change().rolling(cfg["vol_win"]).std()
    thr = rv.shift(1).rolling(cfg["vol_lookback"],
                            min_periods=cfg["vol_lookback"] // 2).quantile(
                            cfg["vol_q"])
    hot = bool(len(thr.dropna()) and rv.iloc[-1] > thr.iloc[-1])
    why = "+".join(x for x, on in (("spy<sma", below), ("vol>q", hot)) if on) \
        or "calm"
    return e, why


def final_scores(signals: dict[str, pd.Series], weights: dict[str, float],
                 day: pd.Timestamp) -> pd.Series:
    """Weighted cross-sectional-rank composite for `day`."""
    parts = []
    for s, w in weights.items():
        v = signals.get(s)
        if v is None or v.empty or w <= 0:
            continue
        day_v = v.xs(day, level="datetime") if isinstance(
            v.index, pd.MultiIndex) else v
        parts.append(day_v.rank(pct=True) * w)
    if not parts:
        return pd.Series(dtype=float)
    return sum(parts).dropna().sort_values(ascending=False)


def gbpusd() -> float:
    """GBP->USD from a free FX endpoint; falls back to the research static."""
    try:
        r = requests.get("https://open.er-api.com/v6/latest/GBP", timeout=15)
        rate = float(r.json()["rates"]["USD"])
        if 1.0 < rate < 2.0:
            return rate
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return 1.27
    return 1.27
