"""Phase-6 US signal library — honest, PIT, cross-sectional signals.

All signals are computed from the qlib-format US dump (events.load_field
bin reader) so they share the store's adjusted-price basis and PIT
calendar: a signal at day t uses only data available on day t (fundamental
signals key off the SEC *filed* date, never the period end).

Every signal returns a Series on the (datetime, instrument) grid — the
same convention as walkforward fold preds — so they drop straight into
phase6_us's rank-IC / portfolio machinery.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research import events as ev


def empty() -> pd.Series:
    idx = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([], name="datetime"), pd.Index([], name="instrument")])
    return pd.Series(dtype=float, index=idx)


def _long(df: pd.DataFrame) -> pd.Series:
    s = df.stack()
    s.index.names = ["datetime", "instrument"]
    return s


# ---------------------------------------------------------------------------
# price panels


def load_panels(provider_dir: Path, symbols: list[str],
                cal: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(close, volume) date×symbol frames from the qlib dump."""
    close, vol = {}, {}
    for s in symbols:
        close[s] = ev.load_field(provider_dir, s, cal, "close")
        vol[s] = ev.load_field(provider_dir, s, cal, "volume")
    return (pd.DataFrame(close, index=cal).sort_index(),
            pd.DataFrame(vol, index=cal).sort_index())


def price_signals(close: pd.DataFrame, volume: pd.DataFrame) -> dict:
    """Price/volume-only signals; each is a (date × symbol) frame."""
    r1 = close.pct_change()
    mkt = r1.mean(axis=1)  # equal-weight universe index (SPY proxy for beta)
    var_m = mkt.rolling(60, min_periods=40).var()
    beta = r1.rolling(60, min_periods=40).cov(mkt).div(var_m, axis=0)
    dollar = (close * volume).clip(lower=1.0)
    sigs = {
        # short-horizon reversal: negative 5d return
        "srev5": -(close / close.shift(5) - 1.0),
        # negative 21d return (existing 'rev' — keep for the combos)
        "rev": -(close / close.shift(21) - 1.0),
        # 6-month momentum, skip last month
        "mom6": close.shift(21) / close.shift(126) - 1.0,
        # 12-month momentum, skip last month (existing 'mom')
        "mom": close.shift(21) / close.shift(252) - 1.0,
        # low realised-volatility (20d daily-return std, negated)
        "lowvol": -r1.rolling(20, min_periods=15).std(),
        # low market beta (60d vs equal-weight universe, negated)
        "beta": -beta,
        # Amihud illiquidity: mean(|r1| / dollar_volume, 20d) — higher = more
        # illiquid; the premium pays illiquid names, so the raw level is the
        # signal (kept positive, not negated).
        "amihud": r1.abs().div(dollar).rolling(20, min_periods=15).mean(),
    }
    return sigs


def sector_neutral(sig: pd.Series, sector_map: dict[str, str]) -> pd.Series:
    """Subtract each day's sector mean — pure within-sector ranking."""
    if sig.empty or not sector_map:
        return sig
    df = sig.rename("s").reset_index()
    df["sec"] = df["instrument"].map(sector_map).fillna("UNK")
    df["s"] = df["s"] - df.groupby(["datetime", "sec"])["s"].transform("mean")
    return df.set_index(["datetime", "instrument"])["s"].sort_index()


# ---------------------------------------------------------------------------
# fundamentals (PIT via SEC filed date)


def fundamental_signals(provider_dir: Path, symbols: list[str],
                        cal: pd.DatetimeIndex,
                        fundamentals_path: Path,
                        shares_path: Path | None = None,
                        close: pd.DataFrame | None = None) -> dict:
    """pead (NI YoY surprise, 63d) + qual (NI margin, 95d) + vq composite."""
    from quant_research import fundamentals as F
    out = {"pead": empty(), "qual": empty(), "vq": empty()}
    if not Path(fundamentals_path).exists():
        return out
    f = pd.read_parquet(fundamentals_path)
    f["symbol"] = f["symbol"].map(ev.canon)
    f["filed"] = pd.to_datetime(f["filed"])
    f["end"] = pd.to_datetime(f["end"])
    ni = f[(f["group"] == "ni") & f["form"].isin(
        ["10-Q", "10-K", "10-Q/A", "10-K/A"])]
    rv = f[(f["group"] == "rev") & f["form"].isin(
        ["10-Q", "10-K", "10-Q/A", "10-K/A"])]
    pead_frames, qual_frames = {}, {}
    for s in symbols:
        n = ni[ni["symbol"] == s].sort_values(["end", "filed"])
        n = n.drop_duplicates("end", keep="last").set_index("end")
        q = n[n["fp"].isin(["Q1", "Q2", "Q3", "Q4"])]["val"].sort_index()
        surprise = (q - q.shift(4)) / (q.shift(4).abs() + 1e-9)
        filed = n.loc[surprise.index, "filed"]
        # signal valid from filed+1d (conservative) through the next ~quarter
        ser = pd.Series(surprise.values, index=filed + pd.Timedelta(days=1))
        ser = ser[~ser.index.duplicated()].sort_index()
        if len(ser):
            daily = ser.reindex(cal, method="ffill")
            last_filed = pd.Series(ser.index, index=ser.index).reindex(
                cal, method="ffill")
            mask = (pd.Series(cal, index=cal) - last_filed).dt.days <= 63
            pead_frames[s] = daily.where(mask).values
        r = rv[rv["symbol"] == s].sort_values(["end", "filed"])
        r = r.drop_duplicates("end", keep="last").set_index("end")["val"]
        nq = n["val"].sort_index()
        both = pd.concat({"ni": nq, "rev": r}, axis=1).dropna()
        if len(both):
            margin = both["ni"] / (both["rev"].abs() + 1e-9)
            mfiled = n.loc[margin.index, "filed"]
            mser = pd.Series(margin.values,
                             index=mfiled + pd.Timedelta(days=1))
            mser = mser[~mser.index.duplicated()].sort_index()
            mdaily = mser.reindex(cal, method="ffill")
            mfiled2 = pd.Series(mser.index, index=mser.index).reindex(
                cal, method="ffill")
            mmask = (pd.Series(cal, index=cal) - mfiled2).dt.days <= 95
            qual_frames[s] = mdaily.where(mmask).values
    for name, frames in (("pead", pead_frames), ("qual", qual_frames)):
        if frames:
            m = pd.DataFrame(frames, index=cal).sort_index()
            out[name] = _long(m)
    # ---- vq composite (bm/ep/roe, same math as the phase-4 store) ------
    if close is not None:
        shares = pd.DataFrame()
        if shares_path and Path(shares_path).exists():
            shares = pd.read_parquet(shares_path)
            shares["filed"] = pd.to_datetime(shares["filed"])
            shares["symbol"] = shares["symbol"].map(ev.canon)
        shr_by = {s: g.rename(columns={"val": "shares"})
                  for s, g in shares.groupby("symbol")} \
            if not shares.empty else {}
        f_by = {s: g for s, g in f.groupby("symbol")}
        parts = {c: {} for c in ("vq_bm", "vq_ep", "vq_roe")}
        for s in symbols:
            fs = f_by.get(s)
            if fs is None or fs.empty or s not in close.columns:
                continue
            vqf = F.vq_features(fs, shr_by.get(s),
                                close[s].values, cal)
            for c in parts:  # noqa: PLC0206
                parts[c][s] = vqf[c]
        if parts["vq_bm"]:
            ranks = []
            for c, frames in parts.items():
                m = pd.DataFrame(frames, index=cal)
                ranks.append(_long(m.rank(axis=1, pct=True)))
            out["vq"] = pd.concat(ranks, axis=1).mean(axis=1)
    return out


# ---------------------------------------------------------------------------
# events + news


def event_signals(store_like: pd.DataFrame | None,
                  feats_path: Path) -> dict:
    """ins_net: insider net-buying intensity from the Phase-3 event store."""
    out = {"ins": empty()}
    p = Path(feats_path)
    if p.exists():
        feats = pd.read_parquet(p, columns=["ins_net_usd_90d"])
        s = feats["ins_net_usd_90d"].swaplevel(0, 1).sort_index()
        out["ins"] = s
    return out


def news_signals(news_dir: Path) -> dict:
    """EW FinBERT daily score if a Kaggle sentiment push has landed."""
    out = {"news": empty()}
    files = sorted(Path(news_dir).glob("news_sentiment/*.parquet"))
    if files:
        df = pd.concat([pd.read_parquet(f) for f in files])
        col = "score" if "score" in df.columns else df.columns[-1]
        df["datetime"] = pd.to_datetime(df.get("date", df.get("datetime")))
        sym = df.get("symbol", df.get("instrument")).map(ev.canon)
        s = pd.Series(df[col].values,
                      index=pd.MultiIndex.from_arrays(
                          [df["datetime"], sym],
                          names=["datetime", "instrument"]))
        out["news"] = s.sort_index()
    return out


# ---------------------------------------------------------------------------
# registry


def build_signals(provider_dir: Path, symbols: list[str],
                  cal: pd.DatetimeIndex, names: list[str], *,
                  fundamentals_path: Path, feats_path: Path,
                  news_dir: Path, sector_map: dict[str, str] | None,
                  close: pd.DataFrame | None = None,
                  volume: pd.DataFrame | None = None) -> dict:
    """Assemble every requested signal on the (datetime, instrument) grid."""
    if close is None or volume is None:
        close, volume = load_panels(provider_dir, symbols, cal)
    sigs = {}
    need_price = [n for n in names if n.removeprefix("sn_") in
                  ("srev5", "rev", "mom6", "mom", "lowvol", "beta", "amihud")]
    if need_price:
        sigs.update({k: _long(v) for k, v in
                     price_signals(close, volume).items()})
    if any(n.removeprefix("sn_") in ("pead", "qual", "vq") for n in names):
        sigs.update(fundamental_signals(
            provider_dir, symbols, cal, fundamentals_path,
            shares_path=Path(str(fundamentals_path)).parent / "shares_out.parquet",
            close=close))
    if any(n.removeprefix("sn_") == "ins" for n in names):
        sigs.update(event_signals(None, feats_path))
    if "news" in names:
        sigs.update(news_signals(news_dir))
    # sector-neutral variants: sn_<base> = base - per-day sector mean
    out = {}
    for n in names:
        if n.startswith("sn_"):
            s = sector_neutral(sigs.get(n[3:], empty()), sector_map or {})
        else:
            s = sigs.get(n, empty())
        # event/parquet sources can carry duplicate (datetime, instrument)
        # rows (amended filings etc.) — dedupe before they hit reindex()
        if not s.empty:
            s = s[~s.index.duplicated(keep="first")].sort_index()
        out[n] = s
    return out
