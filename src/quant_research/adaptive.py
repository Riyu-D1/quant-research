"""Phase-4 adaptive strategy engine: monthly retrain + signal weighting.

The Phase-3 critique: every prior model was trained once and frozen for the
whole holdout. This engine tests the *process* a real desk would run:

* signal library — six predeclared signals, each an independent
  (datetime, instrument) score series: ``gbdt`` (Alpha158 LightGBM,
  retrained monthly on a rolling window), ``ev`` (Phase-3 disclosure-event
  LightGBM, retrained monthly), ``mom`` (12-1 momentum), ``rev`` (1-month
  reversal), ``vq`` (XBRL value/quality composite), ``news`` (FinBERT news
  sentiment).
* adaptive weighting — each month m, every signal is scored by the
  exponentially-weighted mean of its daily rank IC over the trailing
  ``ic_months`` before m (strictly: only days < m's first trading day).
  Weights floor at 0 (losing signals switch off) and L1-normalise; an
  all-nonpositive month yields all-zero weights -> the book sits in cash.
* regime guard — predeclared: exposure is cut when SPY is below its 200-day
  average or trailing realised vol is in its top decile (both measured on
  closes strictly before the day).
* portfolio — the Phase-3 ``MicroTop5`` £10 fractional-share simulator,
  extended with a per-day exposure multiplier.

Honesty contract:
  * a signal value on day D uses only data known before D (price features
    are point-in-time; filings key on filed < D; headlines key on publish
    date < trading day).
  * month m's weights use only rank ICs from days < first trading day of m.
  * month m's GBDTs train on rows with datetime < first trading day of m.
  * config selection (window, decay, weighting, regime action) happens on
    the 2021-2024 simulation only; the chosen process is scored on the
    2025+ holdout exactly once.

This module is deliberately qlib-free so it runs (and unit-tests) without a
qlib init: it consumes the Phase-4 feature *store* parquet + lightgbm.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RANK_SCALE = 3.46  # qlib CSRankNorm: (rank_pct - 0.5) * 3.46 ~ N(0,1)

STATIC_SIGNALS = ("mom", "rev", "vq", "news")
MODEL_SIGNALS = ("gbdt", "ev")
FROZEN_SIGNAL = "gbdt_frozen"
ALL_SIGNALS = ("gbdt", "ev", "mom", "rev", "vq", "news")


def empty_signal() -> pd.Series:
    """All signal series must live on the (datetime, instrument) grid —
    even empty ones, so cs_rank/combine never see a bare RangeIndex."""
    idx = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=["datetime", "instrument"])
    return pd.Series(dtype=float, index=idx)


# ---------------------------------------------------------------------------
# calendar / label helpers


def month_blocks(days: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(first, last) trading day of each calendar month present in `days`."""
    s = pd.Series(days, index=days)
    g = s.groupby([s.dt.year, s.dt.month])
    return [(pd.Timestamp(v.iloc[0]), pd.Timestamp(v.iloc[-1]))
            for _, v in g]


def cs_rank_norm(y: pd.Series) -> pd.Series:
    """qlib CSRankNorm on labels: per-day pct-rank -> (r - .5) * 3.46."""
    return (y.groupby(level="datetime").rank(pct=True) - 0.5) * RANK_SCALE


def cs_rank(s: pd.Series) -> pd.Series:
    return s.groupby(level="datetime").rank(pct=True)


def signal_metrics(pred: pd.Series, label: pd.Series) -> dict:
    """Same definition as walkforward.signal_metrics (kept qlib-free)."""
    df = pd.concat({"p": pred, "y": label.reindex(pred.index)}, axis=1).dropna()
    if df.empty:
        return {"IC": None, "ICIR": None, "RankIC": None, "RankICIR": None}
    by_day = df.groupby(level="datetime")
    ic = by_day.apply(lambda g: g["p"].corr(g["y"]))
    ric = by_day.apply(lambda g: g["p"].corr(g["y"], method="spearman"))
    return {"IC": float(ic.mean()), "ICIR": float(ic.mean() / ic.std()),
            "RankIC": float(ric.mean()),
            "RankICIR": float(ric.mean() / ric.std())}


def daily_rank_ic(score: pd.Series, label: pd.Series) -> pd.Series:
    df = pd.concat({"p": score, "y": label.reindex(score.index)}, axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float,
                         index=pd.DatetimeIndex([], name="datetime"))
    return df.groupby(level="datetime").apply(
        lambda g: g["p"].corr(g["y"], method="spearman")).rename("ic")


# ---------------------------------------------------------------------------
# monthly LightGBM retraining


LGB_PARAMS = {
    "objective": "mse", "verbosity": -1,
    "colsample_bytree": 0.8879, "learning_rate": 0.2,
    "subsample": 0.8789, "bagging_freq": 1,
    "lambda_l1": 205.6999, "lambda_l2": 580.9768,
    "max_depth": 8, "num_leaves": 210, "seed": 0,
}


def month_lgbm(X: pd.DataFrame, y: pd.Series, m0: pd.Timestamp,
               m1: pd.Timestamp, train_months: int, threads: int = 4,
               valid_days: int = 42, params: dict | None = None) -> pd.Series:
    """Train LGBM on [m0 - train_months, m0) -> predict days [m0, m1].

    The last `valid_days` trading days of the train window are the early-stop
    set. Features are used RAW: RobustZScoreNorm is a per-column monotone
    transform and LightGBM splits are monotone-invariant, so skipping it (and
    Fillna — LGBM routes NaN natively) changes nothing material while keeping
    the no-lookahead contract trivially intact (declared simplification).
    Labels are per-day CSRankNormed exactly like the qlib learn pipeline.
    """
    import lightgbm as lgb

    days = X.index.get_level_values("datetime")
    t0 = m0 - pd.DateOffset(months=train_months)
    tr_mask = (days >= t0) & (days < m0)
    te_mask = (days >= m0) & (days <= m1)
    tr_days = days[tr_mask].unique().sort_values()
    if len(tr_days) < 30 or te_mask.sum() == 0:
        return pd.Series(dtype=float)
    v0 = tr_days[-valid_days] if len(tr_days) > valid_days else tr_days[-1]
    lab = y.reindex(X.index)
    fit_m = tr_mask & (days < v0) & lab.notna()
    val_m = tr_mask & (days >= v0) & lab.notna()
    dtrain = lgb.Dataset(X[fit_m], cs_rank_norm(lab[fit_m]))
    dval = lgb.Dataset(X[val_m], cs_rank_norm(lab[val_m]))
    p = dict(LGB_PARAMS if params is None else params)
    p["num_threads"] = threads
    model = lgb.train(p, dtrain, num_boost_round=1000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    return pd.Series(model.predict(X[te_mask]), index=X[te_mask].index)


# ---------------------------------------------------------------------------
# adaptive weighting


def ic_history(signals: dict[str, pd.Series], label: pd.Series) -> pd.DataFrame:
    """Per-day rank IC of every signal -> DataFrame (day x signal)."""
    return pd.DataFrame({k: daily_rank_ic(v, label)
                         for k, v in signals.items()})


def ew_ic_weights(hist: pd.DataFrame, before: pd.Timestamp, months: int,
                  halflife_days: float, signals: list[str]) -> pd.Series:
    """EW mean rank-IC over [before - months, before); floored at 0 and
    L1-normalised. All-nonpositive -> all-zero (flat month, book in cash)."""
    lo = before - pd.DateOffset(months=months)
    w = hist.loc[(hist.index >= lo) & (hist.index < before)]
    if w.empty:
        return pd.Series(0.0, index=signals)
    idx = pd.DatetimeIndex(w.index)
    age = np.asarray((idx.max() - idx) / pd.Timedelta(days=1), dtype=float)
    lam = np.power(0.5, age / max(halflife_days, 1e-9))
    lam_s = pd.Series(lam, index=w.index)
    num = w.mul(lam_s, axis=0).sum()
    den = w.notna().mul(lam_s, axis=0).sum().replace(0, np.nan)
    ew = (num / den).reindex(signals).fillna(0.0)
    pos = ew.clip(lower=0.0)
    return pos / pos.sum() if pos.sum() > 0 else pos


def monthly_weights(hist: pd.DataFrame, months: list[tuple], weighting: str,
                    ic_months: int, halflife_days: float,
                    signals: list[str]) -> list[pd.Series]:
    """Weight vector per eval month, each using only days < that month."""
    out = []
    for m0, _m1 in months:
        if weighting == "ewic":
            w = ew_ic_weights(hist, m0, ic_months, halflife_days, signals)
        elif weighting == "equal":
            w = pd.Series(1.0 / len(signals), index=signals)
        elif weighting.startswith("single:"):
            name = weighting.split(":", 1)[1]
            w = pd.Series({s: (1.0 if s == name else 0.0) for s in signals})
        else:
            raise ValueError(weighting)
        out.append(w)
    return out


def rank_combine(ranked: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Σ w_i * rank_i on the union grid; missing rank -> 0.5 (neutral)."""
    w = weights[weights > 0]
    if w.empty:
        return pd.Series(dtype=float, name="score")
    r = ranked.reindex(columns=w.index).fillna(0.5)
    return r.mul(w, axis=1).sum(axis=1).rename("score")


# ---------------------------------------------------------------------------
# regime guard (predeclared — see spec yaml; not tuned post-hoc)


def regime_exposure(spy: pd.Series, sma: int = 200, vol_win: int = 21,
                    vol_lookback: int = 504, vol_q: float = 0.90,
                    action: float = 0.5) -> pd.Series:
    """Per-day exposure in {action, 1.0} from SPY closes strictly < day.

    Guard fires when either: SPY close < its 200d SMA (both lagged one day),
    or the trailing `vol_win`-day realised vol exceeds the `vol_q` quantile
    of its own trailing `vol_lookback`-day history.
    """
    spy = spy.sort_index()
    ma = spy.rolling(sma, min_periods=sma // 2).mean()
    rv = spy.pct_change().rolling(vol_win).std()
    thr = rv.shift(1).rolling(vol_lookback, min_periods=vol_lookback // 2) \
             .quantile(vol_q)
    risk = (spy.shift(1) < ma.shift(1)) | (rv.shift(1) > thr)
    return pd.Series(np.where(risk.fillna(False), action, 1.0),
                     index=spy.index, name="exposure")


# ---------------------------------------------------------------------------
# process simulation


def combined_score(ranked: pd.DataFrame, hist: pd.DataFrame,
                   months: list[tuple], weighting: str, ic_months: int,
                   halflife_days: float, signals: list[str]) -> tuple:
    """Monthly adaptive weights -> per-day combined score + weight ledger.

    ranked: per-day pct-rank frame (grid x signals). hist: day x signal
    rank ICs; month m's weights use only days < m's first trading day.
    """
    days = ranked.index.get_level_values("datetime")
    wlist = monthly_weights(hist, months, weighting, ic_months,
                            halflife_days, signals)
    parts, ledger = [], {}
    for (m0, m1), w in zip(months, wlist):
        mask = (days >= m0) & (days <= m1)
        parts.append(rank_combine(ranked[mask], w))
        ledger[str(m0.date())] = {k: round(float(v), 4)
                                  for k, v in w.items() if v > 0}
    score = pd.concat(parts).sort_index() if parts \
        else pd.Series(dtype=float, name="score")
    return score, ledger


def run_process(*, ranked: pd.DataFrame, hist: pd.DataFrame, label: pd.Series,
                months: list[tuple], weighting: str, ic_months: int,
                halflife_days: float, signals: list[str],
                exposure: pd.Series | None, closes: pd.DataFrame,
                micro_kw: dict, log=lambda m: None) -> dict:
    """One full monthly simulation of the adaptive process."""
    from quant_research import micro as mc

    score, ledger = combined_score(ranked, hist, months, weighting,
                                   ic_months, halflife_days, signals)
    met = mc.micro_backtest(score, closes, exposure=exposure, **micro_kw)
    sig = signal_metrics(score, label)
    out = {**sig, **{k: v for k, v in met.items() if k != "daily_excess"},
           "weight_ledger": ledger, "daily_rets": met["daily_excess"],
           "score": score}
    log(f"  {weighting} ic={sig['RankIC']} ir={met['information_ratio']:.4f} "
        f"ar={met['annualized_return']:.4f} mdd={met['max_drawdown']:.4f}")
    return out


# ---------------------------------------------------------------------------
# monthly model-prediction cache (resume across runner kills)


def preds_cache(root: Path, name: str, wmonths: int, ym: str,
                make: callable, log=lambda m: None) -> pd.Series:
    """Checkpointed monthly preds: <root>/<name>__w<w>__<YYYY-MM>.pkl."""
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{name}__w{wmonths}__{ym}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    s = make()
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(s, f)
    tmp.replace(p)
    log(f"    trained {p.stem} ({len(s)} rows)")
    return s


def model_signal_series(X: pd.DataFrame, y: pd.Series, name: str,
                        months: list[tuple], train_months: int, cache: Path,
                        threads: int, log=lambda m: None) -> pd.Series:
    """Monthly-retrained LGBM preds over the given month blocks, cached."""
    parts = []
    for m0, m1 in months:
        s = preds_cache(
            cache, name, train_months, f"{m0.year}-{m0.month:02d}",
            lambda m0=m0, m1=m1: month_lgbm(X, y, m0, m1, train_months,
                                            threads=threads), log)
        if not s.empty:
            parts.append(s)
    return pd.concat(parts).sort_index() if parts else empty_signal()


def frozen_signal_series(X: pd.DataFrame, y: pd.Series, m0: pd.Timestamp,
                         eval_days: pd.DatetimeIndex, train_months: int,
                         threads: int) -> pd.Series:
    """Train once on [m0 - train_months, m0) -> predict all eval_days (the
    frozen-GBDT comparator mirroring the Phase-2/3 approach)."""
    return month_lgbm(X, y, m0, eval_days.max(), train_months,
                      threads=threads,
                      params={**LGB_PARAMS, "objective": "mse"})


# ---------------------------------------------------------------------------
# static (price/fundamental/news) signal construction


def static_signals(store: pd.DataFrame,
                   news_daily: pd.DataFrame | None = None) -> dict:
    """The model-free signals from store columns + optional news table.

    store index is (instrument, datetime); signals returned on the
    (datetime, instrument) grid. vq = mean of per-day ranks of the three
    value/quality columns (documented composite, no fitting).
    """
    def rank3(*cols):
        parts = [store[c].swaplevel(0, 1).sort_index()
                 .groupby(level="datetime").rank(pct=True) for c in cols]
        return sum(parts) / len(parts)

    out = {
        "mom": store["mom_12_1"].swaplevel(0, 1).sort_index(),
        "rev": store["rev_1m"].swaplevel(0, 1).sort_index(),
        "vq": rank3("vq_bm", "vq_ep", "vq_roe"),
    }
    if news_daily is not None and not news_daily.empty:
        from quant_research import news as nw
        out["news"] = nw.daily_signal(news_daily, store)
    else:
        # all-NaN on the grid: neutral 0.5 rank downstream, no IC evidence
        out["news"] = out["mom"] * np.nan
    return out
