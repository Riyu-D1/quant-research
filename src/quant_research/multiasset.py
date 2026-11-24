"""Phase-10 multi-asset / factor building blocks (pure, causal, unit-tested).

Every weight function reads a wide close panel `P` (days x tickers,
total-return adjusted) and returns the target book for ONE decision day
using rows <= that day only. `monthly_book` turns a per-day weight
function into the {day: {sym: weight}} map the phase-8 micro sim trades
(recomputed on the first trading day of each month, held in between —
the sim's 2% band handles drift). Weights are fractions of equity;
anything not allocated is parked in the cash ETF (BIL) explicitly.

Long-only, unlevered by default (gross <= 1). `gross_cap` > 1 is only
used for the flagged leverage sensitivities; the borrowed share is
charged financing by `financing_drag`.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

Book = dict[str, float]


# ---------------------------------------------------------------------------
# calendars and estimators


def month_start_mask(days: pd.DatetimeIndex) -> np.ndarray:
    """True on the first trading day of each calendar month."""
    ym = days.year * 100 + days.month
    m = np.ones(len(days), dtype=bool)
    m[1:] = ym[1:] != ym[:-1]
    return m


def trailing_return(P: pd.DataFrame, n: int, skip: int = 0) -> pd.DataFrame:
    """P[t-skip] / P[t-n] - 1 (NaN until n rows of history)."""
    return P.shift(skip) / P.shift(n) - 1.0


def realised_vol(P: pd.DataFrame, win: int) -> pd.DataFrame:
    """Annualised std of the last `win` daily returns (full window only)."""
    return P.pct_change().rolling(win, min_periods=win).std() * math.sqrt(252)


def sma(P: pd.DataFrame, n: int) -> pd.DataFrame:
    return P.rolling(n, min_periods=n).mean()


def cov_at(R: pd.DataFrame, t: int, win: int, cols: list[str]) -> np.ndarray | None:
    """Annualised covariance of daily returns over rows (t-win, t]."""
    blk = R[cols].iloc[max(0, t - win + 1):t + 1].dropna()
    if len(blk) < max(20, int(0.8 * win)):
        return None
    return blk.cov().values * 252


def port_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(math.sqrt(max(0.0, w @ cov @ w)))


def erc_weights(cov: np.ndarray, iters: int = 2000, tol: float = 1e-12) -> np.ndarray:
    """Long-only equal-risk-contribution weights (sum 1) by the cyclical
    coordinate method of Griveau-Billion, Richard & Roncalli (2013):
    each w_i solves w_i (Sigma w)_i = b_i with b = 1/n, then normalise."""
    n = cov.shape[0]
    b = np.full(n, 1.0 / n)
    w = 1.0 / np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    w /= w.sum()
    for _ in range(iters):
        w_old = w.copy()
        for i in range(n):
            c = cov[i] @ w - cov[i, i] * w[i]
            a = cov[i, i]
            w[i] = (-c + math.sqrt(c * c + 4 * a * b[i])) / (2 * a)
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w / w.sum()


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    mrc = cov @ w
    tot = w @ mrc
    return w * mrc / tot if tot > 0 else np.zeros_like(w)


def vol_target(w: np.ndarray, cov: np.ndarray, target: float,
               gross_cap: float = 1.0) -> np.ndarray:
    """Scale risky weights to `target` annual vol, gross <= gross_cap."""
    g = float(w.sum())
    if g <= 0:
        return w
    pv = port_vol(w, cov)
    s = gross_cap / g if pv <= 0 else min(target / pv, gross_cap / g)
    return w * s


def with_cash(book: Book, cash: str) -> Book:
    """Drop ~0 weights; park 1 - gross in the cash ETF (if gross < 1)."""
    out = {k: float(v) for k, v in book.items() if v > 1e-9}
    rest = 1.0 - sum(out.values())
    if rest > 1e-9:
        out[cash] = out.get(cash, 0.0) + rest
    return out


# ---------------------------------------------------------------------------
# per-day weight rules (row t of the panels = the decision close)


class Ctx:
    """Pre-computed causal panels shared by every rule."""

    def __init__(self, P: pd.DataFrame, est: dict, cash: str):
        self.P = P
        self.R = P.pct_change()
        self.cash = cash
        self.est = est
        self.vol = realised_vol(P, est["vol_win"])
        self.r12 = trailing_return(P, est["mom_long"])
        self.r12_1 = trailing_return(P, est["mom_long"], est["mom_skip"])
        self.ret_h = {h: trailing_return(P, h) for h in est["trend_horizons"]}
        self.sma = sma(P, est["sma_days"])

    def xs_ret(self, h: int, t: int, cols: list[str]) -> pd.Series:
        """h-day return in excess of the cash ETF's h-day return."""
        r = self.ret_h[h] if h in self.ret_h else trailing_return(self.P, h)
        c = r[self.cash].iloc[t] if self.cash in r.columns else 0.0
        return r[cols].iloc[t] - (c if np.isfinite(c) else 0.0)


def _ready(row: pd.Series, cols: list[str]) -> list[str]:
    return [c for c in cols if c in row.index and np.isfinite(row[c])]


def w_hold(weights: Book) -> Callable[[Ctx, int], Book]:
    return lambda ctx, t: dict(weights)


def w_tsmom(cols: list[str], gross_cap: float = 1.0):
    """Moskowitz-Ooi-Pedersen 12m TSMOM, long-only: assets with positive
    12m return in excess of cash get inverse-vol weight (share of the
    whole universe's 1/sigma), off assets -> cash; then vol-targeted."""
    def f(ctx: Ctx, t: int) -> Book:
        vol = ctx.vol.iloc[t]
        ok = _ready(vol, cols)
        xs = ctx.xs_ret(ctx.est["mom_long"], t, ok)
        ok = [c for c in ok if np.isfinite(xs.get(c, np.nan))]
        if not ok:
            return {ctx.cash: 1.0}
        iv = 1.0 / vol[ok]
        base = iv / iv.sum()
        return _finish(ctx, t, base.where(xs[ok] > 0, 0.0), gross_cap, base)
    return f


def w_trend(cols: list[str], gross_cap: float = 1.0):
    """Hurst-Ooi-Pedersen style multi-horizon trend: signal = mean of
    sign(excess return) over 1/3/12 months, long-only part, scaling the
    inverse-vol weight; vol-targeted."""
    def f(ctx: Ctx, t: int) -> Book:
        vol = ctx.vol.iloc[t]
        ok = _ready(vol, cols)
        X = pd.concat([ctx.xs_ret(h, t, ok) for h in ctx.est["trend_horizons"]],
                      axis=1).dropna()
        if X.empty:
            return {ctx.cash: 1.0}
        sig = np.sign(X).mean(axis=1).clip(lower=0.0)
        iv = 1.0 / vol[X.index]
        base = iv / iv.sum()
        return _finish(ctx, t, base * sig, gross_cap, base)
    return f


def w_gtaa(cols: list[str]):
    """Faber (2007): each asset 1/N when close > ~10-month SMA, else cash."""
    def f(ctx: Ctx, t: int) -> Book:
        p, m = ctx.P.iloc[t], ctx.sma.iloc[t]
        w = {c: 1.0 / len(cols) for c in cols
             if np.isfinite(p.get(c, np.nan)) and np.isfinite(m.get(c, np.nan))
             and p[c] > m[c]}
        return with_cash(w, ctx.cash)
    return f


def w_gem(us: str = "SPY", intl: str = "EFA", bonds: str = "AGG"):
    """Antonacci global equities momentum: if US 12m > cash 12m hold the
    better of US / international (12m), else bonds."""
    def f(ctx: Ctx, t: int) -> Book:
        r = ctx.r12.iloc[t]
        cash_r = r.get(ctx.cash, np.nan)
        if not all(np.isfinite(r.get(c, np.nan)) for c in (us, intl)) \
                or not np.isfinite(cash_r):
            return {ctx.cash: 1.0}
        if r[us] > cash_r:
            return {us if r[us] >= r[intl] else intl: 1.0}
        return {bonds: 1.0}
    return f


def w_dualmom(cols: list[str], k: int):
    """Top-k by 12m return, each 1/k, each needing 12m > cash's (else its
    slot goes to cash)."""
    def f(ctx: Ctx, t: int) -> Book:
        r = ctx.r12.iloc[t]
        ok = _ready(r, cols)
        cash_r = r.get(ctx.cash, np.nan)
        if not ok or not np.isfinite(cash_r):
            return {ctx.cash: 1.0}
        top = r[ok].sort_values(ascending=False, kind="stable").index[:k]
        return with_cash({c: 1.0 / k for c in top if r[c] > cash_r}, ctx.cash)
    return f


def w_erc(cols: list[str], trend_filter: bool = False, gross_cap: float = 1.0):
    """ERC across the universe (252d covariance); optional 12m trend
    filter zeroes assets with non-positive excess return (no renorm);
    vol-targeted, rest to cash."""
    def f(ctx: Ctx, t: int) -> Book:
        ok = _ready(ctx.P.iloc[t], cols)
        cov = cov_at(ctx.R, t, ctx.est["cov_win"], ok) if ok else None
        if cov is None:
            return {ctx.cash: 1.0}
        base = pd.Series(erc_weights(cov), index=ok)
        w = base
        if trend_filter:
            xs = ctx.xs_ret(ctx.est["mom_long"], t, ok)
            w = base.where(xs > 0, 0.0)
        return _finish(ctx, t, w, gross_cap, base, cov=cov)
    return f


def w_sector(cols: list[str], k: int, abs_filter: bool = False):
    """Sector rotation: top-k by 12-1 momentum, 1/k each; abs_filter needs
    the pick's 12m return > cash's, else its slot goes to cash."""
    def f(ctx: Ctx, t: int) -> Book:
        s = ctx.r12_1.iloc[t]
        ok = _ready(s, cols)
        if not ok:
            return {ctx.cash: 1.0}
        top = s[ok].sort_values(ascending=False, kind="stable").index[:k]
        if abs_filter:
            r = ctx.r12.iloc[t]
            cash_r = r.get(ctx.cash, np.nan)
            top = [c for c in top if np.isfinite(cash_r) and r[c] > cash_r]
        return with_cash({c: 1.0 / k for c in top}, ctx.cash)
    return f


def _finish(ctx: Ctx, t: int, w: pd.Series, gross_cap: float,
            base: pd.Series, cov: np.ndarray | None = None) -> Book:
    """Vol-target on the STRATEGIC book `base` (every asset on): one
    scalar s = min(target / vol(base), gross_cap / sum(base)) is applied
    to the signal-filtered weights `w`, so switched-off assets go to cash
    rather than being re-spread over the survivors."""
    if (w > 0).sum() == 0:
        return {ctx.cash: 1.0}
    C = cov if cov is not None else cov_at(ctx.R, t, ctx.est["cov_win"],
                                           list(base.index))
    if C is None:
        return with_cash((w / max(1.0, base.sum())).to_dict(), ctx.cash)
    b = base.values
    s = float(vol_target(b, C, ctx.est["target_vol"], gross_cap).sum() / b.sum())
    return with_cash((w * s).to_dict(), ctx.cash)


# ---------------------------------------------------------------------------
# books, combos, overlays, leverage accounting


def monthly_book(ctx: Ctx, rule: Callable[[Ctx, int], Book],
                 days: pd.DatetimeIndex, freq: str = "monthly") -> dict:
    """{day: book} on `days` (must be a subset of ctx.P.index). The rule
    runs on the first trading day of each month (or every day) at that
    day's close; the book is held until the next review."""
    pos = ctx.P.index.get_indexer(days)
    if (pos < 0).any():
        raise ValueError("days must be on the panel's calendar")
    ms = month_start_mask(ctx.P.index)
    out, cur = {}, None
    for d, t in zip(days, pos):
        if cur is None or freq == "daily" or ms[t]:
            cur = rule(ctx, int(t))
        out[d] = dict(cur)
    return out


def combine(parts: list[tuple[float, dict]]) -> dict:
    """Weighted sum of several {day: book} maps -> one book per day."""
    days = sorted(set().union(*[set(b) for _, b in parts]))
    out = {}
    for d in days:
        acc: Book = {}
        for w, books in parts:
            for s, v in books.get(d, {}).items():
                acc[s] = acc.get(s, 0.0) + w * v
        out[d] = {s: v for s, v in acc.items() if v > 1e-12}
    return out


def crash_exposure(spy: pd.Series, days: pd.DatetimeIndex, sma_n: int,
                   vol_win: int, vol_lookback: int, vol_q: float,
                   risk_mult: float, min_hold: int) -> pd.Series:
    """Crash overlay: risk-off when SPY close < SMA(sma_n) AND SPY's
    vol_win-day realised vol is above its trailing vol_lookback-day
    vol_q quantile — both read at the decision close. Exposure is
    risk_mult while risk-off, 1 otherwise; a switch holds >= min_hold
    days."""
    s = spy.sort_index()
    ma = s.rolling(sma_n, min_periods=sma_n).mean()
    v = s.pct_change().rolling(vol_win, min_periods=vol_win).std()
    q = v.rolling(vol_lookback, min_periods=vol_lookback // 2).quantile(vol_q)
    off = ((s < ma) & (v > q)).reindex(days).fillna(False).values
    out, state, held = [], False, min_hold
    for o in off:
        held += 1
        if held >= min_hold and bool(o) != state:
            state, held = bool(o), 0
        out.append(risk_mult if state else 1.0)
    return pd.Series(out, index=days)


def scale_book(books: dict, exposure: pd.Series, cash: str) -> dict:
    """Multiply every non-cash weight by exposure[d]; freed share -> cash."""
    out = {}
    for d, b in books.items():
        e = float(exposure.get(d, 1.0))
        nb = {s: v * e for s, v in b.items() if s != cash}
        nb[cash] = b.get(cash, 0.0) + (1.0 - e) * sum(
            v for s, v in b.items() if s != cash)
        out[d] = {s: v for s, v in nb.items() if v > 1e-12}
    return out


def financing_drag(books: dict, days: pd.DatetimeIndex, cash_ret: pd.Series,
                   spread: float, cash: str) -> pd.Series:
    """Daily financing cost of the borrowed share: a day-t return is
    earned on the book decided at t-1 (filled at t's open), so the
    borrowed fraction max(0, gross - 1) of the PREVIOUS decision pays
    cash_ret[t] + spread/252."""
    out = pd.Series(0.0, index=days)
    prev = None
    for d in days:
        if prev is not None and prev in books:
            g = sum(v for s, v in books[prev].items() if s != cash)
            b = max(0.0, g - 1.0)
            if b > 0:
                cr = cash_ret.get(d, 0.0)
                out[d] = b * ((cr if np.isfinite(cr) else 0.0) + spread / 252)
        prev = d
    return out


# ---------------------------------------------------------------------------
# single-name signals (S&P panel)


def residual_vol(close: pd.DataFrame, mkt: pd.Series, win: int,
                 min_obs: int | None = None) -> pd.DataFrame:
    """Rolling idiosyncratic vol vs the market (CAPM residual std),
    vectorised: var(r) - cov(r, m)^2 / var(m) over the last `win` days."""
    r = close.pct_change()
    m = mkt.reindex(close.index).pct_change()
    mp = min_obs or int(0.8 * win)
    ok = r.notna()
    mm = pd.DataFrame(np.where(ok, m.values[:, None], np.nan),
                      index=r.index, columns=r.columns)
    roll = lambda x: x.rolling(win, min_periods=mp).mean()  # noqa: E731
    er, em = roll(r), roll(mm)
    cov = roll(r * mm) - er * em
    vr = roll(r * r) - er * er
    vm = roll(mm * mm) - em * em
    res = vr - cov * cov / vm.where(vm > 0)
    return np.sqrt(res.clip(lower=0.0)) * math.sqrt(252)


def sue_events(facts: pd.DataFrame, max_q_days: int = 130,
               min_hist: int = 4, hist: int = 8) -> pd.DataFrame:
    """Standardised unexpected earnings from quarterly XBRL net income
    (seasonal random walk, Bernard & Thomas 1989 / Foster 1977):

        SUE_q = (NI_q - NI_{q-4}) / std(last `hist` such differences)

    Event date = the FIRST `filed` date of quarter q (the original
    10-Q/10-K). Every input to an event is a version filed strictly
    before or on that date; the std uses only earlier events' diffs and
    needs >= min_hist of them. -> rows (filed, end, sue)."""
    if facts.empty:
        return pd.DataFrame(columns=["filed", "end", "sue"])
    f = facts.dropna(subset=["start", "end", "filed", "val"]).copy()
    f = f[(f["end"] - f["start"]).dt.days <= max_q_days]
    if f.empty:
        return pd.DataFrame(columns=["filed", "end", "sue"])
    f = f.sort_values(["end", "filed"])
    first = f.groupby("end", as_index=False).first()[["end", "filed", "val"]]
    rows, diffs = [], []
    for e, fd, v in first.itertuples(index=False):
        prior = f[(f["filed"] <= fd)
                  & ((e - f["end"]).dt.days.between(345, 385))]
        if prior.empty:
            continue
        pv = prior.sort_values(["end", "filed"]).iloc[-1]["val"]
        d = float(v) - float(pv)
        past = diffs[-hist:]
        if len(past) >= min_hist:
            sd = float(np.std(past, ddof=1))
            if sd > 0:
                rows.append({"filed": fd, "end": e, "sue": d / sd})
        diffs.append(d)
    return pd.DataFrame(rows, columns=["filed", "end", "sue"])


def sue_panel(events: dict[str, pd.DataFrame], days: pd.DatetimeIndex,
              max_age_days: int) -> pd.DataFrame:
    """Day x symbol SUE of the latest event filed strictly before the day,
    NaN once it is older than max_age_days TRADING days."""
    d64 = days.values.astype("datetime64[D]")
    cols = {}
    for s, ev in events.items():
        if ev.empty:
            continue
        ev = ev.sort_values("filed")
        filed = ev["filed"].values.astype("datetime64[D]")
        pos = np.searchsorted(filed, d64, side="left") - 1     # filed < D
        val = np.where(pos >= 0, ev["sue"].values[np.clip(pos, 0, None)], np.nan)
        fpos = np.searchsorted(d64, filed, side="right")        # first day > filed
        age = np.arange(len(days)) - np.where(
            pos >= 0, fpos[np.clip(pos, 0, None)], 0)
        cols[s] = np.where((pos >= 0) & (age < max_age_days), val, np.nan)
    return pd.DataFrame(cols, index=days)


# ---------------------------------------------------------------------------
# metrics


def perf(daily: pd.Series, rf: pd.Series | None = None,
         book_usd: float = 100_000.0) -> dict:
    """AR, Sharpe (excess of rf when given), MDD and $ P&L on the book."""
    r = daily.dropna()
    if r.empty:
        return {"ar": 0.0, "sharpe": 0.0, "sharpe_rf0": 0.0, "mdd": 0.0,
                "total": 0.0, "pnl_usd": 0.0, "n_days": 0,
                "start": None, "end": None}
    eq = (1 + r).cumprod()
    x = r - (rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0)
    sd = lambda s: float(s.mean() / s.std() * math.sqrt(252)) \
        if s.std() > 0 else 0.0  # noqa: E731
    return {"ar": float(eq.iloc[-1] ** (252 / len(r)) - 1),
            "sharpe": sd(x), "sharpe_rf0": sd(r),
            "mdd": float((eq / eq.cummax() - 1).min()),
            "total": float(eq.iloc[-1] - 1),
            "pnl_usd": float(book_usd * (eq.iloc[-1] - 1)),
            "n_days": int(len(r)),
            "start": str(r.index[0].date()), "end": str(r.index[-1].date())}
