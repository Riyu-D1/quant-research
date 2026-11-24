"""Phase-4 value/quality factors from SEC XBRL companyfacts (free, no key).

Same endpoint already used for shares-outstanding in Phase 3
(data.sec.gov/api/xbrl/companyfacts/CIK{n}.json); this pulls a tiny tag set
and keeps every fact's `filed` date so the features are point-in-time —
a value at day D is the latest fact whose `filed` is strictly before D.

Factors (per symbol, per day):
    vq_bm   book equity / market cap          (value: B/M)
    vq_ep   net income TTM / market cap       (value: E/P)
    vq_roe  net income TTM / book equity      (quality: ROE)

market cap = latest-filed shares outstanding (Phase-3 shares_out table)
            x close. Quarterly tags are selected by period length (~90d);
TTM = sum of the last 4 quarterly facts visible at D.

Versioning is point-in-time per period end: at day D each period end
contributes its latest `filed` < D version. (Earlier code deduped on the
whole frame first, so a post-D amendment both overwrote the visible
value AND made the period invisible until the amendment's filed date —
the phase-8 truncation audit caught it: 7,361 versioned cells.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# first tag found wins (XBRL tag names drifted across taxonomies)
BOOK_TAGS = ["StockholdersEquity",
             "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
             "CommonStockholdersEquity"]
NI_TAGS = ["NetIncomeLoss", "ProfitLoss"]
REV_TAGS = ["RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues", "SalesRevenueNet"]
MAX_QUARTER_DAYS = 130          # ~90d periods; tolerates 53-week years
VQ_GROUPS = (("book", BOOK_TAGS), ("ni", NI_TAGS), ("rev", REV_TAGS))
# phase 11 value/quality tilt (EBIT/TEV, gross profitability, ROA)
QV_GROUPS = (("ebit", ["OperatingIncomeLoss"]),
             ("assets", ["Assets"]),
             ("cash", ["CashAndCashEquivalentsAtCarryingValue",
                       "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]),
             ("debt", ["LongTermDebt", "LongTermDebtNoncurrent"]),
             ("gp", ["GrossProfit"]),
             ("cogs", ["CostOfRevenue", "CostOfGoodsAndServicesSold"]))


def parse_companyfacts(payload: dict, groups=VQ_GROUPS) -> pd.DataFrame:
    """companyfacts JSON -> tidy rows (tag, end, start, filed, val, fp, form).

    Only us-gaap facts for the declared tag set, USD/share units only."""
    gaap = payload.get("facts", {}).get("us-gaap", {})
    rows = []
    for group, tags in groups:
        for tag in tags:
            fact = gaap.get(tag)
            if not fact:
                continue
            for unit, vals in fact.get("units", {}).items():
                for v in vals:
                    rows.append({"group": group, "tag": tag, "unit": unit,
                                 "start": v.get("start"), "end": v.get("end"),
                                 "filed": v.get("filed"), "fy": v.get("fy"),
                                 "fp": v.get("fp"), "form": v.get("form"),
                                 "val": v.get("val")})
            break          # first available tag in the preference list
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    return df.dropna(subset=["end", "filed", "val"])


def _latest_point(facts: pd.DataFrame, days: pd.DatetimeIndex) -> np.ndarray:
    """Latest `val` visible at day D (point facts: book equity).

    PIT per period end: each `end` contributes its latest `filed` < D
    version; the score sees the most recent end with a visible version."""
    d64 = days.values.astype("datetime64[D]")
    out = np.full(len(days), np.nan)
    for _end, g in facts.sort_values(["end", "filed"]).groupby("end"):
        filed = g["filed"].values.astype("datetime64[D]")
        vals = g["val"].values.astype(float)
        pos = np.searchsorted(filed, d64, side="left") - 1   # filed < D
        v = np.where(pos >= 0,
                     vals[np.clip(pos, 0, len(vals) - 1)], np.nan)
        out = np.where(pos >= 0, v, out)      # ascending end: later wins
    return out


def _ttm(facts: pd.DataFrame, days: pd.DatetimeIndex) -> np.ndarray:
    """Trailing-twelve-month sum of quarterly facts visible at day D.

    Quarterly = period length <= MAX_QUARTER_DAYS. Each period end
    contributes its latest `filed` < D version (PIT per end); the TTM is
    the sum of the 4 most recent ends with a visible version."""
    if facts.empty:
        return np.full(len(days), np.nan)
    f = facts.copy()
    span = (f["end"] - f["start"]).dt.days
    f = f[span <= MAX_QUARTER_DAYS].sort_values(["end", "filed"])
    if f.empty:
        return np.full(len(days), np.nan)
    d64 = days.values.astype("datetime64[D]")
    cols = []
    for _end, g in f.groupby("end"):          # ascending end order
        filed = g["filed"].values.astype("datetime64[D]")
        vals = g["val"].values.astype(float)
        pos = np.searchsorted(filed, d64, side="left") - 1
        cols.append(np.where(pos >= 0,
                             vals[np.clip(pos, 0, len(vals) - 1)], np.nan))
    M = np.array(cols).T                       # day x quarter-end
    out = np.full(len(days), np.nan)
    for i in range(len(d64)):
        v = M[i][np.isfinite(M[i])]
        if len(v) >= 4:
            out[i] = v[-4:].sum()
        elif len(v) >= 2:
            # partial window: scale only when at least 2 quarters known
            out[i] = v.sum() * 4.0 / len(v)
    return out


def vq_features(facts: pd.DataFrame, shares: pd.DataFrame | None,
                close: np.ndarray, days: pd.DatetimeIndex) -> dict:
    """One symbol -> {vq_bm, vq_ep, vq_roe} arrays on `days`.

    facts: parse_companyfacts output. shares: (filed, shares) rows (Phase-3
    shares_out schema). close: adjusted close aligned to `days`."""
    out = {c: np.full(len(days), np.nan) for c in ("vq_bm", "vq_ep", "vq_roe")}
    if facts.empty:
        return out
    book = _latest_point(facts[facts["group"] == "book"], days)
    ni = _ttm(facts[facts["group"] == "ni"], days)
    mcap = np.full(len(days), np.nan)
    if shares is not None and not shares.empty:
        sh = shares.dropna(subset=["filed", "shares"]).sort_values("filed")
        filed = sh["filed"].values.astype("datetime64[D]")
        vals = sh["shares"].values.astype(float)
        d64 = days.values.astype("datetime64[D]")
        pos = np.searchsorted(filed, d64, side="left") - 1      # filed < D
        so = np.where(pos >= 0, vals[np.clip(pos, 0, len(vals) - 1)], np.nan)
        so = pd.Series(so).ffill().values
        mcap = so * close
    with np.errstate(divide="ignore", invalid="ignore"):
        out["vq_bm"] = book / mcap
        out["vq_ep"] = ni / mcap
        out["vq_roe"] = ni / np.where(np.abs(book) > 0, book, np.nan)
    return out
