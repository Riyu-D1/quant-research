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
TTM = sum of the last 4 quarterly facts visible at D (a rolling sum over
facts sorted by period end then filed — documented approximation when
filings are late/amended).
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


def parse_companyfacts(payload: dict) -> pd.DataFrame:
    """companyfacts JSON -> tidy rows (tag, end, start, filed, val, fp, form).

    Only us-gaap facts for the declared tag set, USD/share units only."""
    gaap = payload.get("facts", {}).get("us-gaap", {})
    rows = []
    for group, tags in (("book", BOOK_TAGS), ("ni", NI_TAGS),
                        ("rev", REV_TAGS)):
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
    """Latest `val` with filed < D per day (point facts: book equity)."""
    d64 = days.values.astype("datetime64[D]")
    f = facts.sort_values(["end", "filed"])
    # dedupe same period end -> keep latest FILED version
    f = f[~f.duplicated("end", keep="last")]
    filed = f["filed"].values.astype("datetime64[D]")
    vals = f["val"].values.astype(float)
    pos = np.searchsorted(filed, d64, side="left") - 1     # filed < D
    out = np.where(pos >= 0, vals[np.clip(pos, 0, len(vals) - 1)], np.nan)
    return pd.Series(out).ffill().values


def _ttm(facts: pd.DataFrame, days: pd.DatetimeIndex) -> np.ndarray:
    """Trailing-twelve-month sum of quarterly facts visible at day D.

    Quarterly = period length <= MAX_QUARTER_DAYS. At each day D the TTM is
    the sum of the 4 quarterly facts with the largest period-end among those
    filed < D."""
    if facts.empty:
        return np.full(len(days), np.nan)
    f = facts.copy()
    span = (f["end"] - f["start"]).dt.days
    f = f[span <= MAX_QUARTER_DAYS].sort_values(["end", "filed"])
    f = f[~f.duplicated("end", keep="last")]
    if f.empty:
        return np.full(len(days), np.nan)
    d64 = days.values.astype("datetime64[D]")
    filed = f["filed"].values.astype("datetime64[D]")
    ends = f["end"].values.astype("datetime64[D]")
    vals = f["val"].values.astype(float)
    out = np.full(len(days), np.nan)
    order = np.argsort(ends)                 # oldest -> newest period end
    for i, d in enumerate(d64):
        vis = order[(filed[order] < d)]
        if len(vis) >= 1:
            tail = vals[vis][-4:]
            if len(tail) == 4:
                out[i] = tail.sum()
            elif len(tail) >= 2:
                # partial window: scale only when at least 2 quarters known
                out[i] = tail.sum() * 4.0 / len(tail)
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
