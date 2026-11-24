"""Point-in-time S&P 500 membership for the backtest (phase 11 fix).

Before phase 11 the B50 walk-forward ranked every name that was *ever* an
S&P 500 member in 2016-2026, so a 2024 decision could pick a name that
only joined later (SMCI 2024-03-18, APP 2025-09-22, CVNA 2025-12-22).
Here a name is pickable on signal day d only if it was a member on d.

Sources (free, no key):
  * fja05680/sp500 `sp500_ticker_start_end.csv`, pinned to a commit
    (one row per membership spell: ticker, start_date, end_date; an empty
    end_date means still a member). end_date is the effective removal
    date, so a spell covers start <= d < end.
  * Wikipedia "List of S&P 500 companies" revision history, used to
    carry the pinned file forward to the sample end: the constituents
    table is read from the last revision of each calendar day after the
    pin, and a membership flip counts only once it holds for `persist`
    consecutive daily snapshots (filters vandalism / edit wars). New
    names start on their "Date added" when that falls inside the gap.

Symbols are canonical qlib form (upper case, '-' class separator).
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from quant_research import events as ev

FJA_URL = ("https://raw.githubusercontent.com/fja05680/sp500/{sha}/"
           "sp500_ticker_start_end.csv")
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_TITLE = "List of S&P 500 companies"
UA = "quant-research/1.0 (academic research; contact: riyanshdiwan@gmail.com)"


def fja_windows(sha: str, session: requests.Session) -> pd.DataFrame:
    """Pinned fja05680 spells -> (symbol, start, end[NaT = open])."""
    r = session.get(FJA_URL.format(sha=sha), timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return pd.DataFrame({"symbol": df["ticker"].map(ev.canon),
                         "start": pd.to_datetime(df["start_date"]),
                         "end": pd.to_datetime(df["end_date"])})


def wiki_daily_revisions(start: str, end: str,
                         session: requests.Session) -> dict[pd.Timestamp, int]:
    """{calendar day: id of the page's last revision at that day's end}
    for every day in [start, end] (UTC)."""
    params = {"action": "query", "prop": "revisions", "titles": WIKI_TITLE,
              "rvprop": "ids|timestamp", "rvlimit": "max",
              "rvstart": f"{end}T23:59:59Z",
              "rvend": f"{pd.Timestamp(start) - pd.Timedelta(days=90):%Y-%m-%d}"
                       "T00:00:00Z",
              "format": "json", "formatversion": "2"}
    revs, cont = [], {}
    while True:
        d = session.get(WIKI_API, params={**params, **cont}, timeout=60).json()
        revs += d["query"]["pages"][0].get("revisions", [])
        if "continue" not in d:
            break
        cont = d["continue"]
    s = pd.Series({pd.Timestamp(r["timestamp"]).tz_localize(None): r["revid"]
                   for r in revs}).sort_index()
    out = {}
    for day in pd.date_range(start, end, freq="D"):
        upto = s.loc[:day + pd.Timedelta(hours=23, minutes=59, seconds=59)]
        if len(upto):
            out[day] = int(upto.iloc[-1])
    return out


def wiki_constituents(revid: int, session: requests.Session) -> pd.DataFrame:
    """The constituents table (symbol, date_added) of one revision."""
    d = session.get(WIKI_API, params={"action": "parse", "oldid": revid,
                                      "prop": "text", "format": "json",
                                      "formatversion": "2"}, timeout=60).json()
    t = pd.read_html(io.StringIO(d["parse"]["text"]), match="Symbol")[0]
    return pd.DataFrame({"symbol": t["Symbol"].map(ev.canon),
                         "date_added": pd.to_datetime(t.get("Date added"),
                                                      errors="coerce")})


def debounce(on: np.ndarray, persist: int) -> np.ndarray:
    """Accept a state flip only once the new state holds for `persist`
    consecutive observations (or to the end of the series)."""
    out = np.empty_like(on)
    state = bool(on[0])
    for i in range(len(on)):
        if bool(on[i]) != state:
            run = on[i:i + persist]
            if (run == on[i]).all():
                state = bool(on[i])
        out[i] = state
    return out


def patch_windows(win: pd.DataFrame, snaps: dict[pd.Timestamp, pd.DataFrame],
                  persist: int = 5) -> pd.DataFrame:
    """Carry the pinned spells forward with daily Wikipedia snapshots.

    The state before the first snapshot is the pinned file's open set.
    A name leaving on day d gets end = d (exclusive). A name joining on
    day d opens a new spell from its Date added (if inside the gap) or d.
    """
    days = sorted(snaps)
    base = set(win.loc[win["end"].isna(), "symbol"])
    names = sorted(base.union(*[set(s["symbol"]) for s in snaps.values()]))
    added = {}
    for d in days:
        for r in snaps[d].dropna(subset=["date_added"]).itertuples():
            added[r.symbol] = r.date_added
    on = pd.DataFrame(False, index=days, columns=names)
    for d in days:
        on.loc[d, list(set(snaps[d]["symbol"]))] = True
    out = win.copy()
    new_rows = []
    for s in names:
        series = np.concatenate([[s in base], on[s].values])
        state = debounce(series, persist)[1:]
        prev, start = s in base, None
        for d, now in zip(days, state):
            if prev and not now:
                if start is None:              # closes the pinned open spell
                    out.loc[(out["symbol"] == s) & out["end"].isna(), "end"] = d
                else:
                    new_rows.append({"symbol": s, "start": start, "end": d})
                    start = None
            elif now and not prev:
                a = added.get(s)
                start = a if a is not None and days[0] <= a <= d else d
            prev = now
        if start is not None:
            new_rows.append({"symbol": s, "start": start, "end": pd.NaT})
    if new_rows:
        out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)
    return out.sort_values(["symbol", "start"]).reset_index(drop=True)


def member_mask(win: pd.DataFrame, days: pd.DatetimeIndex,
                cols: pd.Index) -> pd.DataFrame:
    """days x cols: True on days the name is a member (start <= d < end;
    an open spell runs through the last day)."""
    m = pd.DataFrame(False, index=days, columns=cols)
    for r in win.itertuples():
        if r.symbol not in m.columns:
            continue
        ok = days >= r.start
        if pd.notna(r.end):
            ok &= days < r.end
        m.loc[ok, r.symbol] = True
    return m


def non_member_picks(picks: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    """Rows of (date, symbol) picks that were not members on `date`."""
    days = pd.DatetimeIndex(sorted(picks["date"].unique()))
    m = member_mask(win, days, pd.Index(sorted(picks["symbol"].unique())))
    ok = [bool(m.at[d, s]) for d, s in zip(picks["date"], picks["symbol"])]
    return picks[~np.array(ok, dtype=bool)]
