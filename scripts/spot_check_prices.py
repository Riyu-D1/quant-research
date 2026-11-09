"""Spot-check adjusted closes in the Alpaca CSVs against a second source.

Second sources tried (free, no key): Yahoo Finance chart API, then the
Nasdaq public quote API. Compares our adjustment=all close on N shared dates
per symbol; prints a table and exits 0 with a WARN if no source answered —
the check is informational and must not block the data build.

    python scripts/spot_check_prices.py --csv-dir data/us_alpaca_csv \
        --tickers AAPL MSFT MON CELG TWX
"""
import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd
import requests

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
TOL = 0.02          # adjusted closes across vendors drift; 2% is the flag bar
N_DATES = 3


def yahoo_adjclose(sym: str, start: str, end: str) -> pd.Series:
    p1 = int(dt.datetime.fromisoformat(start).timestamp())
    p2 = int(dt.datetime.fromisoformat(end).timestamp()) + 86400
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
           f"&includeAdjustedClose=true")
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"]
    if not res:
        return pd.Series(dtype=float)
    res = res[0]
    ts = pd.to_datetime(res["timestamp"], unit="s").strftime("%Y-%m-%d")
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    return pd.Series(adj, index=ts).dropna()


def nasdaq_close(sym: str, start: str, end: str) -> pd.Series:
    url = (f"https://api.nasdaq.com/api/quote/{sym}/historical"
           f"?assetclass=stocks&fromdate={start}&todate={end}&limit=200")
    h = {"User-Agent": UA, "Accept": "application/json",
         "Accept-Language": "en-US,en;q=0.9"}
    r = requests.get(url, headers=h, timeout=30)
    r.raise_for_status()
    rows = r.json()["data"]["tradesTable"]["rows"]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    # nasdaq returns split-adjusted closes, no dividend adjustment
    return pd.Series(df["close"].str.replace("$", "", regex=False)
                     .str.replace(",", "", regex=False).astype(float)
                     .values, index=df["date"])


def check(sym: str, ours: pd.Series, ours_raw: pd.Series | None) -> list[dict]:
    idx = ours.dropna().index
    # spread probes: earliest, middle, latest of the symbol's range
    dates = [idx[0], idx[len(idx) // 2], idx[-1]][:N_DATES]
    span = f"{dates[0]}:{dates[-1]}"
    try:
        ref = yahoo_adjclose(sym, dates[0], dates[-1])
        kind = "yahoo-adjclose"
    except Exception:  # noqa: BLE001 — any failure falls through to nasdaq
        try:
            ref = nasdaq_close(sym, dates[0], dates[-1])
            kind = "nasdaq-close(split-adj only)"
        except Exception:  # noqa: BLE001 — inconclusive check, don't block
            return [{"symbol": sym, "date": d, "ours": ours.get(d),
                     "ref": None, "diff_pct": None,
                     "note": "no second source reachable"} for d in dates]
    out = []
    for d in dates:
        ours_v = ours.get(d)
        ref_v = ref.get(d)
        if ref_v is None or ours_v is None or pd.isna(ours_v) or pd.isna(ref_v):
            out.append({"symbol": sym, "date": d, "ours": ours_v, "ref": ref_v,
                        "diff_pct": None, "note": f"{kind}: no bar on {d} "
                        f"({span})"})
            continue
        diff = abs(ours_v / ref_v - 1)
        flag = "OK" if diff <= TOL else "MISMATCH"
        note = f"{kind} {flag}"
        out.append({"symbol": sym, "date": d, "ours": round(float(ours_v), 4),
                    "ref": round(float(ref_v), 4),
                    "diff_pct": round(diff * 100, 3), "note": note})
        time.sleep(0.2)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv-dir", type=Path, required=True)
    p.add_argument("--tickers", nargs="+",
                   default=["AAPL", "MSFT", "MON", "TWX", "CELG"])
    args = p.parse_args()

    rows = []
    for sym in args.tickers:
        f = args.csv_dir / f"{sym.lower().replace('.', '-')}.csv"
        if not f.exists():
            rows.append({"symbol": sym, "date": "-", "ours": None, "ref": None,
                         "diff_pct": None, "note": "no CSV (not downloaded)"})
            continue
        df = pd.read_csv(f, dtype={"date": str}).set_index("date")
        rows += check(sym, df["close"], df.get("close_raw"))

    tbl = pd.DataFrame(rows)
    with pd.option_context("display.width", 160):
        print(tbl.to_string(index=False))
    ok = tbl["diff_pct"].dropna()
    n_ok = (ok <= TOL * 100).sum()
    print(f"\nspot-check: {n_ok}/{len(ok)} compared dates within "
          f"{TOL:.0%} of second source")
    if tbl["note"].str.contains("no second source").all():
        print("WARN: second source unreachable — check inconclusive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
