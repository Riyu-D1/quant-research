"""Build a US qlib dump from Alpaca Market Data (v2 stock bars).

Subcommands:
  universe  build a point-in-time S&P 500 membership table -> CSV
            (symbol, in_date, out_date, sector, source).
  fetch     download daily bars via GET /v2/stocks/bars (multi-symbol) for the
            universe + a delisted watchlist + SPY (benchmark). Writes one CSV
            per symbol: date,open,high,low,close,volume,close_raw where OHLCV
            are adjustment=all and close_raw is the unadjusted close.
            Requires env ALPACA_API_KEY / ALPACA_API_SECRET — never logged.
  finalize  run after scripts/us_data_to_qlib.py: rewrites factor.day.bin as
            adjclose/rawclose (community-dump convention: raw = close/factor)
            and instruments/<universe>.txt with the PIT membership windows.

Alpaca notes:
  * feed=sip works on the free plan for HISTORICAL data as long as `end` is
    older than ~15 min; we always request through yesterday.
  * SIP history starts 2016 — names delisted before then (LEH, BSC, WYE)
    cannot exist in this source.
  * `asof` pins symbol->entity resolution; used for delisted watchlist names
    so a recycled ticker doesn't silently return the wrong company.

  python scripts/alpaca_us_data.py universe --start 2016-01-01 --out build/universe_sp500.csv
  python scripts/alpaca_us_data.py fetch --universe-csv build/universe_sp500.csv \
      --csv-dir data/us_alpaca_csv --start 2016-01-01 --end 2026-11-08
  python scripts/us_data_to_qlib.py --input-dir data/us_alpaca_csv \
      --output-dir data/qlib/us_alpaca --universe sp500
  python scripts/alpaca_us_data.py finalize --qlib-dir data/qlib/us_alpaca \
      --csv-dir data/us_alpaca_csv --universe-csv build/universe_sp500.csv \
      --universe sp500
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
# PIT S&P 500 membership (ticker,start_date,end_date since 1996) + current
# constituents w/ GICS sectors — github.com/fja05680/sp500, free, no key.
SP500_BASE = "https://raw.githubusercontent.com/fja05680/sp500/master/"
SP500_WINDOWS = SP500_BASE + "sp500_ticker_start_end.csv"
SP500_CURRENT = SP500_BASE + "sp500.csv"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# Delisted names used as the coverage gate. `asof` pins the symbol->entity
# lookup to before delisting so recycled tickers don't return a different
# company. Names delisted before Alpaca's 2016 SIP start (LEH, BSC, WYE)
# are expected to come back empty — we still ask, and report it.
DELISTED_WATCH = {
    # sym: (approx delist date, note)
    "LEH":  ("2008-09-15", "Lehman Brothers — pre-2016, expected absent"),
    "BSC":  ("2008-05-30", "Bear Stearns — pre-2016, expected absent"),
    "WYE":  ("2009-10-15", "Wyeth — pre-2016, expected absent"),
    "RSH":  ("2015-02-02", "RadioShack — pre-2016, expected absent"),
    "MON":  ("2018-06-07", "Monsanto -> Bayer"),
    "TWX":  ("2018-06-15", "Time Warner -> AT&T (now WBD)"),
    "CA":   ("2018-11-05", "CA Technologies -> Broadcom"),
    "ESRX": ("2018-12-20", "Express Scripts -> Cigna"),
    "LLL":  ("2019-06-28", "L3 -> L3Harris (LHX)"),
    "CELG": ("2019-11-20", "Celgene -> BMS"),
    "AGN":  ("2020-05-08", "Allergan -> AbbVie"),
    "RTN":  ("2020-04-03", "Raytheon Co -> RTX"),
    "XLNX": ("2022-02-14", "Xilinx -> AMD"),
    "ATVI": ("2023-10-13", "Activision -> Microsoft"),
    "SGEN": ("2023-12-14", "Seagen -> Pfizer"),
    "SIVB": ("2023-03-10", "SVB Financial — FDIC receivership"),
    "FRC":  ("2023-05-01", "First Republic -> JPM"),
}
EXTRA_SYMBOLS = ["SPY"]          # backtest benchmark
MAX_SYMBOLS_PER_REQ = 80         # multi-symbol bars accepts <=100; keep margin
PAGE_LIMIT = 10000               # bars per page (across all symbols)


# ---------------------------------------------------------------- universe
def build_pit_windows(start: str, end: str) -> pd.DataFrame:
    """symbol -> [in_date, out_date] membership windows over [start, end].

    Source: fja05680/sp500 ticker_start_end.csv (1996->, updated for index
    changes) joined to the same repo's current-constituents file for GICS
    sectors. Tickers keep Alpaca's '.' class-share format (BRK.B).
    """
    import io
    win = pd.read_csv(io.StringIO(requests.get(
        SP500_WINDOWS, headers={"User-Agent": UA}, timeout=30).text))
    win["end_date"] = win["end_date"].fillna(end)
    win = win.rename(columns={"ticker": "symbol",
                              "start_date": "in_date", "end_date": "out_date"})
    win["in_date"] = win["in_date"].clip(lower=start)
    win["out_date"] = win["out_date"].clip(upper=end)
    win = win[win["in_date"] <= win["out_date"]]

    cur = pd.read_csv(io.StringIO(requests.get(
        SP500_CURRENT, headers={"User-Agent": UA}, timeout=30).text))
    sectors = cur.set_index("Symbol")[
        ["GICS Sector", "GICS Sub-Industry"]].to_dict("index")
    win["sector"] = win["symbol"].map(
        lambda s: sectors.get(s, {}).get("GICS Sector", ""))
    win["sub_industry"] = win["symbol"].map(
        lambda s: sectors.get(s, {}).get("GICS Sub-Industry", ""))
    win["source"] = "fja05680/sp500"
    return win.sort_values(["symbol", "in_date"]).reset_index(drop=True)


# ------------------------------------------------------------------ fetch
class Alpaca:
    def __init__(self, feed: str):
        key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_API_SECRET")
        if not key or not secret:
            sys.exit("ALPACA_API_KEY / ALPACA_API_SECRET not set in environment")
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": key,
                               "APCA-API-SECRET-KEY": secret, "User-Agent": UA})
        self.requests = 0
        self.feed = self._negotiate_feed(feed)

    def _get(self, params: dict) -> dict:
        for attempt in range(4):
            r = self.s.get(BARS_URL, params=params, timeout=60)
            self.requests += 1
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code in (401, 403):
                body = r.text[:300]
                if "subscription" in body.lower() or "does not permit" in body.lower():
                    raise SubscriptionError(body)
                raise AuthError(f"{r.status_code} {body}")
            r.raise_for_status()
            return r.json()
        raise RateLimitError("still rate-limited after retries")

    def _negotiate_feed(self, feed: str) -> str:
        """One probe request picks a working feed. Never upgrades to paid."""
        if feed != "auto":
            return feed
        probe = {"symbols": "AAPL", "timeframe": "1Day", "limit": 5,
                 "adjustment": "raw", "sort": "asc"}
        try:
            r = self.s.get(BARS_URL, params={**probe, "feed": "sip",
                                             "start": "2019-01-01",
                                             "end": "2019-01-15"}, timeout=60)
            self.requests += 1
            if r.status_code == 200:
                print("[alpaca] feed=sip OK (historical SIP available on this plan)")
                return "sip"
            body = r.text[:300]
            if r.status_code in (401, 403) and "permit" not in body.lower() \
                    and "subscription" not in body.lower():
                raise AuthError(f"{r.status_code} {body}")
            print(f"[alpaca] sip rejected ({r.status_code}: {body}) -> falling back to iex")
        except AuthError:
            raise
        except requests.RequestException as e:
            print(f"[alpaca] sip probe failed ({e}) -> falling back to iex")
        r = self.s.get(BARS_URL, params={**probe, "feed": "iex",
                                         "start": "2019-01-01",
                                         "end": "2019-01-15"}, timeout=60)
        self.requests += 1
        if r.status_code != 200:
            raise AuthError(f"iex probe failed: {r.status_code} {r.text[:300]}")
        print("[alpaca] feed=iex (IEX-only, ~2.5% of volume — coverage gaps expected)")
        return "iex"

    def bars(self, symbols: list[str], start: str, end: str,
             adjustment: str, asof: str | None = None) -> dict[str, list[dict]]:
        """All daily bars for `symbols` in [start, end] -> {SYM: [bar,...]}."""
        out: dict[str, list[dict]] = {}
        token = None
        while True:
            params = {"symbols": ",".join(symbols), "timeframe": "1Day",
                      "start": start, "end": end, "adjustment": adjustment,
                      "feed": self.feed, "limit": PAGE_LIMIT, "sort": "asc"}
            if asof:
                params["asof"] = asof
            if token:
                params["page_token"] = token
            data = self._get(params)
            for sym, rows in data.get("bars", {}).items():
                out.setdefault(sym.upper(), []).extend(rows)
            token = data.get("next_page_token")
            if not token:
                return out
            time.sleep(0.35)


class AuthError(Exception):
    pass


class SubscriptionError(Exception):
    pass


class RateLimitError(Exception):
    pass


def cmd_fetch(args) -> int:
    uni = pd.read_csv(args.universe_csv)
    pit_syms = sorted(uni["symbol"].unique())
    end = args.end or (dt.date.today() - dt.timedelta(days=1)).isoformat()

    # Split symbols by whether they left the index before `end`: ex-members
    # are queried with asof pinned INSIDE their membership window (30d before
    # the exit) — pinning at the exit date itself can already resolve to the
    # recycled ticker (MON showed the post-Monsanto entity that way).
    def minus_30d(d: str) -> str:
        return (dt.date.fromisoformat(d) - dt.timedelta(days=30)).isoformat()

    last_out = uni.groupby("symbol")["out_date"].max()
    first_in = uni.groupby("symbol")["in_date"].min()
    current_syms = sorted(set(last_out[last_out >= end].index) | set(EXTRA_SYMBOLS))
    past_by_asof: dict[str, list[str]] = {}
    for sym, out_d in last_out[last_out < end].items():
        past_by_asof.setdefault(minus_30d(out_d), []).append(sym)
    watch = {s: d for s, (d, _n) in DELISTED_WATCH.items()
             if s not in set(pit_syms)}

    client = Alpaca(feed=args.feed)
    n_past = sum(len(v) for v in past_by_asof.values())
    print(f"[fetch] {len(current_syms)} listed + {n_past} ex-member symbols "
          f"(asof-grouped) + {len(watch)} watch names, "
          f"{args.start}..{end}, feed={client.feed}")

    adj: dict[str, dict] = {}
    raw: dict[str, dict] = {}

    def merge(dst: dict, data: dict) -> None:
        """data = {sym: {date: bar}} — fold date-keyed bars into dst."""
        for sym, by_date in data.items():
            dst.setdefault(sym, {}).update(by_date)

    def fetch_group(syms: list[str], adjustment: str,
                    asof: str | None = None) -> dict:
        """{sym: {date: bar}} for a chunk of symbols (list bars -> date map)."""
        res: dict[str, dict] = {s: {} for s in syms}
        for i in range(0, len(syms), MAX_SYMBOLS_PER_REQ):
            for sym, rows in client.bars(
                    syms[i:i + MAX_SYMBOLS_PER_REQ],
                    args.start, end, adjustment, asof=asof).items():
                res.setdefault(sym, {}).update({b["t"][:10]: b for b in rows})
            time.sleep(0.35)
        return res

    # listed names + SPY
    merge(adj, fetch_group(current_syms, "all"))
    if args.raw_close:
        merge(raw, fetch_group(current_syms, "raw"))
    # ex-members, asof pinned 30d inside the membership window
    for asof_d, syms in sorted(past_by_asof.items()):
        merge(adj, fetch_group(syms, "all", asof=asof_d))
        if args.raw_close:
            merge(raw, fetch_group(syms, "raw", asof=asof_d))
    # delisted watch names not already covered by the universe
    for sym, delist in watch.items():
        try:
            merge(adj, fetch_group([sym], "all", asof=minus_30d(delist)))
            if args.raw_close:
                merge(raw, fetch_group([sym], "raw", asof=minus_30d(delist)))
        except SubscriptionError as e:
            print(f"[fetch] {sym}: feed rejected ({e}) — skipped")

    # Clamp ex-member/watch bars to the entity's last membership day — a dead
    # company cannot trade after delisting, so later bars are the recycled
    # ticker (a different entity that reuses the symbol).
    bar_cap = {s: min(o, end) for s, o in last_out[last_out < end].items()}
    bar_cap.update({s: min(delist, end)
                    for s, (delist, _n) in DELISTED_WATCH.items()})
    for sym, cap in bar_cap.items():
        if sym in adj:
            adj[sym] = {d: b for d, b in adj[sym].items() if d <= cap}
        if sym in raw:
            raw[sym] = {d: b for d, b in raw[sym].items() if d <= cap}

    # Second chance: empty universe symbols retry with asof = 30d after their
    # first membership date (covers mapping gaps like BCR/CCEP/DOW).
    retry = [s for s in last_out.index if not adj.get(s)]
    if retry:
        print(f"[fetch] second-chance asof retry for {len(retry)} empty "
              f"symbols: {retry}")
        for s in retry:
            merge(adj, fetch_group([s], "all", asof=first_in[s]))
            if args.raw_close:
                merge(raw, fetch_group([s], "raw", asof=first_in[s]))
            # re-apply the cap for ex-members
            if s in bar_cap:
                adj[s] = {d: b for d, b in adj.get(s, {}).items()
                          if d <= bar_cap[s]}
                raw[s] = {d: b for d, b in raw.get(s, {}).items()
                          if d <= bar_cap[s]}

    args.csv_dir.mkdir(parents=True, exist_ok=True)
    coverage = {}
    symbols = sorted(set(current_syms) | set(last_out.index))
    all_syms = sorted(set(symbols) | set(watch))
    for sym in all_syms:
        rows = adj.get(sym, {})
        if not rows:
            coverage[sym] = {"bars": 0}
            continue
        df = pd.DataFrame([{"date": d, "open": b["o"], "high": b["h"],
                            "low": b["l"], "close": b["c"], "volume": b["v"]}
                           for d, b in sorted(rows.items())])
        if args.raw_close and sym in raw and raw[sym]:
            df["close_raw"] = df["date"].map(
                {d: b["c"] for d, b in raw[sym].items()})
        fname = sym.lower().replace(".", "-") + ".csv"
        df.to_csv(args.csv_dir / fname, index=False)
        coverage[sym] = {"bars": len(df), "first": df["date"].iloc[0],
                         "last": df["date"].iloc[-1]}

    # gate on the FULL watchlist — names that are also index members (MON,
    # TWX, CELG...) arrive via the asof-grouped universe fetch, not `watch`
    found = [s for s in DELISTED_WATCH if coverage.get(s, {}).get("bars")]
    missing = [s for s in DELISTED_WATCH if not coverage.get(s, {}).get("bars")]
    have = sum(1 for s in symbols if coverage.get(s, {}).get("bars"))
    report = {
        "feed": client.feed, "start": args.start, "end": end,
        "universe_symbols": len(symbols), "with_bars": have,
        "delisted_found": found, "delisted_missing": missing,
        "empty_symbols": [s for s in symbols
                          if not coverage.get(s, {}).get("bars")],
        "coverage": coverage, "requests": client.requests,
    }
    args.coverage_json.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_json.write_text(json.dumps(report, indent=2))
    print(f"[fetch] wrote {have}/{len(symbols)} symbols -> {args.csv_dir}; "
          f"{client.requests} api requests")
    print(f"[coverage] delisted found: {found or 'NONE'}")
    print(f"[coverage] delisted missing: {missing or 'none'}")
    if not found:
        print("[coverage] ZERO delisted names returned data — Alpaca feed "
              "lacks the required coverage; stopping per policy (no paid plan)")
        return 3
    return 0


# --------------------------------------------------------------- finalize
def cmd_finalize(args) -> int:
    """Fix factor bins + write PIT instruments file after conversion."""
    root = args.qlib_dir
    cal = pd.read_csv(root / "calendars/day.txt", header=None)[0].tolist()

    uni = pd.read_csv(args.universe_csv)
    watch_syms = set(DELISTED_WATCH)
    n_fixed, rows = 0, []
    for f in sorted(root.glob("features/*/close.day.bin")):
        sym = f.parent.name
        csv = args.csv_dir / f"{sym}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv, dtype={"date": str}).set_index("date").reindex(cal)
        a = np.fromfile(f, dtype=np.float32)
        start = int(a[0])
        vals = a[1:]
        sl = df.iloc[start:start + len(vals)]
        if "close_raw" in sl and sl["close_raw"].notna().any():
            factor = (sl["close"] / sl["close_raw"]
                      .where(sl["close_raw"] > 0)).ffill().fillna(1.0)
            np.concatenate([np.array([start], np.float32),
                            factor.astype(np.float32).values]
                           ).tofile(f.parent / "factor.day.bin")
            n_fixed += 1
        # keep data-span row for all.txt; PIT membership row for <universe>.txt
        present = df["close"].dropna()
        if len(present):
            rows.append({"symbol": sym.upper(), "in_date": present.index[0],
                         "out_date": present.index[-1]})

    all_df = pd.DataFrame(rows).sort_values("symbol")
    (root / "instruments" / "all.txt").write_text(
        "\n".join(f"{r.symbol}\t{r.in_date}\t{r.out_date}" for r in all_df.itertuples()) + "\n")

    # PIT membership windows for the traded universe; delisted watch names get
    # their data span as the window (they left the index when they died).
    # qlib looks instruments up as lowercased feature-dir names, so '.' class
    # shares (BRK.B -> features/brk-b) must be written in file form.
    have = set(all_df["symbol"])
    span = all_df.set_index("symbol")
    uni = uni.copy()
    uni["sym_file"] = uni["symbol"].str.upper().str.replace(".", "-")
    pit = uni[uni["sym_file"].isin(have)].copy()
    for sym in watch_syms & have:
        if sym not in set(pit["sym_file"]):
            pit.loc[len(pit)] = {"symbol": sym, "sym_file": sym,
                                 "in_date": span.loc[sym, "in_date"],
                                 "out_date": span.loc[sym, "out_date"],
                                 "sector": "", "source": "delisted-watch"}
    pit["in_date"] = pit["in_date"].clip(lower=cal[0])
    pit["out_date"] = pit["out_date"].clip(upper=cal[-1])
    pit = pit[pit["in_date"] <= pit["out_date"]]
    pit = pit.sort_values(["sym_file", "in_date"])
    out = root / "instruments" / f"{args.universe}.txt"
    out.write_text("\n".join(f"{r.sym_file}\t{r.in_date}\t{r.out_date}"
                             for r in pit.itertuples()) + "\n")
    print(f"[finalize] factors rewritten for {n_fixed} symbols; "
          f"instruments: all.txt={len(all_df)} rows, "
          f"{args.universe}.txt={len(pit)} PIT rows / {pit['symbol'].nunique()} syms")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("universe", help="scrape wikipedia -> PIT membership csv")
    u.add_argument("--start", default="2016-01-01")
    u.add_argument("--end", default=None)
    u.add_argument("--out", type=Path, default=Path("build/universe_sp500.csv"))

    f = sub.add_parser("fetch", help="download alpaca daily bars -> csvs")
    f.add_argument("--universe-csv", type=Path, required=True)
    f.add_argument("--csv-dir", type=Path, default=Path("data/us_alpaca_csv"))
    f.add_argument("--start", default="2016-01-01")
    f.add_argument("--end", default=None, help="default: yesterday")
    f.add_argument("--feed", default="auto", choices=["auto", "sip", "iex"])
    f.add_argument("--raw-close", action=argparse.BooleanOptionalAction,
                   default=True, help="second pass for raw closes (factor)")
    f.add_argument("--coverage-json", type=Path,
                   default=Path("results/us_alpaca_coverage.json"))

    z = sub.add_parser("finalize", help="fix factor bins + PIT instruments")
    z.add_argument("--qlib-dir", type=Path, required=True)
    z.add_argument("--csv-dir", type=Path, default=Path("data/us_alpaca_csv"))
    z.add_argument("--universe-csv", type=Path, required=True)
    z.add_argument("--universe", default="sp500")

    args = p.parse_args()
    if args.cmd == "universe":
        end = args.end or dt.date.today().isoformat()
        df = build_pit_windows(args.start, end)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"[universe] {df['symbol'].nunique()} symbols, {len(df)} "
              f"membership rows over {args.start}..{end} -> {args.out}")
        return
    if args.cmd == "fetch":
        try:
            code = cmd_fetch(args)
        except AuthError as e:
            print(f"[fetch] AUTH FAILED: {e} — keys rejected; stopping "
                  "(no retry, no paid plan)")
            code = 2
        except (SubscriptionError, RateLimitError) as e:
            print(f"[fetch] FAILED: {e}")
            code = 4
        sys.exit(code)
    if args.cmd == "finalize":
        sys.exit(cmd_finalize(args))


if __name__ == "__main__":
    main()
