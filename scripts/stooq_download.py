"""Download US daily OHLCV CSVs from stooq (free, no account).

Stooq (stooq.com / stooq.pl) serves per-symbol CSVs at
    https://stooq.com/q/d/l/?s=<ticker>.us&i=d
but gates them behind a JavaScript proof-of-work: find n such that
sha256(challenge + n) starts with `d` zeros, POST it to /__verify, then the
session is cleared for normal requests. We solve the same PoW a browser
would — once per session — then download politely (~1 req/s).

Usage:
    python scripts/stooq_download.py --tickers AAPL MSFT LEH BSC
    python scripts/stooq_download.py --tickers-file data/us_universe.txt
"""
import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://stooq.com/q/d/l/"
VERIFY = "https://stooq.com/__verify"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
CHALLENGE_RE = re.compile(r'const c="([^"]+)",d=(\d+)')


def looks_like_challenge(resp: requests.Response) -> re.Match | None:
    if "requires JavaScript" not in resp.text:
        return None
    return CHALLENGE_RE.search(resp.text)


def solve_pow(session: requests.Session, resp: requests.Response) -> bool:
    m = looks_like_challenge(resp)
    if not m:
        return False
    c, d = m.group(1), int(m.group(2))
    target = "0" * d
    n = 0
    while True:
        h = hashlib.sha256((c + str(n)).encode()).hexdigest()
        if h.startswith(target):
            break
        n += 1
    r = session.post(VERIFY, data={"c": c, "n": n},
                     headers={"Content-Type":
                              "application/x-www-form-urlencoded"})
    return r.ok


def fetch_csv(session: requests.Session, ticker: str) -> str | None:
    """Return CSV text, or None if unavailable/challenge failed."""
    url = f"{BASE}?s={ticker.lower()}.us&i=d"
    resp = session.get(url, timeout=30)
    if looks_like_challenge(resp):
        if not solve_pow(session, resp):
            return None
        resp = session.get(url, timeout=30)
        if looks_like_challenge(resp):
            return None
    text = resp.text.strip()
    if not resp.ok or not text.startswith("Date,"):
        return None
    rows = text.splitlines()
    if len(rows) < 2 or "Exceeded the daily hits limit" in text:
        return None
    return text + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*", default=[])
    p.add_argument("--tickers-file", type=Path, default=None,
                   help="one ticker per line; '#' comments allowed")
    p.add_argument("--output-dir", type=Path, default=Path("data/us_csv"))
    p.add_argument("--delay", type=float, default=1.0)
    args = p.parse_args()

    tickers = list(args.tickers)
    if args.tickers_file:
        tickers += [l.strip() for l in args.tickers_file.read_text().splitlines()
                    if l.strip() and not l.startswith("#")]
    if not tickers:
        sys.exit("no tickers given")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = UA

    ok, failed, skipped = [], [], []
    for i, t in enumerate(tickers):
        t = t.upper()
        out = args.output_dir / f"{t.lower()}.us.csv"
        if out.exists() and out.stat().st_size > 100:
            skipped.append(t)
            continue
        text = fetch_csv(session, t)
        if text is None:
            failed.append(t)
            print(f"  [{i+1}/{len(tickers)}] {t}: FAILED", flush=True)
        else:
            out.write_text(text)
            n = len(text.splitlines()) - 1
            last = text.splitlines()[-1].split(",")[0]
            ok.append(t)
            print(f"  [{i+1}/{len(tickers)}] {t}: {n} rows, last {last}",
                  flush=True)
        time.sleep(args.delay)

    print(f"\ndone: {len(ok)} fetched, {len(skipped)} cached, "
          f"{len(failed)} failed")
    if failed:
        print("failed:", " ".join(failed))


if __name__ == "__main__":
    main()
