"""Phase-4 data fetch — Alpaca news headlines + XBRL fundamentals.

Subcommands (each idempotent; caches under data/news/raw/ + data/events/raw/):
  news          Alpaca /v1beta1/news, monthly pages -> data/news/headlines.parquet
                (needs ALPACA_API_KEY + ALPACA_API_SECRET env; paper keys,
                never printed)
  fundamentals  companyfacts XBRL book/NI/revenue per universe CIK ->
                data/events/fundamentals.parquet (no key; EDGAR polite ~8rps)

Outputs land in gitignored data/; the phase4 workflow ships them as
artifacts. A manifest tracks processed ids so reruns are incremental.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quant_research import edgar as E
from quant_research import fundamentals as F
from quant_research import news as NW

REPO = Path(__file__).resolve().parents[1]
NEWS_DIR = REPO / "data" / "news"
EVENTS_DIR = REPO / "data" / "events"
RAW = EVENTS_DIR / "raw"


def log(msg: str) -> None:
    print(f"[phase4-fetch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmd_news(args) -> None:
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET") or os.environ.get(
        "ALPACA_SECRET_KEY")
    if not key or not secret:
        sys.exit("ALPACA_API_KEY / ALPACA_API_SECRET env vars missing — "
                 "the news fetch needs the paper keys (GitHub Environment "
                 "'ALPACA_API_KEY').")
    ex = NW.fetch_news(key, secret, args.start, args.end,
                       NEWS_DIR / "raw", NEWS_DIR / "headlines.parquet",
                       rps=args.rps)
    log(f"news rows: {len(ex)}")


def cmd_fundamentals(ed: E.Edgar, args) -> None:
    mpath = EVENTS_DIR / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    done = set(manifest.get("fundamentals_ciks", []))
    cmap = pd.read_parquet(EVENTS_DIR / "cik_map.parquet")
    RAW.mkdir(parents=True, exist_ok=True)
    for r in cmap.dropna(subset=["cik"]).itertuples():
        cik = int(r.cik)
        if cik in done:
            continue
        cache = RAW / f"fund_{cik}.parquet"
        try:
            d = ed.get(f"https://data.sec.gov/api/xbrl/companyfacts/"
                       f"CIK{cik:010d}.json").json()
            df = F.parse_companyfacts(d)
            df["symbol"] = r.symbol
            df["cik"] = cik
            df.to_parquet(cache)
            done.add(cik)
        except Exception as e:  # noqa: BLE001
            log(f"fundamentals {r.symbol} ({cik}): {e}")
        if len(done) % 100 == 0:
            manifest["fundamentals_ciks"] = sorted(done)
            mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True))
            log(f"fundamentals: {len(done)} issuers")
    manifest["fundamentals_ciks"] = sorted(done)
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    frames = [pd.read_parquet(p) for p in sorted(RAW.glob("fund_*.parquet"))]
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(
            EVENTS_DIR / "fundamentals.parquet", index=False)
        log(f"fundamentals: {sum(len(f) for f in frames)} rows -> "
            f"{EVENTS_DIR/'fundamentals.parquet'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["news", "fundamentals", "all"])
    p.add_argument("--start", default="2020-09-01",
                   help="news fetch start (default ~4mo before eval start)")
    p.add_argument("--end", default="",
                   help="news fetch end (blank = yesterday)")
    p.add_argument("--rps", type=float, default=3.0,
                   help="Alpaca requests/sec (free tier 200/min)")
    args = p.parse_args()
    end = args.end or str(pd.Timestamp.now().date() - pd.Timedelta(days=1))
    args.end = end
    if args.cmd in ("news", "all"):
        cmd_news(args)
    if args.cmd in ("fundamentals", "all"):
        cmd_fundamentals(E.Edgar(rps=8.0), args)
    log("done")


if __name__ == "__main__":
    main()
