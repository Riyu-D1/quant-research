"""Build the live signal snapshots the paper bot consumes.

    .venv/bin/python scripts/live_snapshot.py

Outputs (tracked, committed by the refresh_snapshot CI job):
  state/live/vq_snapshot.parquet    symbol, score, asof — latest XBRL
                                  value/quality composite (mean rank of
                                  book-to-market, earnings-to-price, ROE)
  state/live/news_snapshot.parquet  symbol, score, asof — latest daily
                                  FinBERT sentiment, when the Kaggle
                                  results branch has landed

Inputs resolved from workflow artifacts (GH_TOKEN) or local data/ when
present: phase4-inputs (fundamentals.parquet), phase3-events
(shares_out.parquet, cik_map.parquet), phase4-store manifest for prices.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from quant_research import fundamentals as F
from quant_research import micro as MC

OUT = REPO / "state" / "live"


def log(msg: str) -> None:
    import time
    print(f"[snapshot {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def latest_vq(fund: pd.DataFrame, shares: pd.DataFrame,
              closes: pd.DataFrame, day: pd.Timestamp) -> pd.Series:
    """Per-symbol vq score at `day` = mean rank of vq_bm/ep/roe."""
    rows = {}
    sh = shares.rename(columns={"val": "shares"}) if "val" in shares.columns \
        else shares
    days = pd.DatetimeIndex([day])
    for sym, f in fund.groupby("symbol"):
        px = closes[sym].dropna() if sym in closes.columns else pd.Series()
        if px.empty:
            continue
        s_rows = sh[sh["symbol"] == sym]
        v = F.vq_features(f, s_rows, np.array([float(px.iloc[-1])]), days)
        rows[sym] = {k: v[k][0] for k in v}
    df = pd.DataFrame(rows).T
    return df.rank(pct=True).mean(axis=1).rename("score")


def main() -> None:
    provider = REPO / "data" / "qlib" / "us_alpaca"
    cal = pd.DatetimeIndex(
        pd.read_csv(provider / "calendars/day.txt", header=None)[0])
    long_cl = MC.load_closes(provider, "sp500")   # (datetime, instrument)
    closes = long_cl["close"].unstack("instrument")  # wide date x symbol
    day = cal.max()
    log(f"snapshot day: {day.date()}")

    fund = pd.read_parquet(REPO / "data/events/fundamentals.parquet")
    shares = pd.read_parquet(REPO / "data/events/shares_out.parquet")
    from quant_research.news import canon
    fund["symbol"] = fund["symbol"].map(canon)
    shares["symbol"] = shares["symbol"].map(canon)
    OUT.mkdir(parents=True, exist_ok=True)

    vq = latest_vq(fund, shares, closes, day).dropna()
    pd.DataFrame({"symbol": vq.index, "score": vq.values,
                  "asof": day}).to_parquet(OUT / "vq_snapshot.parquet",
                                          index=False)
    log(f"vq snapshot: {len(vq)} symbols -> {OUT/'vq_snapshot.parquet'}")

    news_path = REPO / "results" / "news_sentiment" / "daily_sentiment.parquet"
    if news_path.exists():
        d = pd.read_parquet(news_path)
        latest = (d.sort_values("pub_date").groupby("symbol")
                  .tail(1).set_index("symbol")["mean_score"])
        pd.DataFrame({"symbol": latest.index, "score": latest.values,
                      "asof": d["pub_date"].max()}).to_parquet(
            OUT / "news_snapshot.parquet", index=False)
        log(f"news snapshot: {len(latest)} symbols "
            f"(asof {d['pub_date'].max().date()})")
    else:
        log("no news sentiment yet — skipping news snapshot")

    meta = {"built": str(pd.Timestamp.now()), "day": str(day.date()),
            "vq_symbols": len(vq)}
    (OUT / "snapshot_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
