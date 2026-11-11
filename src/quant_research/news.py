"""Phase-4 news/sentiment signal from free sources.

Fetch: Alpaca Market Data news API (/v1beta1/news) with the existing paper
keys — same credentials + GitHub Environment as the us-alpaca-data workflow.
Score: ProsusAI/finbert (small pretrained BERT) on a Kaggle GPU.

Honesty contract: a headline is keyed on its publish timestamp and becomes
visible on the first trading day STRICTLY AFTER its publish calendar date —
same convention as the Phase-3 filings (overnight/pre-market alpha is
deliberately discarded, which is conservative and documented).

Signal: per (symbol, day) the exponentially-weighted mean of article scores
(pos - neg probability) with a 5-trading-day halflife, plus the count of
articles still contributing. A symbol with no visible news gets NaN
(neutral rank downstream), not 0.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
PAGE_LIMIT = 50                     # API max
FINBERT = "ProsusAI/finbert"


def log(msg: str) -> None:
    print(f"[news {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def canon(sym: str) -> str:
    return str(sym).upper().strip().replace(".", "-").replace("_", "-") \
        .replace("/", "-")


# ---------------------------------------------------------------------------
# Alpaca fetch — paginated monthly pulls, cached per month for resume


def _month_windows(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1)
    while cur <= end:
        nxt = cur + pd.DateOffset(months=1)
        lo, hi = max(cur, start), min(nxt - pd.Timedelta(microseconds=1), end)
        yield lo, hi, f"{cur.year}-{cur.month:02d}"
        cur = nxt


def fetch_month(key: str, secret: str, start: pd.Timestamp,
                end: pd.Timestamp, rps: float = 3.0,
                timeout: int = 60) -> list[dict]:
    """All news in [start, end] (no symbol filter — the per-ticker feeds miss
    multi-name articles). Pagination via next_page_token, sort=asc."""
    hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    token, out, gap = None, [], 1.0 / rps
    while True:
        t0 = time.time()
        # RFC-3339 with explicit Z — naive isoformat() gets a 400
        params = {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "limit": PAGE_LIMIT, "sort": "asc",
                  "include_content": "false"}
        if token:
            params["page_token"] = token
        r = requests.get(NEWS_URL, headers=hdr, params=params, timeout=timeout)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2.0)
            r = requests.get(NEWS_URL, headers=hdr, params=params,
                             timeout=timeout)
        if not r.ok:
            raise RuntimeError(
                f"news {r.status_code}: {r.text[:300]} "
                f"(params {params | {'page_token': '...'}})")
        d = r.json()
        out.extend(d.get("news", []))
        token = d.get("next_page_token")
        if not token:
            return out
        time.sleep(max(0.0, gap - (time.time() - t0)))


def fetch_news(key: str, secret: str, start: str, end: str,
               raw_dir: Path, out_path: Path, rps: float = 3.0) -> pd.DataFrame:
    """Headline/summary rows per (article, symbol) for the sp500 universe.

    raw_dir caches raw monthly pulls (news_YYYY-MM.parquet); reruns resume.
    Rows: (symbol, published_at, headline, summary, source, article_id).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for lo, hi, ym in _month_windows(pd.Timestamp(start), pd.Timestamp(end)):
        cache = raw_dir / f"news_{ym}.parquet"
        if cache.exists():
            frames.append(pd.read_parquet(cache))
            continue
        arts = fetch_month(key, secret, lo, hi, rps=rps)
        rows = []
        for a in arts:
            syms = [canon(s) for s in (a.get("symbols") or [])]
            rows.append({"article_id": a.get("id"),
                         "published_at": a.get("created_at") or a.get("updated_at"),
                         "headline": a.get("headline") or "",
                         "summary": a.get("summary") or "",
                         "source": a.get("source") or "",
                         "symbols": syms})
        df = pd.DataFrame(rows)
        df.to_parquet(cache, index=False)
        frames.append(df)
        log(f"news {ym}: {len(df)} articles")
    if not frames:
        return pd.DataFrame()
    alln = pd.concat(frames, ignore_index=True)
    alln["published_at"] = pd.to_datetime(alln["published_at"], utc=True)
    alln["pub_date"] = alln["published_at"].dt.date
    ex = alln.explode("symbols").rename(columns={"symbols": "symbol"})
    ex = ex[ex["symbol"].notna() & (ex["symbol"] != "")]
    ex.to_parquet(out_path, index=False)
    log(f"news: {len(alln)} articles, {len(ex)} symbol rows -> {out_path}")
    return ex


# ---------------------------------------------------------------------------
# FinBERT scoring (Kaggle GPU entry)


def _text_for_score(df: pd.DataFrame) -> pd.Series:
    return (df["headline"].fillna("") + ". " + df["summary"].fillna("")).str.strip()


def score_finbert(articles: pd.DataFrame, out_path: Path,
                  model: str = FINBERT, batch: int = 64,
                  max_len: int = 128) -> pd.DataFrame:
    """Score unique article texts with ProsusAI/finbert -> score = P(pos)-P(neg).

    Input: the exploded (per-symbol) headline table; dedup is on text so a
    headline covering 3 tickers is scored once. Output parquet carries one
    row per (symbol, pub_date) article occurrence + the score columns.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForSequenceClassification.from_pretrained(model)
    device = 0 if torch.cuda.is_available() else -1
    pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, device=device,
                                      top_k=3, truncation=True,
                                      max_length=max_len)

    df = articles.copy()
    df["text"] = _text_for_score(df)
    df["thash"] = df["text"].map(
        lambda t: hashlib.md5(t.encode()).hexdigest())
    uniq = df.drop_duplicates("thash")[["thash", "text"]].reset_index(drop=True)
    log(f"finbert: {len(df)} rows -> {len(uniq)} unique texts "
        f"(device={device})")
    scores = {}
    for i in range(0, len(uniq), batch):
        chunk = uniq["text"].iloc[i:i + batch].tolist()
        for h, res in zip(uniq["thash"].iloc[i:i + batch],
                          pipe(chunk, batch_size=batch)):
            p = {r["label"].lower(): r["score"] for r in res}
            scores[h] = p.get("positive", 0.0) - p.get("negative", 0.0)
        if (i // batch) % 100 == 0:
            log(f"finbert {i + len(chunk)}/{len(uniq)}")
    df["score"] = df["thash"].map(scores)
    out = df[["article_id", "symbol", "pub_date", "published_at", "source",
              "score"]].dropna(subset=["score"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log(f"finbert: {len(out)} scored rows -> {out_path}")
    return out


def daily_table(scored: pd.DataFrame) -> pd.DataFrame:
    """(symbol, pub_date) -> mean score + article count for that publish
    calendar date. This is what the Kaggle run pushes back."""
    g = (scored.groupby(["symbol", "pub_date"], as_index=False)
               .agg(mean_score=("score", "mean"), n=("score", "size")))
    g["symbol"] = g["symbol"].map(canon)
    g["pub_date"] = pd.to_datetime(g["pub_date"])
    return g.sort_values(["symbol", "pub_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# daily signal: EW mean of visible article scores, 5-trading-day halflife


def daily_signal(daily: pd.DataFrame, store: pd.DataFrame,
                 halflife_td: float = 5.0) -> pd.Series:
    """(datetime, instrument) sentiment series on the store's trading grid.

    daily: output of daily_table — (symbol, pub_date, mean_score, n).
    An article published on calendar date P is visible on the first trading
    day STRICTLY AFTER P; its weight then decays 0.5^(elapsed_td/halflife).
    Per symbol this is the recursion acc[d] = acc[d-1]*decay + S[d], where
    S[d] sums the scores of articles that become visible on day d.
    """
    days = store.index.get_level_values("datetime").unique().sort_values()
    d64 = days.values.astype("datetime64[D]")
    decay = 0.5 ** (1.0 / halflife_td)
    grid = (store.index.swaplevel(0, 1)
            if store.index.names[0] == "instrument" else store.index)

    sig_parts = {}
    by_sym = {s: g.sort_values("pub_date")
              for s, g in daily.groupby("symbol")}
    inst = store.index.get_level_values("instrument").unique()
    for sym in inst:
        g = by_sym.get(sym)
        if g is None:
            continue
        pub = pd.to_datetime(g["pub_date"]).values.astype("datetime64[D]")
        # first trading day strictly after the publish date
        vis = np.searchsorted(d64, pub, side="right")
        ok = vis < len(d64)
        # weighted sum at visibility day: score * count
        s_acc = np.zeros(len(days))
        np.add.at(s_acc, vis[ok],
                  g["mean_score"].values[ok] * g["n"].values[ok])
        n_acc = np.zeros(len(days))
        np.add.at(n_acc, vis[ok], g["n"].values[ok])
        acc, cnt = 0.0, 0.0
        sig = np.full(len(days), np.nan)
        for d in range(len(days)):
            acc = acc * decay + s_acc[d]
            cnt = cnt * decay + n_acc[d]
            if cnt > 1e-9:
                sig[d] = acc / cnt
        sig_parts[sym] = sig
    if not sig_parts:
        return pd.Series(np.nan, index=grid, name="news")
    df = pd.DataFrame(sig_parts, index=days)
    s = df.stack()
    s.index.names = ["datetime", "instrument"]
    return s.rename("news").reindex(grid)


# ---------------------------------------------------------------------------
# helpers for the Kaggle entry point


def gh_artifact_download(repo: str, name: str, token: str, dest: Path) -> Path:
    """Latest non-expired workflow artifact `name` -> extracted under dest."""
    api = f"https://api.github.com/repos/{repo}/actions/artifacts"
    hdr = {"Authorization": f"Bearer {token}",
           "Accept": "application/vnd.github+json"}
    r = requests.get(api, headers=hdr, params={"name": name, "per_page": 100},
                     timeout=60)
    r.raise_for_status()
    arts = [a for a in r.json()["artifacts"] if not a["expired"]]
    if not arts:
        raise RuntimeError(f"no '{name}' artifact on {repo}")
    url = max(arts, key=lambda a: a["created_at"])["archive_download_url"]
    import io
    import zipfile
    z = requests.get(url, headers=hdr, timeout=900)
    z.raise_for_status()
    dest.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(z.content)).extractall(dest)
    log(f"artifact {name} -> {dest}")
    return dest
