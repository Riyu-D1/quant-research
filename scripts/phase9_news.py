#!/usr/bin/env python3
"""Phase-9 FinBERT headline sentiment — sharded CPU scoring on GitHub Actions.

    python scripts/phase9_news.py prep  --headlines H.parquet --shards K
    python scripts/phase9_news.py score --shard k [--max-minutes M]
    python scripts/phase9_news.py merge

prep   universe-filter the phase4-news headlines (configs/live_universe.txt),
       dedupe on headline text, split unique texts into K hash shards.
score  ProsusAI/finbert, headline only, dynamic int8 quantisation of the
       Linear layers, length-sorted batches (CPU throughput); score =
       P(positive) - P(negative). A shard that hits --max-minutes writes
       what it scored and says so in its meta file (coverage is reported,
       never silently dropped).
merge  join scores back to (symbol, pub_date) rows -> news.daily_table ->
       results/news_sentiment/daily_sentiment.parquet (+ coverage json).

FinBERT's pre-training (Reuters TRC2, 2008-2010) and fine-tuning (Financial
PhraseBank, Malo et al. 2014) predate the 2021+ test years, so unlike a
GPT/Claude scorer it cannot have memorised what happened after a headline.
Same alternative on GPU: Kaggle MODELS="phase4_news_score".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from quant_research import news as NW  # noqa: E402

WORK = REPO / "data" / "news9"
OUT = REPO / "results" / "news_sentiment"


def log(msg: str) -> None:
    print(f"[phase9-news {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def universe() -> set[str]:
    lines = (REPO / "configs" / "live_universe.txt").read_text().splitlines()
    return {NW.canon(s) for s in lines if s.strip() and not s.startswith("#")}


def thash(t: str) -> str:
    return hashlib.md5(t.encode()).hexdigest()


def cmd_prep(headlines: Path, shards: int) -> dict:
    df = pd.read_parquet(headlines)
    n0 = len(df)
    df["symbol"] = df["symbol"].map(NW.canon)
    df = df[df["symbol"].isin(universe())].copy()
    df["text"] = df["headline"].fillna("").str.strip()
    df = df[df["text"].str.len() > 0]
    df["thash"] = df["text"].map(thash)
    rows = df[["article_id", "symbol", "pub_date", "thash"]]
    uniq = df.drop_duplicates("thash")[["thash", "text"]]
    uniq = uniq.assign(shard=uniq["thash"].map(lambda h: int(h[:8], 16)
                                               % shards))
    WORK.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(WORK / "rows.parquet", index=False)
    for k, g in uniq.groupby("shard"):
        g[["thash", "text"]].to_parquet(WORK / f"shard_{k}.parquet",
                                        index=False)
    meta = {"headline_symbol_rows": int(n0), "universe_rows": int(len(rows)),
            "unique_texts": int(len(uniq)), "shards": shards,
            "per_shard": uniq["shard"].value_counts().sort_index().tolist(),
            "pub_date_min": str(pd.to_datetime(rows["pub_date"]).min().date()),
            "pub_date_max": str(pd.to_datetime(rows["pub_date"]).max().date())}
    (WORK / "prep_meta.json").write_text(json.dumps(meta, indent=1))
    log(json.dumps(meta))
    return meta


def cmd_score(shard: int, max_minutes: float, batch: int = 64,
              max_len: int = 64) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(os.cpu_count() or 2)
    t0 = time.time()
    df = pd.read_parquet(WORK / f"shard_{shard}.parquet")
    tok = AutoTokenizer.from_pretrained(NW.FINBERT)
    mdl = AutoModelForSequenceClassification.from_pretrained(NW.FINBERT)
    mdl.eval()
    mdl = torch.quantization.quantize_dynamic(mdl, {torch.nn.Linear},
                                              dtype=torch.qint8)
    lab = {i: l.lower() for i, l in mdl.config.id2label.items()}
    pos = next(i for i, l in lab.items() if l.startswith("pos"))
    neg = next(i for i, l in lab.items() if l.startswith("neg"))
    order = df["text"].str.len().sort_values().index
    texts = df.loc[order, "text"].tolist()
    hashes = df.loc[order, "thash"].tolist()
    out_h, out_p, out_n = [], [], []
    complete = True
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            if (time.time() - t0) / 60 > max_minutes:
                complete = False
                break
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            pr = torch.softmax(mdl(**enc).logits, dim=-1).numpy()
            out_h += hashes[i:i + batch]
            out_p += pr[:, pos].tolist()
            out_n += pr[:, neg].tolist()
            if (i // batch) % 200 == 0:
                rate = len(out_h) / max(time.time() - t0, 1e-9)
                log(f"shard {shard}: {len(out_h)}/{len(texts)} "
                    f"({rate:.0f}/s)")
    res = pd.DataFrame({"thash": out_h, "p_pos": out_p, "p_neg": out_n})
    res["score"] = res["p_pos"] - res["p_neg"]
    res.to_parquet(WORK / f"scored_{shard}.parquet", index=False)
    meta = {"shard": shard, "n_texts": len(texts), "n_scored": len(res),
            "complete": complete, "minutes": round((time.time() - t0) / 60, 1),
            "texts_per_s": round(len(res) / max(time.time() - t0, 1e-9), 1)}
    (WORK / f"scored_{shard}.json").write_text(json.dumps(meta))
    log(json.dumps(meta))
    return meta


def cmd_merge() -> dict:
    rows = pd.read_parquet(WORK / "rows.parquet")
    sc = pd.concat([pd.read_parquet(p)
                    for p in sorted(WORK.glob("scored_*.parquet"))])
    metas = [json.loads(p.read_text())
             for p in sorted(WORK.glob("scored_*.json"))]
    m = rows.merge(sc[["thash", "score"]], on="thash", how="left")
    cov = float(m["score"].notna().mean())
    scored = m.dropna(subset=["score"])
    daily = NW.daily_table(scored)
    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT / "daily_sentiment.parquet", index=False)
    meta = {"rows": int(len(m)), "row_coverage": cov,
            "daily_rows": int(len(daily)),
            "symbols": int(daily["symbol"].nunique()),
            "first_day": str(daily["pub_date"].min().date()),
            "last_day": str(daily["pub_date"].max().date()),
            "score_mean": float(scored["score"].mean()),
            "frac_pos": float((scored["score"] > 0.5).mean()),
            "frac_neg": float((scored["score"] < -0.5).mean()),
            "shards": metas, "model": NW.FINBERT,
            "text": "headline only, max_len 64, int8 dynamic quantisation"}
    (OUT / "coverage.json").write_text(json.dumps(meta, indent=1))
    log(json.dumps({k: v for k, v in meta.items() if k != "shards"}))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "score", "merge"])
    ap.add_argument("--headlines", default="data/news/headlines.parquet")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=300)
    a = ap.parse_args()
    if a.cmd == "prep":
        cmd_prep(Path(a.headlines), a.shards)
    elif a.cmd == "score":
        cmd_score(a.shard, a.max_minutes)
    else:
        cmd_merge()


if __name__ == "__main__":
    main()
