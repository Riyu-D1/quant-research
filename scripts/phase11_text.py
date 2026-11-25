#!/usr/bin/env python3
"""Phase-11 text layer — 10-K risk-factor sections, embedded by a
chronologically consistent LLM (ChronoBERT, knowledge cutoff 2016-12-31).

    python scripts/phase11_text.py fetch                 # EDGAR, no key
    python scripts/phase11_text.py embed --shard i --of k
    python scripts/phase11_text.py merge

Heavy — GitHub Actions only (.github/workflows/phase11.yml).

fetch  every 10-K (not amendments) filed 2015-01-01 .. sample_end by a
       universe CIK; Item 1A "Risk Factors" cut from the primary document
       -> data/text/tenk_1a.parquet (cik, symbols, accession, filed, text).
embed  mean-pooled ChronoBERT embedding of up to `chunks` evenly spaced
       512-token windows of each section, averaged and L2-normalised.
merge  shards -> data/text/tenk_emb.parquet.

The model never sees a return, and it was trained only on text dated
before 2017-01-01, so its embeddings cannot encode any 2017+ outcome
(docs/PHASE11_LIT.md section 6).
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

TEXT = REPO / "data" / "text"
SECTIONS = TEXT / "tenk_1a.parquet"
EMB = TEXT / "tenk_emb.parquet"
SPEC_PATH = REPO / "configs" / "experiments" / "phase11_us.yaml"

ITEM_1A = re.compile(r"item\s*1a\s*[\.\:\-\u2013\u2014]?\s*risk\s+factors", re.I)
ITEM_END = re.compile(r"item\s*(1b|1c|2)\s*[\.\:\-\u2013\u2014]?\s*"
                      r"(unresolved|cybersecurity|properties)", re.I)
MIN_CHARS = 2000


def log(msg: str) -> None:
    print(f"[phase11-text {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def html_text(raw: bytes) -> str:
    """Primary-document HTML -> plain text (inline-XBRL header dropped).
    Regex tag stripping: the section cut only needs approximate text."""
    s = raw.decode("utf-8", errors="ignore")
    s = re.sub(r"<ix:header>.*?</ix:header>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s).replace("\xa0", " "))


def risk_factors(text: str) -> str:
    """The longest 'Item 1A. Risk Factors' .. 'Item 1B/1C/2' span (the table
    of contents gives short spans, the section itself the long one)."""
    best = ""
    for m in ITEM_1A.finditer(text):
        e = ITEM_END.search(text, m.end())
        if e is None:
            continue
        span = text[m.end(): e.start()]
        if len(span) > len(best):
            best = span
    best = best.strip()
    return best if len(best) >= MIN_CHARS else ""


def cmd_fetch(spec: dict) -> None:
    import phase11_us as P11
    from quant_research import edgar as E
    ed = E.Edgar()
    cmap = P11.universe_ciks(spec, ed).assign(cik=lambda d: d["cik"].astype(int)) \
        .groupby("cik")["symbol"].agg(lambda x: ";".join(sorted(x))).reset_index()
    TEXT.mkdir(parents=True, exist_ok=True)
    rows, misses = [], 0
    for i, r in enumerate(cmap.itertuples()):
        cik = int(r.cik)
        try:
            subs = E.fetch_submissions(ed, cik, since="2015-01-01")
        except Exception as e:  # noqa: BLE001
            log(f"submissions {r.symbol}: {e}")
            continue
        if subs.empty:
            continue
        k = subs[(subs["form"] == "10-K") & (subs["filing_date"] <= spec["sample_end"])]
        for f in k.itertuples():
            url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                   f"{f.accession.replace('-', '')}/{f.primary_doc}")
            try:
                sec = risk_factors(html_text(ed.get(url).content))
            except Exception as e:  # noqa: BLE001
                log(f"10-K {r.symbol} {f.accession}: {e}")
                sec = ""
            misses += not sec
            if sec:
                rows.append({"cik": cik, "symbols": r.symbol, "accession": f.accession,
                             "filed": f.filing_date, "text": sec})
        if i % 50 == 0:
            log(f"{i}/{len(cmap)} issuers, {len(rows)} sections, {misses} without 1A")
    df = pd.DataFrame(rows)
    df["filed"] = pd.to_datetime(df["filed"])
    df.to_parquet(SECTIONS, index=False)
    log(f"wrote {len(df)} sections ({df['cik'].nunique()} issuers, "
        f"{misses} 10-Ks without a usable Item 1A) -> {SECTIONS}")


def chunk_ids(ids: list[int], size: int, k: int) -> list[list[int]]:
    """Up to k evenly spaced, non-overlapping windows of `size` tokens."""
    n = max(1, -(-len(ids) // size))
    picks = sorted(set(np.linspace(0, n - 1, min(k, n)).round().astype(int)))
    return [ids[p * size:(p + 1) * size] for p in picks]


def cmd_embed(spec: dict, shard: int, of: int) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer
    cfg = spec["text"]
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModel.from_pretrained(cfg["model"], revision=cfg["revision"]).eval()
    df = pd.read_parquet(SECTIONS)
    df = df.iloc[shard::of]
    size, k = cfg["chunk_tokens"], cfg["chunks"]
    body = size - 2                      # room for [CLS] / [SEP]
    out, t0 = [], time.time()
    for i, r in enumerate(df.itertuples()):
        ids = tok(r.text, add_special_tokens=False)["input_ids"]
        batch = [[tok.cls_token_id, *c, tok.sep_token_id] for c in chunk_ids(ids, body, k)]
        L = max(len(b) for b in batch)
        x = torch.full((len(batch), L), tok.pad_token_id)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, b in enumerate(batch):
            x[j, :len(b)] = torch.tensor(b)
            att[j, :len(b)] = 1
        with torch.no_grad():
            h = model(input_ids=x, attention_mask=att).last_hidden_state
        v = ((h * att[..., None]).sum(1) / att.sum(1, keepdim=True)).mean(0).numpy()
        out.append({"accession": r.accession, "cik": r.cik, "symbols": r.symbols,
                    "filed": r.filed,
                    "n_tokens": len(ids), "emb": (v / np.linalg.norm(v)).astype(np.float32)})
        if i % 100 == 0:
            log(f"shard {shard}: {i}/{len(df)} ({time.time() - t0:.0f}s)")
    pd.DataFrame(out).to_parquet(TEXT / f"tenk_emb_{shard}.parquet", index=False)
    log(f"shard {shard}: {len(out)} embeddings in {time.time() - t0:.0f}s")


def cmd_merge() -> None:
    parts = sorted(TEXT.glob("tenk_emb_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(EMB, index=False)
    log(f"merged {len(parts)} shards -> {len(df)} embeddings -> {EMB}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "embed", "merge"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args()
    spec = yaml.safe_load(SPEC_PATH.read_text())
    if args.cmd == "fetch":
        cmd_fetch(spec)
    elif args.cmd == "embed":
        cmd_embed(spec, args.shard, args.of)
    else:
        cmd_merge()


if __name__ == "__main__":
    main()
