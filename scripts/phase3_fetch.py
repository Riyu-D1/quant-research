"""Phase-3 data fetch: top-trader disclosures -> tidy event tables.

Subcommands (each idempotent; cached under data/events/raw/):
  cik-map    universe symbol -> issuer CIK (company_tickers.json for listed
             names + form345 ISSUERTRADINGSYMBOL/ISSUERCIK seen inside each
             symbol's PIT membership window for delisted ones)
  insiders   DERA form345 quarterly zips -> insider_events.parquet
  f13f       fixed fund list 13F-HR info tables -> f13f_events.parquet
  shares     dei shares-outstanding per issuer CIK -> shares_out.parquet
  house      Clerk FD index + PTR PDFs -> congress_events.parquet
  all        everything, in that order

Outputs land in data/events/ (gitignored; rebuild by rerunning). A manifest
tracks processed ids so reruns are incremental.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quant_research import edgar as E
from quant_research import house as H

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "events" / "raw"
OUT = REPO / "data" / "events"

# Pre-declared 13F fund list (chosen before any backtest — see docs/PHASE3.md).
FUNDS = {
    "Berkshire Hathaway": 1067983,
    "Renaissance Technologies": 1037389,
    "Citadel Advisors": 1423053,
    "Bridgewater Associates": 1350694,
    "Two Sigma Investments": 1179392,
    "D.E. Shaw & Co": 1009207,
    "Millennium Management": 1273087,
    "Point72 Asset Management": 1603466,
    "Baupost Group": 1061768,
    "Pershing Square Capital": 1336528,
    "Tiger Global Management": 1167483,
    "Coatue Management": 1135730,
    "Viking Global Investors": 1103804,
    "Lone Pine Capital": 1061165,
    "Soros Fund Management": 1029160,
    "Third Point": 1040273,
    "Elliott Investment Management": 1048445,
    "AQR Capital Management": 1167557,
    "ARK Investment Management": 1697748,
    "Appaloosa Management": 1656456,
}

FORM345_START_YEAR = 2015   # features are used from 2016 (walk-forward start)


def log(msg: str) -> None:
    print(f"[phase3-fetch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def universe() -> pd.DataFrame:
    """PIT S&P 500 universe: prefer the build CSV, else derive from the qlib
    instruments file (symbol, membership start, end)."""
    csv = REPO / "build" / "universe_sp500.csv"
    if csv.exists():
        return pd.read_csv(csv)
    inst = REPO / "data/qlib/us_alpaca/instruments/sp500.txt"
    rows = []
    for line in inst.read_text().splitlines():
        if not line.strip():
            continue
        s, a, b = line.split("\t")
        rows.append({"symbol": s, "in_date": a, "out_date": b})
    return pd.DataFrame(rows)


def qlib_sym(sym: str) -> str:
    return sym.upper().replace(".", "-")


def manifest_path() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / "manifest.json"


def load_manifest() -> dict:
    p = manifest_path()
    return json.loads(p.read_text()) if p.exists() else {}


def save_manifest(m: dict) -> None:
    manifest_path().write_text(json.dumps(m, indent=1, sort_keys=True))


# --------------------------------------------------------------------------
# cik-map


def cmd_cik_map(ed: E.Edgar, args) -> None:
    uni = universe()
    uni["esym"] = uni["symbol"].map(E.symbol_to_edgar_ticker)
    tick = E.ticker_cik_map(ed)
    tick_by_t = tick.set_index("ticker")

    rows = {}
    for r in uni.itertuples():
        if r.esym in tick_by_t.index:
            t = tick_by_t.loc[r.esym]
            if isinstance(t, pd.DataFrame):
                t = t.iloc[0]
            rows[r.symbol] = {"cik": int(t["cik"]), "issuer": t["title"],
                              "cik_source": "company_tickers"}
    missing = uni[~uni["symbol"].isin(rows)]
    log(f"cik-map: {len(rows)} listed via company_tickers; "
        f"{len(missing)} delisted need form345 lookup")

    # Delisted names: scan cached form345 submissions for (tradingSymbol, cik)
    # pairs whose filings fall inside the symbol's membership window.
    subs_all = []
    if len(missing) or args.refresh_map:
        for p in sorted(RAW.glob("form345_*_SUBMISSION.parquet")):
            subs_all.append(pd.read_parquet(
                p, columns=["ISSUERCIK", "ISSUERTRADINGSYMBOL",
                            "ISSUERNAME", "FILING_DATE"]))
        if not subs_all and len(missing):
            log("no cached form345 submissions — run `insiders` first; "
                "delisted CIKs unresolved for now")
    if subs_all:
        sub = pd.concat(subs_all, ignore_index=True).dropna(
            subset=["ISSUERTRADINGSYMBOL", "ISSUERCIK"])
        sub["esym"] = sub["ISSUERTRADINGSYMBOL"].str.upper().str.strip()
        sub["cik"] = sub["ISSUERCIK"].astype(int)
        sub["fd"] = sub["FILING_DATE"]
        for r in missing.itertuples():
            cand = sub[sub["esym"] == r.esym]
            if cand.empty:
                rows[r.symbol] = {"cik": None, "issuer": None,
                                  "cik_source": "unresolved"}
                continue
            in_d, out_d = str(r.in_date)[:10], str(r.out_date)[:10]
            inside = cand[(cand["fd"] >= in_d) & (cand["fd"] <= out_d)]
            pick = inside if len(inside) else cand
            cik = int(pick["cik"].value_counts().idxmax())
            issuer = pick[pick["cik"] == cik]["ISSUERNAME"].mode().iloc[0]
            rows[r.symbol] = {"cik": cik, "issuer": issuer,
                              "cik_source": "form345"}
    out = pd.DataFrame([{"symbol": s, **v} for s, v in rows.items()])
    out.to_parquet(OUT / "cik_map.parquet", index=False)
    n_un = (out["cik"].isna()).sum()
    log(f"cik-map: wrote {len(out)} rows -> {OUT/'cik_map.parquet'}; "
        f"unresolved: {n_un} {sorted(out.loc[out['cik'].isna(),'symbol'])[:20]}")


# --------------------------------------------------------------------------
# insiders — form345 bulk -> tidy events


def cmd_insiders(ed: E.Edgar, args) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    m = load_manifest()
    done = set(m.get("form345_quarters", []))
    now = pd.Timestamp.now()
    cur = (now.year, (now.month - 1) // 3 + 1)
    for y, q in E.form345_quarters(args.start_year):
        if (y, q) > cur:
            break
        key = f"{y}q{q}"
        # always re-fetch the in-progress quarter — new filings keep landing
        if key in done and (y, q) != cur:
            continue
        tabs = E.fetch_form345(ed, y, q)
        if tabs is None:
            log(f"form345 {key}: not published yet — skipped")
            continue
        for name, df in tabs.items():
            df.to_parquet(RAW / f"form345_{key}_{name}.parquet")
        done.add(key)
        m["form345_quarters"] = sorted(done)
        save_manifest(m)          # per-quarter checkpoint
        log(f"form345 {key}: {len(tabs['SUBMISSION'])} filings")
    # seed the manifest from the cache too — a crash between download and
    # manifest save must not force a re-download
    for p in RAW.glob("form345_*_SUBMISSION.parquet"):
        done.add(p.name.split("_")[1])
    m["form345_quarters"] = sorted(done)
    save_manifest(m)

    # ---- tidy events ----------------------------------------------------
    subs = pd.concat(
        [pd.read_parquet(p) for p in sorted(RAW.glob("form345_*_SUBMISSION.parquet"))],
        ignore_index=True)
    owners = pd.concat(
        [pd.read_parquet(p) for p in sorted(RAW.glob("form345_*_REPORTINGOWNER.parquet"))],
        ignore_index=True)
    txs = pd.concat(
        [pd.read_parquet(p) for p in sorted(RAW.glob("form345_*_NONDERIV_TRANS.parquet"))],
        ignore_index=True)

    subs = subs.drop_duplicates("ACCESSION_NUMBER")
    subs["filing_date"] = pd.to_datetime(subs["FILING_DATE"], format="%d-%b-%Y",
                                         errors="coerce")
    # amendments (4/A) supersede the original — keep latest filing per
    # (issuer, owner?, period) is hard; simplest honest rule: keep all, but
    # when the same accession appears twice drop dupes (done above).
    owners_s = (owners[["ACCESSION_NUMBER", "RPTOWNERNAME",
                        "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"]]
                .drop_duplicates())
    txs = txs.merge(owners_s, on="ACCESSION_NUMBER", how="left")
    txs = txs.merge(
        subs[["ACCESSION_NUMBER", "filing_date", "ISSUERCIK",
              "ISSUERTRADINGSYMBOL", "ISSUERNAME", "REMARKS", "AFF10B5ONE",
              "DOCUMENT_TYPE"]],
        on="ACCESSION_NUMBER", how="left")

    ev = pd.DataFrame({
        "accession": txs["ACCESSION_NUMBER"],
        "symbol": txs["ISSUERTRADINGSYMBOL"].str.upper().str.strip(),
        "issuer_name": txs["ISSUERNAME"],
        "issuer_cik": txs["ISSUERCIK"],
        "filing_date": txs["filing_date"].dt.strftime("%Y-%m-%d"),
        "txn_date": txs["TRANS_DATE"],
        "code": txs["TRANS_CODE"],
        "form_type": txs["TRANS_FORM_TYPE"],
        "shares": pd.to_numeric(txs["TRANS_SHARES"], errors="coerce"),
        "price": pd.to_numeric(txs["TRANS_PRICEPERSHARE"], errors="coerce"),
        "acq_disp": txs["TRANS_ACQUIRED_DISP_CD"],
        "direct": txs["DIRECT_INDIRECT_OWNERSHIP"],
        "owner": txs["RPTOWNERNAME"],
        "role": txs["RPTOWNER_RELATIONSHIP"],
        "title": txs["RPTOWNER_TITLE"],
        "is10b5_filing": txs["AFF10B5ONE"].fillna("0").astype(str) != "0",
        "doc_type": txs["DOCUMENT_TYPE"],
        "security": txs["SECURITY_TITLE"],
        "remarks": txs["REMARKS"],
    })
    ev["usd"] = ev["shares"] * ev["price"]
    ev.to_parquet(OUT / "insider_events.parquet", index=False)
    log(f"insiders: {len(ev)} txns -> {OUT/'insider_events.parquet'}")


# --------------------------------------------------------------------------
# f13f


def cmd_f13f(ed: E.Edgar, args) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    m = load_manifest()
    done = set(m.get("f13f_accessions", []))
    frames = []
    since = args.since
    for fund, cik in FUNDS.items():
        try:
            subs = E.fetch_submissions(ed, cik, since=since)
        except Exception as e:  # noqa: BLE001
            log(f"13f {fund}: submissions failed ({e}) — skipped")
            continue
        fil = subs[subs["form"].str.startswith("13F-HR")].copy()
        fil["year"] = fil["filing_date"].str[:4].astype(int)
        for r in fil.itertuples():
            if r.accession in done:
                continue
            cache = RAW / f"f13f_{cik}_{r.accession.replace('-','')}.parquet"
            try:
                tab = E.fetch_13f_infotable(ed, cik, r.accession)
            except Exception as e:  # noqa: BLE001
                log(f"13f {fund} {r.accession}: infotable failed ({e})")
                continue
            tab["fund"] = fund
            tab["fund_cik"] = cik
            tab["accession"] = r.accession
            tab["filing_date"] = r.filing_date
            tab["report_date"] = r.report_date
            tab["form"] = r.form
            tab.to_parquet(cache)
            done.add(r.accession)
        n_cached = len(list(RAW.glob(f"f13f_{cik}_*.parquet")))
        log(f"13f {fund}: {n_cached} filings cached")
        time.sleep(0.1)
    m["f13f_accessions"] = sorted(done)
    save_manifest(m)

    frames = [pd.read_parquet(p) for p in sorted(RAW.glob("f13f_*.parquet"))]
    if not frames:
        log("13f: no filings cached")
        return
    allf = pd.concat(frames, ignore_index=True)
    allf = allf[allf["put_call"].isna()]   # long holdings only, drop options
    # amendments supersede: keep latest filing per (fund, report_date)
    allf = (allf.sort_values("filing_date")
                .groupby(["fund", "report_date"], as_index=False)
                .apply(lambda g: g[g["filing_date"] == g["filing_date"].max()])
                .reset_index(drop=True))
    allf["shares"] = pd.to_numeric(allf["shares"], errors="coerce")
    allf["value"] = pd.to_numeric(allf["value_x1000"], errors="coerce")
    allf["weight"] = allf["value"] / allf.groupby(
        ["fund", "report_date"])["value"].transform("sum")
    allf.to_parquet(OUT / "f13f_events.parquet", index=False)
    log(f"13f: {len(allf)} holding rows, {allf['fund'].nunique()} funds, "
        f"{allf['report_date'].nunique()} quarters -> "
        f"{OUT/'f13f_events.parquet'}")


# --------------------------------------------------------------------------
# shares outstanding


def cmd_shares(ed: E.Edgar, args) -> None:
    m = load_manifest()
    done = set(m.get("shares_ciks", []))
    cmap = pd.read_parquet(OUT / "cik_map.parquet")
    for r in cmap.dropna(subset=["cik"]).itertuples():
        cik = int(r.cik)
        if cik in done:
            continue
        cache = RAW / f"shares_{cik}.parquet"
        try:
            df = E.fetch_shares_outstanding(ed, cik)
            df["symbol"] = r.symbol
            df["cik"] = cik
            df.to_parquet(cache)
            done.add(cik)
        except Exception as e:  # noqa: BLE001
            log(f"shares {r.symbol} ({cik}): {e}")
        if len(done) % 100 == 0:
            m["shares_ciks"] = sorted(done)
            save_manifest(m)
            log(f"shares: {len(done)} issuers")
    m["shares_ciks"] = sorted(done)
    save_manifest(m)
    frames = [pd.read_parquet(p) for p in sorted(RAW.glob("shares_*.parquet"))]
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(
            OUT / "shares_out.parquet", index=False)
        log(f"shares: {sum(len(f) for f in frames)} rows -> "
            f"{OUT/'shares_out.parquet'}")


# --------------------------------------------------------------------------
# house


def cmd_house(h: H.House, args) -> None:
    from pypdf import PdfReader
    import io as _io
    RAW.mkdir(parents=True, exist_ok=True)
    m = load_manifest()
    done = set(m.get("house_docs", []))
    failed = set(m.get("house_failed", []))
    frames = []
    for year in range(args.house_start, pd.Timestamp.now().year + 1):
        idx_cache = RAW / f"house_index_{year}.parquet"
        if idx_cache.exists():
            idx = pd.read_parquet(idx_cache)
        else:
            idx = h.fd_index(year)
            idx.to_parquet(idx_cache)
        ptr = idx[idx["FilingType"].isin(["P"])]
        log(f"house {year}: {len(ptr)} PTRs ({len(idx)} filings total)")
        new = [r for r in ptr.itertuples()
               if str(r.DocID) not in done and str(r.DocID) not in failed]
        for i, r in enumerate(new):
            doc = str(r.DocID)
            cache = RAW / f"ptr_{doc}.parquet"
            try:
                pdf = h.ptr_pdf(year, doc)
                text = "\n".join(pg.extract_text() or ""
                                 for pg in PdfReader(_io.BytesIO(pdf)).pages)
                rows = H.parse_ptr_text(text)
                rec = pd.DataFrame(rows)
                if rec.empty:
                    failed.add(doc)
                else:
                    rec["doc_id"] = doc
                    rec["filing_date"] = r.FilingDate
                    rec["filer"] = H.ptr_name_from_text(text) or \
                        f"{r.First} {r.Last}"
                    rec.to_parquet(cache)
                    done.add(doc)
            except Exception as e:  # noqa: BLE001
                failed.add(doc)
                if len(failed) % 50 == 1:
                    log(f"house {doc}: {e}")
            if (i + 1) % 500 == 0:
                m["house_docs"] = sorted(done)
                m["house_failed"] = sorted(failed)
                save_manifest(m)
                log(f"house {year}: {i+1}/{len(new)} parsed "
                    f"({len(done)} ok / {len(failed)} no-text)")
        m["house_docs"] = sorted(done)
        m["house_failed"] = sorted(failed)
        save_manifest(m)
    frames = [pd.read_parquet(p) for p in sorted(RAW.glob("ptr_*.parquet"))]
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(
            OUT / "congress_events.parquet", index=False)
        log(f"house: {sum(len(f) for f in frames)} txn rows -> "
            f"{OUT/'congress_events.parquet'}")


# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["cik-map", "insiders", "f13f", "shares",
                                   "house", "all"])
    p.add_argument("--start-year", type=int, default=FORM345_START_YEAR)
    p.add_argument("--since", default="2019-01-01",
                   help="13F filings since (filing date)")
    p.add_argument("--house-start", type=int, default=2019)
    p.add_argument("--rps", type=float, default=8.0)
    p.add_argument("--refresh-map", action="store_true")
    args = p.parse_args()

    ed = E.Edgar(rps=args.rps)
    h = H.House(rps=4.0)

    if args.cmd in ("cik-map", "all"):
        cmd_cik_map(ed, args)
    if args.cmd in ("insiders", "all"):
        cmd_insiders(ed, args)
    if args.cmd in ("f13f", "all"):
        cmd_f13f(ed, args)
    if args.cmd in ("shares", "all"):
        cmd_shares(ed, args)
    if args.cmd in ("house", "all"):
        cmd_house(h, args)
    log(f"done — {ed.requests} edgar / {h.requests} house requests")


if __name__ == "__main__":
    main()
