"""SEC EDGAR fetchers for Phase-3 top-trader features.

Sources (all public, no key; polite rate limiting ~8 req/s + proper UA):
  * company_tickers.json            ticker -> CIK map (whole universe, 1 req)
  * DERA insider-transactions zips  quarterly Form 3/4/5 bulk tables
      https://www.sec.gov/dera/data/form-345 -> .../{YYYY}q{Q}_form345.zip
      (SUBMISSION / REPORTINGOWNER / NONDERIV_TRANS / DERIV_TRANS / FOOTNOTES)
  * data.sec.gov/submissions/CIK{n}.json  per-issuer or per-manager filing list
      (acceptanceDateTime for precise "known-at"; files[] pages for history)
  * /Archives/edgar/data/{cik}/{acc}/index.json + information-table XML  13F
  * /api/xbrl/companyfacts/CIK{n}.json     shares outstanding (dei)

PIT note: the DERA bulk tables carry FILING_DATE but not the EDGAR
acceptance timestamp. Filing date is therefore treated as the information
date and every event is made visible to trading days STRICTLY AFTER it
(filing accepted Friday night is usable Monday; Monday filing is usable
Tuesday). That is conservative vs the real dissemination time and matches
the "accepted before the trading day" rule.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

SEC_UA = ("quant-research/1.0 (academic research; "
          "contact: riyanshdiwan@gmail.com)")
FORM345_BASE = ("https://www.sec.gov/files/structureddata/data/"
                "insider-transactions-data-sets/{y}q{q}_form345.zip")
FORM345_BASE_ALT = ("https://www.sec.gov/files/datastandardsinnovation/data/"
                    "insider-transactions-data-sets/{y}q{q}_form345.zip")


class Edgar:
    """Polite SEC session: ~8 req/s, exponential backoff on 429/5xx."""

    def __init__(self, rps: float = 8.0, timeout: int = 60):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": SEC_UA,
                               "Accept-Encoding": "gzip, deflate"})
        self.min_interval = 1.0 / rps
        self.timeout = timeout
        self._last = 0.0
        self.requests = 0

    def get(self, url: str, **kw) -> requests.Response:
        for attempt in range(5):
            dt = self.min_interval - (time.time() - self._last)
            if dt > 0:
                time.sleep(dt)
            r = self.s.get(url, timeout=self.timeout, **kw)
            self._last = time.time()
            self.requests += 1
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        raise RuntimeError(f"edgar: still failing after retries: {url}")


# ---------------------------------------------------------------------------
# ticker <-> CIK


def ticker_cik_map(edgar: Edgar) -> pd.DataFrame:
    """Universe of SEC-registered tickers -> (cik, title)."""
    d = edgar.get("https://www.sec.gov/files/company_tickers.json").json()
    df = pd.DataFrame(d).T.rename(columns={"cik_str": "cik"})[["ticker", "cik", "title"]]
    return df.reset_index(drop=True)


def symbol_to_edgar_ticker(sym: str) -> str:
    """Alpaca/qlib symbol -> EDGAR ticker (class shares: BRK.B -> BRK-B)."""
    return sym.upper().replace(".", "-").replace("_", "-")


# ---------------------------------------------------------------------------
# Form 3/4/5 quarterly bulk (DERA insider transactions dataset)


def form345_quarters(start_year: int, end_year: int | None = None) -> list[tuple[int, int]]:
    end_year = end_year or pd.Timestamp.now().year
    out = []
    for y in range(start_year, end_year + 1):
        for q in range(1, 5):
            out.append((y, q))
    return out


def fetch_form345(edgar: Edgar, year: int, q: int) -> dict[str, pd.DataFrame] | None:
    """One quarter of Form 3/4/5 bulk tables -> {table: DataFrame}.

    Returns None when the quarter's zip isn't published yet (in-progress
    quarters 404 on both host paths)."""
    r = None
    for base in (FORM345_BASE, FORM345_BASE_ALT):
        url = base.format(y=year, q=q)
        try:
            r = edgar.get(url)
            break
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                continue
            raise
    if r is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    keep = ["SUBMISSION", "REPORTINGOWNER", "NONDERIV_TRANS",
            "DERIV_TRANS", "FOOTNOTES"]
    out = {}
    for k in keep:
        try:
            out[k] = pd.read_csv(z.open(f"{k}.tsv"), sep="\t", dtype=str,
                                 low_memory=False)
        except KeyError:
            out[k] = pd.DataFrame()
    return out


# ---------------------------------------------------------------------------
# issuer / manager submissions index


def fetch_submissions(edgar: Edgar, cik: int,
                      since: str | None = None) -> pd.DataFrame:
    """Filing index for one CIK -> DataFrame(form, accession, filingDate,
    acceptanceDateTime, primaryDocument, reportDate).

    Pages through the older 'files' array until covered `since` date.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    d = edgar.get(url).json()
    rows = []

    def add(recent: dict) -> None:
        n = len(recent.get("accessionNumber", []))
        for i in range(n):
            rows.append({
                "form": recent["form"][i],
                "accession": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "accepted": recent["acceptanceDateTime"][i],
                "primary_doc": recent.get("primaryDocument", [None] * n)[i],
                "report_date": recent.get("reportDate", [None] * n)[i],
            })

    add(d["filings"]["recent"])
    if since:
        for f in d["filings"].get("files", []):
            # file covers [filingFrom, filingTo]; only fetch if needed
            if f.get("filingTo", "9999") >= since:
                extra = edgar.get("https://data.sec.gov/submissions/" + f["name"]).json()
                add(extra)
    df = pd.DataFrame(rows)
    if since and not df.empty:
        df = df[df["filing_date"] >= since]
    return df


# ---------------------------------------------------------------------------
# 13F-HR information tables


def fetch_13f_infotable(edgar: Edgar, cik: int, accession: str) -> pd.DataFrame:
    """Parse a 13F-HR(-like) filing's information table.

    Returns rows (cusip, issuer, value_x1000, shares, put_call) — empty frame
    when the filing carries no XML info table (e.g. 13F-NT notices).
    """
    acc_nodash = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}"
    idx = edgar.get(f"{base}/index.json").json()
    names = [it["name"] for it in idx["directory"]["item"]]
    # the info table is an .xml whose name is not the primary doc —
    # try every xml until one parses as an information table
    cands = [n for n in names if n.lower().endswith(".xml")]
    cands.sort(key=lambda n: ("index" in n.lower(), "primary" in n.lower()))
    import xml.etree.ElementTree as ET

    def loc(el) -> str:
        return el.tag.lower().split("}")[-1]

    for name in cands:
        try:
            raw = edgar.get(f"{base}/{name}").content
            root = ET.fromstring(raw)
        except Exception:  # noqa: BLE001 — not XML / not the info table
            continue
        if loc(root) != "informationtable":
            continue
        rows = []
        for it in root.iter():
            if loc(it) != "infotable":
                continue

            def t(x, it=it):
                for e in it.iter():
                    if e is not it and loc(e) == x:
                        return e.text.strip() if e.text else None
                return None

            rows.append({
                "issuer": t("nameofissuer"),
                "class": t("titleofclass"),
                "cusip": t("cusip"),
                "value_x1000": t("value"),
                "shares": t("sshprnamt"),
                "put_call": t("putcall"),
                "investment_discretion": t("investmentdiscretion"),
                "voting_sole": t("sole"),
            })
        return pd.DataFrame(rows)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# shares outstanding (for market-cap normalization)


def fetch_shares_outstanding(edgar: Edgar, cik: int) -> pd.DataFrame:
    """dei:EntityCommonStockSharesOutstanding history -> (end, val, filed).

    Point-in-time: each row carries the EDGAR `filed` date; a value is known
    only from its filed date onward.
    """
    d = edgar.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()
    dei = d.get("facts", {}).get("dei", {})
    rows = []
    for fact in dei.get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", []):
        rows.append({"end": fact.get("end"), "val": fact.get("val"),
                     "filed": fact.get("filed"), "fy": fact.get("fy"),
                     "form": fact.get("form")})
    return pd.DataFrame(rows)
