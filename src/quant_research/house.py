"""U.S. House PTR (Periodic Transaction Report) fetcher + PDF parser.

Official source: Office of the Clerk public disclosures.

  * Year index:  https://disclosures-clerk.house.gov/public_disc/
                 financial-pdfs/{YEAR}FD.zip  -> {YEAR}FD.txt/.xml
                 (filer name, FilingType, FilingDate, DocID)
                 FilingType 'P' = Periodic Transaction Report.
  * Report PDF:  https://disclosures-clerk.house.gov/public_disc/
                 ptr-pdfs/{YEAR}/{DocID}.pdf

E-filed PTRs (the ~overwhelming majority since 2021) are generated PDFs with
extractable text; the transaction table row shape is

    [owner] asset name (TICKER) [TYPECODE] P|S|E mm/dd/yyyy mm/dd/yyyy $lo - $hi

which may wrap across lines. Scanned paper filings exist pre-~2019 and for a
few members; they yield no text and are counted in the manifest as
unparsed rather than silently dropped.

Senate note: the Senate eFD search API is Akamai-blocked for scripted access
(403 at the edge, GET/POST alike, browser UAs alike — verified 2026-11-11)
and the community mirrors that covered it (senate-stock-watcher-data) went
stale in 2021-03, so this module covers the House only. See docs/PHASE3.md.
"""
from __future__ import annotations

import io
import re
import zipfile

import pandas as pd
import requests

HOUSE_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
INDEX_URL = ("https://disclosures-clerk.house.gov/public_disc/"
             "financial-pdfs/{year}FD.zip")
PDF_URL = ("https://disclosures-clerk.house.gov/public_disc/"
           "ptr-pdfs/{year}/{doc}.pdf")

STOCK_CODES = {"ST", "SO", "OP"}   # stock / stock-option / other equity


class House:
    def __init__(self, rps: float = 4.0, timeout: int = 60):
        import time
        self._time = time
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": HOUSE_UA})
        self.min_interval = 1.0 / rps
        self.timeout = timeout
        self._last = 0.0
        self.requests = 0

    def get(self, url: str) -> requests.Response:
        for attempt in range(4):
            dt = self.min_interval - (self._time.time() - self._last)
            if dt > 0:
                self._time.sleep(dt)
            r = self.s.get(url, timeout=self.timeout)
            self._last = self._time.time()
            self.requests += 1
            if r.status_code == 429 or r.status_code >= 500:
                self._time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        raise RuntimeError(f"house: still failing after retries: {url}")

    def fd_index(self, year: int) -> pd.DataFrame:
        """Year's FD index -> one row per filing (all types)."""
        z = zipfile.ZipFile(io.BytesIO(self.get(INDEX_URL.format(year=year)).content))
        txt = z.read(f"{year}FD.txt").decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(txt), sep="\t", dtype=str)
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            df[c] = df[c].str.strip()
        df["year"] = year
        return df

    def ptr_pdf(self, year: int, doc_id: str) -> bytes:
        return self.get(PDF_URL.format(year=year, doc=doc_id)).content


# ---------------------------------------------------------------------------
# PDF text -> transaction rows

_AMOUNT = r"\$[\d,]+\s*-\s*\$[\d,]+"
_DATE = r"\d{2}/\d{2}/\d{4}"
# A transaction row ends with: TYPE DATE DATE AMOUNT  (cap-gains flag ignored)
_ROW_END = re.compile(
    rf"(?P<type>P|E|X|S(?:\s*\(partial\))?)\s+(?P<tdate>{_DATE})\s+"
    rf"(?P<ndate>{_DATE})\s+(?P<amount>{_AMOUNT})")
_OWNER = {"JT", "SP", "DC", "CS", "JT/SP"}
_TYPE_CODE = re.compile(r"\[([A-Z]{2})\]")
_TICKER = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,7})\)\s*$")


def _amount_mid(s: str) -> tuple[float, float]:
    lo, hi = re.findall(r"[\d,]+", s)
    return float(lo.replace(",", "")), float(hi.replace(",", ""))


def parse_ptr_text(text: str) -> list[dict]:
    """Extract transaction rows from an e-filed House PTR's extracted text.

    Rows can wrap; we find each row's trailing "TYPE date date amount"
    anchor, then take the text since the previous anchor (or header) as the
    owner+asset field. Returns [] when nothing parses (scanned PDF).
    """
    # de-space the weird letter-spread headers ("F I L E R" etc. are headers
    # anyway) and collapse whitespace so wrapped rows join up
    t = text.replace("\x00", "")
    t = re.sub(r"\s+", " ", t)
    rows = []
    prev_end = 0
    for m in _ROW_END.finditer(t):
        head = t[prev_end:m.start()]
        prev_end = m.end()
        # head = "… <owner?> <asset> [(TICKER)] [CODE]" possibly with prior
        # row's bookkeeping ("FILING STATUS:", "S O:", "* For the…")
        # keep only the tail from the last asset-type bracket
        code_m = None
        for cm in _TYPE_CODE.finditer(head):
            code_m = cm
        if code_m is None:
            continue
        asset = head[:code_m.start()].strip()
        code = code_m.group(1)
        # strip boilerplate that can precede the asset (footer lines,
        # column header, section titles)
        asset = re.split(
            r"ID Owner Asset Transaction Type Date Notification Date Amount "
            r"Cap\.?\s*Gains\s*>\s*\$200\??|"
            r"FILING STATUS:|SOURCE OFFICE:|SOURCE:|S?O:|SUBLOCATION:|LOCATION:|"
            r"DESCRIPTION:|Type Date Notification Date Amount|"
            r"Cap\.?|Gains\s*>\s*\$200\??|"
            r"Filing ID|asset type abbreviations|"
            r"DIGITALLY|Digitally Signed|I CERTIFY|"
            r"Name:|Status:|State/District:",
            asset)[-1].strip()
        tick_m = _TICKER.search(asset)
        ticker = tick_m.group(1) if tick_m else None
        if tick_m:
            asset = asset[:tick_m.start()].strip()
        owner = "Self"
        parts = asset.split(None, 1)
        if len(parts) == 2 and parts[0].upper() in _OWNER:
            owner, asset = parts[0].upper(), parts[1]
        if not asset:
            continue
        lo, hi = _amount_mid(m.group("amount"))
        rows.append({"owner": owner, "asset": asset.strip(" ,"),
                     "ticker": ticker, "asset_code": code,
                     "txn_type": m.group("type"),
                     "txn_date": m.group("tdate"),
                     "notif_date": m.group("ndate"),
                     "amount_lo": lo, "amount_hi": hi})
    return rows


def ptr_name_from_text(text: str) -> str | None:
    m = re.search(r"Name:\s*(.+)", text)
    return m.group(1).strip() if m else None
