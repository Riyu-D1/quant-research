"""Phase-3 tests: PIT no-lookahead, event feature builder, symbol handling.

Honesty contract under test (src/quant_research/events.py):
  a feature at trading day D uses ONLY events whose info date (EDGAR
  filing date / House filing date) is a trading day STRICTLY BEFORE D.
  test_no_lookahead_recompute recomputes sampled real-panel rows from the
  raw event tables restricted to events known before the row's date and
  fails if any feature row reflects a filing accepted after its date.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quant_research import events as ev  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/events/features_us.parquet"
EVENTS_DIR = ROOT / "data/events"

DAYS = pd.DatetimeIndex(pd.bdate_range("2021-01-01", "2021-03-31"))


def _ins(rows):
    df = pd.DataFrame(rows)
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


def test_symbol_normalization():
    assert ev.canon("BRK.B") == "BRK-B"
    assert ev.canon("brk-b") == "BRK-B"
    assert ev.canon("BF/B") == "BF-B"
    assert ev.canon(" aapl ") == "AAPL"
    assert ev.canon("BRK-B") == "BRK-B"


def test_role_weight():
    assert ev.role_weight("Officer", "Chief Executive Officer") == 2.0
    assert ev.role_weight("Officer", "VP Sales") == 1.5
    assert ev.role_weight("Director", "") == 1.0
    assert ev.role_weight("TenPercentOwner", "") == 0.75
    assert ev.role_weight("Other", "") == 0.5


def test_insider_windows_strict():
    """A filing is visible from the NEXT trading day; windows are calendar
    days; 10b5-1-flagged sales are excluded; planned buys still count."""
    ins = _ins([
        # planned buy filed day 3 — counts (exclusion is for sales only)
        dict(filing_date=DAYS[3], code="P", usd=50.0, owner="off c",
             role="Officer", title="VP", is10b5_filing=True),
        # open-market buy filed day 4
        dict(filing_date=DAYS[4], code="P", usd=1000.0, owner="ceo a",
             role="Officer", title="Chief Executive Officer",
             is10b5_filing=False),
        # open-market sale filed day 39
        dict(filing_date=DAYS[39], code="S", usd=1000.0, owner="dir b",
             role="Director", title="", is10b5_filing=False),
    ])
    close = np.full(len(DAYS), 100.0)
    shares = pd.DataFrame({"filed": [DAYS[0] - pd.Timedelta(days=10)],
                           "shares": [1e6]})
    f = ev.insider_features(ins, DAYS, shares, close)
    i = {d: k for k, d in enumerate(DAYS)}

    # filing on day D is NOT visible on D itself
    assert f["ins_buy_usd_90d"][i[DAYS[4]]] == 50.0    # only day-3 buy live
    assert f["ins_buy_usd_90d"][i[DAYS[5]]] == 1050.0  # both buys live
    assert f["ins_sell_usd_90d"][i[DAYS[39]]] == 0.0   # sell not yet public
    assert f["ins_sell_usd_90d"][i[DAYS[40]]] == 1000.0
    assert f["ins_net_usd_90d"][i[DAYS[40]]] == pytest.approx(50.0)

    # distinct buyers within 90d; cluster = distinct P owners in 30d
    assert f["ins_nbuyers_90d"][i[DAYS[5]]] == 2
    assert f["ins_cluster_30d"][i[DAYS[5]]] == 2
    # 30d later both buys still inside 30d window? c[4]+30d > c[5]+30d…
    assert f["ins_cluster_30d"][i[DAYS[40]]] == 0  # >30 calendar days gone

    # role-weighted net: 0.4*50... VP->1.5? 'VP' not in C-suite regex -> 1.5
    # role_net at day5 = 50*1.5 + 1000*2.0 = 2075
    assert f["ins_role_net_90d"][i[DAYS[5]]] == pytest.approx(2075.0)
    # mcap: 1e6 sh * 100 = 1e8; net 1050/1e8 = 1.05e-5
    assert f["ins_net_mcap_90d"][i[DAYS[5]]] == pytest.approx(1.05e-5)
    # days_since_p: last P visible as of day5 -> 0
    assert f["ins_days_since_p"][i[DAYS[5]]] == 0
    assert f["ins_days_since_p"][i[DAYS[6]]] == 1
    # window expiry: day-3 buy leaves the 90d window after fd+90
    gone = np.searchsorted(DAYS.values.astype("datetime64[D]"),
                           (DAYS[3] + pd.Timedelta(days=90))
                           .to_datetime64().astype("datetime64[D]"),
                           side="right")
    if gone < len(DAYS):
        assert f["ins_buy_usd_90d"][gone] == 0.0


def test_planned_sale_excluded():
    ins = _ins([dict(filing_date=DAYS[4], code="S", usd=1000.0,
                     owner="dir b", role="Director", title="",
                     is10b5_filing=True)])
    f = ev.insider_features(ins, DAYS, None, np.full(len(DAYS), 100.0))
    i = {d: k for k, d in enumerate(DAYS)}
    assert f["ins_sell_usd_90d"][i[DAYS[5]]] == 0
    assert f["ins_nsellers_90d"][i[DAYS[5]]] == 0
    assert f["ins_net_usd_90d"][i[DAYS[5]]] == 0
    assert f["ins_role_net_90d"][i[DAYS[5]]] == 0


def test_f13f_features():
    f13 = pd.DataFrame({
        "fund": ["fund1", "fund2", "fund1"],
        "filing_date": [DAYS[9], DAYS[19], DAYS[29]],
        "weight": [0.10, 0.05, 0.12],
    })
    f = ev.f13f_features(f13, DAYS)
    i = {d: k for k, d in enumerate(DAYS)}
    assert f["f13f_nholders"][i[DAYS[9]]] == 0    # not visible yet
    assert f["f13f_nholders"][i[DAYS[10]]] == 1
    assert f["f13f_nholders"][i[DAYS[20]]] == 2
    assert f["f13f_wsum"][i[DAYS[20]]] == pytest.approx(0.15)
    # fund1 re-files a higher weight -> latest-known value
    assert f["f13f_wsum"][i[DAYS[30]]] == pytest.approx(0.17)
    assert f["f13f_new_91d"][i[DAYS[10]]] == 1
    assert f["f13f_nholders_chg"][i[DAYS[20]]] == 2  # vs ~91d before


def test_congress_features():
    cong = pd.DataFrame({
        "filing_date": [DAYS[14], DAYS[24]],
        "amount_lo": [1001.0, 15001.0],
        "amount_hi": [15000.0, 50000.0],
        "txn_type": ["P", "S"],
    })
    f = ev.congress_features(cong, DAYS)
    i = {d: k for k, d in enumerate(DAYS)}
    assert f["cong_nbuy_45d"][i[DAYS[14]]] == 0
    assert f["cong_nbuy_45d"][i[DAYS[15]]] == 1
    assert f["cong_nsell_45d"][i[DAYS[25]]] == 1
    # midpoint 8000.5 buy then -32500.5 sale
    assert f["cong_net_usd_45d"][i[DAYS[15]]] == pytest.approx(8000.5)
    assert f["cong_net_usd_45d"][i[DAYS[25]]] == pytest.approx(8000.5 - 32500.5)


def _tiny_qlib(tmp: Path) -> Path:
    """Fake qlib dump: calendar + instruments + close bin for 'X' at 100."""
    d = tmp / "qlib"
    (d / "calendars").mkdir(parents=True)
    (d / "instruments").mkdir()
    (d / "features" / "x").mkdir(parents=True)
    DAYS.to_series().dt.strftime("%Y-%m-%d").to_csv(
        d / "calendars" / "day.txt", index=False, header=False)
    (d / "instruments" / "sp500.txt").write_text(
        f"X\t{DAYS[0].date()}\t{DAYS[-1].date()}\n")
    arr = np.concatenate([[0.0], np.full(len(DAYS), 100.0, np.float32)])
    arr.astype(np.float32).tofile(d / "features" / "x" / "close.day.bin")
    return d


def test_build_panel_end_to_end(tmp_path):
    d = _tiny_qlib(tmp_path)
    ed = tmp_path / "events"
    ed.mkdir()
    _ins([dict(filing_date=DAYS[4], code="P", usd=1000.0, owner="ceo a",
               role="Officer", title="CEO", is10b5_filing=False)]
         ).assign(symbol="X", issuer_name="Xco", issuer_cik="1",
                  txn_date=DAYS[3], shares=100.0, price=10.0
                  ).to_parquet(ed / "insider_events.parquet")
    close = np.full(len(DAYS), 100.0)
    out = tmp_path / "panel.parquet"
    panel = ev.build_panel(d, "sp500", ed, out)
    assert out.exists()
    assert panel.index.names == ["instrument", "datetime"]
    row = panel.loc[("X", DAYS[4])]
    assert row["ins_buy_usd_90d"] == 0          # filed D -> not visible on D
    assert panel.loc[("X", DAYS[5]), "ins_buy_usd_90d"] == 1000.0
    assert panel.loc[("X", DAYS[5]), "rule_ins_buy_hold60"] == 1
    # LABEL0 = close[d+2]/close[d+1]-1 = 0 for flat prices; last 2 days NaN
    assert panel.loc[("X", DAYS[10]), "LABEL0"] == pytest.approx(0.0)
    assert np.isnan(panel.loc[("X", DAYS[-1]), "LABEL0"])
    assert np.isnan(panel.loc[("X", DAYS[-2]), "LABEL0"])
    assert set(panel.columns) >= set(ev.ALL_COLS) | {"LABEL0"}


# ---------------------------------------------------------------------------
# sampled recompute of the REAL panel — the no-lookahead audit


@pytest.mark.skipif(not PANEL.exists(), reason="events panel not built")
def test_no_lookahead_recompute():
    rng = np.random.default_rng(7)
    panel = pd.read_parquet(PANEL)
    ins = pd.read_parquet(EVENTS_DIR / "insider_events.parquet")
    ins["symbol"] = ins["symbol"].map(ev.canon)
    ins["filing_date"] = pd.to_datetime(ins["filing_date"])
    n = 400
    idx = panel.index[rng.integers(0, len(panel), n)]
    for inst, d in idx:
        row = panel.loc[(inst, d)]
        past = ins[(ins["symbol"] == inst) & (ins["filing_date"] < d)]
        for w, bcol, ncol in ((90, "ins_buy_usd_90d", "ins_nbuyers_90d"),
                              (30, None, "ins_cluster_30d")):
            cut = past[past["filing_date"] >= d - pd.Timedelta(days=w)]
            buys = cut[cut["code"].astype(str).str.upper() == "P"]
            if bcol:
                assert row[bcol] == pytest.approx(
                    buys["usd"].sum(), rel=1e-6, abs=1e-4), \
                    f"{inst} {d.date()} {bcol}: {row[bcol]} != {buys['usd'].sum()}"
            if ncol:
                assert row[ncol] == buys["owner"].nunique(), \
                    f"{inst} {d.date()} {ncol}: {row[ncol]}"
        cut = past[(past["filing_date"] >= d - pd.Timedelta(days=90))]
        is_p = cut["code"].astype(str).str.upper() == "P"
        is_s_unflag = (cut["code"].astype(str).str.upper() == "S") & \
                      (~cut["is10b5_filing"].fillna(False).astype(bool))
        assert row["ins_net_usd_90d"] == pytest.approx(
            cut[is_p]["usd"].sum() - cut[is_s_unflag]["usd"].sum(),
            rel=1e-6, abs=1e-4)   # fp accumulation order differs from .sum()


@pytest.mark.skipif(not (EVENTS_DIR / "insider_events.parquet").exists(),
                    reason="insider events not fetched")
def test_insider_event_schema():
    df = pd.read_parquet(EVENTS_DIR / "insider_events.parquet")
    assert {"symbol", "filing_date", "txn_date", "code", "shares", "price",
            "usd", "role", "title", "is10b5_filing", "owner",
            "issuer_name"} <= set(df.columns)
    fd = pd.to_datetime(df["filing_date"], errors="coerce")
    td = pd.to_datetime(df["txn_date"], errors="coerce")
    ok = fd.notna() & td.notna()
    assert (fd[ok] >= td[ok]).mean() > 0.98  # filings follow the transaction
    assert (df["usd"].dropna() >= 0).all()   # magnitude; sign applied by code
