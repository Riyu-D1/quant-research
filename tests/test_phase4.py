"""Phase-4 tests: adaptive engine correctness + no-lookahead audits on
tiny synthetic fixtures. No qlib / no real data needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quant_research import adaptive as ad
from quant_research import fundamentals as F
from quant_research import micro as mc
from quant_research import news as nw

DAYS = pd.DatetimeIndex(pd.bdate_range("2021-01-01", "2021-06-30"))


def _grid(cols, days=DAYS, insts=("A", "B", "C")):
    idx = pd.MultiIndex.from_product([days, list(insts)],
                                     names=["datetime", "instrument"])
    return pd.DataFrame(np.random.default_rng(0).normal(size=(len(idx), cols)),
                        index=idx)


# ---------------------------------------------------------------------------
# calendar + scoring helpers


def test_month_blocks():
    months = ad.month_blocks(DAYS)
    assert len(months) == 6
    for m0, m1 in months:
        assert m0.month == m1.month
        assert m0.day <= 3 and m1.day >= 25
    assert months[0][0] == DAYS[0] and months[-1][1] == DAYS[-1]


def test_ew_ic_weights_floor_and_pit():
    ic = pd.DataFrame({"a": np.full(len(DAYS), 0.02),
                       "b": np.full(len(DAYS), -0.01)}, index=DAYS)
    w = ad.ew_ic_weights(ic, DAYS[120], months=3, halflife_days=30,
                         signals=["a", "b"])
    assert w["a"] == pytest.approx(1.0) and w["b"] == 0.0
    # all-nonpositive -> all-zero flat month
    w2 = ad.ew_ic_weights(ic.assign(a=-0.02), DAYS[120], 3, 30, ["a", "b"])
    assert (w2 == 0).all()
    # no-lookahead: days >= `before` must not matter
    ic_future = ic.copy()
    ic_future.loc[DAYS[120]:, "a"] = -99.0
    w3 = ad.ew_ic_weights(ic_future, DAYS[120], months=3, halflife_days=30,
                          signals=["a", "b"])
    pd.testing.assert_series_equal(w, w3)


def test_monthly_weights_modes():
    hist = pd.DataFrame({"a": 0.01, "b": 0.02}, index=DAYS)
    months = ad.month_blocks(DAYS)
    eq = ad.monthly_weights(hist, months, "equal", 3, 30, ["a", "b"])
    assert all(abs(v["a"] - 0.5) < 1e-9 for v in eq)
    sg = ad.monthly_weights(hist, months, "single:b", 3, 30, ["a", "b"])
    assert all(v["b"] == 1.0 and v["a"] == 0.0 for v in sg)


def test_rank_combine_neutral_fill():
    r = pd.DataFrame({"a": [0.9, np.nan], "b": [0.1, 0.4]},
                     index=pd.MultiIndex.from_tuples(
                         [(DAYS[0], "X"), (DAYS[0], "Y")],
                         names=["datetime", "instrument"]))
    w = pd.Series({"a": 1.0, "b": 0.0})
    s = ad.rank_combine(r, w)
    assert s.iloc[0] == pytest.approx(0.9)
    assert s.iloc[1] == pytest.approx(0.5)      # missing -> neutral 0.5


# ---------------------------------------------------------------------------
# regime guard


def test_regime_exposure_sma_only():
    spy = pd.Series(np.r_[np.full(300, 100.0), np.linspace(99, 50, 60)],
                    index=pd.bdate_range("2020-01-01", periods=360))
    e = ad.regime_exposure(spy, sma=200, vol_win=21, vol_lookback=504,
                           vol_q=0.99, action=0.5)   # vol leg ~never fires
    assert (e.iloc[:300] == 1.0).all()
    assert e.iloc[-1] == 0.5                        # deep under the SMA
    # the guard must not see same-day close: first day below the MA is still 1.0
    below = spy.index[spy < spy.rolling(200, min_periods=100).mean()][0]
    assert e.loc[below] == 1.0
    nxt = spy.index[spy.index.get_loc(below) + 1]
    assert e.loc[nxt] == 0.5


def test_regime_exposure_vol():
    # deterministic calm period (constant |ret| = 0.005) then a vol spike;
    # sma=10000 makes the SMA leg inert so only the vol leg can fire.
    calm = np.tile([0.005, -0.005], 350)
    rng = np.random.default_rng(1)
    rets = np.r_[calm, rng.normal(0, 0.06, 100)]
    spy = pd.Series(100 * np.exp(np.cumsum(rets)),
                    index=pd.bdate_range("2019-01-01", periods=800))
    e = ad.regime_exposure(spy, sma=10000, vol_win=21, vol_lookback=504,
                           vol_q=0.90, action=0.0)
    assert e.iloc[-1] == 0.0                        # recent vol spike -> cash
    assert e.iloc[400] == 1.0                       # calm day -> invested


# ---------------------------------------------------------------------------
# monthly LGBM retraining — PIT audit


def test_month_lgbm_no_eval_leak():
    rng = np.random.default_rng(3)
    X = _grid(4, insts=("A", "B"))
    y = pd.Series(X[0].values * 0.5 + rng.normal(0, .1, len(X)), index=X.index)
    m0, m1 = pd.Timestamp("2021-05-03"), pd.Timestamp("2021-05-31")
    p1 = ad.month_lgbm(X, y, m0, m1, train_months=4, valid_days=10)
    assert not p1.empty
    assert (p1.index.get_level_values("datetime") >= m0).all()
    assert (p1.index.get_level_values("datetime") <= m1).all()
    # corrupt eval-window labels + post-window features: preds must be identical
    y_bad = y.copy()
    y_bad[y_bad.index.get_level_values("datetime") >= m0] = 1e6
    X_bad = X.copy()
    X_bad[X_bad.index.get_level_values("datetime") > m1] = 0.0
    p2 = ad.month_lgbm(X_bad, y_bad, m0, m1, train_months=4, valid_days=10)
    pd.testing.assert_series_equal(p1, p2)
    # train rows must stop before m0: shrink window edge case
    p3 = ad.month_lgbm(X, y, DAYS[5], DAYS[10], train_months=1)
    assert p3.empty                                     # <30 train days -> skip


# ---------------------------------------------------------------------------
# micro sim exposure


def test_micro_exposure_scaling():
    days = pd.bdate_range("2021-02-01", periods=6)
    idx = pd.MultiIndex.from_product([days, ["A", "B"]],
                                     names=["datetime", "instrument"])
    pred = pd.Series(1.0, index=idx)
    closes = pd.DataFrame({"close": 100.0}, index=idx)
    exp = pd.Series(0.5, index=days)
    r = mc.micro_backtest(pred, closes, n=2, cash_gbp=10, gbpusd=1.0,
                          cost_bps=0.0, exposure=exp)
    # ~half of ~$10 invested across 2 names -> trades ~2.5 each
    buys = [t for t in r["trades"] if t["side"] == "buy"]
    assert buys and all(t["notional"] == pytest.approx(2.5, abs=0.01)
                        for t in buys[:2])
    r0 = mc.micro_backtest(pred, closes, n=2, cash_gbp=10, gbpusd=1.0,
                           cost_bps=0.0,
                           exposure=pd.Series(0.0, index=days))
    assert not any(t["side"] == "buy" for t in r0["trades"])
    assert r0["final_equity_usd"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# news daily signal — visibility strictly after publish date


def test_news_visibility_and_decay():
    days = pd.bdate_range("2021-03-01", periods=10)
    store = pd.DataFrame(
        index=pd.MultiIndex.from_product([["A"], days],
                                         names=["instrument", "datetime"]))
    daily = pd.DataFrame({"symbol": ["A"], "pub_date": [days[2]],
                          "mean_score": [0.8], "n": [2]})
    s = nw.daily_signal(daily, store, halflife_td=2.0)
    assert np.isnan(s.loc[days[2], "A"])            # published day 2 -> not visible
    assert s.loc[days[3], "A"] == pytest.approx(0.8)
    assert s.loc[days[4], "A"] == pytest.approx(0.8)   # decay affects weights,
    assert s.loc[days[9], "A"] == pytest.approx(0.8)   # not the EW mean itself


def test_news_weekend_publish_lands_monday():
    days = pd.bdate_range("2021-03-01", periods=10)
    sat = days[4] + pd.Timedelta(days=1)            # Fri 03-05 + 1d = Sat
    store = pd.DataFrame(
        index=pd.MultiIndex.from_product([["A"], days],
                                         names=["instrument", "datetime"]))
    daily = pd.DataFrame({"symbol": ["A"], "pub_date": [sat],
                          "mean_score": [0.5], "n": [1]})
    s = nw.daily_signal(daily, store, halflife_td=2.0)
    assert np.isnan(s.loc[days[4], "A"])            # not visible Friday
    assert s.loc[days[5], "A"] == pytest.approx(0.5)  # visible Monday


# ---------------------------------------------------------------------------
# fundamentals — filed < D visibility


def test_vq_point_in_time():
    days = pd.bdate_range("2021-01-01", periods=20)
    facts = pd.DataFrame({
        "group": ["book"] * 4 + ["ni"] * 4,
        "tag": ["StockholdersEquity"] * 4 + ["NetIncomeLoss"] * 4,
        "unit": ["USD"] * 8,
        "start": [pd.NaT] * 4 + list(days[[0, 5, 10, 15]] - pd.Timedelta(days=80)),
        "end": [days[0]] * 4 + list(days[[4, 9, 14, 18]]),
        "filed": [days[2]] * 4 + list(days[[5, 10, 15, 19]]),
        "fy": [2020] * 8, "fp": ["FY"] * 4 + ["Q1", "Q2", "Q3", "Q4"],
        "form": ["10-K"] * 8,
        "val": [100.0, 100.0, 100.0, 100.0, 5.0, 5.0, 5.0, 5.0]})
    shares = pd.DataFrame({"filed": [days[0]], "shares": [10.0]})
    close = np.full(len(days), 10.0)
    vq = F.vq_features(facts, shares, close, days)
    assert np.isnan(vq["vq_bm"][2])
    assert vq["vq_bm"][3] == pytest.approx(100.0 / (10 * 10.0))
    # TTM needs 4 quarterly facts -> last day is the first complete one
    assert vq["vq_ep"][-1] == pytest.approx(20.0 / 100.0)


def test_vq_no_lookahead_corruption():
    """The audit that fails on look-ahead: mutating or adding facts filed
    AFTER day D must not change any vq value on days <= D."""
    days = pd.bdate_range("2021-01-01", periods=20)
    facts = pd.DataFrame({
        "group": ["book"] * 4 + ["ni"] * 4,
        "tag": ["StockholdersEquity"] * 4 + ["NetIncomeLoss"] * 4,
        "unit": ["USD"] * 8,
        "start": [pd.NaT] * 4 + list(days[[0, 5, 10, 15]] - pd.Timedelta(days=80)),
        "end": [days[0]] * 4 + list(days[[4, 9, 14, 18]]),
        "filed": [days[2]] * 4 + list(days[[5, 10, 15, 19]]),
        "fy": [2020] * 8, "fp": ["FY"] * 4 + ["Q1", "Q2", "Q3", "Q4"],
        "form": ["10-K"] * 8,
        "val": [100.0, 100.0, 100.0, 100.0, 5.0, 5.0, 5.0, 5.0]})
    shares = pd.DataFrame({"filed": [days[0]], "shares": [10.0]})
    close = np.full(len(days), 10.0)
    cut = 10
    v0 = F.vq_features(facts, shares, close, days)
    # corrupt every value filed after `cut`, and add a huge future filing
    fut = facts.copy()
    fut.loc[fut["filed"] > days[cut], "val"] = 999.0
    extra = pd.DataFrame({
        "group": ["book"], "tag": ["StockholdersEquity"], "unit": ["USD"],
        "start": [pd.NaT], "end": [days[19]], "filed": [days[19]],
        "fy": [2021], "fp": ["FY"], "form": ["10-K"], "val": [1e12]})
    v1 = F.vq_features(pd.concat([fut, extra]), shares, close, days)
    for c in ("vq_bm", "vq_ep", "vq_roe"):
        assert np.allclose(v0[c][:cut + 1], v1[c][:cut + 1],
                           equal_nan=True), c


def test_parse_companyfacts_shape():
    payload = {"facts": {"us-gaap": {
        "StockholdersEquity": {"units": {"USD": [
            {"end": "2020-12-31", "filed": "2021-02-01", "val": 1e9,
             "fy": 2020, "fp": "FY", "form": "10-K"}]}},
        "NetIncomeLoss": {"units": {"USD": [
            {"start": "2020-10-01", "end": "2020-12-31", "filed": "2021-02-01",
             "val": 5e7, "fy": 2020, "fp": "Q4", "form": "10-Q"}]}}}}}
    df = F.parse_companyfacts(payload)
    assert set(df["group"]) == {"book", "ni"}
    assert pd.api.types.is_datetime64_any_dtype(df["filed"])
