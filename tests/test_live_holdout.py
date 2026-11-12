"""Live-bot holdout tests — parity of the vectorised PIT helpers with the
exact functions the paper bot calls (LV.momentum / LV.final_scores /
ad.regime_exposure), plus the report plumbing. All synthetic; no network,
no qlib.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import live_holdout as LH  # noqa: E402

from quant_research import live as LV  # noqa: E402
from quant_research import adaptive as ad  # noqa: E402
from quant_research import micro as mc  # noqa: E402

DAYS = pd.bdate_range("2023-01-02", "2025-06-30")
SYMS = [f"S{i}" for i in range(12)]


def fixture_closes(syms=SYMS, seed=1, gaps=False):
    rng = np.random.default_rng(seed)
    data = {}
    for i, s in enumerate(syms):
        drift = 0.0002 + (i % 5) * 0.0003
        px = 50 * np.exp(np.cumsum(rng.normal(drift, 0.012, len(DAYS))))
        data[s] = px
    df = pd.DataFrame(data, index=DAYS)
    if gaps:                     # delist-style holes -> dropna positional test
        df.iloc[300:340, 1] = np.nan
        df.iloc[400:, 2] = np.nan
    df["SPY"] = 100 * np.exp(np.cumsum(
        np.full(len(DAYS), 0.0005)))
    return df


def test_bars_score_matches_live_momentum():
    cl = fixture_closes(gaps=True).drop(columns=["SPY"])
    mom = LH.bars_score(cl, "mom").unstack("instrument")
    rev = LH.bars_score(cl, "rev").unstack("instrument")
    for d in DAYS[-60:]:
        for s in SYMS:
            lm = LV.momentum(cl[s], d)
            lr = LV.reversal(cl[s], d)
            m, r = mom.loc[d, s], rev.loc[d, s]
            assert (np.isnan(m) and np.isnan(lm)) or m == pytest.approx(lm)
            assert (np.isnan(r) and np.isnan(lr)) or r == pytest.approx(lr)


def test_composite_matches_final_scores():
    rng = np.random.default_rng(2)
    idx = pd.MultiIndex.from_product([DAYS[-40:], SYMS],
                                   names=["datetime", "instrument"])
    a = pd.Series(rng.normal(size=len(idx)), index=idx)
    b = pd.Series(rng.normal(size=len(idx)), index=idx)
    b.iloc[::7] = np.nan          # patchy coverage -> intersection semantics
    sigs = {"mom": a, "vq": b}
    months = ad.month_blocks(pd.DatetimeIndex(DAYS[-40:]))
    comp = LH.composite(sigs, [{"mom": 0.5, "vq": 0.5}] * len(months),
                        months)
    for d in DAYS[-40::9]:
        ref = LV.final_scores(sigs, {"mom": 0.5, "vq": 0.5}, d)
        got = comp.xs(d, level="datetime")
        # same covered symbols, same scores (tie ordering may differ)
        assert set(got.index) == set(ref.index)
        pd.testing.assert_series_equal(
            got.sort_index(), ref.sort_index(),
            check_names=False, check_dtype=False)


def test_monthly_weights_equal_and_ewic_fallback():
    cfg = {"weighting": "equal", "ic_months": 12, "halflife_frac": 0.5}
    months = [("m", "x"), ("m2", "x2")]
    sigs = {"mom": pd.Series(dtype=float), "vq": pd.Series(dtype=float)}
    wl = LH.monthly_weights(cfg, sigs, None, months)
    assert all(w == {"mom": 0.5, "vq": 0.5} for w in wl)
    cfg["weighting"] = "ewic"
    wl = LH.monthly_weights(cfg, sigs, pd.Series(dtype=float), months)
    assert all(w == {"mom": 0.5, "vq": 0.5} for w in wl)   # snapshot fallback


def test_roundtrips_fifo():
    tr = [{"sym": "A", "side": "buy", "qty": 1.0, "price": 10.0},
          {"sym": "A", "side": "buy", "qty": 1.0, "price": 12.0},
          {"sym": "A", "side": "sell", "qty": 1.5, "price": 15.0}]
    rt = LH.roundtrips(tr)
    assert rt["round_trips"] == 2 and rt["win_rate"] == 1.0
    assert rt["realised_pnl_usd"] == pytest.approx(1 * 5 + 0.5 * 3)


def test_run_window_regime_variants():
    cl = fixture_closes()
    cfg = {"signals": ["mom"], "weighting": "equal",
           "regime": "cash", "n_positions": 5, "budget_gbp": 10.0,
           "cost_bps": 10.0,
           "regime_cfg": {"sma": 200, "vol_win": 21, "vol_lookback": 504,
                          "vol_q": 0.90,
                          "actions": {"none": 1.0, "half": 0.5, "cash": 0.0}}}
    wide = cl.copy()
    spy = wide.pop("SPY")
    sigs = {"mom": LH.bars_score(wide, "mom").dropna()}
    long = wide.stack().rename("close").to_frame()
    long.index.names = ["datetime", "instrument"]
    days = pd.DatetimeIndex(DAYS[DAYS >= "2025-01-01"])
    kw = {"n": 5, "cash_gbp": 10.0, "gbpusd": 1.27, "cost_bps": 10.0}
    met_none, _ = LH.run_window(cfg, sigs, None, long, spy, days, "none", kw)
    met_cash, _ = LH.run_window(cfg, sigs, None, long, spy, days, "cash", kw)
    assert met_none["n_days"] == len(days)
    assert met_none["final_equity_usd"] > 0
    rep = LH.bot_metrics(met_none, 1.27, spy)
    assert set(("total_return", "final_gbp", "hit_rate", "ir_xs_spy",
                "spy", "round_trips", "win_rate")) <= set(rep)
    # calm SPY fixture -> cash regime should ~track the unguarded run here
    assert met_cash["n_days"] == len(days)


def test_spy_metrics_shape():
    cl = fixture_closes()
    spy = cl["SPY"]
    m = LH.spy_metrics(spy, pd.DatetimeIndex(DAYS[-100:]))
    assert m["n_days"] == 100 and 0 <= m["hit_rate"] <= 1
    assert m["total_return"] > 0        # monotonic up fixture


# ---------------------------------------------------------------------------
# honest-micro sim: next-open fills, delist liquidation, rebalance freq, costs


def _long(wide: pd.DataFrame, col: str) -> pd.DataFrame:
    f = wide.stack().rename(col).to_frame()
    f.index.names = ["datetime", "instrument"]
    return f


def test_next_open_fills():
    """Decisions on day D must fill at day D+1's OPEN, not D's close."""
    days = pd.bdate_range("2024-01-01", periods=6)
    cl = pd.DataFrame({"A": [10, 10, 10, 10, 10, 10],
                       "B": [10, 10, 10, 10, 10, 10]}, index=days,
                      dtype=float)
    # day-2 open gaps +10% over day-1 close — the fill must pay the open
    op = cl.copy()
    op.loc[days[1]] = [11.0, 11.0]
    idx = pd.MultiIndex.from_product([days, ["A", "B"]],
                                     names=["datetime", "instrument"])
    score = pd.Series({"A": 2.0, "B": 1.0}, name="score") \
        .reindex(idx).fillna(0.0)
    met = mc.micro_backtest(score, _long(cl, "close"),
                            exec_prices=_long(op, "open"),
                            fill="next_open", n=1, cash_gbp=10.0,
                            gbpusd=1.27, cost_bps=0.0)
    buys = [t for t in met["trades"] if t["side"] == "buy"]
    assert buys and buys[0]["date"] == days[1]     # day-1 signal -> day-2 open
    assert buys[0]["price"] == 11.0                # the OPEN, not day-1 close
    # close-fill reference: same signal fills day 1 at close 10
    met_c = mc.micro_backtest(score, _long(cl, "close"), fill="close",
                              n=1, cash_gbp=10.0, gbpusd=1.27, cost_bps=0.0)
    assert met_c["trades"][0]["date"] == days[0]
    assert met_c["trades"][0]["price"] == 10.0


def test_delist_liquidation():
    """A held name whose priced history ends must pay out at its last close,
    not zombie to zero."""
    days = pd.bdate_range("2024-01-01", periods=8)
    cl = pd.DataFrame({"A": np.linspace(10, 12, 8),
                       "B": np.full(8, 10.0)}, index=days)
    cl.loc[days[5]:, "B"] = np.nan            # B delists after day 4
    idx = pd.MultiIndex.from_product([days, ["A", "B"]],
                                     names=["datetime", "instrument"])
    score = pd.Series({"A": 1.0, "B": 2.0}).reindex(idx).fillna(0.0)
    met = mc.micro_backtest(score, _long(cl, "close"), n=2, cash_gbp=10.0,
                            gbpusd=1.27, cost_bps=0.0,
                            delist_liquidate=True)
    liq = [t for t in met["trades"] if t.get("reason") == "delist-liquidate"]
    assert len(liq) == 1 and liq[0]["sym"] == "B"
    assert liq[0]["date"] == days[5]            # first day with no bar
    assert liq[0]["price"] == 10.0              # last close, not 0
    # without liquidation the position is silently valued at 0
    met_z = mc.micro_backtest(score, _long(cl, "close"), n=2, cash_gbp=10.0,
                              gbpusd=1.27, cost_bps=0.0)
    assert met["final_equity_usd"] > met_z["final_equity_usd"]


def test_rebalance_freq():
    """weekly/monthly trade only on each period's first priced day."""
    days = pd.bdate_range("2024-01-01", periods=20)
    cl = pd.DataFrame({"A": np.linspace(10, 12, 20),
                       "B": np.linspace(12, 10, 20),
                       "C": np.linspace(11, 11, 20)}, index=days)
    idx = pd.MultiIndex.from_product([days, ["A", "B", "C"]],
                                     names=["datetime", "instrument"])
    rng = np.random.default_rng(0)
    score = pd.Series(rng.normal(size=len(idx)), index=idx)
    for freq, expect_days in (("daily", 20), ("weekly", 4), ("monthly", 1)):
        met = mc.micro_backtest(score, _long(cl, "close"), n=2,
                                cash_gbp=10.0, gbpusd=1.27, cost_bps=0.0,
                                rebalance=freq)
        tdays = {t["date"] for t in met["trades"]}
        assert len(tdays) <= expect_days
        if freq == "monthly":
            assert tdays == {days[0]}


def test_liquidity_cost_map():
    days = pd.bdate_range("2024-01-02", periods=2)
    cl = pd.DataFrame({"MEGA": [100.0, 100.0], "MID": [100.0, 100.0],
                       "ILL": [100.0, 100.0]}, index=days)
    vol = pd.DataFrame({"MEGA": [2e7, 2e7], "MID": [2e5, 2e5],
                        "ILL": [5e3, 5e3]}, index=days)
    cm = mc.liquidity_cost_map(_long(cl, "close"), _long(vol, "volume"),
                               flat_bps=10.0)
    assert cm[(days[0], "MEGA")] == pytest.approx((1.0 + 10.0) / 1e4)
    assert cm[(days[0], "MID")] == pytest.approx((5.0 + 10.0) / 1e4)
    assert cm[(days[0], "ILL")] == pytest.approx((20.0 + 10.0) / 1e4)


def test_per_name_cost_dict():
    """rebalance_step accepts {sym: frac} costs — illiquid names pay more."""
    scores = pd.Series({"A": 1.0, "B": 0.9})
    step = mc.rebalance_step(scores, 100.0, 2, {"A": 10.0, "B": 10.0}, {},
                           {"A": 0.0, "B": 0.10})
    pa = next(t for t in step["trades"] if t["sym"] == "A")["price"]
    pb = next(t for t in step["trades"] if t["sym"] == "B")["price"]
    assert pa == 10.0 and pb == pytest.approx(11.0)


def test_robustness_helpers():
    days = pd.bdate_range("2021-01-04", periods=400)
    rng = np.random.default_rng(5)
    spy = pd.Series(100 * np.exp(np.cumsum(
        rng.normal(0.0004, 0.01, len(days)))), index=days)
    bot = spy.pct_change().fillna(0) + 0.0002
    bot.iloc[0] = 0.0
    py = LH.per_year_table(bot, spy)
    assert set(py) >= {2021}
    ri = LH.rolling_xs_ir(bot, spy)
    assert ri["n_windows"] > 0 and ri["median"] > 0
    bs = LH.bootstrap_xs(bot, spy, n_boot=200)
    assert bs["ci_lo"] < bs["xs_ann_mean"] < bs["ci_hi"]
