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
