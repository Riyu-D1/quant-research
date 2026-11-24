"""Phase-9 tests — the adaptive machinery is causal and the fallback works.

All synthetic — no data files, no network. Each look-ahead test corrupts
data that a day-t decision must not see and asserts the decision is
bit-identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import phase7_us as P7  # noqa: E402
import phase8_us as P8  # noqa: E402
import phase9_us as P9  # noqa: E402

SPEC = yaml.safe_load((REPO / "configs/experiments/phase9_us.yaml").read_text())


def _panel(n_days=300, n_sym=40, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_sym)]
    return pd.DataFrame(rng.normal(size=(n_days, n_sym)), index=days,
                        columns=cols)


def _close(n_days=300, n_sym=40, seed=1):
    r = _panel(n_days, n_sym, seed) * 0.01
    return 100 * (1 + r).cumprod()


# ---------------------------------------------------------------------------
# spec contract


class TestSpec:
    def test_baseline_and_anchor_present(self):
        names = [c["name"] for c in SPEC["candidates"]]
        assert "B50_momvq" in names and "B50_micro" in names
        assert SPEC["adoption"]["baseline"] == "B50_momvq"
        assert SPEC["adoption"]["min_dsr"] >= 0.95

    def test_every_sleeve_candidate_declares_n(self):
        for c in SPEC["candidates"] + SPEC["news_candidates"]:
            if c.get("comp"):
                assert int(c["n"]) > 0, c["name"]

    def test_micro_n8_floor_arithmetic(self):
        # the audit's n=8 collapse: 0.5 * $12.70 / 8 < $1 order floor
        b = SPEC["books"]["micro"]
        per_name = 0.5 * b["cash_gbp"] * b["gbpusd"] / 8
        assert per_name < SPEC["min_order_usd"]
        p = SPEC["books"]["paper"]
        assert 0.5 * p["cash_gbp"] * p["gbpusd"] / 20 > 1000


# ---------------------------------------------------------------------------
# building blocks


class TestBlocks:
    def test_rank_ic_matches_phase8(self):
        s, f = _panel(60, 30, 2), _panel(60, 30, 3)
        s.iloc[5, :4] = np.nan
        f.iloc[7, 3:9] = np.nan
        a = P9.rank_ic(s, f)
        b = P8.daily_ic(s, f)
        pd.testing.assert_series_equal(a.dropna(), b, check_names=False,
                                       check_freq=False, atol=1e-12)

    def test_weighted_blend_equals_p7_blend(self):
        mom, vq = _panel(30, 20, 4), _panel(30, 20, 5)
        mom.iloc[3, 2] = np.nan
        vq.iloc[4, 5] = np.nan
        ranks = {"mom": P9.row_rank(mom), "vq": P9.row_rank(vq)}
        W = pd.DataFrame({"mom": 0.5, "vq": 0.5}, index=mom.index)
        got = P9.weighted_blend(ranks, W, ["mom", "vq"])
        exp = P7.blend({"mom": mom, "vq": vq}, ["mom", "vq"])
        pd.testing.assert_frame_equal(got, exp, atol=1e-12)

    def test_blend_neutral_fill_for_optional_leg(self):
        mom, vq, x = _panel(5, 6, 6), _panel(5, 6, 7), _panel(5, 6, 8)
        x.iloc[:, 0] = np.nan
        ranks = {k: P9.row_rank(v) for k, v in
                 {"mom": mom, "vq": vq, "x": x}.items()}
        W = pd.DataFrame({"mom": 1 / 3, "vq": 1 / 3, "x": 1 / 3},
                         index=mom.index)
        got = P9.weighted_blend(ranks, W, ["mom", "vq"])
        c = mom.columns[0]
        exp = (ranks["mom"][c] + ranks["vq"][c] + 0.5) / 3
        np.testing.assert_allclose(got[c].values, exp.values)

    def test_usable_shift_is_h_plus_one(self):
        days = pd.bdate_range("2021-01-01", periods=40)
        ic = pd.Series(np.arange(40.0), index=days)
        u = P9.usable(ic, days, 21)
        assert np.isnan(u.iloc[21])
        assert u.iloc[22] == 0.0 and u.iloc[39] == 17.0

    def test_month_starts(self):
        days = pd.DatetimeIndex(["2021-01-29", "2021-02-01", "2021-02-02",
                                 "2021-03-01"])
        assert P9.month_starts(days) == [0, 1, 3]


# ---------------------------------------------------------------------------
# idea 3 — online IC weights


class TestICWeights:
    cfg = {"ic_days": 60, "halflife_days": 20, "min_obs": 30, "lam": 0.5,
           "prior": {"mom": 0.5, "vq": 0.5}}

    def _ic(self, seed=0):
        rng = np.random.default_rng(seed)
        days = pd.bdate_range("2021-01-01", periods=200)
        return pd.DataFrame({"mom": 0.05 + rng.normal(0, .02, 200),
                             "vq": rng.normal(0, .02, 200),
                             "lowvol": -0.05 + rng.normal(0, .02, 200)},
                            index=days)

    def test_prior_until_min_obs(self):
        ic = self._ic()
        ic.iloc[25:] = np.nan
        W, led = P9.ic_weight_table(ic, self.cfg)
        assert all(r["src"] == "prior" for r in led)
        assert W["lowvol"].max() == 0.0
        np.testing.assert_allclose(W["mom"].dropna().unique(), [0.5])

    def test_negative_ic_leg_gets_zero_ic_share(self):
        W, led = P9.ic_weight_table(self._ic(), self.cfg)
        last = led[-1]
        assert last["src"] == "ic"
        assert last["w"]["lowvol"] == 0.0
        assert last["w"]["mom"] > 0.5            # prior 0.5 + IC share
        assert abs(sum(last["w"].values()) - 1) < 1e-9

    def test_future_ic_cannot_move_month_weights(self):
        ic = self._ic()
        W, _ = P9.ic_weight_table(ic, self.cfg)
        p = P9.month_starts(ic.index)[5]
        bad = ic.copy()
        bad.iloc[p + 1:] = 9.0
        W2, _ = P9.ic_weight_table(bad, self.cfg)
        pd.testing.assert_frame_equal(W.iloc[:p + 1], W2.iloc[:p + 1])


# ---------------------------------------------------------------------------
# idea 4 — LightGBM walk-forward is purged


class TestGBM:
    cfg = {"features": ["f1", "f2"], "stride": 1, "cv_stride": 1,
           "embargo": 2, "min_train_days": 40, "cv_min_days": 10_000,
           "cv_folds": 3,
           "base": {"learning_rate": 0.1, "num_leaves": 4, "n_estimators": 20,
                    "min_child_samples": 20, "subsample": 1.0,
                    "subsample_freq": 0, "colsample_bytree": 1.0,
                    "reg_lambda": 0.0, "seed": 0},
           "grid": [{"num_leaves": 4, "n_estimators": 20}]}
    h = 5

    def _data(self):
        f1, f2 = _panel(160, 30, 10), _panel(160, 30, 11)
        lab = P9.row_rank(f1 * 0.7 + _panel(160, 30, 12))
        valid = pd.DataFrame(True, index=f1.index, columns=f1.columns)
        return {"f1": P9.row_rank(f1), "f2": P9.row_rank(f2)}, lab, valid

    def test_no_model_before_min_train(self):
        feats, lab, valid = self._data()
        out, led = P9.gbm_composite(feats, lab, valid, self.cfg, self.h,
                                    n_jobs=1)
        first = pd.Timestamp(led[0]["month"])
        assert out.loc[:first - pd.Timedelta(days=1)].isna().all().all()
        assert out.loc[first:].notna().any().any()

    def test_labels_after_cut_cannot_change_month(self):
        feats, lab, valid = self._data()
        out, led = P9.gbm_composite(feats, lab, valid, self.cfg, self.h,
                                    n_jobs=1)
        m = led[1]
        p = out.index.get_loc(pd.Timestamp(m["month"]))
        cut = p - (self.h + 1) - self.cfg["embargo"]
        assert out.index[cut] == pd.Timestamp(m["train_last_day"])
        bad = lab.copy()
        bad.iloc[cut + 1:] = 1.0 - bad.iloc[cut + 1:]   # invert the future
        out2, _ = P9.gbm_composite(feats, bad, valid, self.cfg, self.h,
                                   n_jobs=1)
        q = P9.month_starts(out.index)
        q = q[q.index(p) + 1]
        pd.testing.assert_frame_equal(out.iloc[p:q], out2.iloc[p:q])

    def test_future_features_cannot_change_month(self):
        feats, lab, valid = self._data()
        out, led = P9.gbm_composite(feats, lab, valid, self.cfg, self.h,
                                    n_jobs=1)
        p = out.index.get_loc(pd.Timestamp(led[0]["month"]))
        q = P9.month_starts(out.index)
        q = q[q.index(p) + 1]
        bad = {k: v.copy() for k, v in feats.items()}
        for v in bad.values():
            v.iloc[q:] = 0.123
        out2, _ = P9.gbm_composite(bad, lab, valid, self.cfg, self.h,
                                   n_jobs=1)
        pd.testing.assert_frame_equal(out.iloc[p:q], out2.iloc[p:q])

    def test_purged_cv_picks_from_grid(self):
        feats, lab, valid = self._data()
        cfg = dict(self.cfg, grid=[{"num_leaves": 2, "n_estimators": 5},
                                   {"num_leaves": 4, "n_estimators": 30}])
        names = cfg["features"]
        X = np.stack([feats[f].values for f in names], -1).astype(np.float32)
        Y = lab.values.astype(np.float32)
        V = valid.values
        params, scores = P9.purged_cv(X, Y, V, np.arange(120), cfg, self.h, 1)
        assert len(scores) == 2
        assert {params["num_leaves"], params["n_estimators"]} <= {2, 4, 5, 30}


# ---------------------------------------------------------------------------
# idea 2 — overlays; drift fallback


class TestOverlays:
    def test_shadow_sleeve_uses_previous_close_picks(self):
        close = _close(10, 5, 3)
        comp = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        comp.iloc[3, 2] = 1.0       # S02 tops the composite on day 3 only
        r = P9.shadow_sleeve(comp, close, 1)
        exp = close.iloc[4, 2] / close.iloc[3, 2] - 1
        assert r.iloc[4] == pytest.approx(exp)

    def test_vol_exposure_causal_and_capped(self):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2019-01-01", periods=900)
        r = pd.Series(rng.normal(0, .01, 900), index=idx)
        r.iloc[600:700] *= 4                       # a vol spike
        cfg = SPEC["overlays"]["vol"]
        e = P9.vol_exposure(r, cfg)
        assert e.max() <= 1.0 and e.min() >= 0.0
        assert e.iloc[650:700].mean() < 0.5        # de-risks in the spike
        bad = r.copy()
        bad.iloc[700:] = 0.5
        pd.testing.assert_series_equal(e.iloc[:700],
                                       P9.vol_exposure(bad, cfg).iloc[:700])

    def test_trend_exposure_lagged(self):
        days = pd.bdate_range("2020-01-01", periods=300)
        spy = pd.Series(np.linspace(100, 200, 300), index=days)
        spy.iloc[250:] = 50.0                      # crash on day 250
        e = P9.trend_exposure(spy, days, {"sma": 200, "risk_off_mult": 0.5})
        assert e.iloc[250] == 1.0 and e.iloc[251] == 0.5


class TestDrift:
    cfg = dict(SPEC["drift"], act_win=20, ic_win=20, ic_min_obs=10,
               bxs_win=40, min_hold=5)

    def _series(self, n=200):
        idx = pd.bdate_range("2020-01-01", periods=n)
        z = pd.Series(0.0, index=idx)
        return idx, z

    def test_underperformance_triggers_b50_then_recovers(self):
        idx, z = self._series()
        r_ad = z.copy()
        r_ad.iloc[30:60] = -0.01                    # adaptive lags B50
        r_ad.iloc[60:100] = 0.01                    # then recovers
        m = P9.drift_modes(r_ad, z, z, z + 0.01, self.cfg)
        assert m["mode"].iloc[25] == "adaptive"
        assert "fallback_b50" in set(m["mode"].iloc[40:70])
        assert m["mode"].iloc[-1] == "adaptive"

    def test_negative_ic_triggers_fallback(self):
        idx, z = self._series()
        m = P9.drift_modes(z, z, z, z - 0.02, self.cfg)
        assert (m["mode"].iloc[20:] == "fallback_b50").all()

    def test_b50_collapse_routes_sleeve_to_spy(self):
        idx, z = self._series()
        r_b = z.copy()
        r_b.iloc[10:60] = -0.01                    # B50 sleeve -50pt vs SPY
        m = P9.drift_modes(r_b, r_b, z, z + 0.01, self.cfg)
        assert (m["mode"] == "fallback_spy").any()
        first = m.index[m["mode"] == "fallback_spy"][0]
        # bxs needs a full 40d window before it can fire
        assert m.index.get_loc(first) >= 39

    def test_mode_uses_only_past(self):
        idx, z = self._series()
        r_ad = z.copy()
        r_ad.iloc[30:60] = -0.01
        a = P9.drift_modes(r_ad, z, z, z + 0.01, self.cfg)
        bad = r_ad.copy()
        bad.iloc[61:] = 0.2
        b = P9.drift_modes(bad, z, z, z + 0.01, self.cfg)
        assert (a["mode"].iloc[:61] == b["mode"].iloc[:61]).all()


# ---------------------------------------------------------------------------
# adoption rule


class TestAdoption:
    def _ev(self, dsr_act, dsr_xs, year_frac=0.8, act_ar=0.02, gate=True):
        per = {"B50_momvq": {"dsr_xs": {"DSR": 0.36}},
               "X": {"gate": {"passed": gate}, "dsr_xs": {"DSR": dsr_xs},
                     "dsr_active": {"DSR": dsr_act},
                     "active": {"ar": act_ar, "year_frac": year_frac}}}
        res = {"B50_momvq": {"sleeve_signal": "b50", "book": "paper"},
               "X": {"sleeve_signal": "ens", "book": "paper"}}
        return {"per": per}, res

    def test_keeps_baseline_on_weak_dsr(self):
        ev_, res = self._ev(dsr_act=0.7, dsr_xs=0.5)
        a = P9.adoption(ev_, res, SPEC)
        assert a["keep_baseline"] and a["adopted"] is None
        assert not a["checks"]["X"]["dsr_active_ok"]

    def test_adopts_only_when_every_leg_holds(self):
        ev_, res = self._ev(dsr_act=0.97, dsr_xs=0.6)
        assert P9.adoption(ev_, res, SPEC)["adopted"] == "X"
        for kw in ({"gate": False}, {"year_frac": 0.5}, {"act_ar": -0.01},
                   {"dsr_xs": 0.2}):
            args = {"dsr_act": 0.97, "dsr_xs": 0.6} | kw
            ev_, res = self._ev(**args)
            assert P9.adoption(ev_, res, SPEC)["adopted"] is None, kw

    def test_micro_anchor_never_adoptable(self):
        ev_, res = self._ev(dsr_act=0.99, dsr_xs=0.9)
        res["X"]["book"] = "micro"
        assert P9.adoption(ev_, res, SPEC)["adopted"] is None


# ---------------------------------------------------------------------------
# end-to-end on a synthetic feed: composites -> plan -> micro sim -> eval


def _synthetic_inputs(n_days=520, n_sym=30):
    rng = np.random.default_rng(42)
    days = pd.bdate_range("2019-01-01", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_sym)]
    close = pd.DataFrame(100 * np.cumprod(
        1 + rng.normal(0.0004, 0.015, (n_days, n_sym)), axis=0),
        index=days, columns=cols)
    close["SPY"] = 300 * np.cumprod(1 + rng.normal(0.0004, 0.01, n_days))
    opn = close.shift(1).fillna(close) * (1 + rng.normal(0, 0.002,
                                                         close.shape))

    def long(df, name):
        s = df.stack()
        s.index.names = ["datetime", "instrument"]
        return s.rename(name).to_frame()
    bars = {"close": long(close, "close"), "open": long(opn, "open"),
            "volume": long(close * 0 + 1e6, "volume")}
    feed = {"bars": bars, "spy": close["SPY"], "days": days,
            "close": close, "cal": days}
    stocks = close[cols]
    panels = {k: pd.DataFrame(rng.normal(size=stocks.shape), index=days,
                              columns=cols).reindex(columns=close.columns)
              for k in ("mom", "vq", "mom6", "rev", "srev5", "lowvol",
                        "beta", "vq_bm", "vq_ep", "vq_roe")}
    ranks = {k: P9.row_rank(v) for k, v in panels.items()}
    return {"feed": feed, "smap": {}, "panels": panels, "ranks": ranks,
            "fwd": P8.fwd_wide(close, 1, 21), "cmap": None, "days": days}


def test_end_to_end_synthetic():
    spec = yaml.safe_load(yaml.safe_dump(SPEC))
    spec["gbm"].update({"min_train_days": 100, "cv_min_days": 10_000,
                        "stride": 5})
    spec["gbm"]["base"].update({"n_estimators": 10, "min_child_samples": 20})
    spec["ic_weights"].update({"min_obs": 40})
    spec["drift"].update({"act_win": 40, "bxs_win": 60, "ic_win": 40,
                          "ic_min_obs": 20})
    inp = _synthetic_inputs()
    comps, led = P9.composites(spec, inp, {"b50", "ic", "gbm", "ens"})
    assert {"b50", "ic", "gbm", "ens"} <= set(comps)
    assert led["gbm"] and led["ic_weights"]
    years = [2020]
    names = ["A_spy", "B50_momvq", "B50_micro", "R50_n15_ivol",
             "T50_trendvol", "AD50_full"]
    res = {}
    for c in spec["candidates"]:
        if c["name"] not in names:
            continue
        plan = P9.candidate_plan(spec, inp, comps, c)
        res[c["name"]] = P9.score_candidate(spec, inp, c, plan, years)
        assert res[c["name"]]["wf"]["n_days"] > 200
    spy_by = {2020: {"annualized_return": 0.1}}
    spy_wf = {"max_drawdown": -0.3}
    ev_ = P9.evaluate(res, inp["feed"]["spy"], "B50_momvq", 500, years,
                      spec, spy_by, spy_wf)
    assert "dsr_active" in ev_["per"]["AD50_full"]
    a = P9.adoption(ev_, res, spec)
    assert "B50_micro" not in a["checks"]          # micro anchor excluded
    # the 100% SPY book holds only SPY
    assert res["A_spy"]["by_year"][2020]["concentration"] is None


def test_rerun_does_not_double_count(tmp_path, monkeypatch):
    log = tmp_path / "holdout_log.jsonl"
    log.write_text('{"spec": "phase8_us.yaml", "n_trials": 392}\n'
                   '{"spec": "phase9_us.yaml", "n_trials": 600}\n')
    monkeypatch.setattr(P9, "LOG_PATH", log)
    assert P9.prior_trials_ex_phase9() == 392
    P9.drop_phase9_log_rows()
    assert log.read_text() == '{"spec": "phase8_us.yaml", "n_trials": 392}\n'
