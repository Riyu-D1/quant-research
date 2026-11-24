"""Phase-10 tests — multi-asset rules are causal, the books are well formed,
the holdout is sealed and the trial/adoption accounting follows the spec.

All synthetic — no data files, no network. Look-ahead tests corrupt rows a
day-t decision must not see and assert the decision is unchanged.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import phase10_us as P10  # noqa: E402
from quant_research import multiasset as MA  # noqa: E402

SPEC = yaml.safe_load((REPO / "configs/experiments/phase10_us.yaml").read_text())
EST = SPEC["est"]
U8 = SPEC["universes"]["u8"]


def _prices(n_days=700, cols=(*U8, "AGG", "BIL"), seed=0, drift=None,
            start="2016-01-04"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    mu = np.array([(drift or {}).get(c, 0.0003) for c in cols])
    vol = np.linspace(0.005, 0.02, len(cols))
    r = rng.normal(size=(n_days, len(cols))) * vol + mu
    if "BIL" in cols:
        r[:, list(cols).index("BIL")] = 0.0001
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=days,
                        columns=list(cols))


def _ctx(P):
    return MA.Ctx(P, EST, "BIL")


def _corrupt_after(P, t):
    Q = P.copy()
    Q.iloc[t + 1:] = Q.iloc[t + 1:] * np.random.default_rng(9).uniform(
        0.2, 5.0, size=Q.iloc[t + 1:].shape)
    return Q


RULES = {
    "tsmom": MA.w_tsmom(U8), "trend": MA.w_trend(U8),
    "gtaa": MA.w_gtaa(SPEC["universes"]["gtaa5"]), "gem": MA.w_gem(),
    "dualmom": MA.w_dualmom(U8, 3), "erc": MA.w_erc(U8),
    "erc_trend": MA.w_erc(U8, trend_filter=True),
    "erc_x15": MA.w_erc(U8, gross_cap=1.5),
}


# ---------------------------------------------------------------------------
# spec contract


class TestSpec:
    def test_holdout_is_last_12_months(self):
        h0 = pd.Timestamp(SPEC["holdout"]["start"])
        h1 = pd.Timestamp(SPEC["holdout"]["end"])
        assert h1 == pd.Timestamp(SPEC["sample_end"])
        assert 360 <= (h1 - h0).days <= 366

    def test_addendum_proxies_exact(self):
        names = {p["name"] for p in SPEC["proxies"]}
        assert names == {f"X_{s}" for s in ("MTUM", "QUAL", "USMV", "VLUE",
                                            "DBMF", "KMLM", "QMOM", "QVAL",
                                            "SPY", "6040")}
        sixty = next(p for p in SPEC["proxies"] if p["name"] == "X_6040")
        assert sixty["weights"] == {"SPY": 0.6, "AGG": 0.4}

    def test_every_ticker_is_fetched(self):
        have = {s for v in SPEC["etfs"].values() for s in v} | {"SPY"}
        need = {s for u in SPEC["universes"].values() for s in u}
        need |= {s for p in SPEC["proxies"] for s in p["weights"]}
        need |= {"AGG", "BIL", "EFA"}
        assert need <= have

    def test_no_leverage_outside_flagged_sensitivities(self):
        for c in SPEC["candidates"]:
            if float(c.get("gross_cap", 1.0)) > 1.0:
                assert c.get("leverage_sensitivity"), c["name"]
                assert float(c["gross_cap"]) <= 1.5
                assert not P10.eligible(SPEC, c)

    def test_combo_parts_exist_and_sum_to_one(self):
        names = {c["name"] for c in SPEC["candidates"]}
        for c in SPEC["candidates"]:
            if c["kind"] == "combo":
                assert set(c["parts"]) <= names
                assert math.isclose(sum(c["parts"].values()), 1.0)

    def test_new_single_name_books_use_pit_members(self):
        for c in SPEC["candidates"]:
            if c["kind"] in ("lowvol", "strev", "pead"):
                assert c.get("members") == "pit", c["name"]

    def test_adoption_rule_strict(self):
        a = SPEC["adoption"]
        assert a["baseline"] == "B50_momvq" and a["min_dsr"] >= 0.95
        assert a["min_proxies_beaten"] > len(SPEC["proxies"]) / 2


# ---------------------------------------------------------------------------
# estimators


class TestEstimators:
    def test_erc_equalises_risk(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(6, 6))
        cov = A @ A.T / 6 + np.eye(6) * 0.01
        w = MA.erc_weights(cov)
        rc = MA.risk_contributions(w, cov)
        assert np.all(w > 0) and math.isclose(w.sum(), 1.0)
        assert np.allclose(rc, 1 / 6, atol=1e-6)

    def test_erc_diagonal_is_inverse_vol(self):
        sig = np.array([0.05, 0.1, 0.2])
        w = MA.erc_weights(np.diag(sig ** 2))
        iv = (1 / sig) / (1 / sig).sum()
        assert np.allclose(w, iv, atol=1e-8)

    def test_vol_target_and_cap(self):
        cov = np.diag([0.04, 0.04])
        w = np.array([0.5, 0.5])                 # vol = 0.1414
        v = MA.vol_target(w, cov, 0.10, 1.0)
        assert math.isclose(MA.port_vol(v, cov), 0.10, rel_tol=1e-9)
        lo = np.diag([0.0001, 0.0001])           # tiny vol: cap binds
        assert math.isclose(MA.vol_target(w, lo, 0.10, 1.0).sum(), 1.0)
        assert math.isclose(MA.vol_target(w, lo, 0.10, 1.5).sum(), 1.5)

    def test_with_cash_fills_to_one(self):
        b = MA.with_cash({"SPY": 0.3, "TLT": 0.2, "X": 0.0}, "BIL")
        assert b == {"SPY": 0.3, "TLT": 0.2, "BIL": 0.5}
        assert "BIL" not in MA.with_cash({"SPY": 1.2}, "BIL")

    def test_month_start_mask(self):
        d = pd.bdate_range("2020-01-01", "2020-03-31")
        m = MA.month_start_mask(d)
        assert list(d[m].month) == [1, 2, 3] and m[0]


# ---------------------------------------------------------------------------
# rules: well-formed books and causality


class TestRules:
    @pytest.mark.parametrize("name", list(RULES))
    def test_book_is_long_only_and_sums(self, name):
        P = _prices()
        ctx = _ctx(P)
        cap = 1.5 if name.endswith("x15") else 1.0
        for t in (300, 450, 650):
            b = RULES[name](ctx, t)
            assert all(v >= 0 for v in b.values())
            g = sum(b.values())
            assert g <= cap + 1e-9
            if cap == 1.0:
                assert math.isclose(g, 1.0, abs_tol=1e-9), (name, b)

    @pytest.mark.parametrize("name", list(RULES))
    def test_no_lookahead(self, name):
        P = _prices(seed=3)
        t = 500
        a = RULES[name](_ctx(P), t)
        b = RULES[name](_ctx(_corrupt_after(P, t)), t)
        assert a.keys() == b.keys()
        assert all(math.isclose(a[k], b[k], rel_tol=1e-12) for k in a)

    def test_tsmom_drops_downtrend_to_cash(self):
        drift = {c: 0.001 for c in U8}
        drift["TLT"] = -0.003
        P = _prices(drift=drift, seed=4)
        b = MA.w_tsmom(U8)(_ctx(P), 600)
        assert "TLT" not in b and b.get("BIL", 0) > 0

    def test_insufficient_history_is_cash(self):
        P = _prices(n_days=100)
        for name in ("tsmom", "trend", "gem", "dualmom", "erc"):
            assert RULES[name](_ctx(P), 50) == {"BIL": 1.0}, name

    def test_gem_switches_to_bonds(self):
        drift = {c: -0.002 for c in U8}
        P = _prices(drift=drift, seed=5)
        assert MA.w_gem()(_ctx(P), 600) == {"AGG": 1.0}
        drift = {c: 0.002 for c in U8}
        drift["EFA"] = 0.004
        P = _prices(drift=drift, seed=5)
        assert MA.w_gem()(_ctx(P), 600) == {"EFA": 1.0}

    def test_sector_rotation_top_k(self):
        secs = SPEC["universes"]["sectors9"]
        drift = {s: 0.0001 * i for i, s in enumerate(secs)}
        P = _prices(cols=(*secs, "BIL"), drift=drift, seed=6)
        b = MA.w_sector(secs, 3)(_ctx(P), 600)
        assert len(b) == 3 and all(math.isclose(v, 1 / 3) for v in b.values())
        drift = {s: -0.003 for s in secs}
        P = _prices(cols=(*secs, "BIL"), drift=drift, seed=6)
        assert MA.w_sector(secs, 3, abs_filter=True)(_ctx(P), 600) == {"BIL": 1.0}


# ---------------------------------------------------------------------------
# books, combos, overlays


class TestBooks:
    def test_monthly_book_only_changes_at_month_starts(self):
        P = _prices()
        ctx = _ctx(P)
        days = P.index[300:500]
        bk = MA.monthly_book(ctx, MA.w_tsmom(U8), days)
        ms = MA.month_start_mask(P.index)[300:500]
        for i in range(1, len(days)):
            if not ms[i]:
                assert bk[days[i]] == bk[days[i - 1]]

    def test_monthly_book_rejects_off_calendar_days(self):
        P = _prices()
        with pytest.raises(ValueError):
            MA.monthly_book(_ctx(P), MA.w_hold({"SPY": 1}),
                            pd.DatetimeIndex(["1999-01-01"]))

    def test_combine_weights(self):
        d = pd.Timestamp("2020-01-02")
        out = MA.combine([(0.7, {d: {"SPY": 0.5, "AAA": 0.5}}),
                          (0.3, {d: {"TLT": 0.6, "BIL": 0.4}})])
        assert math.isclose(sum(out[d].values()), 1.0)
        assert math.isclose(out[d]["SPY"], 0.35)
        assert math.isclose(out[d]["TLT"], 0.18)

    def test_scale_book_moves_freed_share_to_cash(self):
        d = pd.Timestamp("2020-01-02")
        b = {d: {"SPY": 0.5, "AAA": 0.4, "BIL": 0.1}}
        out = MA.scale_book(b, pd.Series({d: 0.5}), "BIL")
        assert math.isclose(out[d]["SPY"], 0.25)
        assert math.isclose(out[d]["BIL"], 0.55)
        assert math.isclose(sum(out[d].values()), 1.0)

    def test_crash_exposure_causal_and_holds(self):
        P = _prices(n_days=900, seed=7)
        spy = P["SPY"].copy()
        spy.iloc[600:640] *= np.linspace(1, 0.6, 40)   # crash
        days = spy.index[300:]
        e = MA.crash_exposure(spy, days, 200, 21, 504, 0.8, 0.5, 10)
        assert set(e.unique()) <= {0.5, 1.0} and (e == 0.5).any()
        t = 700
        spy2 = spy.copy()
        spy2.iloc[t + 1:] *= 3.0
        e2 = MA.crash_exposure(spy2, days, 200, 21, 504, 0.8, 0.5, 10)
        assert e.loc[:spy.index[t]].equals(e2.loc[:spy.index[t]])
        runs = (e != e.shift()).cumsum()
        lens = e.groupby(runs).size().iloc[1:-1]
        assert (lens >= 10).all()

    def test_financing_drag_uses_previous_decision(self):
        days = pd.bdate_range("2020-01-01", periods=4)
        books = {days[0]: {"SPY": 1.5}, days[1]: {"SPY": 1.0, "BIL": 0.0},
                 days[2]: {"SPY": 1.2}, days[3]: {"SPY": 1.2}}
        cr = pd.Series(0.0001, index=days)
        dr = MA.financing_drag(books, days, cr, 0.0252, "BIL")
        assert dr.iloc[0] == 0.0
        assert math.isclose(dr.iloc[1], 0.5 * (0.0001 + 0.0001))
        assert dr.iloc[2] == 0.0
        assert math.isclose(dr.iloc[3], 0.2 * (0.0001 + 0.0001))


# ---------------------------------------------------------------------------
# single-name signals


class TestSingleName:
    def test_residual_vol_recovers_idio_vol(self):
        rng = np.random.default_rng(2)
        n = 600
        days = pd.bdate_range("2018-01-01", periods=n)
        m = rng.normal(0, 0.01, n)
        idio = np.array([0.005, 0.02])
        r = m[:, None] * np.array([1.5, 0.5]) + rng.normal(size=(n, 2)) * idio
        close = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=days,
                             columns=["A", "B"])
        mkt = pd.Series(100 * np.cumprod(1 + m), index=days)
        iv = MA.residual_vol(close, mkt, 252).iloc[-1]
        assert np.allclose(iv.values, idio * math.sqrt(252), rtol=0.15)
        close2 = close.copy()
        close2.iloc[400:] *= 2.0
        a = MA.residual_vol(close, mkt, 252).iloc[399]
        b = MA.residual_vol(close2, mkt, 252).iloc[399]
        assert np.allclose(a.values, b.values)

    @staticmethod
    def _facts():
        rows = []
        for y in range(2015, 2021):
            for q, (sm, em) in enumerate([(1, 3), (4, 6), (7, 9)]):
                start = pd.Timestamp(y, sm, 1)
                end = pd.Timestamp(y, em, 28)
                val = 100 + 10 * (y - 2015) + q + (50 if y == 2020 else 0)
                rows.append({"start": start, "end": end,
                             "filed": end + pd.Timedelta(days=35), "val": val})
        f = pd.DataFrame(rows)
        # a restatement of 2019Q1 filed in 2021 must never enter 2020Q1's SUE
        f.loc[len(f)] = {"start": pd.Timestamp(2019, 1, 1),
                         "end": pd.Timestamp(2019, 3, 28),
                         "filed": pd.Timestamp(2021, 6, 1), "val": -1e9}
        return f

    def test_sue_events_pit_and_sign(self):
        ev = MA.sue_events(self._facts(), min_hist=4)
        assert not ev.empty
        assert (ev["filed"] == ev["end"] + pd.Timedelta(days=35)).all()
        last = ev.sort_values("filed").iloc[-1]
        assert last["sue"] > 0                    # 2020 jump is a positive surprise
        assert np.isfinite(ev["sue"]).all() and abs(ev["sue"]).max() < 1e6

    def test_sue_panel_strictly_after_filing_and_ages_out(self):
        days = pd.bdate_range("2020-01-01", "2020-12-31")
        ev = pd.DataFrame({"filed": [pd.Timestamp("2020-05-05")],
                           "end": [pd.Timestamp("2020-03-31")], "sue": [2.0]})
        p = MA.sue_panel({"A": ev}, days, 63)["A"]
        assert np.isnan(p.loc["2020-05-05"]) and p.loc["2020-05-06"] == 2.0
        assert p.notna().sum() == 63

    def test_perf_basic(self):
        d = pd.Series([0.01, -0.005, 0.02],
                      index=pd.bdate_range("2020-01-01", periods=3))
        m = MA.perf(d)
        assert math.isclose(m["total"], 1.01 * 0.995 * 1.02 - 1)
        assert math.isclose(m["pnl_usd"], 100_000 * m["total"])
        assert m["mdd"] == pytest.approx(-0.005)


# ---------------------------------------------------------------------------
# harness: holdout sealing, finalists, trials, adoption


def _feed(n=600, start="2024-06-03"):
    days = pd.bdate_range(start, periods=n)
    P = _prices(n_days=n, cols=("SPY", "AAA"), start=start)
    bars = {}
    for f in ("close", "open", "volume"):
        v = P if f != "volume" else P * 0 + 1e7
        s = v.stack()
        s.index.names = ["datetime", "instrument"]
        bars[f] = s.rename(f).to_frame()
    return {"bars": bars, "spy": P["SPY"], "cal": pd.Series(days),
            "days": days, "close": P}


class TestHarness:
    def test_slice_feed_drops_holdout(self):
        feed = _feed()
        h0 = pd.Timestamp(SPEC["holdout"]["start"])
        s = P10.slice_feed(feed, h0)
        assert s["days"][-1] < h0 and s["close"].index[-1] < h0
        assert s["spy"].index[-1] < h0 and (s["cal"] < h0).all()
        for b in s["bars"].values():
            assert b.index.get_level_values("datetime").max() < h0

    def test_dev_episodes_yearly(self):
        days = pd.bdate_range("2016-01-04", "2025-11-21")
        eps = P10.dev_episodes(days, 2017)
        assert list(eps) == [str(y) for y in range(2017, 2026)]
        assert eps["2025"][-1] == pd.Timestamp("2025-11-21")

    def test_finalists_rule(self):
        summ = {c["name"]: {"sharpe": 0.1, "mdd": -0.30}
                for c in SPEC["candidates"]}
        summ["B50_momvq"] = {"sharpe": 1.2, "mdd": -0.32}
        summ["TS1_tsmom12"] = {"sharpe": 2.0, "mdd": -0.50}     # MDD too deep
        summ["LX1_erc_x15"] = {"sharpe": 3.0, "mdd": -0.10}     # never eligible
        summ["B50_pit"] = {"sharpe": 3.0, "mdd": -0.10}         # diagnostic
        summ["RP1_erc"] = {"sharpe": 0.9, "mdd": -0.12}
        summ["PF1_b50_rp30"] = {"sharpe": 1.3, "mdd": -0.25}
        summ["DM1_gem"] = {"sharpe": 0.8, "mdd": -0.33}         # within tol
        f = P10.pick_finalists(SPEC, summ)
        assert f["finalists"] == ["PF1_b50_rp30", "RP1_erc", "DM1_gem"]
        assert "TS1_tsmom12" in f["excluded_mdd"]

    def test_prior_trials_ignores_phase10_rows(self, tmp_path, monkeypatch):
        lp = tmp_path / "log.jsonl"
        lp.write_text("\n".join(json.dumps(r) for r in [
            {"spec": "phase9_us.yaml", "n_trials": 574},
            {"spec": "phase10_us.yaml", "stage": "dev", "n_trials": 800}]) + "\n")
        monkeypatch.setattr(P10, "LOG_PATH", lp)
        assert P10.prior_trials() == 574
        P10.write_log([{"spec": "phase10_us.yaml", "stage": "dev",
                        "n_trials": 760}], "dev")
        rows = [json.loads(x) for x in lp.read_text().splitlines()]
        assert [r["n_trials"] for r in rows] == [574, 760]

    def test_holdout_refuses_second_run(self, tmp_path, monkeypatch):
        hp = tmp_path / "h.json"
        hp.write_text("{}")
        monkeypatch.setattr(P10, "HOLD_OUT", hp)
        with pytest.raises(SystemExit):
            P10.cmd_holdout(SPEC)

    def test_adoption_needs_every_leg(self):
        def m(sh, mdd, dsr=None):
            return {"sharpe": sh, "mdd": mdd,
                    "dsr": {"DSR": dsr} if dsr is not None else None}
        hs = {"B50_momvq": m(1.0, -0.15), "A_spy": m(0.8, -0.10),
              "C1": m(1.6, -0.12), "C2": m(1.6, -0.30)}
        for i, p in enumerate(SPEC["proxies"]):
            hs[p["name"]] = m(0.5 + 0.2 * i, -0.1)            # 0.5 .. 2.3
        full = {"B50_momvq": m(1.2, -0.32, 0.97), "C1": m(1.3, -0.25, 0.99),
                "C2": m(1.3, -0.25, 0.99)}
        dev = {"B50_momvq": m(1.2, -0.3), "C1": m(1.25, -0.2),
               "C2": m(1.25, -0.2)}
        a = P10.adoption(SPEC, hs, full, dev, ["C1", "C2"])
        assert a["checks"]["C1"]["proxies_ok"]                 # beats 6 of 10
        assert a["adopted"] == "C1"
        assert not a["checks"]["C2"]["holdout_mdd_ok"]
        full["C1"]["dsr"]["DSR"] = 0.96                        # < B50's 0.97
        a = P10.adoption(SPEC, hs, full, dev, ["C1", "C2"])
        assert a["keep_baseline"] and a["adopted"] is None

    def test_sim_runs_etf_books_end_to_end(self):
        """A hold book and a vol-targeted book through the phase-8 sim."""
        feed = _feed(n=400, start="2020-01-01")
        etf = _prices(n_days=400, cols=(*U8[1:], "AGG", "BIL"), start="2020-01-01")
        rows = etf.stack().rename("close").reset_index()
        rows.columns = ["datetime", "instrument", "close"]
        rows["open"] = rows["close"]
        rows["volume"] = 1e7
        sf = P10.sim_feed(feed, rows)
        P = sf["close"]
        inp = {"sfeed": sf, "feed": feed, "P": P, "ctx": MA.Ctx(P, EST, "BIL"),
               "cmap": None}
        days = P.index[300:]
        spec = {**SPEC, "candidates": SPEC["candidates"]}
        hold = {"name": "X", "weights": {"SPY": 0.6, "AGG": 0.4}}
        r = P10.run_episode(spec, inp, hold, days, {}, {})
        assert r["n_trades"] >= 2 and len(r["daily"]) == len(days) - 1
        tsc = next(c for c in SPEC["candidates"] if c["name"] == "TS1_tsmom12")
        r2 = P10.run_episode(spec, inp, tsc, days, {}, {})
        assert np.isfinite(r2["daily"]).all() and r2["avg_gross_ex_cash"] <= 1.0
        lev = next(c for c in SPEC["candidates"] if c["name"] == "LX1_erc_x15")
        r3 = P10.run_episode(spec, inp, lev, days, {}, {})
        assert np.isfinite(r3["daily"]).all()
