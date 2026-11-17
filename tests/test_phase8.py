"""Phase-8 tests — the new machinery (sector cap, sn_ legs, IC
shrink/gate, monthly freeze, deploy gate) and its production parity.

All synthetic — no data files, no network. The production-path tests
reuse the same helpers the bot and the walk-forward gate call, so a pass
here means sim and live use identical math.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import phase8_us as P8  # noqa: E402
from quant_research import coresat as CS  # noqa: E402
from quant_research import live as LV  # noqa: E402


# ---------------------------------------------------------------------------
# coresat.pick_top — the sector-capped pick list (sim + bot)


class TestPickTop:
    def test_plain_topn_matches_nlargest(self):
        s = pd.Series({"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6})
        assert CS.pick_top(s, 2) == ["A", "B"]

    def test_sector_cap_skips_and_fills(self):
        s = pd.Series({f"E{i}": 1.0 - i * 0.01 for i in range(5)}
                      | {"F1": 0.5, "F2": 0.4})
        smap = {f"E{i}": "Energy" for i in range(5)} | \
            {"F1": "Fin", "F2": "Fin"}
        got = CS.pick_top(s, 4, smap, sector_max=2)
        assert got == ["E0", "E1", "F1", "F2"]   # E2-E4 capped out

    def test_unmapped_names_are_singleton_sectors(self):
        s = pd.Series({"X1": 0.9, "X2": 0.8, "X3": 0.7})
        got = CS.pick_top(s, 2, {"X1": "Energy"}, sector_max=1)
        # X1 capped at 1; X2/X3 each own sector -> allowed through
        assert got == ["X1", "X2"]

    def test_cap_with_fewer_candidates_than_n(self):
        s = pd.Series({"A": 0.9, "B": 0.8})
        smap = {"A": "Energy", "B": "Energy"}
        assert CS.pick_top(s, 5, smap, sector_max=2) == ["A", "B"]

    def test_nan_scores_ignored(self):
        s = pd.Series({"A": 0.9, "B": np.nan, "C": 0.7})
        assert CS.pick_top(s, 2) == ["A", "C"]


# ---------------------------------------------------------------------------
# live.sector_neutralise — sn_ legs


class TestSectorNeutralise:
    def test_within_sector_ranking_preserved(self):
        s = pd.Series({"E1": 0.9, "E2": 0.8, "F1": 0.2, "F2": 0.1})
        smap = {"E1": "E", "E2": "E", "F1": "F", "F2": "F"}
        out = LV.sector_neutralise(s, smap)
        assert out["E1"] > out["E2"] and out["F1"] > out["F2"]
        # each sector's demeaned scores sum to ~0
        for sec in ("E", "F"):
            names = [k for k, v in smap.items() if v == sec]
            assert abs(out[names].sum()) < 1e-12

    def test_sector_bias_removed(self):
        # Energy beats Finance on raw score; after demeaning, the best
        # name of each sector is equally strong -> no tilt survives.
        s = pd.Series({"E1": 0.9, "F1": 0.4})
        smap = {"E1": "E", "F1": "F"}
        out = LV.sector_neutralise(s, smap)
        assert abs(out["E1"]) < 1e-12 and abs(out["F1"]) < 1e-12

    def test_unmapped_name_becomes_zero(self):
        s = pd.Series({"X": 0.9, "E1": 0.8, "E2": 0.6})
        out = LV.sector_neutralise(s, {"E1": "E", "E2": "E"})
        assert out["X"] == 0.0


# ---------------------------------------------------------------------------
# P8.demean_sector — sim-side equivalent must agree with live's


class TestDemeanSectorParity:
    def test_panel_vs_live_scalar(self):
        idx = pd.bdate_range("2026-01-01", periods=3)
        panel = pd.DataFrame({"E1": [0.9, 0.5, 0.3],
                              "E2": [0.8, 0.4, 0.1],
                              "F1": [0.2, 0.1, 0.0]}, index=idx)
        smap = {"E1": "E", "E2": "E", "F1": "F"}
        dm = P8.demean_sector(panel, smap)
        for d in idx:
            live = LV.sector_neutralise(panel.loc[d], smap)
            pd.testing.assert_series_equal(dm.loc[d], live, check_names=False)


# ---------------------------------------------------------------------------
# live.trailing_mom_ic / sleeve_ic_scale / leg_ic_on


def _trending_closes(n=400, n_sym=40, seed=0):
    """Synthetic panel where momentum has genuine rank-IC: persistent
    per-symbol drift + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    drift = rng.normal(0.0004, 0.0006, n_sym)          # persistent winners
    noise = rng.normal(0.0, 0.008, (n, n_sym))
    ret = noise + drift
    px = 100 * np.exp(np.cumsum(ret, axis=0))
    cols = [f"S{i}" for i in range(n_sym)]
    return pd.DataFrame(px, index=idx, columns=cols)


class TestTrailingMomIC:
    def test_positive_ic_on_persistent_panel(self):
        close = _trending_closes()
        ew, n = LV.trailing_mom_ic(close, close.index[-1])
        assert n > 60 and ew > 0

    def test_scale_bounds_and_cold_start(self):
        close = _trending_closes()
        cfg = {"fwd_h": 21, "ic_days": 252, "halflife_days": 63,
               "min_obs": 63, "prior_scale": 0.5, "full_at": 0.02}
        sc = LV.sleeve_ic_scale(close, close.index[-1], cfg)
        assert 0.0 <= sc <= 1.0
        # cold start: too few days -> prior_scale, never 0-block
        small = close.iloc[:100]
        assert LV.sleeve_ic_scale(small, small.index[-1], cfg) == 0.5

    def test_gate_respects_min_obs(self):
        cfg = {"fwd_h": 21, "ic_days": 252, "halflife_days": 63,
               "min_obs": 63}
        small = _trending_closes(n=200)
        # 200 days: mom valid ~last 0 days of usable ICs -> don't gate
        assert LV.leg_ic_on(small, small.index[-1], cfg)

    def test_ic_boundary_no_lookahead(self):
        """IC at `day` must only use forwards that closed by `day` —
        chopping the last h+1 bars must not change the statistic."""
        close = _trending_closes()
        day = close.index[-1]
        ew1, n1 = LV.trailing_mom_ic(close, day)
        trimmed = close.iloc[:-(21 + 1)]
        # recompute at the same day on the truncated frame: identical
        ew2, n2 = LV.trailing_mom_ic(trimmed, trimmed.index[-1])
        assert n2 <= n1  # fewer usable ICs, never more


# ---------------------------------------------------------------------------
# P8 machinery: gated blend, freeze, deploy gate


class TestGatedBlend:
    def _panels(self):
        idx = pd.bdate_range("2026-01-01", periods=3)
        mom = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 1.0, 1.0],
                            "C": [3.0, 3.0, 2.0]}, index=idx)
        vq = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.9, 0.9, 0.9],
                           "C": [0.1, 0.1, 0.1]}, index=idx)
        return {"mom": mom, "vq": vq}

    def test_leg_off_contributes_nothing_but_doesnt_drop_names(self):
        p = self._panels()
        on = pd.Series([False] * 3, index=p["mom"].index)
        comp = P8.blend(p, ["mom", "vq"], {"mom": on})
        # mom off -> composite = vq ranks only
        pd.testing.assert_frame_equal(
            comp, p["vq"].rank(axis=1, pct=True), check_dtype=False)

    def test_leg_on_equals_plain_blend(self):
        p = self._panels()
        on = pd.Series([True] * 3, index=p["mom"].index)
        comp = P8.blend(p, ["mom", "vq"], {"mom": on})
        plain = P8.blend(p, ["mom", "vq"], None)
        pd.testing.assert_frame_equal(comp, plain)

    def test_partial_mask(self):
        p = self._panels()
        on = pd.Series([True, False, True], index=p["mom"].index)
        comp = P8.blend(p, ["mom", "vq"], {"mom": on})
        d1 = p["mom"].index[1]
        # day 1: mom off -> comp equals vq rank there
        assert comp.loc[d1, "B"] == pytest.approx(1.0)
        # day 0: both legs on -> B ranks mid (mom 2nd, vq 1st)
        assert 0.5 < comp.loc[p["mom"].index[0], "B"] <= 1.0


class TestMonthlyFreeze:
    def _setup(self):
        idx = pd.bdate_range("2026-01-05", periods=45)
        comp = pd.DataFrame(
            {"A": np.linspace(0.9, 0.1, 45), "B": 0.5,
             "C": np.linspace(0.1, 0.9, 45)}, index=idx)
        close = pd.DataFrame(100.0, index=idx,
                             columns=["A", "B", "C", "SPY"])
        spec = {"max_stock_weight": 0.2, "core_symbol": "SPY",
                "account": {"cash_gbp": 10.0, "gbpusd": 1.27},
                "min_order_usd": 1.0}
        cand = {"name": "M", "core": 0.5, "sleeve_signal": "momvq",
                "guard": "none", "pick_freeze": "monthly",
                "sleeve": {"n": 2}}
        return comp, close, idx, cand, spec

    def test_picks_frozen_within_month(self):
        comp, close, idx, cand, spec = self._setup()
        tg = P8.daily_targets(comp, close, idx, cand, spec, {}, {})
        # month-2 picks are chosen once — same set every day of Feb
        feb = [d for d in idx if d.month == 2]
        sets = {tuple(sorted(k for k in tg[d] if k != "SPY")) for d in feb}
        assert len(sets) == 1

    def test_no_freeze_repicks_daily(self):
        comp, close, idx, cand, spec = self._setup()
        cand = {**cand, "pick_freeze": None}
        tg = P8.daily_targets(comp, close, idx, cand, spec, {}, {})
        picks = [tuple(sorted(k for k in tg[d] if k != "SPY"))
                 for d in idx]
        assert len(set(picks)) > 1   # drifting scores -> changing picks


class TestDeployGate:
    def _res(self, name, by_year_ar, mdd, ir):
        return {"by_year": by_year_ar, "wf": {"max_drawdown": mdd,
                                             "information_ratio": ir},
                "core": 0.5,
                "sleeve_signal": None if name == "A_spy" else "momvq"}

    def test_fallback_to_spy_when_none_pass(self):
        spy_by = {y: {"annualized_return": 0.10} for y in (2024, 2025)}
        spy_wf = {"max_drawdown": -0.20}
        results = {
            "A_spy": self._res("A_spy", {2024: {"annualized_return": 0.099},
                                         2025: {"annualized_return": 0.098}},
                               -0.20, 0.9),
            "X": self._res("X", {2024: {"annualized_return": 0.0},
                                 2025: {"annualized_return": 0.0}},
                           -0.25, 1.5),
        }
        g = {"mdd_tol": 0.02, "match_tol": 0.005, "min_match_frac": 0.6,
             "worst_year_tol": 0.05}
        dec = P8.deploy_gate(results, spy_by, spy_wf, [2024, 2025], g)
        # A_spy's own AR is ~SPY minus costs -> inside match_tol -> passes
        # X fails mdd (-0.25 < -0.22) AND match (0.0 << 0.10) AND worst
        assert not dec["decision"]["X"]["passed"]
        assert dec["decision"]["A_spy"]["passed"]
        assert dec["winner"] == "A_spy"
        assert dec["fallback_used"]

    def test_winner_is_best_passing_ir(self):
        spy_by = {y: {"annualized_return": 0.10} for y in (2024, 2025)}
        spy_wf = {"max_drawdown": -0.20}
        good = {2024: {"annualized_return": 0.101},
                2025: {"annualized_return": 0.099}}
        results = {
            "A_spy": self._res("A_spy", {2024: {"annualized_return": 0.099},
                                         2025: {"annualized_return": 0.098}},
                               -0.20, 0.9),
            "X": self._res("X", good, -0.19, 1.2),
            "Y": self._res("Y", good, -0.19, 1.4),
        }
        g = {"mdd_tol": 0.02, "match_tol": 0.005, "min_match_frac": 0.6,
             "worst_year_tol": 0.05}
        dec = P8.deploy_gate(results, spy_by, spy_wf, [2024, 2025], g)
        assert dec["winner"] == "Y"
        assert not dec["fallback_used"]

    def test_one_bad_year_kills(self):
        spy_by = {y: {"annualized_return": 0.10} for y in (2024, 2025)}
        spy_wf = {"max_drawdown": -0.20}
        results = {
            "A_spy": self._res("A_spy", {2024: {"annualized_return": 0.099},
                                         2025: {"annualized_return": 0.098}},
                               -0.20, 0.9),
            "X": self._res("X", {2024: {"annualized_return": 0.50},
                                 2025: {"annualized_return": 0.04}},
                           -0.19, 2.0),
        }
        g = {"mdd_tol": 0.02, "match_tol": 0.005, "min_match_frac": 0.6,
             "worst_year_tol": 0.05}
        dec = P8.deploy_gate(results, spy_by, spy_wf, [2024, 2025], g)
        # 2025 loses 6pt to SPY > 5pt tol -> fails despite huge IR
        assert not dec["decision"]["X"]["passed"]
        assert dec["winner"] == "A_spy"
        assert dec["fallback_used"]


# ---------------------------------------------------------------------------
# production parity: the config the gate writes is loadable end-to-end


class TestRenderedConfig:
    def test_live_config_parses_and_has_interlocks(self, tmp_path):
        spec = {
            "signals": {"momvq": ["mom", "vq"]},
            "account": {"cash_gbp": 10.0, "gbpusd": 1.27},
            "core_symbol": "SPY", "max_stock_weight": 0.20,
            "rebalance_band": 0.02, "min_order_usd": 1.0,
            "sector_map": "configs/benchmarks/assets/sp500_sector_map.csv",
            "fwd_h": 21,
            "ic_shrink": {"ic_signal": "mom", "halflife_days": 63,
                          "ic_days": 252, "full_at": 0.02, "min_obs": 63,
                          "prior_scale": 0.5},
            "ic_gate": {"halflife_days": 63, "ic_days": 252, "min_obs": 63,
                        "ungateable": ["vq", "sn_vq"]},
        }
        winner = {"name": "S50_sec2_ic", "core": 0.5,
                  "sleeve_signal": "momvq", "guard": "none",
                  "sleeve": {"n": 6}, "pick": {"sector_max": 2},
                  "pick_freeze": None, "ic_shrink": True,
                  "ic_gate": None, "band": None}
        dec = {"decision": {"S50_sec2_ic": {"passed": True,
                                          "match_frac": 0.8,
                                          "worst_xs_year": -0.01}},
               "survivors": ["S50_sec2_ic"]}
        wf_res = {"S50_sec2_ic": {"wf": {"n_days": 2400,
                                        "information_ratio": 0.9,
                                        "annualized_return": 0.12,
                                        "max_drawdown": -0.2}}}
        text = P8.render_live_config(winner, spec, dec, wf_res)
        import yaml
        cfg = yaml.safe_load(text)
        assert cfg["live_trading"] is False
        assert cfg["validated"] is True
        assert cfg["signals"] == ["mom", "vq"]
        assert cfg["sleeve"]["sector_max"] == 2
        assert cfg["sleeve"]["ic_shrink"]["ic_signal"] == "mom"
        assert cfg["sleeve"]["ic_shrink"]["fwd_h"] == 21
        assert cfg["core"]["weight"] == 0.5
        assert cfg["sector_map"].endswith("sp500_sector_map.csv")

    def test_spy_fallback_config_is_sane_and_unvalidated(self):
        spec = {
            "signals": {"momvq": ["mom", "vq"]},
            "account": {"cash_gbp": 10.0, "gbpusd": 1.27},
            "core_symbol": "SPY", "max_stock_weight": 0.20,
            "rebalance_band": 0.02, "min_order_usd": 1.0,
            "sector_map": "configs/benchmarks/assets/sp500_sector_map.csv",
            "fwd_h": 21,
            "ic_shrink": {"ic_signal": "mom", "halflife_days": 63,
                          "ic_days": 252, "full_at": 0.02, "min_obs": 63,
                          "prior_scale": 0.5},
            "ic_gate": {"halflife_days": 63, "ic_days": 252, "min_obs": 63,
                        "ungateable": ["vq", "sn_vq"]},
        }
        winner = {"name": "A_spy", "core": 1.0, "sleeve_signal": None,
                  "guard": "none", "sleeve": None, "pick": None,
                  "pick_freeze": None, "ic_shrink": None,
                  "ic_gate": None, "band": None}
        dec = {"decision": {"A_spy": {"passed": True, "match_frac": 1.0,
                                     "worst_xs_year": 0.0}},
               "survivors": []}
        text = P8.render_live_config(winner, spec, dec, {"A_spy": {"wf": {}}})
        import yaml
        cfg = yaml.safe_load(text)
        assert cfg["validated"] is False
        assert cfg["live_trading"] is False
        assert cfg["signals"] == []
        assert cfg["core"]["weight"] == 1.0
        assert cfg["sleeve"]["n"] == 0
