"""Phase-7 tests — core+satellite allocator, the 2%/​$1 rebalance band,
one-order-per-symbol-per-day, a mocked multi-day weight-targets sim, the
predeclared selection rule, and an end-to-end core+satellite dry-run.
All synthetic; no network, no keys.
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

import paper_trade as pt  # noqa: E402
import phase7_us as P7  # noqa: E402

from quant_research import coresat as CS  # noqa: E402
from quant_research import micro as MC  # noqa: E402

DAYS = pd.bdate_range("2025-01-02", periods=40)
SYMS = ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
# coresat.build_targets / invvol / cap_weights


def test_build_targets_basic_sums_to_one():
    t = CS.build_targets(["A", "B", "C"], 0.50, core_sym="SPY")
    assert t["SPY"] == pytest.approx(0.50)
    assert all(t[s] == pytest.approx(0.50 / 3) for s in "ABC")
    assert sum(t.values()) == pytest.approx(1.0)


def test_build_targets_empty_or_full_core_is_all_spy():
    assert CS.build_targets([], 0.50, core_sym="SPY") == {"SPY": 1.0}
    assert CS.build_targets(["A"], 1.00, core_sym="SPY") == {"SPY": 1.0}
    # core symbol can never double-enter the sleeve
    t = CS.build_targets(["A", "SPY"], 0.0, core_sym="SPY")
    assert set(t) == {"A", "SPY"}
    assert t["A"] == pytest.approx(0.20)     # capped
    assert t["SPY"] == pytest.approx(0.80)   # residual -> core


def test_build_targets_cap_overflow_goes_to_core():
    # 1 pick holding a 0.5 sleeve would take 0.50 of equity — the 20%
    # single-stock cap binds and the overflow lands on the core
    t = CS.build_targets(["A"], 0.50, cap=0.20, core_sym="SPY")
    assert t["A"] == pytest.approx(0.20)
    assert t["SPY"] == pytest.approx(0.80)
    assert sum(t.values()) == pytest.approx(1.0)
    # a 3-pick 0.5 sleeve at ~0.167 each is under the cap — untouched
    t2 = CS.build_targets(["A", "B", "C"], 0.50, cap=0.20, core_sym="SPY")
    assert t2["A"] == pytest.approx(0.50 / 3)
    assert t2["SPY"] == pytest.approx(0.50)


def test_build_targets_risk_off_to_core_vs_cash():
    picks = ["A", "B"]
    t_core = CS.build_targets(picks, 0.50, exposure=0.0,
                              risk_off="to_core", core_sym="SPY")
    assert t_core == {"SPY": 1.0}            # whole sleeve -> core
    t_half = CS.build_targets(picks, 0.50, exposure=0.5,
                              risk_off="to_core", core_sym="SPY")
    assert t_half["SPY"] == pytest.approx(0.50 + 0.50 * 0.5)
    assert sum(t_half.values()) == pytest.approx(1.0)
    t_cash = CS.build_targets(picks, 0.50, exposure=0.0,
                              risk_off="cash", core_sym="SPY")
    assert t_cash == {"SPY": 0.50}           # freed sleeve stays cash


def test_build_targets_invvol_weights_applied():
    w = CS.build_targets(["A", "B"], 0.0,
                         invvol_w={"A": 0.75, "B": 0.25}, core_sym="SPY")
    assert w["A"] == pytest.approx(0.20)     # cap binds at 0.20
    assert w["B"] == pytest.approx(0.20)     # cap binds at 0.25->0.20
    assert w["SPY"] == pytest.approx(0.60)   # residual -> core
    w2 = CS.build_targets(["A", "B", "C", "D", "E", "F", "G", "H"], 0.0,
                          invvol_w={"A": 0.5, "B": 0.1, "C": 0.1, "D": 0.05,
                                    "E": 0.05, "F": 0.1, "G": 0.05,
                                    "H": 0.05}, core_sym="SPY")
    assert w2["A"] == pytest.approx(0.20)    # 0.50 capped at 0.20
    assert abs(sum(w2.values()) - 1.0) < 1e-9


def test_invvol_inverse_volatility():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-01-01", periods=60)
    calm = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.005, len(days))))
    wild = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.04, len(days))))
    cl = pd.DataFrame({"CALM": calm, "WILD": wild}, index=days)
    w = CS.invvol(cl, ["CALM", "WILD"], days[-1])
    assert w["CALM"] > w["WILD"] > 0
    assert sum(w.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# micro.weights_step — band, $1 floor, sells-first, one-order-per-symbol


def test_weights_step_initial_buys_hit_targets():
    prices = {"A": 10.0, "B": 20.0, "SPY": 100.0}
    targets = {"A": 0.25, "B": 0.25, "SPY": 0.50}
    st = MC.weights_step(targets, 12.70, prices, {}, 0.0, band=0.02)
    buys = {t["sym"]: t for t in st["trades"]}
    assert set(buys) == {"A", "B", "SPY"}
    assert buys["SPY"]["notional"] == pytest.approx(12.70 * 0.50, abs=0.01)
    assert buys["A"]["notional"] == pytest.approx(12.70 * 0.25, abs=0.01)


def test_weights_step_band_skips_small_drift():
    prices = {"A": 10.0, "SPY": 10.0}
    # held A at w=0.505, target 0.50 -> |dw| 0.005 < 0.02 band -> no trade
    held = {"A": 0.635}                      # $6.35 of a $12.70 book = 0.50
    targets = {"A": 0.50, "SPY": 0.50}
    st = MC.weights_step(targets, 12.70, prices, held, 0.0, band=0.02)
    assert all(t["sym"] != "A" for t in st["trades"])
    # a 15% gap on the same book clears both the band and the $1 floor
    st2 = MC.weights_step({"A": 0.65, "SPY": 0.35}, 12.70, prices,
                          dict(held), 0.0, band=0.02)
    assert any(t["sym"] == "A" and t["side"] == "buy"
               for t in st2["trades"])


def test_weights_step_exit_ignores_band():
    """Target-0 exits ALWAYS trade — Alpaca exempts full-position
    close-outs from the $1 floor, and the band is a drift filter,
    not a freeze."""
    prices = {"A": 10.0, "SPY": 10.0}
    st = MC.weights_step({"SPY": 1.0}, 12.70, prices, {"A": 0.05},
                         0.0, band=0.02)
    sells = [t for t in st["trades"]
             if t["sym"] == "A" and t["side"] == "sell"]
    assert len(sells) == 1 and "A" not in st["held"]
    st2 = MC.weights_step({"SPY": 1.0}, 12.70, prices, {"A": 0.2},
                          0.0, band=0.02)
    sells2 = [t for t in st2["trades"]
              if t["sym"] == "A" and t["side"] == "sell"]
    assert len(sells2) == 1 and "A" not in st2["held"]


def test_weights_step_min_order_floor():
    prices = {"A": 10.0, "SPY": 10.0}
    # 3% drift on a $12.70 book = $0.38 < $1 -> skipped even past the band
    st = MC.weights_step({"A": 0.53, "SPY": 0.47}, 12.70, prices,
                         {"A": 0.635}, 0.0, band=0.02)
    assert all(t["sym"] != "A" for t in st["trades"])


def test_weights_step_sells_first_one_order_per_symbol():
    prices = {"A": 10.0, "B": 10.0, "SPY": 10.0}
    held = {"A": 1.0, "B": 0.5}              # $10 + $5
    targets = {"B": 0.90, "SPY": 0.10}       # A out, B up, SPY new
    st = MC.weights_step(targets, 15.0, prices, held, 0.0, band=0.0)
    sides = [t["side"] for t in st["trades"]]
    assert sides == sorted(sides, key=lambda s: s != "sell")  # sells first
    per_sym = {}
    for t in st["trades"]:
        per_sym.setdefault(t["sym"], set()).add(t["side"])
    assert all(len(v) == 1 for v in per_sym.values())   # PDT-safe


# ---------------------------------------------------------------------------
# multi-day mocked sim — weight_targets through micro_backtest


def _bars(days, syms):
    """(datetime, instrument) close/open/volume frames, flat then drift."""
    rows_c, rows_o, rows_v = [], [], []
    for i, d in enumerate(days):
        for j, s in enumerate(syms):
            px = 10.0 * (1 + 0.001 * i) * (1 + 0.05 * j)
            rows_c.append((d, s, px))
            rows_o.append((d, s, px * 1.001))
            rows_v.append((d, s, 1e7))
    def long(rows, col):
        df = pd.DataFrame(rows, columns=["datetime", "instrument", col])
        return df.set_index(["datetime", "instrument"]).sort_index()
    return {"close": long(rows_c, "close"), "open": long(rows_o, "open"),
            "volume": long(rows_v, "volume")}


def test_micro_backtest_weight_targets_multiday():
    days = DAYS[:4]
    syms = ["A", "B", "SPY"]
    bars = _bars(days, syms)
    # day1 targets (fill day2 open): half A, half SPY.
    # day2 targets (fill day3): rotate A->B.
    tgt = {days[0]: {"A": 0.5, "SPY": 0.5},
           days[1]: {"B": 0.5, "SPY": 0.5},
           days[2]: {"B": 0.5, "SPY": 0.5},
           days[3]: {"B": 0.5, "SPY": 0.5}}
    met = MC.micro_backtest(
        pd.Series(dtype=float, index=pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
            names=["datetime", "instrument"])),
        bars["close"], exec_prices=bars["open"], fill="next_open",
        cost_bps=0.0, weight_targets=tgt, band=0.0, delist_liquidate=True)
    led = pd.DataFrame(met["trades"])
    assert not led.empty
    # nothing traded on day 1 (no prior signal day)
    assert led["date"].min() == days[1]
    d2 = led[led["date"] == days[1]]
    assert {t for t in d2["sym"]} == {"A", "SPY"}
    assert (d2["side"] == "buy").all()
    d3 = led[led["date"] == days[2]]
    assert any(t["sym"] == "A" and t["side"] == "sell"
               for _, t in d3.iterrows())
    assert any(t["sym"] == "B" and t["side"] == "buy"
               for _, t in d3.iterrows())
    # no same-day same-symbol buy+sell (PDT by construction)
    assert not (d3.groupby("sym")["side"].nunique() > 1).any()
    assert met["n_days"] == 4 and met["final_equity_usd"] > 0
    assert "daily_cost" in met and "daily_traded" in met


def test_micro_backtest_band_cuts_turnover():
    days = DAYS[:6]
    bars = _bars(days, ["A", "SPY"])
    flat_targets = {d: {"A": 0.5, "SPY": 0.5} for d in days}
    met = MC.micro_backtest(
        pd.Series(dtype=float, index=pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
            names=["datetime", "instrument"])),
        bars["close"], exec_prices=bars["open"], fill="next_open",
        cost_bps=0.0, weight_targets=flat_targets, band=0.02,
        delist_liquidate=True)
    led = pd.DataFrame(met["trades"])
    # fills happen day 2 (day-1 targets); afterwards the static targets
    # keep drift inside the 2% band -> no more trading
    assert set(led["date"].unique()) == {days[1]}


# ---------------------------------------------------------------------------
# the predeclared selection rule (unit, no data)


def _row(name, ir, mdd, roll, core=0.5):
    return {"name": name, "core": core, "sleeve_signal": "vq",
            "guard": "none", "information_ratio": ir,
            "max_drawdown": mdd, "roll12_win": roll}


def test_select_winner_rule():
    spy_oof = {"max_drawdown": -0.20}
    rows = [_row("A_spy", 0.80, -0.20, None, core=1.0),
            _row("X_good", 0.90, -0.15, 0.60),
            _row("X_deep", 1.50, -0.25, 0.90),      # mdd worse than SPY
            _row("X_losing", 1.20, -0.10, 0.40)]    # <50% roll wins
    w, sel = P7.select_winner(rows, spy_oof,
                              {"rolling_12m_min_win_rate": 0.50})
    assert w["name"] == "X_good"
    assert sel["survivors"] == ["X_good"]
    assert set(sel["dropped"]) == {"X_deep", "X_losing"}
    assert not sel["fallback_used"]


def test_select_winner_fallback_to_spy():
    spy_oof = {"max_drawdown": -0.20}
    rows = [_row("A_spy", 0.80, -0.20, None, core=1.0),
            _row("X_deep", 1.50, -0.25, 0.90)]
    w, sel = P7.select_winner(rows, spy_oof,
                              {"rolling_12m_min_win_rate": 0.50})
    assert w["name"] == "A_spy" and sel["fallback_used"]


# ---------------------------------------------------------------------------
# end-to-end: core+satellite dry-run through the real bot path


FXDAY = "2026-06-15"
FXDAYS = pd.bdate_range(end=pd.Timestamp(FXDAY), periods=620)
FSYMS = [f"M{i}" for i in range(8)] + [f"X{i}" for i in range(25)]


def _fx_closes(symbols, crash_tail=0):
    rng = np.random.default_rng(3)
    data = {}
    for s in symbols:
        drift = (0.003 + int(s[1]) * 0.0005) if s.startswith("M") else 0.0
        r = rng.normal(drift, 0.01, len(FXDAYS))
        data[s] = 30 * np.exp(np.cumsum(r)) + 10
    spy = 100 * np.exp(np.cumsum(np.full(len(FXDAYS), 0.0004)))
    if crash_tail:
        spy[-crash_tail:] = spy[-crash_tail - 1] * \
            np.cumprod(np.full(crash_tail, 0.985))
    data["SPY"] = spy
    return pd.DataFrame(data, index=FXDAYS)


def _core_cfg(tmp, **over):
    import yaml
    cfg = {"validated": False, "signals": ["mom"], "weighting": "equal",
           "ic_months": 12, "halflife_frac": 0.5, "regime": "none",
           "regime_cfg": {"sma": 200, "vol_win": 21, "vol_lookback": 504,
                          "vol_q": 0.90,
                          "actions": {"none": 1.0, "half": 0.5, "cash": 0.0,
                                      "to_core": 0.0}},
           "core": {"symbol": "SPY", "weight": 0.50},
           "sleeve": {"n": 3, "weighting": "equal", "risk_off": "to_core"},
           "n_positions": 3, "budget_gbp": 10.0, "cost_bps": 0.0,
           "rebalance_band": 0.0,
           "risk": {"max_position_weight": 0.20, "min_universe": 5,
                    "max_orders": 12, "daily_loss": 0.05,
                    "max_drawdown": 0.25, "max_daily_notional_frac": 1.0,
                    "max_price_move": 0.50},
           "universe": str(tmp / "universe.txt")}
    cfg.update(over)
    p = tmp / "core.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_core_satellite_dry_run(tmp_path):
    cl = _fx_closes(FSYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    (tmp_path / "universe.txt").write_text("\n".join(FSYMS))
    cfg = _core_cfg(tmp_path)
    broker, journal = pt.FakeBroker(), pt.FakeJournal()
    argv = sys.argv
    sys.argv = ["paper_trade.py", "--dry-run", "--config", str(cfg),
                "--fixture", str(fx), "--state-dir", str(tmp_path / "st"),
                "--date", FXDAY]
    old_b, old_j = pt.FakeBroker, pt.FakeJournal
    pt.FakeBroker, pt.FakeJournal = lambda: broker, lambda: journal
    try:
        pt.main()
    finally:
        sys.argv = argv
        pt.FakeBroker, pt.FakeJournal = old_b, old_j
    buys = {o["sym"] for o in broker.orders if o["side"] == "buy"}
    assert "SPY" in buys                      # the core buys
    assert 3 <= len(buys - {"SPY"}) <= 4      # 3 sleeve names (mom picks)
    spy_buy = next(o for o in broker.orders
                   if o["sym"] == "SPY" and o["side"] == "buy")
    assert spy_buy["notional"] == pytest.approx(12.70 * 0.50, abs=0.30)
    # single-stock cap: no sleeve buy above 20% of the $12.70 book
    sleeve_notional = [o["notional"] for o in broker.orders
                       if o["side"] == "buy" and o["sym"] != "SPY"]
    assert all(n <= 12.70 * 0.20 + 0.05 for n in sleeve_notional)


def test_core_breaker_still_buys_core(tmp_path):
    """Drawdown breaker: sleeve buys halt, sells proceed, the SPY core
    still buys — freed sleeve cash migrates into the core."""
    cl = _fx_closes(FSYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    (tmp_path / "universe.txt").write_text("\n".join(FSYMS))
    cfg = _core_cfg(tmp_path)
    held = {"M0": {"qty": 0.5, "market_value": 5.0, "avg_entry_price": 10.0}}
    broker = pt.FakeBroker(positions=held)
    journal = pt.FakeJournal()
    journal.pages.append(
        {"entry_type": "day-summary", "date": "2026-06-12",
         "recon_state": {"equity_usd": 100.0, "peak": 100.0, "spy": 500.0,
                         "cum_bot": 0.0, "cum_spy": 0.0}})
    broker.acct["portfolio_value"] = "60.00"   # -40% vs peak -> breaker
    argv = sys.argv
    sys.argv = ["paper_trade.py", "--dry-run", "--config", str(cfg),
                "--fixture", str(fx), "--state-dir", str(tmp_path / "st"),
                "--date", FXDAY]
    old_b, old_j = pt.FakeBroker, pt.FakeJournal
    pt.FakeBroker, pt.FakeJournal = lambda: broker, lambda: journal
    try:
        pt.main()
    finally:
        sys.argv = argv
        pt.FakeBroker, pt.FakeJournal = old_b, old_j
    buys = {o["sym"] for o in broker.orders if o["side"] == "buy"}
    assert buys <= {"SPY"}                    # only the core may buy
    assert any(o["sym"] == "SPY" for o in broker.orders
               if o["side"] == "buy")
    sells = [o for o in broker.orders if o["side"] == "sell"]
    assert all(o["sym"] != "SPY" for o in sells)
