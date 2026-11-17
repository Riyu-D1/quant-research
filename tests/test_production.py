"""Production-rule tests — every gate in scripts/paper_trade.py plus a
mocked multi-day simulation (enter, hold, partial fill, rejection, delist,
halt, drawdown breaker, holiday). Zero keys required.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import paper_trade as pt  # noqa: E402
from quant_research import alpaca_bot as AB  # noqa: E402


# ---------------------------------------------------------------------------
# client_order_id


def test_client_order_id_deterministic():
    a = AB.client_order_id("2026-01-05", "AAPL", "buy")
    b = AB.client_order_id(pd.Timestamp("2026-01-05"), "AAPL", "buy")
    c = AB.client_order_id("2026-01-05", "AAPL", "sell")
    assert a == b and a != c and len(a) <= 48


def test_client_order_id_long_symbol_truncated():
    cid = AB.client_order_id("2026-01-05", "VERY-LONG-TICKER-" * 5, "buy")
    assert len(cid) <= 48


def test_paper_client_duplicate_coid_maps_to_order_exists(monkeypatch):
    c = AB.PaperClient("k", "s")

    class R:
        ok = False
        status_code = 422
        text = '{"code":40010001,"message":"client_order_id must be unique"}'

    monkeypatch.setattr(AB.requests, "request", lambda *a, **k: R())
    with pytest.raises(AB.OrderExistsError):
        c._req("POST", "/v2/orders", json={})


def test_paper_client_refuses_live_url():
    with pytest.raises(RuntimeError):
        AB.PaperClient("k", "s", "https://api.alpaca.markets").account()


def test_paper_client_live_url_needs_flag():
    c = AB.PaperClient("k", "s", "https://api.alpaca.markets",
                       allow_live=True)
    assert "api.alpaca" in c.base       # constructed; gate was opt-in


# ---------------------------------------------------------------------------
# order_window


class _ClockBroker(pt.FakeBroker):
    def __init__(self, is_open=True, cal=None, **kw):
        super().__init__(**kw)
        self.clock_ret = {"is_open": is_open}
        self._cal = cal

    def calendar(self, start, end):
        if self._cal is not None:
            return self._cal
        return super().calendar(start, end)


def _et(d, hm):
    return pd.Timestamp(f"{d} {hm}").tz_localize("America/New_York")


def test_order_window_open_regular_day():
    b = _ClockBroker(cal=[{"date": "2026-01-06", "open": "09:30",
                           "close": "16:00"}])
    ok, note = pt.order_window(b, _et("2026-01-06", "10:00"), {})
    assert ok


def test_order_window_before_open_plus_buffer():
    b = _ClockBroker(cal=[{"date": "2026-01-06", "open": "09:30",
                           "close": "16:00"}])
    ok, note = pt.order_window(b, _et("2026-01-06", "09:40"), {})
    assert not ok and "before order window" in note


def test_order_window_after_close():
    b = _ClockBroker(is_open=False)
    ok, _ = pt.order_window(b, _et("2026-01-06", "16:30"), {})
    assert not ok


def test_order_window_half_day_early_close():
    # 13:00 ET half-day close — 14:00 ET must be refused even if the
    # clock endpoint somehow still says open
    b = _ClockBroker(cal=[{"date": "2026-11-27", "open": "09:30",
                           "close": "13:00"}])
    ok, note = pt.order_window(b, _et("2026-11-27", "14:00"), {})
    assert not ok and "close" in note
    ok2, _ = pt.order_window(b, _et("2026-11-27", "10:00"), {})
    assert ok2


def test_order_window_clock_failure_is_fail_closed():
    class Boom(_ClockBroker):
        def clock(self):
            raise RuntimeError("api down")

    ok, note = pt.order_window(Boom(), _et("2026-01-06", "10:00"), {})
    assert not ok and "failed" in note


# ---------------------------------------------------------------------------
# pdt_guard


def test_pdt_blocks_same_day_flip():
    today = [{"symbol": "AAPL", "side": "buy", "status": "filled",
              "submitted_at": "2026-01-06T15:00Z"}]
    trades = [{"sym": "AAPL", "side": "sell"},
              {"sym": "MSFT", "side": "sell"}]
    kept, blocked = pt.pdt_guard(today, trades)
    assert [t["sym"] for t in kept] == ["MSFT"]
    assert any("PDT" in b and "AAPL" in b for b in blocked)


def test_pdt_ignores_cancelled_prior_orders():
    today = [{"symbol": "AAPL", "side": "buy", "status": "canceled"}]
    kept, blocked = pt.pdt_guard(today, [{"sym": "AAPL", "side": "sell"}])
    assert len(kept) == 1 and not blocked


def test_pdt_within_batch_impossible():
    kept, _ = pt.pdt_guard([], [{"sym": "X", "side": "buy"},
                               {"sym": "X", "side": "sell"}])
    assert [t["side"] for t in kept] == ["buy"]   # sell dropped


# ---------------------------------------------------------------------------
# risk_gate


def _closes(n=3):
    days = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame({s: np.linspace(10, 11, n)
                         for s in ["A", "B", "C"]}, index=days)


def _cfg(**rk):
    return {"risk": rk}


def test_risk_drawdown_breaker_blocks_buys_only():
    step = {"trades": [{"sym": "A", "side": "sell", "qty": 1, "price": 10,
                        "notional": 0},
                       {"sym": "B", "side": "buy", "notional": 50,
                        "price": 11}]}
    prev = {"peak": 100.0, "equity_usd": 90.0}
    g = pt.risk_gate(_cfg(max_drawdown=0.25), None, 70.0, prev, step,
                     {"A": 10, "B": 11, "C": 10}, _closes(),
                     _closes().index[-1])
    assert g["order_ok"] and not g["buy_ok"]
    assert any("drawdown" in n for n in g["notes"])


def test_risk_daily_loss_breaker():
    prev = {"peak": 100.0, "equity_usd": 100.0}
    step = {"trades": [{"sym": "A", "side": "buy", "notional": 10,
                        "price": 10}]}
    g = pt.risk_gate(_cfg(daily_loss=0.05), None, 93.0, prev, step,
                     {"A": 10}, _closes(), _closes().index[-1])
    assert not g["buy_ok"] and any("daily" in n for n in g["notes"])


def test_risk_notional_cap_aborts():
    step = {"trades": [{"sym": "A", "side": "buy", "notional": 200,
                        "price": 10}]}
    g = pt.risk_gate(_cfg(max_daily_notional_frac=1.0), None, 100.0, {},
                     step, {"A": 10}, _closes(), _closes().index[-1])
    assert not g["order_ok"] and "notional" in g["checks"]


def test_risk_price_sanity_aborts():
    days = pd.bdate_range("2026-01-01", periods=3)
    c = pd.DataFrame({"A": [10.0, 10.0, 20.0],   # +100% — broken/bad bar
                      "B": [10.0, 10.0, 10.1]}, index=days)
    step = {"trades": [{"sym": "A", "side": "buy", "notional": 10,
                        "price": 20}]}
    g = pt.risk_gate(_cfg(max_price_move=0.5), None, 100.0, {}, step,
                     {"A": 20, "B": 10.1}, c, days[-1])
    assert not g["order_ok"] and "price_sanity" in g["checks"]


def test_risk_max_orders_caps_buys_keeps_sells():
    trades = ([{"sym": f"S{i}", "side": "sell", "qty": 1, "price": 1,
                "notional": 0} for i in range(3)] +
              [{"sym": f"B{i}", "side": "buy", "notional": 1, "price": 1}
               for i in range(20)])
    step = {"trades": trades}
    g = pt.risk_gate(_cfg(max_orders=12), None, 100.0, {}, step,
                     {"S0": 1}, _closes(), _closes().index[-1])
    assert len(step["trades"]) == 12
    assert sum(t["side"] == "sell" for t in step["trades"]) == 3


def test_risk_position_weight_clip():
    step = {"trades": [{"sym": "A", "side": "buy", "notional": 90,
                        "price": 10}]}
    g = pt.risk_gate(_cfg(max_position_weight=0.4), None, 100.0, {}, step,
                     {"A": 10}, _closes(), _closes().index[-1])
    assert step["trades"][0]["notional"] == 40.0


# ---------------------------------------------------------------------------
# corp_actions + drift


def test_corp_action_closes_vanished_position():
    j = pt.FakeJournal()
    j.pages.append({"entry_type": "trade", "date": "2026-01-01",
                    "sym": "GONE", "side": "buy"})
    notes = pt.corp_actions(j, {}, pd.Timestamp("2026-01-06"),
                            {"GONE": 5.0}, {})
    assert any("GONE" in n for n in notes)
    assert any(p["entry_type"] == "close-out" and p["sym"] == "GONE"
               for p in j.pages)


def test_drift_report_flags_off_target():
    held = {"A": {"qty": 10}, "B": {"qty": 0.5}}
    line = pt.drift_report(["A", "C"], 2, 1.0, held,
                           {"A": 10, "B": 10, "C": 10}, 200.0)
    assert "drift" in line and ("C" in line or "B" in line)


# ---------------------------------------------------------------------------
# multi-day scenario — the big one


def _args(tmp, day):
    class A:
        pass
    a = A()
    a.config = str(tmp / "cfg.yaml")
    a.dry_run = True
    a.date = day
    a.fixture = ""
    a.state_dir = str(tmp / "state")
    a.rec_dir = str(tmp / "rec")
    return a


def _cfg_file(tmp, extra=""):
    cfg = f"""
validated: false
live_trading: false
signals: [mom]
weighting: equal
ic_months: 12
halflife_frac: 0.5
regime: none
regime_cfg:
  sma: 200
  vol_win: 21
  vol_lookback: 504
  vol_q: 0.90
  actions: {{none: 1.0, half: 0.5, cash: 0.0}}
n_positions: 3
budget_gbp: 10.0
cost_bps: 10.0
snapshot_stale_days: 45
order_after_open_min: 15
{extra}universe: {tmp}/u.txt
"""
    (tmp / "cfg.yaml").write_text(cfg)
    (tmp / "u.txt").write_text("\n".join(
        ["A", "B", "C", "D", "E", "F", "G"]))


def test_multiday_scenario(tmp_path, monkeypatch):
    """enter -> hold -> partial fill -> rejection -> delist -> halt ->
    drawdown breaker -> holiday — all through run_day with mocks."""
    _cfg_file(tmp_path)
    days = pd.bdate_range("2025-01-02", periods=60)
    px = {s: 20 + i * 15 + np.cumsum(
        np.random.default_rng(i).normal(0.3, 1, len(days)))
        for i, s in enumerate(["A", "B", "C", "D", "E", "F", "G", "SPY"])}
    closes = pd.DataFrame(px, index=days).clip(1)
    fix = tmp_path / "fix.parquet"
    closes.to_parquet(fix)

    # shared fakes across days
    broker = pt.FakeBroker()
    journal = pt.FakeJournal()
    monkeypatch.setattr(pt, "FakeBroker", lambda: broker)
    monkeypatch.setattr(pt, "FakeJournal", lambda: journal)

    d0 = str(days[-10].date())
    # day 1 — enter
    pt.run_day(_args(tmp_path, d0))
    buys = [o for o in broker.orders if o["side"] == "buy"]
    assert len(buys) == 3                       # top-3 entered
    assert all(o["client_order_id"] for o in buys)
    # day 2 — same day rerun is idempotent (no new orders)
    pt.run_day(_args(tmp_path, d0))
    assert len(broker.orders) == len(buys)

    # day 3 — partial fill + rejection recorded via api_orders
    d1 = str(days[-5].date())
    broker.api_orders = [
        {"symbol": "A", "side": "buy", "status": "partially_filled",
         "filled_qty": "0.5", "submitted_at": f"{d1}T15:00:00Z"},
        {"symbol": "B", "side": "buy", "status": "rejected",
         "qty": "1", "submitted_at": f"{d1}T15:00:00Z"}]
    pt.run_day(_args(tmp_path, d1))
    notes = [p for p in journal.pages if p["entry_type"] == "note"]
    assert any("partial" in p["text"] for p in notes)
    assert any("rejected" in p["text"] for p in notes)

    # day 4 — delist: journal thinks we hold GONE, Alpaca doesn't
    journal.pages.append({"entry_type": "trade", "date": d1, "sym": "GONE",
                          "side": "buy"})
    pt.run_day(_args(tmp_path, d1))
    assert any(p["entry_type"] == "close-out" and p["sym"] == "GONE"
               for p in journal.pages)

    # day 5 — BOT_HALT: no orders, halt row written
    d2 = str(days[-2].date())
    monkeypatch.setenv("BOT_HALT", "1")
    n_orders = len(broker.orders)
    pt.run_day(_args(tmp_path, d2))
    monkeypatch.delenv("BOT_HALT")
    assert len(broker.orders) == n_orders
    assert any(p["entry_type"] == "halt" for p in journal.pages)

    # day 6 — drawdown breaker: prev state says peak was far higher
    journal.pages.append({"entry_type": "day-summary", "date": d2,
                          "recon_state": {"equity_usd": 100.0,
                                          "peak": 100.0, "spy": 500.0,
                                          "cum_bot": 0.0, "cum_spy": 0.0}})
    broker.acct["portfolio_value"] = "60.00"   # -40% vs peak -> breaker
    _cfg_file(tmp_path, extra="risk:\n  max_drawdown: 0.25\n  "
                              "daily_loss: 0.99\n")  # isolate dd breaker
    pt.run_day(_args(tmp_path, d2))
    buys_after = [o for o in broker.orders[n_orders:]
                  if o["side"] == "buy"]
    assert not buys_after, "breaker should have blocked all buys"

    # holiday — calendar says today isn't a session; order_window refuses
    b = _ClockBroker(is_open=False)
    ok, _ = pt.order_window(b, _et("2026-01-19", "10:00"), {})
    assert not ok


def test_dry_run_writes_run_record(tmp_path):
    _cfg_file(tmp_path)
    a = _args(tmp_path, "")
    pt.run_day(a)
    recs = list((tmp_path / "rec").glob("*.json"))
    assert len(recs) == 1
    import json
    rec = json.loads(recs[0].read_text())
    assert "gates" in rec and "orders" in rec
    assert rec["validated"] is False


def test_notion_journal_num_sanitises_nonfinite():
    """2026-11-17 real dispatch: the SPY core trade carried score=NaN
    (SPY isn't in the composite) -> `nan or 0` kept nan (NaN is truthy)
    -> the Notion POST crashed AFTER the real Alpaca order filled.
    Every numeric journal prop must be finite."""
    from quant_research import notion_journal as NJ
    n = NJ.NotionJournal._num
    assert n(float("nan")) == 0.0
    assert n(float("inf")) == 0.0
    assert n(float("-inf")) == 0.0
    assert n(None) == 0.0
    assert n("bad") == 0.0
    assert n(1.234567, 2) == 1.23


def test_log_trade_survives_nan_score(monkeypatch):
    """A core (SPY) trade has no composite score — log_trade must
    serialise a finite number, not propagate NaN into the POST body."""
    import json as _json
    from quant_research import notion_journal as NJ
    j = NJ.NotionJournal("tok", "db")
    bodies = []
    monkeypatch.setattr(j, "_page", lambda props: bodies.append(props))
    j.log_trade(pd.Timestamp("2026-11-17"),
                {"sym": "SPY", "side": "buy", "qty": 0.011,
                 "price": 590.1, "notional": 6.59, "score": float("nan"),
                 "reason": "core rebalance"},
                {"weights_str": "mom:0.50", "regime": "none",
                 "validated": False})
    _json.dumps(bodies, allow_nan=False)      # raises if any NaN leaks
    assert bodies[0]["SignalScore"]["number"] == 0.0
