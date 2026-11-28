"""Paper-bot tests — full day end-to-end with mocked Alpaca/Notion.

Nothing here touches the network or real keys: --dry-run drives
FakeBroker + FakeJournal over synthetic/fixture bars.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import paper_trade as PT

from quant_research import alpaca_bot as AB
from quant_research import live as LV

DAY = "2026-06-15"          # last fixture bar == --date (a Monday)
DAYS = pd.bdate_range(end=pd.Timestamp(DAY), periods=620)


def fixture_closes(symbols: list[str], crash_tail: int = 0,
                   days: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Deterministic closes: M0..M7 strong uptrend, rest flat; SPY calm
    unless crash_tail > 0 (then the last crash_tail days fall hard)."""
    days = DAYS if days is None else days
    rng = np.random.default_rng(3)
    data = {}
    for i, s in enumerate(symbols):
        if s.startswith("M"):
            drift = 0.003 + int(s[1]) * 0.0005
        else:
            drift = 0.0000
        r = rng.normal(drift, 0.01, len(days))
        px = 30 * np.exp(np.cumsum(r)) + 10
        data[s] = px
    # deterministic calm SPY — random vol occasionally trips the guard
    spy = 100 * np.exp(np.cumsum(np.full(len(days), 0.0004)))
    if crash_tail:
        spy[-crash_tail:] = spy[-crash_tail - 1] * \
            np.cumprod(np.full(crash_tail, 0.985))
    data["SPY"] = spy
    return pd.DataFrame(data, index=days)


def write_cfg(tmp: Path, **over) -> Path:
    import yaml
    cfg = {"validated": False, "signals": ["mom", "vq"],
           "weighting": "equal", "ic_months": 12, "halflife_frac": 0.5,
           "regime": "cash",
           "regime_cfg": {"sma": 200, "vol_win": 21, "vol_lookback": 504,
                          "vol_q": 0.90,
                          "actions": {"none": 1.0, "half": 0.5, "cash": 0.0}},
           "n_positions": 5, "budget_gbp": 10.0, "cost_bps": 0.0,
           "universe": str(tmp / "universe.txt")}
    cfg.update(over)
    p = tmp / "live_signal.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def write_universe(tmp: Path, symbols: list[str]) -> None:
    (tmp / "universe.txt").write_text("\n".join(symbols))


def write_vq(tmp: Path, scores: dict[str, float], asof="2026-06-01"):
    sd = tmp / "state"
    sd.mkdir(exist_ok=True)
    pd.DataFrame({"symbol": list(scores), "score": list(scores.values()),
                  "asof": asof}).to_parquet(sd / "vq_snapshot.parquet",
                                           index=False)
    return sd


def write_qv50(tmp: Path, symbols: list[str], asof="2026-06-01"):
    """Synthetic QV filing-input snapshot + PIT membership for `symbols`."""
    sd = tmp / "state"
    sd.mkdir(exist_ok=True)
    rows = []
    for i, s in enumerate(symbols):
        rows.append({"symbol": s, "ebit": 100.0 + i,
                     "assets": 1_000.0, "cash": 0.0, "debt": 0.0,
                     "gp": 300.0 + i, "ni": 80.0 + i,
                     "shares": 1_000_000.0, "asof": asof})
    pd.DataFrame(rows).to_parquet(sd / "qv50_snapshot.parquet", index=False)
    pd.DataFrame({"symbol": symbols, "start": "2020-01-01", "end": ""}) \
        .to_csv(tmp / "membership.csv", index=False)
    return sd


def qv50_cfg(tmp: Path, n=20, **over) -> Path:
    cfg = {"signals": ["qv"], "strategy": "QV50", "require_signals": True,
           "membership_csv": str(tmp / "membership.csv"),
           "regime": "none",
           "core": {"symbol": "SPY", "weight": 0.5},
           "sleeve": {"n": n, "weighting": "equal", "risk_off": "to_core"},
           "n_positions": n, "book_size": "account_equity",
           "risk": {"max_orders": 40}}
    cfg.update(over)
    return write_cfg(tmp, **cfg)


def run_bot(cfg: Path, fixture: Path, state: Path, date: str,
            broker=None, journal=None):
    argv = sys.argv
    sys.argv = ["paper_trade.py", "--dry-run", "--config", str(cfg),
                "--fixture", str(fixture), "--state-dir", str(state),
                "--date", date]
    if broker is not None or journal is not None:
        old_b, old_j = PT.FakeBroker, PT.FakeJournal
        PT.FakeBroker = lambda: broker
        PT.FakeJournal = lambda: journal
    try:
        PT.main()
    finally:
        sys.argv = argv
        if broker is not None or journal is not None:
            PT.FakeBroker, PT.FakeJournal = old_b, old_j


SYMS = [f"M{i}" for i in range(8)] + [f"X{i}" for i in range(25)]


def test_full_day_dry_run(tmp_path):
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path)
    run_bot(cfg, fx, state, DAY)
    out = json.loads((REPO / "results/paper_dryrun" / f"{DAY}.json")
                     .read_text())
    assert out["want"] and all(w in SYMS for w in out["want"])
    assert len(out["want"]) == 5
    assert any(o["side"] == "buy" for o in out["orders"])
    types = {p["entry_type"] for p in out["journal"]}
    assert {"trade", "day-summary", "weights-change"} <= types
    assert all(p.get("validated") is False
               for p in out["journal"] if "validated" in p)


def test_idempotent_same_day(tmp_path):
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path)
    broker, journal = PT.FakeBroker(), PT.FakeJournal()
    run_bot(cfg, fx, state, DAY, broker, journal)
    n_orders = len(broker.orders)
    assert n_orders > 0
    run_bot(cfg, fx, state, DAY, broker, journal)
    assert len(broker.orders) == n_orders    # no double-trading same day


def test_crash_regime_liquidates(tmp_path):
    cl = fixture_closes(SYMS, crash_tail=80)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path)
    held = {SYMS[0]: {"qty": 0.5, "market_value": 30.0,
                      "avg_entry_price": 60.0}}
    broker = PT.FakeBroker(positions=held)
    run_bot(cfg, fx, state, DAY, broker, PT.FakeJournal())
    sides = [o["side"] for o in broker.orders]
    assert "buy" not in sides                # cash regime -> no new buys
    out = json.loads((REPO / "results/paper_dryrun" / f"{DAY}.json")
                     .read_text())
    assert out["exposure"] == 0.0


def test_min_notional_skips(tmp_path):
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path, budget_gbp=2.0)   # ~$2.54 -> $0.51/slot
    broker = PT.FakeBroker()
    run_bot(cfg, fx, state, DAY, broker, PT.FakeJournal())
    assert all(o["side"] == "sell" for o in broker.orders) or \
        not broker.orders                     # all buys below $1 skipped


def test_paper_url_guard():
    with pytest.raises(RuntimeError, match="paper"):
        AB.PaperClient("k", "s", "https://api.alpaca.markets")
    with pytest.raises(RuntimeError, match="paper"):
        AB.PaperClient("k", "s", "https://broker-api.sandbox.alpaca.markets")


def test_snapshot_stale_dropped(tmp_path):
    sd = tmp_path / "state"
    sd.mkdir()
    pd.DataFrame({"symbol": ["AAA"], "score": [0.9],
                  "asof": "2020-01-01"}).to_parquet(
        sd / "vq_snapshot.parquet", index=False)
    s, note = LV.snapshot_signal(sd / "vq_snapshot.parquet", "vq",
                                 stale_days=45)
    assert s.empty and "STALE" in note


def test_ewic_weights_sum(tmp_path):
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = tmp_path / "state"
    state.mkdir()
    cfg = write_cfg(tmp_path, signals=["mom", "rev"], weighting="ewic")
    broker = PT.FakeBroker()
    run_bot(cfg, fx, state, DAY, broker, PT.FakeJournal())
    out = json.loads((REPO / "results/paper_dryrun" / f"{DAY}.json")
                     .read_text())
    assert abs(sum(out["weights"].values()) - 1.0) < 1e-6
    assert all(v >= 0 for v in out["weights"].values())




# ---------------------------------------------------------------------------
# QV50 — phase-11 value/quality shadow book (mocked; no keys)


def test_qv50_scores_exact_formula_and_pit_filter():
    day = pd.Timestamp(DAY)
    rows = []
    # ebit ranks B > A > C; quality ranks A > B > C, so the composite is
    # B first, A second, C third. OUT is best on every input but is not a
    # member on the signal day and must be impossible to pick.
    vals = {
        "A": (10e6, 50e6, 10e6),
        "B": (20e6, 30e6, 30e6),
        "C": (5e6, 40e6, 5e6),
        "OUT": (100e6, 100e6, 100e6),
    }
    for s, (ebit, gp, ni) in vals.items():
        rows.append({"symbol": s, "ebit": ebit, "assets": 100e6,
                     "cash": 0.0, "debt": 0.0, "gp": gp, "ni": ni,
                     "shares": 1e6, "asof": "2026-06-01"})
    inp = pd.DataFrame(rows)
    closes = pd.DataFrame({"A": [100.0], "B": [100.0], "C": [100.0],
                           "OUT": [100.0]}, index=[day])
    got = LV.qv50_scores(inp, closes, day, {"A", "B", "C"})
    assert list(got.sort_values(ascending=False).index) == ["B", "A", "C"]
    assert "OUT" not in got.index
    assert got["B"] == pytest.approx(1.0)
    assert got["A"] == pytest.approx(2 / 3)
    assert got["C"] == pytest.approx(1 / 3)


def test_qv50_target_weights_and_pit_exclusion(tmp_path):
    qsyms = SYMS + ["OUT"]
    cl = fixture_closes(qsyms)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, qsyms)
    state = write_qv50(tmp_path, qsyms)
    # OUT would be a valid row in the snapshot but its spell ended long
    # before the signal day; the phase-11 PIT rule is start <= d < end.
    mem = pd.read_csv(tmp_path / "membership.csv")
    mem["end"] = mem["end"].astype(object)
    mem.loc[mem["symbol"] == "OUT", "end"] = "2020-01-02"
    mem.to_csv(tmp_path / "membership.csv", index=False)
    cfg = qv50_cfg(tmp_path)
    broker = PT.FakeBroker()
    broker.acct = {"cash": str(EQUITY), "portfolio_value": str(EQUITY)}
    journal = PT.FakeJournal()
    run_bot(cfg, fx, state, DAY, broker, journal)
    rec = json.loads((REPO / "results/paper_run" / f"{DAY}.json").read_text())
    targets = rec["targets"]
    stocks = {s: w for s, w in targets.items() if s != "SPY"}
    assert rec["strategy"] == "QV50"
    assert targets["SPY"] == pytest.approx(0.5)
    assert len(stocks) == 20
    assert set(stocks).issubset(set(SYMS))
    assert "OUT" not in stocks
    assert all(w == pytest.approx(0.025) for w in stocks.values())
    assert sum(targets.values()) == pytest.approx(1.0)
    assert all(p.get("strategy", "QV50") == "QV50" for p in journal.pages)


def test_qv50_missing_snapshot_fails_loudly(tmp_path):
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = tmp_path / "state"
    state.mkdir()                       # no qv50_snapshot.parquet
    pd.DataFrame({"symbol": SYMS, "start": "2020-01-01", "end": ""}) \
        .to_csv(tmp_path / "membership.csv", index=False)
    cfg = qv50_cfg(tmp_path)
    journal = PT.FakeJournal()
    with pytest.raises(SystemExit, match="no usable signals"):
        run_bot(cfg, fx, state, DAY, PT.FakeBroker(), journal)
    assert any(p["entry_type"] == "error" for p in journal.pages)


def test_qv50_snapshot_staleness_and_future_guards(tmp_path):
    path = tmp_path / "qv.parquet"
    base = {"symbol": ["A"], "ebit": [1.0], "assets": [1.0],
            "cash": [0.0], "debt": [0.0], "gp": [1.0], "ni": [1.0],
            "shares": [1.0]}
    pd.DataFrame({**base, "asof": ["2026-04-01"]}).to_parquet(path)
    out, note = LV.qv_snapshot_inputs(path, stale_days=45,
                                      asof_day=pd.Timestamp("2026-06-15"))
    assert out.empty and "STALE" in note
    pd.DataFrame({**base, "asof": ["2026-06-16"]}).to_parquet(path)
    out, note = LV.qv_snapshot_inputs(path, stale_days=45,
                                      asof_day=pd.Timestamp("2026-06-15"))
    assert out.empty and "after signal day" in note


# ---------------------------------------------------------------------------
# ship-hardening: reconcile, clock/stale guards, error entries, live gate

DAY2 = "2026-06-16"          # Tuesday after DAY


def alpaca_order(sym, side, status="filled", day=DAY, px=100.0, qty=0.02):
    return {"symbol": sym, "side": side, "status": status,
            "submitted_at": f"{day}T14:00:00Z",
            "filled_at": f"{day}T14:31:00Z" if status != "new" else None,
            "filled_qty": qty, "filled_avg_price": px,
            "qty": qty, "notional": round(px * qty, 2)}


def test_reconcile_close_out_and_fill(tmp_path):
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path)
    broker = PT.FakeBroker()          # flat positions -> sell fully exits
    journal = PT.FakeJournal()
    broker.api_orders = [
        alpaca_order("ZZOLD", "sell", day="2026-06-12", px=42.0),
        alpaca_order(SYMS[0], "buy", day=DAY, px=55.0, qty=0.05)]
    run_bot(cfg, fx, state, DAY, broker, journal)
    types = {(p["entry_type"], p.get("sym")) for p in journal.pages}
    assert ("close-out", "ZZOLD") in types        # filled sell -> Outcome
    assert ("fill", SYMS[0]) in types             # buy reconciled to actuals


def test_reconcile_rejected_and_partial_noted(tmp_path):
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path)
    broker = PT.FakeBroker()
    journal = PT.FakeJournal()
    broker.api_orders = [
        alpaca_order("BAD", "buy", status="rejected"),
        alpaca_order("HALF", "sell", status="partially_filled")]
    run_bot(cfg, fx, state, DAY, broker, journal)
    notes = [p for p in journal.pages if p["entry_type"] == "note"]
    assert len(notes) == 2                         # rejected + partial
    assert any("rejected" in n["text"] for n in notes)
    assert any("partial" in n["text"] for n in notes)


def test_equity_state_tracks_across_runs(tmp_path):
    days2 = pd.bdate_range(end=pd.Timestamp(DAY2), periods=621)
    cl = fixture_closes(SYMS, days=days2)   # fixture reaches DAY2
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path)
    broker, journal = PT.FakeBroker(), PT.FakeJournal()
    run_bot(cfg, fx, state, DAY, broker, journal)
    s1 = [p for p in journal.pages
          if p["entry_type"] == "day-summary"][-1]["recon_state"]
    assert s1["equity_usd"] == 12.70 and s1["cum_bot"] == 0.0
    broker.acct = {"cash": "0.5", "portfolio_value": "13.20"}
    broker.day = None
    run_bot(cfg, fx, state, DAY2, broker, journal)
    s2 = [p for p in journal.pages
          if p["entry_type"] == "day-summary"][-1]["recon_state"]
    assert s2["equity_usd"] == 13.20
    assert s2["cum_bot"] == pytest.approx(13.20 / 12.70 - 1, abs=1e-6)
    spy1 = float(cl["SPY"].loc[:pd.Timestamp(DAY)].iloc[-1])
    spy2 = float(cl["SPY"].loc[:pd.Timestamp(DAY2)].iloc[-1])
    assert s2["cum_spy"] == pytest.approx(spy2 / spy1 - 1, abs=1e-6)


def test_market_closed_note(tmp_path):
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path)
    broker = PT.FakeBroker()
    broker.clock_ret = {"is_open": False,
                        "next_open": "2026-06-16T13:30:00Z",
                        "timestamp": "2026-06-15T21:00:00Z"}
    journal = PT.FakeJournal()
    run_bot(cfg, fx, state, DAY, broker, journal)
    summ = [p for p in journal.pages
            if p["entry_type"] == "day-summary"][-1]
    assert "market closed" in summ["summary"]


def test_stale_bars_refuse(tmp_path):
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path)
    broker = PT.FakeBroker()
    broker.clock_ret = {"is_open": False,
                        "timestamp": "2026-08-01T21:00:00Z"}  # bars 46d stale
    journal = PT.FakeJournal()
    with pytest.raises(SystemExit):
        run_bot(cfg, fx, state, DAY, broker, journal)
    assert any(p["entry_type"] == "error" for p in journal.pages)


def test_fail_writes_error_entry():
    j = PT.FakeJournal()
    with pytest.raises(SystemExit):
        PT.fail(j, "boom went the pipeline")
    assert j.pages[-1]["entry_type"] == "error"
    assert "boom" in j.pages[-1]["message"]


def test_crash_exits_nonzero(tmp_path):
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path)

    class BoomBroker(PT.FakeBroker):
        def positions(self):
            raise RuntimeError("broker exploded")

    with pytest.raises(SystemExit) as e:
        run_bot(cfg, fx, state, DAY, BoomBroker(), PT.FakeJournal())
    assert e.value.code == 1                       # fail_loudly -> exit 1


def test_live_trading_interlock(tmp_path, monkeypatch):
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path, live_trading=True)
    monkeypatch.delenv("LIVE_TRADING_ACK", raising=False)
    with pytest.raises(SystemExit, match="LIVE_TRADING_ACK"):
        run_bot(cfg, fx, state, DAY)


def test_allow_live_client_gate():
    with pytest.raises(RuntimeError, match="paper"):
        AB.PaperClient("k", "s", "https://api.alpaca.markets")
    c = AB.PaperClient("k", "s", "https://api.alpaca.markets",
                       allow_live=True)
    assert c.base == "https://api.alpaca.markets"   # gated override only


def test_weights_change_journal_dedup(tmp_path):
    days2 = pd.bdate_range(end=pd.Timestamp(DAY2), periods=621)
    cl = fixture_closes(SYMS, days=days2)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path)
    broker, journal = PT.FakeBroker(), PT.FakeJournal()
    run_bot(cfg, fx, state, DAY, broker, journal)
    n1 = sum(p["entry_type"] == "weights-change" for p in journal.pages)
    assert n1 == 1
    (state / "weights.json").unlink()   # fresh runner: local state lost
    broker.day = None
    run_bot(cfg, fx, state, DAY2, broker, journal)
    n2 = sum(p["entry_type"] == "weights-change" for p in journal.pages)
    assert n2 == 1                                   # deduped via journal


def test_pinned_config_live_off():
    import yaml
    for name in ("live_signal.yaml", "live_signal_test.yaml"):
        cfg = yaml.safe_load((REPO / "configs" / name).read_text())
        assert cfg.get("live_trading") is False, name
        assert "validated" in cfg, name
    # `validated` on live_signal.yaml is whatever the last honest holdout
    # gate produced — CI must not veto a passed gate. The test-only config
    # is always unvalidated.
    test_cfg = yaml.safe_load(
        (REPO / "configs" / "live_signal_test.yaml").read_text())
    assert test_cfg.get("validated") is False


# ---------------------------------------------------------------------------
# cron-backup hardening: same-day reruns, closed market, deduped reconcile


def test_already_traded_today_skips(tmp_path):
    """A backup slot that fires after the day's orders are in must exit
    cleanly — no new orders, and the day still journals one summary."""
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path)
    broker, journal = PT.FakeBroker(), PT.FakeJournal()
    # what an earlier same-day run already submitted for this trading day
    broker.orders.append({"day": pd.Timestamp(DAY), "sym": "AAA",
                          "side": "buy", "notional": 5.0})
    run_bot(cfg, fx, state, DAY, broker, journal)
    assert len(broker.orders) == 1              # nothing new went out
    assert sum(p["entry_type"] == "day-summary"
               for p in journal.pages) == 1     # day still journaled once


def test_market_closed_skips_orders(tmp_path):
    """Closed-market run: reconcile + summary still happen, orders don't."""
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path)
    broker = PT.FakeBroker()
    broker.window_override = (False, "market closed — no orders this run")
    journal = PT.FakeJournal()
    run_bot(cfg, fx, state, DAY, broker, journal)
    assert broker.orders == []
    summ = [p for p in journal.pages if p["entry_type"] == "day-summary"]
    assert len(summ) == 1
    assert "no orders" in summ[-1]["summary"]
    assert "market closed" in summ[-1]["summary"]


def test_reconcile_rerun_writes_no_dup_rows(tmp_path):
    """Reconcile must be safe to run twice (the 22:00 slot + backups):
    no duplicate note or day-summary rows."""
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = write_cfg(tmp_path)
    broker, journal = PT.FakeBroker(), PT.FakeJournal()
    broker.api_orders = [
        alpaca_order("BAD", "buy", status="rejected"),
        alpaca_order("HALF", "sell", status="partially_filled")]
    run_bot(cfg, fx, state, DAY, broker, journal)
    run_bot(cfg, fx, state, DAY, broker, journal)      # backup slot reruns
    notes = [p for p in journal.pages if p["entry_type"] == "note"]
    assert len(notes) == 2                            # rejected + partial
    assert sum(p["entry_type"] == "day-summary"
               for p in journal.pages) == 1


# ---------------------------------------------------------------------------
# book_size: account_equity — the paper book IS the account (~$100k), not
# the £10 live-mirror pin. Same B50 core+sleeve map, same $1 floor, same
# one-order-per-symbol PDT shape — only the sizing source changes.

EQUITY = 100_000.0


def core_cfg(tmp: Path, **over) -> Path:
    """B50-shaped test config: 50% SPY core + 6-name equal sleeve."""
    return write_cfg(
        tmp, regime="none",
        core={"symbol": "SPY", "weight": 0.5},
        sleeve={"n": 6, "weighting": "equal", "risk_off": "to_core"},
        n_positions=6, **over)


def test_book_size_account_equity(tmp_path):
    """book_size=account_equity sizes targets off the account's equity,
    not the £10 budget_gbp pin: SPY core ~50%, 6 sleeve names ~8.3% each."""
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = core_cfg(tmp_path, book_size="account_equity")
    broker = PT.FakeBroker()
    broker.acct = {"cash": str(EQUITY), "portfolio_value": str(EQUITY)}
    journal = PT.FakeJournal()
    run_bot(cfg, fx, state, DAY, broker, journal)
    buys = {o["sym"]: o["notional"] for o in broker.orders
            if o["side"] == "buy"}
    assert buys["SPY"] == pytest.approx(EQUITY * 0.5, rel=0.02)
    sleeve = [v for s, v in buys.items() if s != "SPY"]
    assert len(sleeve) == 6                          # full sleeve funded
    for v in sleeve:
        assert v == pytest.approx(EQUITY / 12, rel=0.02)
    assert sum(buys.values()) > EQUITY * 0.9         # book ~ equity
    summ = [p for p in journal.pages
            if p["entry_type"] == "day-summary"][-1]
    assert "account_equity" in summ["summary"]
    assert "100,000" in summ["summary"]


def test_fixed_gbp_book_unchanged(tmp_path):
    """Without book_size the same B50 book still sizes to the £10 pin —
    the live mirror must not move."""
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = core_cfg(tmp_path)                          # no book_size key
    broker = PT.FakeBroker()
    broker.acct = {"cash": str(EQUITY), "portfolio_value": str(EQUITY)}
    run_bot(cfg, fx, state, DAY, broker, PT.FakeJournal())
    buys = {o["sym"]: o["notional"] for o in broker.orders
            if o["side"] == "buy"}
    assert sum(buys.values()) < 20.0                 # ~£10 -> ~$12.70 book
    assert buys["SPY"] == pytest.approx(12.70 * 0.5, rel=0.05)


def test_scale_up_tiny_book_one_shot(tmp_path):
    """The one-time scale-up: a $13 book under B50 must rebalance up to
    ~equity targets in a single run — buys cover the gap, exits stay
    tiny, no symbol is both bought and sold, and a same-day rerun is
    a clean no-op (PR #10 anchor)."""
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = core_cfg(tmp_path, book_size="account_equity")
    day = pd.Timestamp(DAY)
    px = {s: float(cl[s].loc[:day].iloc[-1]) for s in cl.columns}
    # the legacy tiny book: ~$6.50 SPY + ~$1 in every momentum name
    held = {"SPY": {"qty": 6.5 / px["SPY"], "market_value": 6.5,
                    "avg_entry_price": px["SPY"]}}
    held.update({s: {"qty": 1.0 / px[s], "market_value": 1.0,
                     "avg_entry_price": px[s]} for s in SYMS[:8]})
    broker = PT.FakeBroker(positions=held)
    broker.acct = {"cash": str(EQUITY - 14.5),
                   "portfolio_value": str(EQUITY)}
    journal = PT.FakeJournal()
    run_bot(cfg, fx, state, DAY, broker, journal)
    buys = {o["sym"]: o["notional"] for o in broker.orders
            if o["side"] == "buy"}
    sells = {o["sym"]: o["qty"] * px[o["sym"]] for o in broker.orders
             if o["side"] == "sell"}
    # SPY core scales ~$6.50 -> ~$50k; sleeve names ~$1 -> ~$8.3k
    assert buys["SPY"] == pytest.approx(EQUITY * 0.5 - 6.5, rel=0.02)
    sleeve_buys = [v for s, v in buys.items() if s != "SPY"]
    assert len(sleeve_buys) == 6
    for v in sleeve_buys:
        assert v == pytest.approx(EQUITY / 12, rel=0.03)
    # the two unpicked tiny names exit (tiny sells, well under $5)
    assert sells and all(v < 5.0 for v in sells.values())
    # PDT by construction: no symbol both bought and sold today
    assert not (set(buys) & set(sells))
    assert sum(buys.values()) > EQUITY * 0.9
    # re-run same day: the ET-today anchor makes it a clean no-op
    n_orders = len(broker.orders)
    run_bot(cfg, fx, state, DAY, broker, journal)
    assert len(broker.orders) == n_orders
    assert sum(p["entry_type"] == "day-summary"
               for p in journal.pages) == 1


def test_account_equity_unreadable_refuses(tmp_path):
    """book_size=account_equity with a dead account() must refuse loudly —
    a silent fallback would run a phantom-size book."""
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = core_cfg(tmp_path, book_size="account_equity")

    class DeadAcct(PT.FakeBroker):
        def account(self):
            raise RuntimeError("account API down")

    journal = PT.FakeJournal()
    with pytest.raises(SystemExit):
        run_bot(cfg, fx, state, DAY, DeadAcct(), journal)
    assert any(p["entry_type"] == "error" for p in journal.pages)


def test_paper_config_pins(tmp_path):
    """configs/paper_signal.yaml: same B50 book as the live mirror,
    sized by account_equity; live gates stay off."""
    import yaml
    paper = yaml.safe_load(
        (REPO / "configs" / "paper_signal.yaml").read_text())
    live = yaml.safe_load(
        (REPO / "configs" / "live_signal.yaml").read_text())
    assert paper["book_size"] == "account_equity"
    assert paper["live_trading"] is False
    assert "book_size" not in live
    for k in ("signals", "core", "sleeve", "regime", "rebalance_band"):
        assert paper[k] == live[k], k               # same strategy map
    assert "budget_gbp" in paper                    # fixed_gbp fallback


def test_b50_output_byte_identical_to_pre_qv50_golden(tmp_path):
    """The QV50 refactor must not move B50: canonical bytes of the exact
    pre-change dry-run output (picks, signal weights, target map and order
    batch) are pinned here. Journal rows are deliberately outside this
    golden because B50 rows gain the required B50 book tag."""
    cl = fixture_closes(SYMS)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = core_cfg(tmp_path, book_size="account_equity")
    broker = PT.FakeBroker()
    broker.acct = {"cash": str(EQUITY), "portfolio_value": str(EQUITY)}
    run_bot(cfg, fx, state, DAY, broker, PT.FakeJournal())
    out = json.loads((REPO / "results/paper_dryrun" / f"{DAY}.json")
                     .read_text())
    rec = json.loads((REPO / "results/paper_run" / f"{DAY}.json")
                     .read_text())
    canon = {"want": out["want"], "exposure": out["exposure"],
             "weights": out["weights"], "orders": out["orders"],
             "targets": rec["targets"]}
    text = json.dumps(canon, sort_keys=True, separators=(",", ":"),
                      default=str)
    assert text == '{"exposure":1.0,"orders":[{"client_order_id":"qr-20260615-SPY-buy","day":"2026-06-15 00:00:00","notional":50000.0,"reason":"core rebalance","side":"buy","sym":"SPY","tif":"day"},{"client_order_id":"qr-20260615-M7-buy","day":"2026-06-15 00:00:00","notional":8333.33,"reason":"enter/rebalance","side":"buy","sym":"M7","tif":"day"},{"client_order_id":"qr-20260615-M6-buy","day":"2026-06-15 00:00:00","notional":8333.33,"reason":"enter/rebalance","side":"buy","sym":"M6","tif":"day"},{"client_order_id":"qr-20260615-M5-buy","day":"2026-06-15 00:00:00","notional":8333.33,"reason":"enter/rebalance","side":"buy","sym":"M5","tif":"day"},{"client_order_id":"qr-20260615-M4-buy","day":"2026-06-15 00:00:00","notional":8333.33,"reason":"enter/rebalance","side":"buy","sym":"M4","tif":"day"},{"client_order_id":"qr-20260615-M2-buy","day":"2026-06-15 00:00:00","notional":8333.33,"reason":"enter/rebalance","side":"buy","sym":"M2","tif":"day"},{"client_order_id":"qr-20260615-M0-buy","day":"2026-06-15 00:00:00","notional":8333.33,"reason":"enter/rebalance","side":"buy","sym":"M0","tif":"day"}],"targets":{"M0":0.083333,"M2":0.083333,"M4":0.083333,"M5":0.083333,"M6":0.083333,"M7":0.083333,"SPY":0.5},"want":["M7","M6","M5","M4","M2","M0"],"weights":{"mom":0.5,"vq":0.5}}'


# ---------------------------------------------------------------------------
# schedule-lag hardening (2026-11-19/20 incident: every order slot landed
# 22:00-04:00 UTC, post-close, and refused quietly for two days):
#   - out-of-session runs queue DAY orders for the next session's open
#   - idempotency keys on the TARGET session, not the run date
#   - a trading day that ends unserved + off-target fails loudly
# Friday 2026-06-19 is the last completed close; Monday is 2026-06-22.

FRI = "2026-06-19"
SAT = pd.Timestamp("2026-06-20T14:00:00Z")
MON = pd.Timestamp("2026-06-22")


def friday_book(tmp_path):
    """Fixture ending Friday + the B50 core book on full account equity —
    the same shape the real weekend runs execute."""
    days = pd.bdate_range(end=pd.Timestamp(FRI), periods=620)
    cl = fixture_closes(SYMS, days=days)
    fx = tmp_path / "cl.parquet"
    cl.to_parquet(fx)
    write_universe(tmp_path, SYMS)
    state = write_vq(tmp_path, {s: 0.5 for s in SYMS})
    cfg = core_cfg(tmp_path, book_size="account_equity")
    return fx, state, cfg


def friday_fill():
    """An in-session Friday fill — marks Friday's session as served, so
    the missed-session alarm doesn't (correctly) fire on it."""
    return {"symbol": "SPY", "side": "buy", "status": "filled",
            "submitted_at": "2026-06-19T15:00:00Z",
            "filled_at": "2026-06-19T15:01:00Z",
            "filled_qty": "0.01", "filled_avg_price": "100.0"}


def test_after_close_run_queues_next_session(tmp_path):
    """A run landing out of session (Saturday) computes the signal on the
    latest completed close and queues DAY orders for Monday's open —
    order records and client_order_ids key on the TARGET session."""
    fx, state, cfg = friday_book(tmp_path)
    broker, journal = PT.FakeBroker(), PT.FakeJournal()
    broker.acct = {"cash": str(EQUITY), "portfolio_value": str(EQUITY)}
    broker.api_orders = [friday_fill()]          # Friday was served
    broker.now = SAT                             # Saturday -> closed
    run_bot(cfg, fx, state, FRI, broker, journal)
    assert broker.orders                         # queued, not refused
    assert all(o["day"] == MON for o in broker.orders)
    assert all(o["client_order_id"].startswith("qr-20260622-")
               for o in broker.orders)
    # notional buys / fractional sells can't use opg -> plain `day`,
    # which Alpaca itself queues to the next open
    assert all(o["tif"] == "day" for o in broker.orders)
    assert all("queued for 2026-06-22 open" in o["reason"]
               for o in broker.orders)
    rec = json.loads((REPO / "results/paper_run" / f"{FRI}.json")
                     .read_text())
    assert rec["session_mode"] == "closed"
    assert rec["target_session"] == "2026-06-22"


def test_no_double_order_late_run_plus_morning(tmp_path):
    """Saturday queues Monday's orders; the Monday pre-open slot AND a
    Monday in-session slot must both be clean no-ops — the anchor is the
    target session, not the run date."""
    fx, state, cfg = friday_book(tmp_path)
    broker, journal = PT.FakeBroker(), PT.FakeJournal()
    broker.acct = {"cash": str(EQUITY), "portfolio_value": str(EQUITY)}
    broker.api_orders = [friday_fill()]
    broker.now = SAT
    run_bot(cfg, fx, state, FRI, broker, journal)
    n = len(broker.orders)
    assert n > 0
    # Monday 03:10 ET — the new 08:10 UTC pre-open slot
    broker.now = pd.Timestamp("2026-06-22T08:10:00Z")
    run_bot(cfg, fx, state, FRI, broker, journal)
    assert len(broker.orders) == n
    # Monday 10:00 ET in-session — the 15:00 UTC slot
    broker.now = pd.Timestamp("2026-06-22T14:00:00Z")
    broker.clock_ret = {"is_open": True}
    run_bot(cfg, fx, state, FRI, broker, journal)
    assert len(broker.orders) == n
    assert sum(p["entry_type"] == "day-summary"
               for p in journal.pages) == 1


def test_weekend_and_holiday_queue_targets(tmp_path):
    """Sunday run -> same Monday target; a holiday run -> the next real
    session (the fake calendar omits non-trading days like Alpaca's)."""
    fx, state, cfg = friday_book(tmp_path)
    broker, journal = PT.FakeBroker(), PT.FakeJournal()
    broker.acct = {"cash": str(EQUITY), "portfolio_value": str(EQUITY)}
    broker.api_orders = [friday_fill()]
    broker.now = pd.Timestamp("2026-06-21T18:00:00Z")   # Sunday
    run_bot(cfg, fx, state, FRI, broker, journal)
    assert broker.orders
    assert all(o["day"] == MON for o in broker.orders)

    class _Cal(PT.FakeBroker):
        def __init__(self, cal, **kw):
            super().__init__(**kw)
            self._cal = cal

        def calendar(self, start, end):
            s, e = pd.Timestamp(start), pd.Timestamp(end)
            s = s.tz_localize(None).normalize() if s.tzinfo \
                else s.normalize()
            e = e.tz_localize(None).normalize() if e.tzinfo \
                else e.normalize()
            return [r for r in self._cal
                    if s <= pd.Timestamp(r["date"]) <= e]

    # Tuesday 2026-01-20 follows the MLK-holiday Monday — the calendar
    # has no 01-19 row, exactly like Alpaca's
    cal = [{"date": "2026-01-16", "open": "09:30", "close": "16:00"},
           {"date": "2026-01-20", "open": "09:30", "close": "16:00"},
           {"date": "2026-01-21", "open": "09:30", "close": "16:00"}]
    b = _Cal(cal)
    b.acct = {"cash": str(EQUITY), "portfolio_value": str(EQUITY)}
    b.api_orders = [{"symbol": "SPY", "side": "buy", "status": "filled",
                     "submitted_at": "2026-01-16T15:00:00Z",
                     "filled_at": "2026-01-16T15:01:00Z",
                     "filled_qty": "0.01", "filled_avg_price": "100.0"}]
    b.now = pd.Timestamp("2026-01-19T15:00:00Z")   # holiday daytime
    days = pd.bdate_range(end="2026-01-16", periods=620)
    cl = fixture_closes(SYMS, days=days)
    fx2 = tmp_path / "cl2.parquet"
    cl.to_parquet(fx2)
    run_bot(cfg, fx2, state, "2026-01-16", b, PT.FakeJournal())
    assert b.orders
    assert all(o["day"] == pd.Timestamp("2026-01-20")
               for o in b.orders)


def test_missed_session_fails_loudly(tmp_path):
    """Friday ended with zero orders and the book is off target — the
    Saturday run writes an error row for Friday and exits 1, while still
    queuing Monday's orders (recovery isn't blocked by the alarm)."""
    fx, state, cfg = friday_book(tmp_path)
    broker, journal = PT.FakeBroker(), PT.FakeJournal()
    broker.acct = {"cash": str(EQUITY), "portfolio_value": str(EQUITY)}
    broker.now = SAT
    with pytest.raises(SystemExit) as e:
        run_bot(cfg, fx, state, FRI, broker, journal)
    assert "missed session 2026-06-19" in str(e.value.code)
    errs = [p for p in journal.pages if p["entry_type"] == "error"]
    assert len(errs) == 1 and errs[0]["date"] == FRI
    assert broker.orders                     # Monday orders still queued
    assert all(o["day"] == MON for o in broker.orders)

    # rerun Sunday: the flag is deduped — row exists, exit is green,
    # and the queued orders make the run a clean no-op
    broker.now = pd.Timestamp("2026-06-21T14:00:00Z")
    n_orders = len(broker.orders)
    run_bot(cfg, fx, state, FRI, broker, journal)
    assert len(broker.orders) == n_orders
    assert sum(p["entry_type"] == "error" for p in journal.pages) == 1


def test_served_session_not_flagged(tmp_path):
    """The counterpart of the missed-day test: when Friday DID trade (the
    Friday order exists), the weekend run queues quietly — no error row,
    exit 0."""
    fx, state, cfg = friday_book(tmp_path)
    broker, journal = PT.FakeBroker(), PT.FakeJournal()
    broker.acct = {"cash": str(EQUITY), "portfolio_value": str(EQUITY)}
    broker.api_orders = [friday_fill()]
    broker.now = SAT
    run_bot(cfg, fx, state, FRI, broker, journal)   # no SystemExit
    assert not any(p["entry_type"] == "error" for p in journal.pages)
