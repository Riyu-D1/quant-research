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
        assert cfg.get("validated") is False, name
