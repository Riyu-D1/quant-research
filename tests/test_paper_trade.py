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


def fixture_closes(symbols: list[str], crash_tail: int = 0) -> pd.DataFrame:
    """Deterministic closes: M0..M7 strong uptrend, rest flat; SPY calm
    unless crash_tail > 0 (then the last crash_tail days fall hard)."""
    rng = np.random.default_rng(3)
    data = {}
    for i, s in enumerate(symbols):
        if s.startswith("M"):
            drift = 0.003 + int(s[1]) * 0.0005
        else:
            drift = 0.0000
        r = rng.normal(drift, 0.01, len(DAYS))
        px = 30 * np.exp(np.cumsum(r)) + 10
        data[s] = px
    # deterministic calm SPY — random vol occasionally trips the guard
    spy = 100 * np.exp(np.cumsum(np.full(len(DAYS), 0.0004)))
    if crash_tail:
        spy[-crash_tail:] = spy[-crash_tail - 1] * \
            np.cumprod(np.full(crash_tail, 0.985))
    data["SPY"] = spy
    return pd.DataFrame(data, index=DAYS)


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
