"""Phase-4 paper-trading bot — ONE daily run, Alpaca paper only.

    .venv/bin/python scripts/paper_trade.py --dry-run          # mocked day
    .venv/bin/python scripts/paper_trade.py --date 2026-11-12  # live paper

Daily flow (config pinned in configs/live_signal.yaml):
  SPY regime guard -> signal scores (mom/rev from Alpaca bars, vq/news from
  state/live snapshots) -> adaptive weights (equal or trailing-EWIC) ->
  top-N equal-weight targets -> reconcile vs paper positions (sells first,
  then notional buys >= $1) -> Notion journal (trade pages + day summary +
  what-changed on monthly weight shifts).

Honesty contract: `validated` from the pinned config is stamped on every
log line and every journal entry — until the holdout gate passes it stays
false and the journal says UNVALIDATED.

Idempotent: if any Alpaca order was already submitted today, the run
places NO new orders (re-runs are safe). Journalling is also date-guarded.

Env: ALPACA_API_KEY, ALPACA_SECRET_KEY (paper), NOTION_TOKEN, and
NOTION_DB_ID or NOTION_PARENT_PAGE_ID. --dry-run needs none.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml

from quant_research import alpaca_bot as AB
from quant_research import live as LV
from quant_research import micro as MC
from quant_research import notion_journal as NJ


def log(msg: str) -> None:
    print(f"[paper {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# fakes for --dry-run / tests


class FakeBroker:
    """In-memory paper broker: canned positions, records orders."""

    def __init__(self, positions: dict | None = None):
        self.pos = positions or {}
        self.orders: list[dict] = []
        self.day = None

    def clock(self):
        return {"is_open": False}

    def account(self):
        return {"cash": "12.70", "portfolio_value": "12.70"}

    def positions(self):
        return self.pos

    def traded_today(self, day):
        # >= semantics like PaperClient: orders recorded with the signal day
        # they were placed for; anything since `day` means already traded
        return [o for o in self.orders if o["day"] >= day]

    def buy_notional(self, sym, usd, reason=""):
        if usd < AB.MIN_NOTIONAL_USD:
            log(f"fake skip buy {sym}: ${usd:.2f} < min")
            return None
        o = {"day": self.day, "sym": sym, "side": "buy", "notional": usd,
             "reason": reason}
        self.orders.append(o)
        return o

    def sell_qty(self, sym, qty, reason=""):
        o = {"day": self.day, "sym": sym, "side": "sell", "qty": qty,
             "reason": reason}
        self.orders.append(o)
        return o


class FakeJournal:
    """Records journal pages; dumps to results/paper_dryrun/<day>.json."""

    def __init__(self):
        self.pages: list[dict] = []

    def day_logged(self, day, entry_type="day-summary"):
        return any(p["entry_type"] == entry_type
                   and p["date"] == day.strftime("%Y-%m-%d")
                   for p in self.pages)

    def log_trade(self, day, t, ctx):
        self.pages.append({"entry_type": "trade", "date":
                           day.strftime("%Y-%m-%d"), **t,
                           "validated": ctx["validated"]})

    def log_day_summary(self, day, ctx):
        self.pages.append({"entry_type": "day-summary",
                           "date": day.strftime("%Y-%m-%d"), **ctx})

    def log_weights_change(self, month, old, new, ctx):
        self.pages.append({"entry_type": "weights-change", "date": month,
                           "old": old, "new": new,
                           "validated": ctx["validated"]})

    def close_out(self, sym, exit_price, reason=""):
        self.pages.append({"entry_type": "close-out", "sym": sym,
                           "exit_price": exit_price, "reason": reason})


def synth_closes(symbols: list[str], days: pd.DatetimeIndex,
                 seed: int = 7) -> pd.DataFrame:
    """Deterministic fake closes for --dry-run without a fixture."""
    rng = np.random.default_rng(seed)
    out = {}
    for i, s in enumerate(symbols):
        r = rng.normal(0.0004, 0.02, len(days)) + (i % 9) * 0.0002
        out[s] = 20 + (i * 37 % 180) + np.cumsum(r * 10)
    return pd.DataFrame(out, index=days).clip(1)


# ---------------------------------------------------------------------------
# run


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg["_path"] = str(path)
    return cfg


def resolve_day(closes: pd.DataFrame, want: str) -> pd.Timestamp:
    """Last bar date <= --date (or latest bar when --date unset)."""
    avail = closes.index
    if not want:
        return avail.max()
    d = pd.Timestamp(want)
    elig = avail[avail <= d]
    if elig.empty:
        sys.exit(f"no bars on/before {want}")
    return elig.max()


def compute_signals(cfg: dict, closes: pd.DataFrame,
                    day: pd.Timestamp, state_dir: Path) -> tuple[dict, dict, list]:
    """Per-signal {sym: score} for `day`, plus IC-history for ewic weights."""
    sig_scalars: dict[str, pd.Series] = {}
    notes: list[str] = []
    sig_series: dict[str, pd.Series] = {}        # only populated for ewic
    days_lookback = closes.index[-min(len(closes.index), 260):]
    for sig in cfg["signals"]:
        if sig in ("mom", "rev"):
            fn = LV.momentum if sig == "mom" else LV.reversal
            sig_scalars[sig] = pd.Series(
                {s: fn(closes[s], day) for s in closes.columns})
            if cfg["weighting"] == "ewic":
                sig_series[sig] = LV.bars_signals(
                    closes, days_lookback, [sig])[sig]
        elif sig in ("vq", "news"):
            snap = state_dir / f"{sig}_snapshot.parquet"
            s_, note = LV.snapshot_signal(
                snap, sig, stale_days=cfg.get("snapshot_stale_days", 45),
                asof_day=day)
            notes.append(note or f"{sig}: snapshot fresh")
            if not s_.empty:
                sig_scalars[sig] = s_
        else:
            notes.append(f"{sig}: not computable live (needs the monthly-"
                         "retrain store pipeline) — dropped")
            log(f"DROPPED {sig}: no live source")
    scalars = {s: v.dropna() for s, v in sig_scalars.items()}
    for s in cfg["signals"]:
        if s not in scalars:
            log(f"signal {s}: no scores — excluded from weights")
    if cfg["weighting"] == "ewic" and scalars:
        labels = LV.labels_from_bars(closes, days_lookback)
        import quant_research.adaptive as ad
        ic_hist = ad.ic_history(sig_series, labels) \
            if sig_series else pd.DataFrame()
        hl = cfg["ic_months"] * 21 * cfg.get("halflife_frac", 0.5)
        w = ad.ew_ic_weights(ic_hist, day, cfg["ic_months"], hl,
                             list(scalars)) if not ic_hist.empty else \
            pd.Series(0.0, index=list(scalars))
        # Snapshot signals have no live IC history — if any are in the mix
        # there is no honest trailing evidence for them, so the whole book
        # falls back to equal weights rather than silently up-weighting bars
        # signals. All-zero ewic (no positive IC) stays zero => flat book.
        if any(s not in sig_series for s in scalars):
            weights = {s: 1.0 / len(scalars) for s in scalars}
            notes.append("ewic: snapshot signal(s) lack IC history — "
                         "equal weighting used instead")
        else:
            weights = {s: float(w.get(s, 0.0)) for s in scalars}
            if sum(weights.values()) <= 0:
                notes.append("ewic: no positive trailing rank-IC — "
                             "zero conviction, book goes flat")
    else:
        weights = {s: 1.0 / len(scalars) for s in scalars}
    return scalars, weights, notes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO / "configs" /
                                           "live_signal.yaml"))
    p.add_argument("--date", default="",
                   help="YYYY-MM-DD; default = latest bar available")
    p.add_argument("--dry-run", action="store_true",
                   help="mocked broker+journal+synthetic bars; no keys needed")
    p.add_argument("--fixture", default="",
                   help="parquet (date x symbol closes) overriding synth bars")
    p.add_argument("--state-dir", default=str(REPO / "state" / "live"))
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    state_dir = Path(args.state_dir)
    validated = bool(cfg.get("validated"))
    banner = "VALIDATED" if validated else "UNVALIDATED"
    log(f"=== paper_trade {banner} config={args.config} "
        f"signals={cfg['signals']} weighting={cfg['weighting']} "
        f"regime={cfg['regime']} ===")

    universe = [l.strip() for l in
                (REPO / cfg["universe"]).read_text().splitlines()
                if l.strip() and not l.startswith("#")]
    log(f"universe: {len(universe)} symbols")

    # -- clients ---------------------------------------------------------------
    dry = args.dry_run
    if dry:
        broker, journal = FakeBroker(), FakeJournal()
        key = secret = ""
    else:
        key = os.environ.get("ALPACA_API_KEY", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            sys.exit("ALPACA_API_KEY / ALPACA_SECRET_KEY missing (paper keys)")
        broker = AB.PaperClient(
            key, secret, os.environ.get("ALPACA_BASE_URL", AB.PAPER_URL))
        token = os.environ.get("NOTION_TOKEN", "")
        journal = NJ.NotionJournal(
            token, os.environ.get("NOTION_DB_ID", ""),
            os.environ.get("NOTION_PARENT_PAGE_ID", ""))

    # -- market data -------------------------------------------------------------
    end = pd.Timestamp(args.date) if args.date else pd.Timestamp.now()
    if dry:
        days = pd.bdate_range(end - pd.Timedelta(days=560 * 1.6), end)
        if args.fixture:
            closes = pd.read_parquet(args.fixture)
        else:
            closes = synth_closes(universe + ["SPY"], days)
    else:
        closes = LV.alpaca_bars(key, secret, universe + ["SPY"],
                              end - pd.Timedelta(days=900), end)
        log(f"bars: {closes.shape}")
    day = resolve_day(closes, args.date)
    if dry:
        broker.day = day
    log(f"signal day: {day.date()}")

    # -- regime -------------------------------------------------------------------
    spy = closes["SPY"].dropna()
    rc = dict(cfg["regime_cfg"])
    rc["action"] = rc["actions"][cfg["regime"]]  # none/half/cash -> exposure
    exposure, why = LV.regime_state(spy, rc, day)
    log(f"regime: {why} -> exposure {exposure}")

    # -- signals + weights ---------------------------------------------------------
    sig_scalars, weights, notes = compute_signals(cfg, closes, day, state_dir)
    for n in notes:
        log(f"  {n}")
    if not sig_scalars:
        sys.exit("no usable signals — nothing to trade")
    weights_str = ", ".join(f"{k}:{v:.2f}" for k, v in weights.items())
    log(f"weights: {weights_str}")

    comp = LV.final_scores(sig_scalars, weights, day)
    held = broker.positions()
    # prices for every scored name AND every held name — held names that fell
    # out of the universe still need a price so they can be sold, not zombied
    px_syms = set(comp.index) | set(held)
    prices = {}
    for s in px_syms:
        if s in closes.columns:
            ser = closes[s].loc[:day].dropna()
            if len(ser):
                prices[s] = float(ser.iloc[-1])
    scores = comp[comp.index.isin(prices)]
    want = list(scores.head(cfg["n_positions"]).index)
    log(f"top-{cfg['n_positions']}: {', '.join(want) or '(none — flat)'}")

    # -- budget --------------------------------------------------------------------
    fx = LV.gbpusd() if not dry else 1.27
    budget_usd = float(cfg["budget_gbp"]) * fx
    if not dry:
        try:
            acct = broker.account()
            equity = float(acct.get("portfolio_value") or
                           acct.get("equity") or budget_usd)
            if equity < budget_usd:
                log(f"account equity ${equity:.2f} < £10 budget "
                    f"${budget_usd:.2f} — sizing to equity")
                budget_usd = equity
        except Exception as e:  # noqa: BLE001
            log(f"account() failed ({e}) — sizing to configured budget")
    log(f"budget: £{cfg['budget_gbp']} -> ${budget_usd:.2f} @ {fx:.4f}")

    # -- idempotency gate ------------------------------------------------------------
    already = broker.traded_today(day)
    journaled = journal.day_logged(day) if hasattr(journal, "day_logged") \
        else False
    ctx = {"weights_str": weights_str, "regime": why,
           "validated": validated, "budget_usd": budget_usd,
           "summary": f"{banner} | picks {want} | exposure {exposure} | "
                      f"fx {fx:.4f} | notes: {'; '.join(notes)}"}
    if already:
        log(f"{len(already)} order(s) already submitted {day.date()} — "
            "idempotent skip (no new orders)")
        if not journaled:
            journal.log_day_summary(day, ctx)
        return
    # -- trade ----------------------------------------------------------------------
    step = MC.rebalance_step(
        scores, budget_usd, cfg["n_positions"], prices,
        {s: p["qty"] for s, p in held.items()},
        cfg.get("cost_bps", 0.0) / 1e4, exposure)
    filled, skipped = [], []
    sells = [t for t in step["trades"] if t["side"] == "sell"]
    buys = [t for t in step["trades"] if t["side"] == "buy"]
    for t in sells + buys:                     # sells free cash first
        sym, side = t["sym"], t["side"]
        t["score"] = float(comp.get(sym, np.nan))
        t["reason"] = ("exit top-5" if side == "sell" and sym not in want
                       else "regime trim" if side == "sell" and
                       exposure < 1 else "enter/rebalance")
        if side == "sell" and buys and t is sells[-1] and not dry:
            time.sleep(3)               # let sells settle -> buying power
        if side == "buy":
            if broker.buy_notional(sym, t["notional"], t["reason"]):
                filled.append(t)
            else:
                skipped.append(f"buy {sym} ${t['notional']:.2f}<min")
                continue
        else:
            qty = float(t["qty"])
            if broker.sell_qty(sym, qty, t["reason"]):
                filled.append(t)
                if step["held"].get(sym, 0.0) <= 1e-9:
                    journal.close_out(sym, t["price"], t["reason"])
            else:
                skipped.append(f"sell {sym} rejected")
                continue
        journal.log_trade(day, t, ctx)
    if skipped:
        ctx["summary"] += " | skipped: " + ", ".join(skipped)

    # -- monthly what-changed --------------------------------------------------------
    wstate = state_dir / "weights.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    old_w = json.loads(wstate.read_text()) if wstate.exists() else {}
    month = day.strftime("%Y-%m")
    if old_w.get("month") != month or old_w.get("weights") != weights:
        journal.log_weights_change(f"{month}-01",
                                   old_w.get("weights", {}), weights, ctx)
        wstate.write_text(json.dumps({"month": month, "weights": weights,
                                      "config": cfg["_path"]}, indent=1))

    # -- journal + dump ----------------------------------------------------------------
    journal.log_day_summary(day, ctx)
    if dry:
        out = REPO / "results" / "paper_dryrun"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{day.strftime('%Y-%m-%d')}.json"
        path.write_text(json.dumps({"day": str(day.date()), "want": want,
                                    "exposure": exposure, "weights": weights,
                                    "orders": broker.orders,
                                    "journal": journal.pages},
                                   indent=1, default=str))
        log(f"dry-run journal -> {path}")
    log(f"done: {len(filled)} order(s), picks {want}, {banner}")


if __name__ == "__main__":
    main()
