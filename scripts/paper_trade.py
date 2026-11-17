"""Phase-4 paper-trading bot — ONE daily run, Alpaca paper only.

    .venv/bin/python scripts/paper_trade.py --dry-run          # mocked day
    .venv/bin/python scripts/paper_trade.py --date 2026-11-12  # live paper

Daily flow (config pinned in configs/live_signal.yaml):
  clock + stale-bars check -> fill reconciliation (actual Alpaca fills and
  positions -> journal Outcome/Lesson + equity-vs-SPY state) -> SPY regime
  guard -> signal scores (mom/rev/srev5/mom6/lowvol/beta from Alpaca bars,
  vq/news from state/live snapshots) -> adaptive weights (equal or
  trailing-EWIC) -> portfolio: either the legacy top-N equal-weight book,
  or (Phase-7 `core:`/`sleeve:` schema) a SPY core + top-N sleeve weight
  map traded toward via the 2% rebalance band and $1 order floor
  (coresat.build_targets + micro.weights_step — the same functions the
  phase-7 simulator scores) -> reconcile vs paper positions (sells first,
  then buys; breaker days still let the core buy) -> Notion journal.

Honesty contract: `validated` from the pinned config is stamped on every
log line and every journal entry — until the holdout gate passes it stays
false and the journal says UNVALIDATED.

Idempotent: if any Alpaca order was already submitted today, the run
places NO new orders (re-runs are safe). Journalling is also date-guarded.
All state that must survive runs lives in Alpaca (positions/orders) or
Notion (summary STATE blobs) — never on runner disk.

Failure contract: any crash -> a `error` entry in the journal (best
effort) AND exit code 1 (workflow fails loudly).

Live trading: `live_trading: true` in the config plus the
LIVE_TRADING_ACK=I_UNDERSTAND_LIVE_RISK secret plus a non-paper
ALPACA_BASE_URL are ALL required — see docs/BOT_SETUP.md. Never set these.

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
from quant_research import coresat as CS
from quant_research import live as LV
from quant_research import micro as MC
from quant_research import notion_journal as NJ


def log(msg: str) -> None:
    print(f"[paper {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# fakes for --dry-run / tests


class FakeBroker:
    """In-memory paper broker: canned positions, records orders.

    `api_orders` holds Alpaca-shaped order dicts for reconcile tests —
    append raw dicts with status/filled_at/filled_qty/filled_avg_price."""

    def __init__(self, positions: dict | None = None):
        self.pos = positions or {}
        self.orders: list[dict] = []
        self.api_orders: list[dict] = []
        self.day = None
        self.clock_ret = {"is_open": False}
        self.acct = {"cash": "12.70", "portfolio_value": "12.70"}

    def clock(self):
        return dict(self.clock_ret)

    def account(self):
        return dict(self.acct)

    def positions(self):
        return self.pos

    def orders_since(self, day, status="all"):
        d0 = pd.Timestamp(day).normalize()
        return [o for o in self.api_orders
                if pd.Timestamp(o.get("submitted_at", d0)).tz_localize(
                    None).normalize() >= d0]

    def traded_today(self, day):
        # >= semantics like PaperClient: orders recorded with the signal day
        # they were placed for; anything since `day` means already traded
        return [o for o in self.orders if o["day"] >= day]

    def buy_notional(self, sym, usd, reason="", coid=None):
        if usd < AB.MIN_NOTIONAL_USD:
            log(f"fake skip buy {sym}: ${usd:.2f} < min")
            return None
        o = {"day": self.day, "sym": sym, "side": "buy", "notional": usd,
             "reason": reason, "client_order_id": coid}
        self.orders.append(o)
        return o

    def sell_qty(self, sym, qty, reason="", coid=None):
        o = {"day": self.day, "sym": sym, "side": "sell", "qty": qty,
             "reason": reason, "client_order_id": coid}
        self.orders.append(o)
        return o

    def calendar(self, start, end):
        """Canned regular session (09:30-16:00 ET) for every business day
        in range — tests inject half-days by overriding this list."""
        out = []
        for d in pd.bdate_range(start, end):
            out.append({"date": d.strftime("%Y-%m-%d"),
                        "open": "09:30", "close": "16:00"})
        return out

    def open_orders(self):
        return [o for o in self.api_orders if o.get("status") == "open"]

    def cancel_order(self, oid):
        for o in self.api_orders:
            if o.get("id") == oid:
                o["status"] = "canceled"

    def cancel_all(self):
        n = 0
        for o in self.open_orders():
            o["status"] = "canceled"
            n += 1
        return n


class FakeJournal:
    """Records journal pages; dumps to results/paper_dryrun/<day>.json."""

    def __init__(self):
        self.pages: list[dict] = []

    def day_logged(self, day, entry_type="day-summary"):
        return any(p["entry_type"] == entry_type
                   and p["date"] == pd.Timestamp(day).strftime("%Y-%m-%d")
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

    def log_error(self, day, message, ctx=None):
        self.pages.append({"entry_type": "error",
                           "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                           "message": message})

    def log_note(self, day, text, ctx=None):
        self.pages.append({"entry_type": "note",
                           "date": day.strftime("%Y-%m-%d"), "text": text})

    def mark_fill(self, sym, day, qty, price):
        if any(p["entry_type"] == "fill" and p["sym"] == sym
               and p["day"] == day for p in self.pages):
            return                                  # already reconciled
        self.pages.append({"entry_type": "fill", "sym": sym, "day": day,
                           "qty": qty, "price": price})

    def last_summary_state(self):
        for p in reversed(self.pages):
            if p["entry_type"] == "day-summary" and p.get("recon_state"):
                return p["recon_state"]
        return {}

    def log_halt(self, day, reason, ctx=None):
        self.pages.append({"entry_type": "halt",
                           "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                           "reason": reason})

    def open_buy(self, sym):
        return None

    def open_buys(self):
        return [p["t"]["sym"] if "t" in p else p.get("sym")
                for p in self.pages
                if p["entry_type"] == "trade" and p.get("side") == "buy"
                and not any(q["entry_type"] == "close-out"
                            and q["sym"] == (p["t"]["sym"] if "t" in p
                                             else p.get("sym"))
                            for q in self.pages)]

    def day_summaries(self, n=10):
        return [p for p in self.pages
                if p["entry_type"] == "day-summary"][-n:]

    def trades_between(self, start, end):
        return [p for p in self.pages
                if p["entry_type"] == "trade"
                and start <= p["date"] <= end]

    def close_out(self, sym, exit_price, reason=""):
        if any(p["entry_type"] == "close-out" and p["sym"] == sym
               for p in self.pages):
            return                                  # already closed out
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


STALE_BAR_DAYS = 7     # latest bar older than this vs the clock -> refuse


def fail(journal, msg: str) -> None:
    """Loud stop: error entry in the journal (best effort) + nonzero exit."""
    log(f"FAIL {msg}")
    try:
        journal.log_error(pd.Timestamp.now(), msg, {})
    except Exception as e:  # noqa: BLE001
        log(f"error journal write failed: {e}")
    sys.exit(msg)


def reconcile(broker, journal, day: pd.Timestamp, held: dict,
              spy_px: float | None, ctx: dict, fx: float = 0.0) -> dict:
    """Read actual Alpaca orders/positions; reconcile the journal.

    - filled sells on a now-flat position -> close_out the open BUY page
      with the ACTUAL fill price (Outcome + Lesson get real numbers)
    - filled buys -> mark_fill patches the trade page to actuals
    - rejected / partially-filled orders -> note entries (never crash)
    - account equity + SPY close -> STATE blob carried in the next
      day-summary, so P&L-vs-SPY tracking persists without local state;
      `peak` is the running equity high for the drawdown breaker
    Returns {"equity", "state", "line", "peak", "prev", "fills", "notes"};
    API failures degrade to notes, never raise.
    """
    notes = []
    try:
        ords = broker.orders_since(day - pd.Timedelta(days=7))
    except Exception as e:  # noqa: BLE001
        ords, notes = [], [f"orders_since failed: {e}"]
    fills = 0
    for o in ords:
        st = o.get("status")
        sym = str(o.get("symbol", "")).replace(".", "-")
        if st == "rejected":
            notes.append(f"rejected {o.get('side')} {sym}")
            journal.log_note(
                day, f"rejected {o.get('side')} {sym} "
                     f"qty={o.get('qty')} notional={o.get('notional')}", ctx)
            continue
        if st == "partially_filled":
            notes.append(f"partial {o.get('side')} {sym} "
                         f"filled_qty={o.get('filled_qty')}")
            journal.log_note(
                day, f"partial fill {o.get('side')} {sym} "
                     f"filled_qty={o.get('filled_qty')}", ctx)
        if st != "filled" or not o.get("filled_at"):
            continue
        fills += 1
        px = float(o.get("filled_avg_price") or 0)
        fq = float(o.get("filled_qty") or 0)
        if o.get("side") == "sell":
            if held.get(sym, {}).get("qty", 0) <= 1e-9:
                journal.close_out(sym, px, "reconciled fill")
                notes.append(f"closed {sym} @ {px:.2f}")
        else:
            journal.mark_fill(sym, str(o.get("submitted_at", ""))[:10],
                              fq, px)
    # equity vs SPY — baseline is the STATE blob on the last day-summary
    try:
        acct = broker.account()
        eq = float(acct.get("portfolio_value") or acct.get("equity") or 0.0)
    except Exception as e:  # noqa: BLE001
        eq, notes = 0.0, notes + [f"account failed: {e}"]
    prev = (journal.last_summary_state()
            if hasattr(journal, "last_summary_state") else {})
    dd_bot = eq / prev["equity_usd"] - 1 if prev.get("equity_usd") else 0.0
    dd_spy = (spy_px / prev["spy"] - 1
              if prev.get("spy") and spy_px else 0.0)
    cum_bot = (1 + prev.get("cum_bot", 0.0)) * (1 + dd_bot) - 1
    cum_spy = (1 + prev.get("cum_spy", 0.0)) * (1 + dd_spy) - 1
    peak = max(float(prev.get("peak", eq) or eq), eq)
    state = {"date": str(day.date()), "equity_usd": round(eq, 4),
             "equity_gbp": round(eq / fx, 4) if fx else None,
             "spy": round(spy_px, 4) if spy_px else None,
             "peak": round(peak, 4),
             "cum_bot": round(cum_bot, 6), "cum_spy": round(cum_spy, 6)}
    line = (f"equity ${eq:.2f} (£{eq / fx:.2f}) ({dd_bot:+.2%} d/d, "
            f"cum {cum_bot:+.2%} vs SPY {cum_spy:+.2%})"
            if eq and fx else f"equity ${eq:.2f}" if eq
            else "equity unavailable")
    return {"equity": eq, "state": state, "line": line, "peak": peak,
            "prev": prev, "fills": fills, "notes": notes}


def market_clock(broker, journal, day: pd.Timestamp, dry: bool) -> str:
    """Clock check + stale-bars guard; returns a note for the summary."""
    try:
        clk = broker.clock() or {}
    except Exception as e:  # noqa: BLE001
        log(f"clock() failed: {e}")
        return "clock unavailable"
    note = f"market {'open' if clk.get('is_open') else 'closed'}"
    if clk.get("next_open"):
        note += f", next open {str(clk['next_open'])[:10]}"
    ref = clk.get("timestamp")
    if ref:
        ref = pd.Timestamp(ref).tz_localize(None)
    elif not dry:
        ref = pd.Timestamp.now()
    if ref is not None and (ref - day).days > STALE_BAR_DAYS:
        fail(journal, f"stale bars: last bar {day.date()} vs clock "
                      f"{ref.date()} (>{STALE_BAR_DAYS}d) — refusing to "
                      "trade stale data")
    return note


# ---------------------------------------------------------------------------
# production gates — every rule is config-driven and independently testable


def order_window(broker, now: pd.Timestamp, cfg: dict,
                 dry: bool = False) -> tuple[bool, str]:
    """May orders go out right now?

    Uses the Alpaca clock + calendar APIs (the source of truth):
      - market must be open (regular session only — the clock's is_open is
        false in pre/post market and on holidays, so extended hours can
        never slip through)
      - now must be >= session open + order_after_open_min (fixed time
        after the open; default 15 min so the open auction settles)
      - now must be <= session close (half-day closes are early and are
        handled automatically by the calendar's close field)
    Returns (allowed, note). Never raises — API failures degrade to
    'closed' (fail safe: no orders beats wrong-time orders).
    """
    after = int(cfg.get("order_after_open_min", 15))
    try:
        clk = broker.clock() or {}
    except Exception as e:  # noqa: BLE001
        return False, f"order-window: clock failed ({e}) — no orders"
    if not clk.get("is_open"):
        return False, "market closed — no orders this run"
    today = now.normalize()
    try:
        cal = broker.calendar(today, today)
    except Exception as e:  # noqa: BLE001
        cal, _ = None, log(f"calendar() failed: {e} — falling back to "
                           "clock-only window")
    if cal:
        import zoneinfo
        et = zoneinfo.ZoneInfo("America/New_York")
        row = cal[0]
        o = pd.Timestamp(f"{row['date']} {row['open']}").tz_localize(et)
        c = pd.Timestamp(f"{row['date']} {row['close']}").tz_localize(et)
        n = now.tz_convert(et) if now.tzinfo else now.tz_localize(
            "UTC").tz_convert(et)
        if n < o + pd.Timedelta(minutes=after):
            return False, (f"before order window (opens {row['open']} ET "
                           f"+{after}m) — no orders")
        if n > c:
            return False, f"after session close {row['close']} ET"
        return True, f"in window {row['open']}+{after}m -> {row['close']} ET"
    return True, f"market open (no calendar; open+{after}m assumed by cron)"


def pdt_guard(orders_today: list[dict], trades: list[dict]) -> tuple[
        list, list]:
    """PDT guard for accounts < $25k: never open and close the same symbol
    the same day. Drops violating trades; returns (kept, blocked_notes).

    Two layers:
      1. within this batch — a symbol can never appear in both a buy and a
         sell list the same day (belt: assert, see below)
      2. vs orders already submitted today — a sell is blocked if a buy
         for that symbol was already submitted/filled today (and vice
         versa), even if the earlier order came from a previous run.
    """
    sides: dict[str, set] = {}
    for o in orders_today:
        s = str(o.get("symbol", "")).replace(".", "-")
        if o.get("status") in ("canceled", "expired", "rejected"):
            continue
        sides.setdefault(s, set()).add(o.get("side"))
    kept, blocked = [], []
    for t in trades:
        prev = sides.get(t["sym"], set())
        if prev and t["side"] not in prev:
            blocked.append(f"PDT: {t['sym']} {t['side']} blocked — "
                           f"{'/'.join(sorted(prev))} already today")
            continue
        sides.setdefault(t["sym"], set()).add(t["side"])
        kept.append(t)
    buys = {t["sym"] for t in kept if t["side"] == "buy"}
    sells = {t["sym"] for t in kept if t["side"] == "sell"}
    assert not (buys & sells), "PDT violation: symbol in both buy+sell"
    return kept, blocked


def risk_gate(cfg: dict, journal, equity: float, prev_state: dict,
              step: dict, prices: dict, closes: pd.DataFrame,
              day: pd.Timestamp) -> dict:
    """All pre-order checks + breakers in one place. Returns a dict:
      {buy_ok, order_ok, notes, checks, drawdown, daily_pnl, peak}
    `order_ok=False` aborts EVERYTHING (sanity failure); `buy_ok=False`
    blocks only buys (breakers — sells still allowed to de-risk).
    Never raises."""
    rk = cfg.get("risk", {})
    notes, checks = [], {}
    order_ok, buy_ok = True, True
    # the SPY core is an index — exempt from the single-stock cap and the
    # universe-size check (both exist to govern the stock sleeve)
    core_sym = (cfg.get("core") or {}).get("symbol")

    # -- sanity: abort everything on failure ---------------------------------
    n_sig = len(step.get("trades", []))
    # prices covers scored+held names; a tiny count means broken bars.
    # Only enforced when a non-core trade is planned — a core-only run
    # (pure SPY book, or signals dropped -> park in core) doesn't depend
    # on the stock universe's data health.
    noncore_trade = any(t["sym"] != core_sym
                        for t in step.get("trades", []))
    if rk.get("min_universe") and noncore_trade \
            and len(prices) < int(rk["min_universe"]):
        order_ok = False
        checks["universe"] = (f"only {len(prices)} priced names < "
                              f"{rk['min_universe']} — data broken?")
    w_sum = sum(t["notional"] for t in step.get("trades", [])
                if t["side"] == "buy")
    if equity > 0 and w_sum > equity * (1 + 1e-6) * \
            float(rk.get("max_daily_notional_frac", 1.0)):
        order_ok = False
        checks["notional"] = (f"buy notional ${w_sum:.2f} > "
                              f"equity ${equity:.2f} cap")
    mm = float(rk.get("max_price_move", 0.5))
    for t in step.get("trades", []):
        s = t["sym"]
        if s not in closes.columns:
            continue
        ser = closes[s].loc[:day].dropna()
        if len(ser) >= 2:
            mv = ser.iloc[-1] / ser.iloc[-2] - 1
            if abs(mv) > mm:
                order_ok = False
                checks.setdefault("price_sanity", []).append(
                    f"{s} {mv:+.0%} vs prior close")
    if not step.get("trades"):
        checks["signal"] = "no trades computed"
    checks["n_planned_orders"] = n_sig
    if any(k in checks for k in ("universe", "notional", "price_sanity")):
        notes.append("SANITY FAIL: " + "; ".join(
            f"{k}={v}" for k, v in checks.items()
            if k != "n_planned_orders"))

    # -- breakers: block buys only -------------------------------------------
    peak = float(prev_state.get("peak", equity) or equity)
    peak = max(peak, equity)
    dd = equity / peak - 1 if peak else 0.0
    max_dd = float(rk.get("max_drawdown", 0.25))
    if dd < -max_dd:
        buy_ok = False
        notes.append(f"BREAKER drawdown {dd:.1%} < -{max_dd:.0%} from "
                     f"peak ${peak:.2f} — new buys halted")
    prev_eq = float(prev_state.get("equity_usd", 0.0) or 0.0)
    daily = equity / prev_eq - 1 if prev_eq else 0.0
    max_daily = float(rk.get("daily_loss", 0.05))
    if prev_eq and daily < -max_daily:
        buy_ok = False
        notes.append(f"BREAKER daily loss {daily:.1%} < -{max_daily:.0%} "
                     "— new buys halted")
    max_w = float(rk.get("max_position_weight", 0.4))
    if equity > 0:
        for t in step.get("trades", []):
            if t["side"] == "buy" and t["sym"] != core_sym \
                    and t["notional"] / equity > max_w:
                t["notional"] = round(equity * max_w, 2)
                notes.append(f"buy {t['sym']} clipped to {max_w:.0%} of "
                             "equity")
    mo = int(rk.get("max_orders", 12))
    if len(step.get("trades", [])) > mo:
        # keep all sells (de-risking), drop lowest-priority buys
        sells = [t for t in step["trades"] if t["side"] == "sell"]
        buys = [t for t in step["trades"] if t["side"] == "buy"]
        buys = buys[:max(0, mo - len(sells))]
        step["trades"] = sells + buys
        notes.append(f"order cap {mo}: dropped excess buys")
    return {"order_ok": order_ok, "buy_ok": buy_ok, "notes": notes,
            "checks": checks, "drawdown": dd, "daily_pnl": daily,
            "peak": peak}


def corp_actions(journal, held: dict, day: pd.Timestamp,
                 prices: dict, ctx: dict) -> list[str]:
    """A ticker with an open BUY page that is no longer an Alpaca position
    vanished without a sell — split-reverse into nothing, delisting, or a
    symbol change. Close it out in the journal so the audit stays honest.
    Returns notes."""
    notes = []
    try:
        open_buys = journal.open_buys() if hasattr(journal, "open_buys") \
            else []
    except Exception as e:  # noqa: BLE001
        return [f"open_buys query failed: {e}"]
    for sym in open_buys:
        if sym not in held or held.get(sym, {}).get("qty", 0) <= 1e-9:
            px = prices.get(sym)
            journal.close_out(sym, px or 0.0,
                              "corporate action: position vanished "
                              "(delist/split/symbol change)")
            notes.append(f"corp-action close {sym} @ "
                         f"{px if px else 'n/a'}")
    return notes


def drift_report(want: list[str], n_positions: int, exposure: float,
                 held: dict, prices: dict, equity: float,
                 targets: dict | None = None) -> str:
    """Actual vs target weights — logged every run so silent drift (missed
    fills, stale positions) is visible in the journal. `targets` = the
    explicit core+satellite map when the book runs that way."""
    if equity <= 0:
        return "drift n/a (no equity)"
    tgt = dict(targets) if targets is not None else \
        {s: exposure / max(len(want), 1) for s in want}
    act = {s: p["qty"] * prices.get(s, 0.0) / equity
           for s, p in held.items() if prices.get(s)}
    rows = []
    for s in sorted(set(tgt) | set(act)):
        d = act.get(s, 0.0) - tgt.get(s, 0.0)
        if abs(d) > 0.02:
            rows.append(f"{s} {d:+.1%}")
    return ("drift ok (max |Δ| <=2%)" if not rows
            else "drift: " + ", ".join(rows))


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
    """Per-signal {sym: score} for `day`, plus IC-history for ewic weights.

    Phase-8 additions: `sn_<base>` legs are the base signal sector-
    demeaned with cfg.sector_map; `sleeve.ic_gate` drops the mom leg on
    days its trailing EW rank-IC has gone non-positive (vq is declared
    ungateable — a snapshot signal has no live IC history)."""
    sig_scalars: dict[str, pd.Series] = {}
    notes: list[str] = []
    sig_series: dict[str, pd.Series] = {}        # only populated for ewic
    days_lookback = closes.index[-min(len(closes.index), 260):]
    smap = LV.load_sector_map(REPO / cfg["sector_map"]) \
        if cfg.get("sector_map") else {}
    for sig in cfg["signals"]:
        base = sig[3:] if sig.startswith("sn_") else sig
        if base in LV._BARS_FNS or base == "beta":
            if base == "beta":
                mkt = closes.pct_change().mean(axis=1)
                sig_scalars[sig] = pd.Series(
                    {s: LV.lowbeta(closes[s], mkt, day)
                     for s in closes.columns})
            else:
                sig_scalars[sig] = pd.Series(
                    {s: LV._BARS_FNS[base](closes[s], day)
                     for s in closes.columns})
            if cfg["weighting"] == "ewic":
                sig_series[sig] = LV.bars_signals(
                    closes, days_lookback, [base])[base]
        elif base in ("vq", "news"):
            snap = state_dir / f"{base}_snapshot.parquet"
            s_, note = LV.snapshot_signal(
                snap, base, stale_days=cfg.get("snapshot_stale_days", 45),
                asof_day=day)
            notes.append(note or f"{sig}: snapshot fresh")
            if not s_.empty:
                sig_scalars[sig] = s_
        else:
            notes.append(f"{sig}: not computable live (needs the monthly-"
                         "retrain store pipeline) — dropped")
            log(f"DROPPED {sig}: no live source")
        if sig.startswith("sn_") and sig in sig_scalars:
            sig_scalars[sig] = LV.sector_neutralise(
                sig_scalars[sig].dropna(), smap)
    scalars = {s: v.dropna() for s, v in sig_scalars.items()}
    for s in cfg["signals"]:
        if s not in scalars:
            log(f"signal {s}: no scores — excluded from weights")
    gate_cfg = (cfg.get("sleeve") or {}).get("ic_gate")
    if gate_cfg and "mom" in scalars:
        gcfg = {**gate_cfg, "fwd_h": int(cfg.get("fwd_h", 21))}
        if LV.leg_ic_on(closes, day, gcfg):
            notes.append("ic_gate: mom trailing rank-IC positive — leg on")
        else:
            scalars.pop("mom")
            notes.append("ic_gate: mom trailing rank-IC <= 0 — leg off")
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


def run_day(args) -> None:

    cfg = load_config(Path(args.config))
    state_dir = Path(args.state_dir)
    # live-trading interlock: needs the config flag AND the ack secret —
    # PaperClient adds the third gate by refusing non-paper URLs unless
    # allow_live is set here. With flag unset this can never go live.
    allow_live = False
    if cfg.get("live_trading"):
        if os.environ.get("LIVE_TRADING_ACK") != "I_UNDERSTAND_LIVE_RISK":
            sys.exit("live_trading: true in config but the "
                     "LIVE_TRADING_ACK secret is not "
                     "I_UNDERSTAND_LIVE_RISK — refusing (see "
                     "docs/BOT_SETUP.md)")
        allow_live = True
        log("WARNING: live_trading flag set + ack present — non-paper "
            "base URLs are now permitted")
    validated = bool(cfg.get("validated"))
    banner = "VALIDATED" if validated else "UNVALIDATED"
    # Phase-7 core+satellite schema: `core` pins a SPY share of equity,
    # `sleeve` sizes/weights the signal book around it. Absent `core` the
    # legacy top-N equal-weight path runs untouched.
    core_cfg = cfg.get("core") or {}
    use_core = bool(core_cfg)
    # None in legacy mode — the core exemption must NOT apply to a
    # top-N book where SPY is just another signal pick
    core_sym = core_cfg.get("symbol", "SPY") if use_core else None
    sleeve_cfg = cfg.get("sleeve") or {}
    sleeve_n = int(sleeve_cfg.get("n", cfg["n_positions"])) \
        if use_core else cfg["n_positions"]
    log(f"=== paper_trade {banner} config={args.config} "
        f"signals={cfg['signals']} weighting={cfg['weighting']} "
        f"regime={cfg['regime']}"
        + (f" core={core_cfg.get('weight'):.0%}{core_sym} "
           f"sleeve={sleeve_n}x{sleeve_cfg.get('weighting', 'equal')}"
           if use_core else ""))

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
            key, secret, os.environ.get("ALPACA_BASE_URL", AB.PAPER_URL),
            allow_live=allow_live)
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

    # -- clock / staleness -----------------------------------------------------
    clock_note = market_clock(broker, journal, day, dry)
    log(clock_note)

    # -- regime -------------------------------------------------------------------
    spy = closes["SPY"].dropna()
    rc = dict(cfg["regime_cfg"])
    rc["action"] = rc["actions"][cfg["regime"]]  # none/half/cash -> exposure
    exposure, why = LV.regime_state(spy, rc, day)
    log(f"regime: {why} -> exposure {exposure}")
    shrink_cfg = sleeve_cfg.get("ic_shrink")
    if shrink_cfg:
        sc = LV.sleeve_ic_scale(
            closes, day, {**shrink_cfg, "fwd_h": int(cfg.get("fwd_h", 21))})
        exposure = round(exposure * sc, 4)
        log(f"ic_shrink: trailing-IC sleeve scale {sc:.2f} "
            f"-> exposure {exposure:.2f}")

    # -- signals + weights ---------------------------------------------------------
    # pick_freeze: monthly — the sleeve list is fixed at the month's first
    # trading day (recomputed there deterministically; needs no state).
    pick_day = day
    if sleeve_cfg.get("pick_freeze") == "monthly":
        mdays = closes.index[(closes.index <= day)
                             & (closes.index.year == day.year)
                             & (closes.index.month == day.month)]
        if len(mdays) and mdays[0] != day:
            pick_day = mdays[0]
            log(f"pick_freeze: sleeve list fixed at month open "
                f"{pick_day.date()}")
    sig_scalars, weights, notes = compute_signals(cfg, closes, pick_day,
                                                  state_dir)
    for n in notes:
        log(f"  {n}")
    if not sig_scalars and not use_core:
        fail(journal, "no usable signals — nothing to trade")
    if not sig_scalars:
        log("no usable sleeve signals — book parks in the core")
        notes.append("no usable sleeve signals — book parks in the core")
    weights_str = ", ".join(f"{k}:{v:.2f}" for k, v in weights.items())
    log(f"weights: {weights_str or '(none — pure core)'}")

    comp = LV.final_scores(sig_scalars, weights, pick_day)
    held = broker.positions()
    # prices for every scored name AND every held name — held names that fell
    # out of the universe still need a price so they can be sold, not zombied
    px_syms = set(comp.index) | set(held) \
        | ({core_sym} if use_core else set())
    prices = {}
    for s in px_syms:
        if s in closes.columns:
            ser = closes[s].loc[:day].dropna()
            if len(ser):
                prices[s] = float(ser.iloc[-1])
    scores = comp[comp.index.isin(prices)]
    smap_w = LV.load_sector_map(REPO / cfg["sector_map"]) \
        if cfg.get("sector_map") else {}
    want = CS.pick_top(scores, sleeve_n, smap_w,
                       sleeve_cfg.get("sector_max"))
    log(f"top-{sleeve_n} (sector_max={sleeve_cfg.get('sector_max')}): "
        f"{', '.join(want) or '(none — flat)'}")

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
    spy_px = float(spy.loc[:day].iloc[-1]) if len(spy.loc[:day]) else None
    ctx = {"weights_str": weights_str, "regime": why,
           "validated": validated, "budget_usd": budget_usd, "fx": fx,
           "summary": f"{banner} | picks {want} | exposure {exposure} | "
                      f"fx {fx:.4f} | {clock_note} | "
                      f"notes: {'; '.join(notes)}"}
    rec = {"day": str(day.date()), "config": args.config,
           "sha": os.environ.get("GITHUB_SHA", "local"),
           "validated": validated, "want": want, "exposure": exposure,
           "fx_gbpusd": round(fx, 6), "gates": {}, "orders": [],
           "notes": list(notes)}

    # -- fill reconciliation (runs every run, even idempotent skips) ------------
    recon = reconcile(broker, journal, day, held, spy_px, ctx, fx=fx)
    for n in recon["notes"]:
        log(f"  recon: {n}")
    ctx["recon_state"] = recon["state"]
    ctx["summary"] += f" | {recon['line']}"
    if recon["notes"]:
        ctx["summary"] += " | recon: " + "; ".join(recon["notes"][:4])
    log(recon["line"])

    # -- corporate actions: held-by-journal symbols that vanished ----------------
    for n in corp_actions(journal, held, day, prices, ctx):
        log(f"  corp: {n}")
        rec["notes"].append(n)

    # -- BOT_HALT kill switch ------------------------------------------------------
    if os.environ.get("BOT_HALT") == "1":
        reason = "BOT_HALT=1 — owner kill switch; run reconciled + " \
                 "journaled but placed no orders"
        log(reason)
        rec["gates"]["bot_halt"] = reason
        try:
            journal.log_halt(day, reason, ctx)
        except Exception as e:  # noqa: BLE001
            log(f"halt journal write failed: {e}")
        ctx["summary"] += f" | HALT ({reason})"
        if not journaled:
            journal.log_day_summary(day, ctx)
        _write_rec(rec, args, day)
        return

    # -- order window (calendar-gated; holidays + half-days handled) -------------
    if dry:
        win_ok, win_note = True, "dry-run: window forced open"
    else:
        now = pd.Timestamp.now(tz="UTC")
        win_ok, win_note = order_window(broker, now, cfg, dry)
    rec["gates"]["order_window"] = win_note
    log(f"order window: {win_note}")

    if already:
        log(f"{len(already)} order(s) already submitted {day.date()} — "
            "idempotent skip (no new orders)")
        if not journaled:
            journal.log_day_summary(day, ctx)
        _write_rec(rec, args, day)
        return
    if not win_ok:
        ctx["summary"] += f" | no orders: {win_note}"
        if not journaled:
            journal.log_day_summary(day, ctx)
        _write_rec(rec, args, day)
        return

    # -- cancel stale open orders before rebalancing -------------------------------
    if not dry:
        try:
            n_cancelled = broker.cancel_all()
            if n_cancelled:
                log(f"cancelled {n_cancelled} stale open order(s)")
                rec["gates"]["cancelled_stale"] = n_cancelled
        except Exception as e:  # noqa: BLE001
            log(f"cancel_all failed (continuing): {e}")

    # -- trade ----------------------------------------------------------------------
    if use_core:
        iv = CS.invvol(closes, want, day) \
            if sleeve_cfg.get("weighting") == "invvol" else None
        targets = CS.build_targets(
            want, float(core_cfg.get("weight", 0.0)), invvol_w=iv,
            exposure=exposure,
            risk_off=sleeve_cfg.get("risk_off", "to_core"),
            core_sym=core_sym,
            cap=float(cfg.get("risk", {}).get("max_position_weight",
                                             0.20)))
        log("targets: " + ", ".join(
            f"{s} {v:.0%}" for s, v in sorted(
                targets.items(), key=lambda kv: -kv[1])))
        rec["targets"] = {s: round(v, 6) for s, v in targets.items()}
        step = MC.weights_step(
            targets, budget_usd, prices,
            {s: p["qty"] for s, p in held.items()},
            cfg.get("cost_bps", 0.0) / 1e4,
            band=float(cfg.get("rebalance_band", 0.02)))
    else:
        targets = None
        step = MC.rebalance_step(
            scores, budget_usd, cfg["n_positions"], prices,
            {s: p["qty"] for s, p in held.items()},
            cfg.get("cost_bps", 0.0) / 1e4, exposure)

    # -- pre-order sanity + breakers -------------------------------------------------
    gate = risk_gate(cfg, journal, recon["equity"], recon["prev"],
                     step, prices, closes, day)
    rec["gates"]["risk"] = {"order_ok": gate["order_ok"],
                            "buy_ok": gate["buy_ok"],
                            "checks": gate["checks"],
                            "drawdown": round(gate["drawdown"], 4),
                            "daily_pnl": round(gate["daily_pnl"], 4)}
    for n in gate["notes"]:
        log(f"  risk: {n}")
    if not gate["order_ok"]:
        ctx["summary"] += " | ABORT: sanity checks failed — no orders"
        journal.log_error(day, "sanity abort: " + "; ".join(
            gate["notes"]) or "see checks", ctx)
        if not journaled:
            journal.log_day_summary(day, ctx)
        _write_rec(rec, args, day)
        return

    # -- PDT: same-day open+close of one symbol is impossible ------------------------
    try:
        today_ords = broker.orders_since(day.normalize())
    except Exception:  # noqa: BLE001
        today_ords = []
    kept, pdt_blocked = pdt_guard(today_ords, step["trades"])
    for n in pdt_blocked:
        log(f"  {n}")
    if pdt_blocked:
        rec["gates"]["pdt_blocked"] = pdt_blocked
        ctx["summary"] += " | " + "; ".join(pdt_blocked)
    trades = kept

    filled, skipped = [], []
    sells = [t for t in trades if t["side"] == "sell"]
    # breaker day: sleeve buys halt but the SPY core still buys — freed
    # sleeve cash migrates into the core instead of idling (phase-7 rule)
    buys = ([t for t in trades if t["side"] == "buy"]
            if gate["buy_ok"]
            else [t for t in trades if t["side"] == "buy"
                  and t["sym"] == core_sym])
    if not gate["buy_ok"]:
        skipped.append("all buys" + (" except the core" if core_sym
                                     else "") +
                       " (breaker active — sells still allowed)")
    for t in sells + buys:                     # sells free cash first
        sym, side = t["sym"], t["side"]
        t["score"] = float(comp.get(sym, np.nan))
        if use_core:
            t["reason"] = (
                "exit sleeve" if side == "sell" and sym not in targets
                else "regime->core" if side == "sell" and exposure < 1
                and sym != core_sym
                else "core rebalance" if sym == core_sym
                else "enter/rebalance")
        else:
            t["reason"] = ("exit top-5" if side == "sell" and sym not in want
                           else "regime trim" if side == "sell" and
                           exposure < 1 else "enter/rebalance")
        coid = AB.client_order_id(day, sym, side)
        t["client_order_id"] = coid
        if side == "sell" and buys and t is sells[-1] and not dry:
            time.sleep(3)               # let sells settle -> buying power
        try:
            if side == "buy":
                res = broker.buy_notional(sym, t["notional"], t["reason"],
                                          coid=coid)
                if res is None:
                    skipped.append(f"buy {sym} ${t['notional']:.2f}<min")
                    continue
                filled.append(t)
            else:
                qty = float(t["qty"])
                res = broker.sell_qty(sym, qty, t["reason"], coid=coid)
                if res is None:
                    skipped.append(f"sell {sym} rejected")
                    continue
                filled.append(t)
                if step["held"].get(sym, 0.0) <= 1e-9:
                    journal.close_out(sym, t["price"], t["reason"])
        except AB.OrderExistsError:
            filled.append(t)
            rec["orders"].append({"sym": sym, "side": side,
                                  "client_order_id": coid,
                                  "status": "already-exists"})
            log(f"{side} {sym}: client_order_id exists — idempotent skip")
        rec["orders"].append({"sym": sym, "side": side,
                              "notional": t.get("notional"),
                              "qty": t.get("qty"),
                              "client_order_id": coid})
        journal.log_trade(day, t, ctx)
    if skipped:
        ctx["summary"] += " | skipped: " + ", ".join(skipped)

    # -- drift: actual vs target weights, logged every run --------------------------
    held_after = broker.positions() if not dry else held
    drift = drift_report(want, cfg["n_positions"], exposure, held_after,
                         prices, recon["equity"], targets=targets)
    log(drift)
    rec["gates"]["drift"] = drift
    ctx["summary"] += f" | {drift}"

    # -- monthly what-changed --------------------------------------------------------
    wstate = state_dir / "weights.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    old_w = json.loads(wstate.read_text()) if wstate.exists() else {}
    month = day.strftime("%Y-%m")
    if old_w.get("month") != month or old_w.get("weights") != weights:
        # the runner's weights.json doesn't persist between CI runs — the
        # journal is the durable record, so dedupe against it too
        mday = pd.Timestamp(f"{month}-01")
        logged = journal.day_logged(mday, "weights-change") \
            if hasattr(journal, "day_logged") else False
        if not logged:
            journal.log_weights_change(
                f"{month}-01", old_w.get("weights", {}), weights, ctx)
        wstate.write_text(json.dumps({"month": month, "weights": weights,
                                      "config": cfg["_path"]}, indent=1))

    # -- journal + dump ----------------------------------------------------------------
    journal.log_day_summary(day, ctx)
    rec["orders_filled"] = len(filled)
    rec["skipped"] = skipped
    _write_rec(rec, args, day)
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


def _write_rec(rec: dict, args, day: pd.Timestamp) -> None:
    """Structured per-run JSON — uploaded as a workflow artifact so every
    gate decision is auditable from a commit SHA."""
    try:
        out = Path(args.rec_dir) if getattr(args, "rec_dir", None) \
            else REPO / "results" / "paper_run"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{day.strftime('%Y-%m-%d')}.json"
        path.write_text(json.dumps(rec, indent=1, default=str))
        log(f"run record -> {path}")
    except Exception as e:  # noqa: BLE001
        log(f"rec write failed (non-fatal): {e}")


def parse_args(argv=None):
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
    p.add_argument("--rec-dir", default="",
                   help="where the structured run JSON lands (default "
                        "results/paper_run/)")
    return p.parse_args(argv)


def fail_loudly(args, e: Exception) -> None:
    """Crash path: error entry in the journal (best effort) + exit 1 so the
    workflow goes red. Journal failures can't mask the original error."""
    log(f"FATAL {type(e).__name__}: {e}")
    try:
        if not getattr(args, "dry_run", False):
            j = NJ.NotionJournal(
                os.environ.get("NOTION_TOKEN", ""),
                os.environ.get("NOTION_DB_ID", ""),
                os.environ.get("NOTION_PARENT_PAGE_ID", ""))
            j.log_error(pd.Timestamp.now(),
                        f"{type(e).__name__}: {e}", {})
    except Exception as e2:  # noqa: BLE001
        log(f"error journal write failed: {e2}")
    sys.exit(1)


def main() -> None:
    args = parse_args()
    try:
        run_day(args)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        fail_loudly(args, e)


if __name__ == "__main__":
    main()
