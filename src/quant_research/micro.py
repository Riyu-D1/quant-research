"""Micro-account backtest: £10-equivalent Alpaca paper account.

Long-only, fractional shares, max N positions, commission-free with a flat
per-side cost modelling spread/slippage. This is deliberately NOT a qlib
strategy — qlib's account model can't express a $13.50 book with fractional
shares. `micro_backtest` returns the same metric dict shape as
walkforward.backtest_signal so it drops into the candidate table under the
"micro5" strategy key, and `rebalance_step` is reused by paper_trade.py for
live sizing.

Alpaca fractional rules honoured (approx.):
  - notional orders, minimum $1.00 — with ~$2.70/slot that's always met
  - fractional quantities to 9 decimal places
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GBPUSD = 1.27            # conservative static rate; paper_trade re-fetches live
MIN_ORDER_USD = 1.0


def rebalance_step(scores: pd.Series, equity: float, n: int,
                   prices: dict[str, float], held: dict[str, float],
                   cost, exposure: float = 1.0) -> dict:
    """One day: pick top-N scored names, equal-weight, fractional fills.

    scores: {instrument: score} for that day (NaN excluded upstream).
    prices: {instrument: price} dealable price for the day.
    held:   {instrument: qty} current book.
    cost:   per-side cost as a scalar fraction, or a {instrument: fraction}
            dict (the liquidity cost model's per-name half-spread + flat).
    exposure: fraction of equity to hold in stock (regime guard); the rest
              stays cash. 0 => liquidate everything.
    Returns dict(trades=[{sym, side, qty, price, notional}], held=..., cash=...).
    """
    c_of = cost.get if isinstance(cost, dict) else lambda _s, _d=None: cost
    want = list(scores.sort_values(ascending=False)
                .head(n).index)
    want = [s for s in want if s in prices and np.isfinite(prices[s])
            and prices[s] > 0]
    slot = equity * exposure / max(len(want), 1)
    trades = []
    new_held = dict(held)

    # sells first (freeing cash), then buys
    for s in list(new_held):
        if s not in want:
            px = prices.get(s)
            if px is None or not np.isfinite(px) or px <= 0:
                continue               # can't trade; keep the zombie position
            qty = new_held.pop(s)
            px = px * (1 - c_of(s, 0.0))
            trades.append({"sym": s, "side": "sell", "qty": qty,
                           "price": px, "notional": qty * px})
    # target notional per wanted name = slot; buy/sell the delta
    cash_used = 0.0
    for s in want:
        px = prices[s] * (1 + c_of(s, 0.0))
        tgt_notional = slot
        cur_notional = new_held.get(s, 0.0) * px
        delta = tgt_notional - cur_notional
        if delta >= MIN_ORDER_USD:
            new_held[s] = new_held.get(s, 0.0) + delta / px
            trades.append({"sym": s, "side": "buy", "qty": delta / px,
                           "price": px, "notional": delta})
            cash_used += delta
        elif delta <= -MIN_ORDER_USD:
            qty = -delta / px
            new_held[s] = max(0.0, new_held.get(s, 0.0) - qty)
            trades.append({"sym": s, "side": "sell", "qty": qty,
                           "price": px, "notional": -delta})
    return {"trades": trades, "held": new_held, "want": want}


def weights_step(targets: dict[str, float], equity: float,
                 prices: dict[str, float], held: dict[str, float],
                 cost, band: float = 0.02,
                 min_order: float = MIN_ORDER_USD) -> dict:
    """One day: trade toward explicit weight targets {sym: equity frac}.

    The core+satellite counterpart of rebalance_step — used by both the
    phase-7 simulator (micro_backtest weight_targets=) and the live bot
    so research and production trade the identical map.

    Band rule (turnover cut): a trade is skipped when the weight gap
    |target - actual| < `band`, OR the notional is < `min_order` (the $1
    Alpaca floor). Outright exits (target 0) always trade — the band is
    a drift filter, not a freeze. Sells run first (frees cash); each
    symbol emits at most one order — the PDT guard's effect by
    construction.
    """
    c_of = cost.get if isinstance(cost, dict) else lambda _s, _d=None: cost
    trades, new_held = [], dict(held)

    for s in sorted(new_held):
        px = prices.get(s)
        if px is None or not np.isfinite(px) or px <= 0:
            continue                       # can't trade; keep the zombie
        cur_w = new_held[s] * px / equity if equity > 0 else 0.0
        tgt_w = float(targets.get(s, 0.0))
        if tgt_w <= 0:                     # exit — band does not apply
            qty = new_held.pop(s)
            px_ = px * (1 - c_of(s, 0.0))
            trades.append({"sym": s, "side": "sell", "qty": qty,
                           "price": px_, "notional": qty * px_})
            continue
        gap = tgt_w - cur_w
        if gap <= -band and -gap * equity >= min_order:
            qty = min(new_held[s],
                      round(-gap * equity / px, 6))
            px_ = px * (1 - c_of(s, 0.0))
            new_held[s] = max(0.0, new_held[s] - qty)
            if new_held[s] <= 1e-9:
                new_held.pop(s)
            trades.append({"sym": s, "side": "sell", "qty": qty,
                           "price": px_, "notional": qty * px_})
    for s in sorted(targets, key=lambda s: -targets[s]):
        px = prices.get(s)
        if px is None or not np.isfinite(px) or px <= 0:
            continue
        cur_w = new_held.get(s, 0.0) * px / equity if equity > 0 else 0.0
        gap = float(targets[s]) - cur_w
        # floor, not round: rounding each buy UP to cents can push the
        # batch total a cent over equity and trip the notional sanity cap
        notional = np.floor(gap * equity * 100) / 100
        if gap >= band and notional >= min_order:
            px_ = px * (1 + c_of(s, 0.0))
            qty = round(notional / px_, 9)
            new_held[s] = new_held.get(s, 0.0) + qty
            trades.append({"sym": s, "side": "buy", "qty": qty,
                           "price": px_, "notional": notional})
    return {"trades": trades, "held": new_held,
            "want": [s for s, v in targets.items() if v > 0]}


# ---------------------------------------------------------------------------
# realistic cost model (no paid bid/ask feed): liquidity-bucket half-spread
#
# Historical NBBO spreads aren't free, so we proxy the half-spread paid on a
# market order from the name's own daily dollar volume — the free, causal
# liquidity measure available in the dump. Buckets are coarse on purpose;
# the flat commission bps is added on top by the caller. Sensitivity is
# always reported across flat 0/10/25/50 bps as well.
HALF_SPREAD_BUCKETS = [  # (min daily dollar volume, half-spread bps)
    (1e9, 1.0),          # mega-liquid (SPY-class names, most S&P megacaps)
    (1e8, 2.0),
    (1e7, 5.0),
    (1e6, 10.0),
    (0.0, 20.0),         # effectively untraded / near-delisted
]


def half_spread_bps(dollar_vol: float) -> float:
    for floor, bps in HALF_SPREAD_BUCKETS:
        if dollar_vol >= floor:
            return bps
    return HALF_SPREAD_BUCKETS[-1][1]


def liquidity_cost_map(closes_long: pd.DataFrame,
                       volumes_long: pd.DataFrame,
                       flat_bps: float = 10.0) -> dict:
    """(datetime, instrument) -> per-side fractional cost = half-spread +
    flat_bps, bucketed by that day's dollar volume.

    closes_long/volumes_long: (datetime, instrument)-indexed frames with a
    'close'/'volume' column (load_bars output). Missing volume -> tightest
    bucket (names with no reported volume print as liquid on most feeds —
    documented approximation, still conservative via the flat bps)."""
    dv = (closes_long["close"] * volumes_long["volume"]).dropna()
    cost = {}
    for (d, s), v in dv.items():
        cost[(d, s)] = (half_spread_bps(float(v)) + flat_bps) / 1e4
    return cost


def _rebalance_day(day: pd.Timestamp, prev: pd.Timestamp | None,
                   freq: str) -> bool:
    """Whether `day` opens a new rebalance period (first priced day of the
    week/month, or every day)."""
    if freq == "daily" or prev is None:
        return True
    if freq == "weekly":
        return day.isocalendar()[:2] != prev.isocalendar()[:2]
    if freq == "monthly":
        return (day.year, day.month) != (prev.year, prev.month)
    raise ValueError(f"unknown rebalance freq {freq!r}")


def micro_backtest(pred: pd.Series, closes: pd.DataFrame, n: int = 5,
                   cash_gbp: float = 10.0, gbpusd: float = GBPUSD,
                   cost_bps: float = 10.0,
                   exposure: pd.Series | dict | None = None,
                   exec_prices: pd.DataFrame | None = None,
                   fill: str = "close",
                   delist_liquidate: bool = False,
                   rebalance: str = "daily",
                   cost_map: dict | None = None,
                   weight_targets: dict | None = None,
                   band: float = 0.02) -> dict:
    """Daily rebalanced long-only top-N on (datetime, instrument) scores.

    When `weight_targets` is given ({sig_day: {sym: equity fraction}},
    from coresat.build_targets), the book trades toward those weights via
    weights_step + `band` instead of top-N scores, and `pred`/`exposure`
    are ignored (the caller bakes regime/exposure into the targets).

    pred:   Series indexed (datetime, instrument), NaN = no opinion.
    closes: DataFrame indexed like pred, column 'close' — end-of-day marks
            and the signal day's price grid.
    exposure: optional per-day fraction of equity to invest (regime guard),
              keyed on the SIGNAL day; missing days default to 1.0.
    exec_prices: like `closes` but column 'open' — required for
            fill='next_open'.
    fill:   'close'     — decide and fill at the signal day's close
                          (the Phase-3/4 research convention);
            'next_open' — decide on the signal day, fill at the NEXT
                          trading day's open (the honest live convention:
                          the bot's queued DAY orders fill at the open).
    delist_liquidate: when a held name's priced history has ended (it left
            the universe), force-sell it at its last known close the next
            priced day — index exits pay out, they don't vanish.
    rebalance: 'daily' | 'weekly' | 'monthly' — trade only on the first
            priced day of each period (reported, not used for picking).
    cost_map: optional {(datetime, instrument): fractional per-side cost}
            overriding the flat cost_bps (liquidity bucket model).
    Returns walkforward-style metrics + trades ledger."""
    cash = cash_gbp * gbpusd
    flat = cost_bps / 1e4
    sig = pred.dropna().unstack()                      # day x inst scores
    px = closes["close"].unstack()
    ox = (exec_prices["open"].unstack()
          if exec_prices is not None else px)
    days = px.index                                    # all priced days
    exp = exposure if exposure is not None else {}
    held: dict[str, float] = {}
    equity_curve, ledger, turnover = [], [], []
    cost_paid, traded_frac = [], []
    cash_pos = cash
    last_px: dict[str, float] = {}                     # last seen close/sym
    last_bar = px.apply(lambda c: c.last_valid_index())  # delist detector
    cmap = cost_map or {}
    prev_day = None

    for day in days:
        marks = px.loc[day].dropna()
        for s, p in marks.items():
            last_px[s] = float(p)
        # fills: signal-day close vs next-day open (decision = prev day)
        if fill == "next_open":
            sig_day = prev_day
            deal = (ox.loc[day].dropna().to_dict()
                    if day in ox.index else {})
        elif fill == "close":
            sig_day, deal = day, marks.to_dict()
        else:
            raise ValueError(f"unknown fill {fill!r}")
        day_cost = (({s: cmap.get((day, s), flat) for s in deal}
                     | {s: cmap.get((day, s), flat) for s in held})
                    if cmap else flat)

        # delisting liquidation BEFORE sizing: a held name whose priced
        # history ended pays out at its last close (cost-adjusted).
        liq = []
        if delist_liquidate:
            for s in list(held):
                lb = last_bar.get(s)
                if lb is not None and lb < day:
                    lp = last_px.get(s)
                    if lp and np.isfinite(lp) and lp > 0:
                        q = held.pop(s)
                        lp *= (1 - (cmap.get((day, s), flat) if cmap
                                    else flat))
                        cash_pos += q * lp
                        liq.append({"date": day, "sym": s, "side": "sell",
                                    "qty": q, "price": lp,
                                    "notional": q * lp,
                                    "reason": "delist-liquidate"})
        ledger.extend(liq)

        # mark-to-market at the deal price for sizing
        positions_val = sum(q * deal.get(s, np.nan)
                            for s, q in held.items()
                            if np.isfinite(deal.get(s, np.nan)))
        equity = cash_pos + positions_val
        day_trades: list[dict] = []
        if _rebalance_day(day, prev_day, rebalance) and sig_day is not None:
            if weight_targets is not None:
                step = weights_step(weight_targets.get(sig_day, {}),
                                    equity, deal, held, day_cost,
                                    band=band)
            else:
                row = (sig.loc[sig_day].dropna() if sig_day in sig.index
                       else pd.Series(dtype=float))
                step = rebalance_step(row, equity, n, deal, held, day_cost,
                                      exposure=float(exp.get(sig_day, 1.0)))
            day_trades = step["trades"]
            bought = sum(t["notional"] for t in day_trades
                         if t["side"] == "buy")
            sold = sum(t["notional"] for t in day_trades
                       if t["side"] == "sell")
            cash_pos += sold - bought
            held = step["held"]
            ledger.extend({"date": day, **t} for t in day_trades)
        else:
            bought = sold = 0.0
        # end-of-day equity: cash + positions at close marks
        positions_val = sum(q * marks.get(s, 0.0) for s, q in held.items())
        equity = cash_pos + positions_val
        equity_curve.append((day, equity))
        day_traded = bought + sold + sum(l["notional"] for l in liq)
        turnover.append(day_traded / max(equity, 1e-9) / 2)
        traded_frac.append(day_traded / max(equity, 1e-9))
        rate = (day_cost.get if isinstance(day_cost, dict)
                else lambda _s, _d=flat: flat)
        cost_paid.append(
            sum(t["notional"] * rate(t["sym"], flat)
                for t in day_trades + liq) / max(equity, 1e-9))
        prev_day = day

    eq = pd.Series(dict(equity_curve)).sort_index()
    rets = eq.pct_change().dropna()
    ar = float((1 + rets).prod() ** (252 / max(len(rets), 1)) - 1) \
        if len(rets) else 0.0
    ir = float(rets.mean() / rets.std() * np.sqrt(252)) \
        if rets.std() > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    return {
        "annualized_return": ar,
        "information_ratio": ir,
        "max_drawdown": mdd,
        "turnover": float(np.mean(turnover)) if turnover else 0.0,
        "n_days": len(eq),
        "final_equity_usd": float(eq.iloc[-1]) if len(eq) else cash,
        "trades": ledger,
        "daily_excess": rets,      # benchmark-free; excess = total return
        # equity-fraction series for cost-sensitivity maths (break-even
        # per-side cost): daily_cost = paid costs, daily_traded = gross
        # two-side traded fraction
        "daily_cost": pd.Series(cost_paid, index=eq.index),
        "daily_traded": pd.Series(traded_frac, index=eq.index),
    }


def load_closes(data_dir, instruments_name: str = "sp500") -> pd.DataFrame:
    """All-universe closes from the qlib bins -> (datetime, instrument)."""
    return load_bars(data_dir, instruments_name, ("close",))["close"]


def load_bars(data_dir, instruments_name: str = "sp500",
              fields: tuple = ("close", "open", "volume"),
              masked: bool = True) -> dict[str, pd.DataFrame]:
    """All-universe qlib bins -> {field: (datetime, instrument) frame}.

    masked=True (default): each symbol is masked to its instruments-file
    membership window — the point-in-time universe (listed/member as of
    each day, delisted names included). masked=False keeps each symbol's
    full data span (for signal math that legitimately reads pre-membership
    price history)."""
    from quant_research import events as ev
    cal = ev.load_calendar(data_dir).rename("datetime")
    inst = ev.load_instruments(data_dir, instruments_name)
    frames = {f: [] for f in fields}
    # re-entrant members appear once per membership window — dedupe them or
    # the unmasked path would duplicate every (datetime, instrument) pair
    seen: set = set()
    for r in inst.itertuples():
        if r.instrument in seen:
            continue
        if not masked:
            seen.add(r.instrument)
        mask = ((cal >= pd.Timestamp(r.start)) & (cal <= pd.Timestamp(r.end))
                if masked else slice(None))
        for f in fields:
            vals = ev.load_field(data_dir, r.instrument, cal, f)
            s = pd.Series(vals, index=cal, name=f)
            frames[f].append(s[mask].to_frame()
                             .assign(instrument=r.instrument)
                             .set_index("instrument", append=True))
    return {f: pd.concat(parts).sort_index() for f, parts in frames.items()}
