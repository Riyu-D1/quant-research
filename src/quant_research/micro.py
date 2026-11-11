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
                   cost: float, exposure: float = 1.0) -> dict:
    """One day: pick top-N scored names, equal-weight, fractional fills.

    scores: {instrument: score} for that day (NaN excluded upstream).
    prices: {instrument: close} dealable price for the day.
    held:   {instrument: qty} current book.
    exposure: fraction of equity to hold in stock (regime guard); the rest
              stays cash. 0 => liquidate everything.
    Returns dict(trades=[{sym, side, qty, price, notional}], held=..., cash=...).
    """
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
            px = px * (1 - cost)
            trades.append({"sym": s, "side": "sell", "qty": qty,
                           "price": px, "notional": qty * px})
    # target notional per wanted name = slot; buy/sell the delta
    cash_used = 0.0
    for s in want:
        px = prices[s] * (1 + cost)
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


def micro_backtest(pred: pd.Series, closes: pd.DataFrame, n: int = 5,
                   cash_gbp: float = 10.0, gbpusd: float = GBPUSD,
                   cost_bps: float = 10.0,
                   exposure: pd.Series | dict | None = None) -> dict:
    """Daily rebalanced long-only top-N on (datetime, instrument) scores.

    pred:   Series indexed (datetime, instrument), NaN = no opinion.
    closes: DataFrame indexed like pred, column 'close' deal price.
    exposure: optional per-day fraction of equity to invest (regime guard);
              missing days default to 1.0, 0.0 = all cash.
    Returns walkforward-style metrics + trades ledger."""
    cash = cash_gbp * gbpusd
    cost = cost_bps / 1e4
    sig = pred.dropna().unstack()                      # day x inst scores
    # iterate every trading day; a day with no scores means hold nothing
    px = closes["close"].unstack()
    days = px.index                                    # all priced days
    exp = exposure if exposure is not None else {}
    held: dict[str, float] = {}
    equity_curve, ledger, turnover = [], [], []
    cash_pos = cash

    for day in days:
        prices = px.loc[day].dropna().to_dict()
        row = (sig.loc[day].dropna() if day in sig.index
               else pd.Series(dtype=float))
        # mark-to-market (zombie positions keep last-known cost basis)
        positions_val = sum(q * prices.get(s, np.nan)
                            for s, q in held.items()
                            if np.isfinite(prices.get(s, np.nan)))
        equity = cash_pos + positions_val
        step = rebalance_step(row, equity, n, prices, held, cost,
                              exposure=float(exp.get(day, 1.0)))
        bought = sum(t["notional"] for t in step["trades"] if t["side"] == "buy")
        sold = sum(t["notional"] for t in step["trades"] if t["side"] == "sell")
        cash_pos += sold - bought
        held = step["held"]
        # end-of-day equity: cash + positions at (cost-adjusted) marks
        positions_val = sum(q * prices.get(s, 0.0) for s, q in held.items())
        equity = cash_pos + positions_val
        equity_curve.append((day, equity))
        ledger.extend({"date": day, **t} for t in step["trades"])
        denom = max(equity, 1e-9)
        turnover.append((bought + sold) / denom / 2)

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
    }


def load_closes(data_dir, instruments_name: str = "sp500") -> pd.DataFrame:
    """All-universe closes from the qlib bins -> (datetime, instrument)."""
    from quant_research import events as ev
    cal = ev.load_calendar(data_dir).rename("datetime")
    inst = ev.load_instruments(data_dir, instruments_name)
    frames = []
    for r in inst.itertuples():
        close = ev.load_close(data_dir, r.instrument, cal)
        s = pd.Series(close, index=cal, name="close")
        mask = (cal >= pd.Timestamp(r.start)) & (cal <= pd.Timestamp(r.end))
        frames.append(s[mask].to_frame()
                      .assign(instrument=r.instrument)
                      .set_index("instrument", append=True))
    return pd.concat(frames).sort_index()
