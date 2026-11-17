"""Core+satellite target construction — the Phase-7 live allocation.

Shared by the honest simulator (scripts/phase7_us.py, via
micro.weights_step) and the daily bot (scripts/paper_trade.py) so the
{symbol: equity fraction} map the research scores is literally the same
map production trades toward.

Book = X% SPY core + (1-X)% signal sleeve of top-N names. On a risk-off
regime day the sleeve's freed share moves into the core (risk_off=
"to_core") or sits in cash (risk_off="cash") — never into a fresh bet.
No single stock may exceed `cap` of total equity (default 20%); the
capped overflow lands on the core symbol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CAP = 0.20


def pick_top(scores: pd.Series, n: int,
             sector_map: dict[str, str] | None = None,
             sector_max: int | None = None) -> list[str]:
    """Top-n names by score, optionally capped at `sector_max` picks per
    sector (greedy in rank order). Unmapped names form their own
    singleton sector ("UNK:SYM") so a missing label can never cap-deny
    a pick. Shared by the walk-forward sim and the daily bot — the same
    list the sim targets is the list the bot buys."""
    desc = scores.dropna().sort_values(ascending=False)
    if not sector_max:
        return desc.head(n).index.tolist()
    picks, cnt = [], {}
    for s in desc.index:
        sec = sector_map.get(s, f"UNK:{s}") if sector_map else f"UNK:{s}"
        if cnt.get(sec, 0) >= sector_max:
            continue
        picks.append(s)
        cnt[sec] = cnt.get(sec, 0) + 1
        if len(picks) >= n:
            break
    return picks


def invvol(closes: pd.DataFrame, picks: list[str], day: pd.Timestamp,
           win: int = 20) -> dict[str, float]:
    """Within-sleeve inverse-volatility weights: w_i ~ 1/sigma_i, where
    sigma_i is the std of the pick's last `win` daily returns on closes
    <= day (the same data LV.lowvol reads). Zero/near-zero or short
    history names drop out."""
    sig = {}
    for s in picks:
        if s not in closes.columns:
            continue
        px = closes[s].loc[:day].dropna()
        if len(px) < win + 1:
            continue
        v = float(px.pct_change().iloc[-win:].std())
        if np.isfinite(v) and v > 0:
            sig[s] = v
    inv = {s: 1.0 / v for s, v in sig.items()}
    tot = sum(inv.values())
    return {s: iv / tot for s, iv in inv.items()} if tot > 0 else {}


def build_targets(picks: list[str], core_w: float, *,
                  invvol_w: dict[str, float] | None = None,
                  exposure: float = 1.0, risk_off: str = "to_core",
                  core_sym: str = "SPY",
                  cap: float = DEFAULT_CAP) -> dict[str, float]:
    """{symbol: equity fraction} for one day's book.

    picks:        the sleeve names (top-N by signal; the core symbol is
                  stripped if it ever appears).
    core_w:       fraction of equity parked in the core (0..1); the rest
                  is the sleeve budget.
    invvol_w:     optional within-sleeve weights (invvol()); absent =>
                  equal weight across picks.
    exposure:     regime multiplier on the sleeve (1.0 = fully invested).
    risk_off:     where the freed (1-exposure) sleeve share goes —
                  "to_core" adds it to the core symbol, "cash" leaves it
                  uninvested (weights sum < 1 -> residual is cash).
    cap:          hard per-single-stock equity cap (0.20 = 20% of the
                  book); capped overflow lands on the core, which is an
                  index and is itself exempt.
    """
    core_w = float(core_w)
    sleeve_w = max(0.0, 1.0 - core_w)
    picks = [p for p in picks if p != core_sym]
    if not picks or sleeve_w <= 0:
        return {core_sym: 1.0}
    if invvol_w is not None:
        w = {s: float(invvol_w.get(s, 0.0)) for s in picks}
        w = {s: v for s, v in w.items() if v > 0}
        if not w:
            return {core_sym: 1.0}
        tot = sum(w.values())
        w = {s: v / tot for s, v in w.items()}
    else:
        w = {s: 1.0 / len(picks) for s in picks}
    invested = sleeve_w * float(exposure)
    freed = sleeve_w - invested
    out = {s: min(invested * wi, cap) for s, wi in w.items()}
    out[core_sym] = (core_w + invested - sum(out.values())
                     + (freed if risk_off == "to_core" else 0.0))
    return {s: v for s, v in out.items() if v > 1e-12}
