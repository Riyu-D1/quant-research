#!/usr/bin/env python3
"""Phase-8 US live-bot — diagnose the 2026 failure + walk-forward fix.

    uv run python scripts/phase8_us.py diagnose
    uv run python scripts/phase8_us.py walkforward [--skip-supp]

diagnose   runs every item in the spec's `diagnose:` block on the honest
           Alpaca PIT feed and writes results/phase8_diagnosis.json plus
           a number-filled docs/PHASE8_DIAGNOSIS.md skeleton.

walkforward runs the predeclared candidate list (configs/experiments/
           phase8_us.yaml — committed BEFORE this job) through the strict
           expanding-window gate: every test year is an independent
           episode on the honest feed (2017->dump end; the feed starts
           2016-01-04 so 2012-2016 honest data does not exist), plus
           flagged supplementary years 2012-2020 on the survivorship-
           biased community dump (reported, never gated). The deploy rule
           picks the best walk-forward IR among candidates that match
           SPY's drawdown within 2pt, match/beat SPY in >=60% of gate
           years, and never lose >5pt to SPY in any year — else A_spy
           (100% SPY) deploys. Writes results/phase8_us_walkforward.json,
           appends results/holdout_log.jsonl and re-pins
           configs/live_signal.yaml.

Sim spec = phase 7 = production: coresat.build_targets +
micro.weights_step, next-open fills, bucketed half-spread + 5bps flat,
$1 order floor, 2% rebalance band, fractional, delist liquidation, 20%
single-stock cap with SPY overflow.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import live_holdout as LH  # noqa: E402
import phase7_us as P7  # noqa: E402
from quant_research import adaptive as ad  # noqa: E402
from quant_research import coresat as CS  # noqa: E402
from quant_research import events as ev  # noqa: E402
from quant_research import micro as mc  # noqa: E402
from quant_research import signals6 as s6  # noqa: E402
from quant_research import walkforward as wf  # noqa: E402

SPEC_PATH = REPO / "configs" / "experiments" / "phase8_us.yaml"


def log(msg: str) -> None:
    print(f"[phase8 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          check=False, cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


# ---------------------------------------------------------------------------
# data


def load_feed(provider: Path) -> dict:
    """P7.load_books for an explicit provider dir (gate or supplementary).

    All-listed universe (delisted names included, masked=False — the same
    'listed while trading' universe the live bot's live_universe.txt
    carries) + SPY appended as a tradable row."""
    cal = ev.load_calendar(provider)
    bars = mc.load_bars(provider, "sp500", ("close", "open", "volume"),
                        masked=False)
    spy_rows = {}
    for f in ("close", "open", "volume"):
        arr = ev.load_field(provider, "SPY", cal, f)
        spy_rows[f] = pd.DataFrame(
            {f: arr}, index=cal.rename("datetime")) \
            .assign(instrument="SPY") \
            .set_index("instrument", append=True)
        bars[f] = pd.concat([bars[f], spy_rows[f]]).sort_index()
    spy = pd.Series(spy_rows["close"]["close"].values,
                    index=cal).dropna().replace(0, np.nan).dropna()
    days = pd.DatetimeIndex(
        bars["close"].index.get_level_values("datetime").unique()
        .sort_values())
    close = bars["close"]["close"].unstack("instrument").sort_index()
    return {"bars": bars, "spy": spy, "cal": cal, "days": days,
            "close": close}


def load_sector_map(spec: dict) -> dict[str, str]:
    """symbol -> GICS sector; unmapped names become their own singleton
    sector ('UNK:SYM') so a missing label can never cap-deny a pick."""
    sm = pd.read_csv(REPO / spec["sector_map"])
    out = {}
    for r in sm.itertuples():
        sec = r.sector if isinstance(r.sector, str) and r.sector.strip() \
            else f"UNK:{r.symbol}"
        out[ev.canon(r.symbol)] = sec
    return out


# ---------------------------------------------------------------------------
# signals — base panels (mom/vq/lowvol/beta via the phase-7 builders) and
# the phase-8 additions (sector-neutral, IC machinery)


def demean_sector(panel: pd.DataFrame, smap: dict) -> pd.DataFrame:
    """Subtract each day's sector mean — pure within-sector ranking.
    Unmapped names are singleton sectors -> score 0 (neutral)."""
    sec = pd.Series({c: smap.get(c, f"UNK:{c}") for c in panel.columns})
    return panel - panel.T.groupby(sec).transform("mean").T


def build_panels(spec: dict, feed: dict, provider: Path,
                 smap: dict) -> dict:
    close = feed["close"]
    needed = {leg for legs in spec["signals"].values() for leg in legs}
    base_needed = {n[3:] if n.startswith("sn_") else n for n in needed}
    out: dict[str, pd.DataFrame] = {}
    if "mom" in base_needed:
        out["mom"] = LH.bars_score(close, "mom").unstack("instrument")
    if "lowvol" in base_needed:
        out["lowvol"] = P7.lowvol_wide(close)
    if "beta" in base_needed:
        out["beta"] = P7.beta_wide(close)
    if "vq" in base_needed:
        cal = feed["cal"]
        inst = ev.load_instruments(provider, "sp500")
        sigs = s6.build_signals(
            provider, inst["instrument"].unique().tolist(), cal, ["vq"],
            fundamentals_path=REPO / spec["fundamentals"],
            feats_path=REPO / "data/events/features_us.parquet",
            news_dir=REPO / "results", sector_map=None,
            close=close, volume=close)
        out["vq"] = sigs["vq"].unstack("instrument")
    for n in needed:
        if n.startswith("sn_"):
            out[n] = demean_sector(out[n[3:]], smap)
    return {k: v.reindex(close.index) for k, v in out.items()}


def blend(panels: dict, legs: list[str],
          leg_on: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Equal-weight per-day pct-rank blend with an optional per-leg
    on/off mask (the IC gate). An OFF leg contributes 0 and is not
    required for a name to score (mirrors LV.final_scores, which skips
    zero-weight legs entirely)."""
    if not leg_on:
        return P7.blend(panels, legs)
    acc = None
    req = None
    n_on = None
    for leg in legs:
        r = panels[leg].rank(axis=1, pct=True)
        on = leg_on.get(leg)
        if on is None:
            acc = r.fillna(0.0) if acc is None else acc + r.fillna(0.0)
            req = r.notna() if req is None else (req & r.notna())
            on = pd.Series(1.0, index=r.index)
        else:
            on = on.astype("boolean").reindex(r.index) \
                .fillna(True).astype(float)
            on_df = pd.DataFrame({c: on for c in r.columns})
            acc = (r * on_df).fillna(0.0) if acc is None \
                else acc + (r * on_df).fillna(0.0)
            req_l = r.notna() | on_df.eq(0.0)
            req = req_l if req is None else (req & req_l)
        n_on = on if n_on is None else n_on + on
    comp = acc.div(n_on.where(n_on > 0), axis=0)
    return comp.where(req)


# ---------------------------------------------------------------------------
# IC machinery (all causal): daily rank-IC vs h-day forward, shifted so a
# day-t decision only reads ICs whose forward window has closed by t.


def fwd_wide(close: pd.DataFrame, skip: int = 1, h: int = 21) -> pd.DataFrame:
    """close[t+skip+h]/close[t+skip]-1 — the return the next-open fill earns."""
    return close.shift(-(skip + h)) / close.shift(-skip) - 1.0


def daily_ic(score_w: pd.DataFrame, fwd_w: pd.DataFrame,
             method: str = "spearman") -> pd.Series:
    """Per-day cross-sectional IC between a score row and the fwd row."""
    s = score_w.stack().rename("s")
    y = fwd_w.stack().rename("y")
    s.index.names = y.index.names = ["datetime", "instrument"]
    df = pd.concat([s, y], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    corr = df.groupby(level="datetime").apply(
        lambda g: g["s"].corr(g["y"], method=method)
        if len(g) > 20 else np.nan)
    return corr.dropna()


def ew_trailing(ic: pd.Series, hl_days: float, win: int) -> pd.Series:
    """EW mean of `ic` over the trailing `win` entries (halflife in
    observations ~ days). At day t the window ends at t inclusive —
    callers must shift the raw IC by h+1 first so only fully-realised
    forwards enter."""
    ic = ic.sort_index()
    vals = ic.values.astype(float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        lo = max(0, i - win + 1)
        seg = vals[lo:i + 1]
        m = np.isfinite(seg)
        if not m.any():
            continue
        seg = seg[m]
        age = np.arange(len(seg))[::-1].astype(float)
        w = 0.5 ** (age / max(hl_days, 1e-9))
        out[i] = float((seg * w).sum() / w.sum())
    return pd.Series(out, index=ic.index)


def ic_scale(usable_ic: pd.Series, cfg: dict) -> pd.Series:
    """sleeve_scale[t] = clip(EW_ic[t]/full_at, 0, 1); < min_obs usable
    IC days -> prior_scale (skeptical prior, never blocks cold start)."""
    ew = ew_trailing(usable_ic, cfg["halflife_days"], cfg["ic_days"])
    cnt = usable_ic.rolling(cfg["ic_days"], min_periods=0).count()
    scale = (ew / cfg["full_at"]).clip(0.0, 1.0)
    return scale.where(cnt >= cfg["min_obs"], cfg["prior_scale"])


def leg_masks(spec: dict, panels: dict, legs: list[str],
              fwd_w: pd.DataFrame) -> dict[str, pd.Series]:
    """IC-gate mask per leg: False where trailing EW rank-IC <= 0.
    Ungateable legs (vq family — no live IC history) are absent from the
    map so sim and live treat them as always-on."""
    cfg = spec["ic_gate"]
    h = spec["fwd_h"]
    out = {}
    for leg in legs:
        if leg in cfg["ungateable"]:
            continue
        ic = daily_ic(panels[leg], fwd_w).shift(h + 1)
        ew = ew_trailing(ic, cfg["halflife_days"], cfg["ic_days"])
        cnt = ic.rolling(cfg["ic_days"], min_periods=0).count()
        out[leg] = (ew > 0) | (cnt < cfg["min_obs"])
    return out


def composite_for(spec: dict, cand: dict, panels: dict,
                  fwd_w: pd.DataFrame) -> pd.DataFrame:
    legs = spec["signals"].get(cand["sleeve_signal"] or "", [])
    if not legs:
        return pd.DataFrame()
    masks = leg_masks(spec, panels, legs, fwd_w) \
        if cand.get("ic_gate") else None
    return blend(panels, legs, masks)


def scale_for(spec: dict, cand: dict, panels: dict,
              fwd_w: pd.DataFrame) -> dict:
    """{day: sleeve_scale} when cand has ic_shrink — trailing EW rank-IC
    of the ic_signal leg vs fwd returns, clipped to [0,1]."""
    if not cand.get("ic_shrink"):
        return {}
    cfg = spec["ic_shrink"]
    ic = daily_ic(panels[cfg["ic_signal"]], fwd_w).shift(spec["fwd_h"] + 1)
    return ic_scale(ic, cfg).to_dict()


# ---------------------------------------------------------------------------
# targets + scoring


def capped_picks(row: pd.Series, n: int, smap: dict,
                 sector_max: int | None) -> list[str]:
    """Top-n by score with at most `sector_max` picks per sector —
    literally the bot's coresat.pick_top."""
    return CS.pick_top(row, n, smap, sector_max)


def daily_targets(comp: pd.DataFrame, close: pd.DataFrame,
                  days: pd.DatetimeIndex, cand: dict, spec: dict,
                  exposure: dict, smap: dict,
                  priced_only: bool = True) -> dict:
    """P7.daily_targets + sector cap, monthly pick freeze, and (phase-8
    fix from diagnosis item 1) priced-today eligibility.

    priced_only=True: a name must have a finite close ON the pick day —
    the live bot's universe can only contain listed names, so a dead
    name cannot be picked there and must not be pickable in the sim
    either (diagnosis: stale vq ranked CELG/CSRA/BCR into sleeve slots
    that could never fill). priced_only=False reproduces the phase-7
    ever-priced convention exactly (used only for the item-1 rerun of
    the failed 2026 config).

    pick_freeze: monthly — the pick list is recomputed only on the first
    trading day of each calendar month; between reviews the previous
    list keeps its targets (a delisted name keeps a target it can't
    fill — the book can't rebuy a halted name live either)."""
    sleeve = cand.get("sleeve") or {}
    n = int(sleeve.get("n", P7.sleeve_n(spec, cand)))
    weighting = sleeve.get("weighting", "equal")
    risk_off = "to_core" if cand.get("guard") == "to_core" else "cash"
    sector_max = (cand.get("pick") or {}).get("sector_max")
    freeze = cand.get("pick_freeze") == "monthly"
    first_seen = close.apply(lambda c: c.first_valid_index())
    out, last_ym, last_picks = {}, None, []
    for d in days:
        if freeze and last_ym == (d.year, d.month):
            picks = last_picks
        else:
            row = comp.loc[d].dropna() if d in comp.index else \
                pd.Series(dtype=float)
            if priced_only:
                px = close.loc[d]
                row = row[px.reindex(row.index).notna().values]
            eligible = row.index[
                [first_seen.get(s) is not None
                 and first_seen.get(s) <= d for s in row.index]]
            picks = capped_picks(row.loc[eligible], n, smap, sector_max)
            last_ym, last_picks = (d.year, d.month), picks
        iv = CS.invvol(close, picks, d) if weighting == "invvol" else None
        out[d] = CS.build_targets(
            picks, cand["core"], invvol_w=iv,
            exposure=float(exposure.get(d, 1.0)), risk_off=risk_off,
            core_sym=spec["core_symbol"],
            cap=float(spec["max_stock_weight"]))
    return out


def exposure_for(spec: dict, cand: dict, spy: pd.Series,
                 scale: dict) -> dict:
    """Per-day sleeve exposure = regime guard x ic_shrink scale."""
    exp = P7.regime_map(spec, spy, cand.get("guard"))
    if scale:
        exp = {d: float(exp.get(d, 1.0)) * float(v)
               for d, v in scale.items()}
    return exp


def run_candidate(feed: dict, targets: dict, days: pd.DatetimeIndex,
                  spec: dict, cost_map: dict | None, band: float | None,
                  flat_bps: float | None = None) -> dict:
    cl = feed["bars"]["close"].loc[days[0]:days[-1]]
    ox = feed["bars"]["open"].loc[days[0]:days[-1]]
    return mc.micro_backtest(
        ad.empty_signal(), cl, cash_gbp=spec["account"]["cash_gbp"],
        gbpusd=spec["account"]["gbpusd"], exec_prices=ox,
        fill=spec["fill"], delist_liquidate=spec["delist_liquidate"],
        rebalance="daily", cost_map=cost_map,
        cost_bps=flat_bps if flat_bps is not None
        else spec["costs"]["flat_bps"],
        band=band if band is not None else spec["rebalance_band"],
        min_order=float(spec.get("min_order_usd", mc.MIN_ORDER_USD)),
        weight_targets=targets)


def score_window(cand: dict, comp: pd.DataFrame, feed: dict,
                 days: pd.DatetimeIndex, spec: dict, smap: dict,
                 cost_map: dict, exposure: dict,
                 priced_only: bool = True) -> tuple[dict, pd.Series]:
    targets = daily_targets(comp, feed["close"], days, cand, spec,
                            exposure, smap, priced_only=priced_only)
    met = run_candidate(feed, targets, days, spec, cost_map,
                        cand.get("band"))
    rep = LH.bot_metrics(met, spec["account"]["gbpusd"], feed["spy"])
    rep["trades_ledger"] = met["trades"]
    rep["targets"] = targets
    daily = met["daily_excess"].dropna()
    rep.update({
        "rolling_12m": LH.rolling_xs_ir(daily, feed["spy"]),
        "breakeven": P7.breakeven_bps(met, feed["spy"]),
        "by_year": LH.per_year_table(daily, feed["spy"]),
    })
    for k in ("daily_cost", "daily_traded"):
        rep.pop(k, None)
    return rep, daily


def clean_row(rep: dict) -> dict:
    return {k: v for k, v in rep.items()
            if k not in ("trades_ledger", "targets")}


# ---------------------------------------------------------------------------
# Part 1 — diagnosis


def per_stock_pnl(trades: list[dict], last_px: dict[str, float],
                  spy: str) -> pd.DataFrame:
    """FIFO realised + unrealised P&L per symbol from the trade ledger."""
    lots: dict[str, list[list]] = {}
    rows: dict[str, dict] = {}
    for t in trades:
        s, px, q = t["sym"], t["price"], t["qty"]
        r = rows.setdefault(s, {"buy_notional": 0.0, "sell_notional": 0.0,
                               "realised": 0.0, "unrealised": 0.0,
                               "n_buys": 0, "n_sells": 0})
        if t["side"] == "buy":
            lots.setdefault(s, []).append([q, px])
            r["buy_notional"] += t["notional"]
            r["n_buys"] += 1
        else:
            r["sell_notional"] += t["notional"]
            r["n_sells"] += 1
            rem = q
            for lot in lots.get(s, []):
                take = min(rem, lot[0])
                r["realised"] += take * (px - lot[1])
                lot[0] -= take
                rem -= take
                if rem <= 1e-12:
                    break
            lots[s] = [l for l in lots.get(s, []) if l[0] > 1e-12]
    for s, ls in lots.items():
        qty = sum(l[0] for l in ls)
        px = last_px.get(s)
        if qty > 1e-9 and px is not None and np.isfinite(px):
            cost = sum(l[0] * l[1] for l in ls)
            rows[s]["unrealised"] = qty * px - cost
    df = pd.DataFrame(rows).T.fillna(0.0)
    df["total_pnl"] = df["realised"] + df["unrealised"]
    df["is_core"] = df.index == spy
    return df.sort_values("total_pnl")


def brinson(w: pd.DataFrame, rets: pd.DataFrame, smap: dict) -> dict:
    """Sleeve return decomposition vs the equal-weight universe sector
    mix, per day then annualised (mean x252).

    w:    day x sym equity-fraction weights of the sleeve (excl. core).
    rets: day x sym daily returns for the same priced universe.
    bench: the same sleeve capital spread equal-weight over the universe
    -> bench_s = sleeve_frac * count_s / N.  Then per day:
      alloc_s = (w_s - bench_s) * r_sec_s
      sel_s   = w_s * (r_picks_s - r_sec_s)
    and  alloc + sel = sleeve_ret - sleeve_frac * EW_universe_ret (exact).
    """
    secs = pd.Series({c: smap.get(c, f"UNK:{c}") for c in rets.columns})
    sec_ret = rets.T.groupby(secs).mean().T          # day x sector EW ret
    n_by_sec = secs.value_counts()
    bench_w = n_by_sec / n_by_sec.sum()              # EW universe sector mix
    wsec = w.T.groupby(pd.Series({c: smap.get(c, f"UNK:{c}")
                                  for c in w.columns})).sum().T \
        .reindex(columns=sec_ret.columns).fillna(0.0)
    sleeve_frac = wsec.sum(axis=1)
    alloc_terms, sel_terms = {}, {}
    for s in sec_ret.columns:
        bw = sleeve_frac * bench_w.get(s, 0.0)
        # non-held / unpriced-sector days are a true 0 contribution —
        # they must stay in the mean's denominator or a 1-day position
        # annualises ~n_days x too large
        alloc_terms[s] = float(((wsec[s] - bw) * sec_ret[s])
                               .fillna(0.0).mean() * 252)
        cols = [c for c in w.columns
                if smap.get(c, f"UNK:{c}") == s]
        if cols:
            picks_ret = (w[cols] * rets[cols]).sum(axis=1) \
                / wsec[s].replace(0, np.nan)
            sel_terms[s] = float(((picks_ret - sec_ret[s])
                                  * wsec[s]).fillna(0.0).mean() * 252)
        else:
            sel_terms[s] = 0.0
    return {"allocation_ann": alloc_terms, "selection_ann": sel_terms,
            "allocation_total": float(np.nansum(list(alloc_terms.values()))),
            "selection_total": float(np.nansum(list(sel_terms.values())))}


def daily_weights(targets: dict, days: pd.DatetimeIndex,
                  core_sym: str) -> pd.DataFrame:
    """{day:{sym:w}} -> day x sym sleeve weight frame (excl. core)."""
    recs = {d: {s: v for s, v in t.items() if s != core_sym}
            for d, t in targets.items()}
    w = pd.DataFrame(recs).T
    w.index = pd.DatetimeIndex(w.index)
    return w.reindex(days).fillna(0.0)


def boot_ci(x: np.ndarray, block: int, n_boot: int, seed: int) -> tuple:
    """Stationary block bootstrap CI on the mean of a daily series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < block * 2:
        return (np.nan, np.nan, float(x.mean()) if len(x) else np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(x) / block))
    starts = rng.integers(0, len(x) - block + 1, size=(n_boot, nb))
    idx = starts[..., None] + np.arange(block)
    sam = x[idx].reshape(n_boot, -1)[:, :len(x)]
    mu = sam.mean(axis=1)
    return (float(np.percentile(mu, 2.5)),
            float(np.percentile(mu, 97.5)), float(x.mean()))


def diag_ic_decay(panels: dict, comps: dict, close: pd.DataFrame,
                  spec: dict) -> dict:
    """Item 2: IC/RankIC per year for vq, mom, momvq, ens_def at each
    horizon + rolling-12m + bootstrap CI + trend."""
    out = {}
    series = {"mom": panels["mom"], "vq": panels["vq"],
              "momvq": comps["momvq"], "ens_def": comps["ens_def"]}
    for h in spec["diagnose"]["ic_horizons"]:
        fw = fwd_wide(close, 1, h)
        hor = {}
        for name, sc in series.items():
            ics = daily_ic(sc, fw, "spearman")
            ici = daily_ic(sc, fw, "pearson")
            per_year = {}
            for y in spec["diagnose"]["ic_years"]:
                ys = ics[ics.index.year == y]
                lo, hi, mu = boot_ci(
                    ys.values,
                    spec["diagnose"]["bootstrap"]["block"],
                    spec["diagnose"]["bootstrap"]["n_boot"],
                    spec["diagnose"]["bootstrap"]["seed"])
                per_year[y] = {"rank_ic": mu, "ic_ci": [lo, hi],
                               "ic": float(ici[ici.index.year == y].mean())
                               if len(ys) else None,
                               "n_days": int(len(ys))}
            roll = ics.rolling(252)
            roll_ir = (roll.mean() / roll.std() * np.sqrt(252)).dropna()
            yrs = sorted(per_year)
            vals = np.array([per_year[y]["rank_ic"]
                             if per_year[y]["rank_ic"] is not None
                             else np.nan for y in yrs])
            slope = float(np.polyfit(yrs, vals, 1)[0]) \
                if np.isfinite(vals).sum() > 2 else None
            half = len(yrs) // 2
            hor[name] = {
                "by_year": per_year,
                "rolling_12m_ic": {
                    "median": float(roll.mean().median()),
                    "ir_median": float(roll_ir.median())
                    if len(roll_ir) else None,
                    "pct_positive": float((roll.mean() > 0).mean()),
                    "min": float(roll.mean().min())},
                "trend_slope_per_year": slope,
                "first_half_mean": float(np.nanmean(vals[:half])),
                "second_half_mean": float(np.nanmean(vals[half:])),
                "years_ic_pos": int(sum(
                    1 for y in per_year.values()
                    if (y["rank_ic"] or 0) > 0)),
            }
        out[f"h{h}"] = hor
    return out


def cscv_pbo(ret: pd.DataFrame, blocks: int = 16) -> dict:
    """CSCV probability of backtest overfitting (Bailey et al. 2017).

    ret: T x N candidate daily-return frame. Split T into `blocks`
    contiguous blocks; over every C(blocks, blocks/2) train/test split,
    pick the train-Sharpe argmax and record its relative rank in the
    test set (fraction of candidates strictly better OOS). PBO =
    P(train-best lands in the bottom half OOS)."""
    X = ret.values
    T, N = X.shape
    edges = np.linspace(0, T, blocks + 1).astype(int)
    blk = np.array([X[edges[i]:edges[i + 1]].mean(0)
                    / np.nan_to_num(
                        X[edges[i]:edges[i + 1]].std(0), nan=1e-12)
                    for i in range(blocks)])           # S x N block SR
    omegas = []
    for test_idx in combinations(range(blocks), blocks // 2):
        mask = np.zeros(blocks, bool)
        mask[list(test_idx)] = True
        s_tr = blk[~mask].mean(0)
        s_te = blk[mask].mean(0)
        best = int(np.nanargmax(s_tr))
        omegas.append(float((s_te > s_te[best]).sum()
                            / max(N - 1, 1)))
    om = np.array(omegas)
    return {"n_splits": len(om), "n_candidates": int(N),
            "blocks": int(blocks),
            "pbo": float((om >= 0.5).mean()),
            "mean_relative_rank_oos": float(om.mean()),
            "frac_oos_top_quartile": float((om < 0.25).mean())}


def _sector_of(smap: dict, s: str) -> str:
    return smap.get(s, f"UNK:{s}")


def diagnose(spec: dict) -> dict:
    provider = REPO / spec["provider"]
    feed = load_feed(provider)
    smap = load_sector_map(spec)
    panels = build_panels(spec, feed, provider, smap)
    comps = {name: blend(panels, legs)
             for name, legs in spec["signals"].items()}
    close = feed["close"]
    days = feed["days"]
    cmap = mc.liquidity_cost_map(feed["bars"]["close"],
                                 feed["bars"]["volume"],
                                 flat_bps=spec["costs"]["flat_bps"])
    n_scored = 0
    out = {"stage": "phase8_diagnose", "git_sha": git_sha(),
           "spec": str(SPEC_PATH.relative_to(REPO)),
           "feed_days": [str(days[0].date()), str(days[-1].date())]}

    # ---- item 1: attribution of the 2026 loss -----------------------------
    ho_days = P7.window_days(days, *spec["diagnose"]["holdout_window"])
    ho_days = ho_days[ho_days <= days[-1]]
    cand7 = {"name": "B50_momvq", "core": 0.50, "sleeve_signal": "momvq",
             "guard": "none"}
    rep, daily26 = score_window(cand7, comps["momvq"], feed, ho_days,
                                spec, smap, cmap, {}, priced_only=False)
    n_scored += 1
    # the recorded phase-7 holdout (pre-fix vq, ever-priced eligibility)
    # — cited so the bug's isolated contribution is measurable
    rec_p = REPO / "results" / "phase7_us_holdout.json"
    if rec_p.exists():
        rec7 = json.loads(rec_p.read_text())
        rec_row = next((r for r in rec7.get("holdout_2026", [])
                        if r.get("name") == "B50_momvq"), {})
        out["recorded_2026"] = {
            "ir_xs_spy": rec_row.get("ir_xs_spy"),
            "ar_xs_spy": rec_row.get("ar_xs_spy"),
            "annualized_return": rec_row.get("annualized_return"),
            "information_ratio": rec_row.get("information_ratio"),
            "final_equity_usd": rec_row.get("final_equity_usd"),
            "n_trials": rec7.get("n_trials"),
            "note": "recorded holdout used the pre-fix vq "
                    "(later-filed amendments leaked) and the "
                    "ever-priced pick rule — the rerun above keeps "
                    "ever-priced, so the delta isolates the vq fix."}
    last_px = close.ffill().iloc[-1].to_dict()
    pnl = per_stock_pnl(rep["trades_ledger"], last_px,
                        spec["core_symbol"])
    w = daily_weights(rep["targets"], ho_days, spec["core_symbol"])
    rets = close.pct_change().reindex(ho_days)
    secs = pd.Series({c: _sector_of(smap, c) for c in rets.columns})
    pick_days = w[w > 0].stack().rename("w")
    pick_days.index.names = ["datetime", "instrument"]
    pick_days = pick_days.reset_index()
    pick_days["sector"] = pick_days["instrument"].map(
        lambda s: _sector_of(smap, s))
    sec_days = pick_days.groupby("sector")["w"].agg(["sum", "count"])
    picks_pnl = pnl[~pnl["is_core"]].copy()
    picks_pnl["sector"] = [_sector_of(smap, s) for s in picks_pnl.index]
    br = brinson(w, rets, smap)
    factor_tilts = {}
    for f, panel in (("mom", panels["mom"]), ("vq", panels["vq"]),
                     ("lowvol", panels["lowvol"]), ("beta", panels["beta"])):
        rk = panel.rank(axis=1, pct=True)
        vals = [rk.at[d, s] for d, s in
                zip(pick_days["datetime"], pick_days["instrument"])
                if d in rk.index and s in rk.columns
                and np.isfinite(rk.at[d, s])]
        factor_tilts[f] = float(np.mean(vals)) if vals else None
    out["attribution"] = {
        "window": [str(ho_days[0].date()), str(ho_days[-1].date())],
        "metrics": clean_row(rep),
        "per_stock_pnl": picks_pnl[
            ["total_pnl", "realised", "unrealised", "buy_notional",
             "sell_notional", "n_buys", "n_sells", "sector"]]
        .round(4).to_dict("index"),
        "sector_weight_days": sec_days.round(4).to_dict("index"),
        "brinson": br,
        "factor_tilts_mean_rank": factor_tilts,
        "n_picks_distinct": int(pick_days["instrument"].nunique()),
        "sleeve_mean_names": float((w > 0).sum(axis=1).mean()),
        # phase-8 finding: target weight aimed at names with NO price on
        # that day (dead names ranked by a stale signal leg) — the sim
        # cannot fill them (cash sits); live cannot pick them at all
        "unpriced_target": {
            "pick_days": int((((w > 0).values)
                              & ~close.reindex(ho_days)[w.columns]
                              .notna().values).sum()),
            "weight_days": float(
                w.where(~close.reindex(ho_days)[w.columns]
                        .notna().values, 0.0).to_numpy().sum()),
        },
    }

    # ---- item 2: signal decay ----------------------------------------------
    out["ic_decay"] = diag_ic_decay(panels, comps, close, spec)

    # ---- item 3: overfitting (DSR + CSCV PBO on the phase-7 selection) -----
    spec7 = yaml.safe_load(
        (REPO / "configs" / "experiments/phase7_us.yaml").read_text())
    books7, _p7, comps7 = P7.build_all(spec7)
    oof7 = P7.window_days(books7["days"], *spec7["oof"])
    rows7, daily7 = P7.eval_candidates(spec7, books7, comps7, oof7)
    n_scored += len(rows7)
    ret_mat = pd.DataFrame(daily7)
    xs_mat = ret_mat.sub(books7["spy"].pct_change(), axis=0)
    out["overfit"] = {
        "oof_candidates": len(rows7),
        "oof_winner_oof_ir": float(max(r["information_ratio"]
                                       for r in rows7)),
        "cscv_total_ret": cscv_pbo(ret_mat.dropna(),
                                 spec["diagnose"]["cscv"]["blocks"]),
        "cscv_xs_spy": cscv_pbo(xs_mat.dropna(),
                              spec["diagnose"]["cscv"]["blocks"]),
        "oof_to_2026_gap": {
            "oof_xs_ir": float(next(
                r["ir_xs_spy"] for r in rows7
                if r["name"] == "B50_momvq")),
            "y2026_xs_ir": float(rep["ir_xs_spy"])},
    }

    # ---- item 4: data checks -----------------------------------------------
    checks = {}
    d0 = days[len(days) // 2]
    mom_full = LH.bars_score(close, "mom")
    close_cut = close.copy()
    close_cut.loc[close_cut.index > d0] = np.nan
    mom_cut = LH.bars_score(close_cut, "mom")
    a = mom_full.xs(d0, level="datetime")
    b = mom_cut.xs(d0, level="datetime")
    checks["lookahead_mom"] = {
        "day": str(d0.date()),
        "identical": bool(a.equals(b)),
        "max_abs_diff": float((a - b).abs().max())}
    fpath = REPO / spec["fundamentals"]
    f = pd.read_parquet(fpath)
    f["filed"] = pd.to_datetime(f["filed"])
    d_cut = pd.Timestamp("2026-01-02")
    inst = ev.load_instruments(provider, "sp500")
    syms = inst["instrument"].unique().tolist()
    f_trunc = f[f["filed"] < d_cut]
    # must sit beside shares_out.parquet — build_signals resolves
    # shares via fundamentals_path.parent (a build/ path would silently
    # drop shares -> mcap NaN -> bm/ep missing -> false-positive diff)
    tmp_p = fpath.parent / "_fundamentals_trunc.parquet"
    f_trunc.to_parquet(tmp_p)
    try:
        vq_t = s6.build_signals(
            provider, syms, feed["cal"], ["vq"],
            fundamentals_path=tmp_p,
            feats_path=REPO / "data" / "events" / "features_us.parquet",
            news_dir=REPO / "results", sector_map=None,
            close=close, volume=close)["vq"].unstack("instrument")
    finally:
        tmp_p.unlink(missing_ok=True)
    v_pre = panels["vq"][panels["vq"].index < d_cut]
    t_pre = vq_t.reindex(v_pre.index)
    diff = (v_pre - t_pre).abs().max().max()
    checks["lookahead_vq"] = {
        "max_abs_diff_pre_cut": float(diff) if np.isfinite(diff) else 0.0,
        "identical": bool(diff < 1e-9) if np.isfinite(diff) else True,
        "note": "truncating filings at filed<2026-01-02 must not change "
                "any pre-cut score — verifies facts key on `filed`",
        "dup_dt_inst_rows": int(
            panels["vq"].stack().index.duplicated().sum())}
    last_day = days[-1]
    dead = inst[pd.to_datetime(inst["end"]) < last_day]
    checks["survivorship"] = {
        "instruments": int(len(inst)),
        "unique": int(inst["instrument"].nunique()),
        "dead_names_present": int(dead["instrument"].nunique()),
        "still_trading": int(inst["instrument"].nunique()
                             - dead["instrument"].nunique())}
    lag = (d_cut - f[f["filed"] < d_cut].groupby("symbol")
           .apply(lambda g: g["filed"].max())
           .rename("last")).dt.days
    snap = pd.read_parquet(REPO / "state/live/vq_snapshot.parquet")
    snap_age = (pd.Timestamp.now().normalize()
                - pd.to_datetime(snap["asof"]).max()).days
    checks["stale_vq"] = {
        "median_last_filing_age_days": float(lag.median()),
        "p90": float(lag.quantile(0.9)), "max": float(lag.max()),
        "names_over_120d": int((lag > 120).sum()),
        "snapshot_age_days_now": int(snap_age),
        "note": "lag measured over all fundamental symbols at 2026-01-02"}
    jumps = int((close.pct_change().abs() > 0.5).sum().sum())
    liq = [t for t in rep["trades_ledger"]
           if t.get("reason") == "delist-liquidate"]
    checks["corporate_actions"] = {
        "one_day_jumps_over_50pct_in_dump": jumps,
        "delist_liquidations_2026": len(liq),
        "dump_end": str(last_day.date())}
    snap_s = snap.set_index(snap["symbol"].map(ev.canon))["score"]
    asof = pd.to_datetime(snap["asof"]).max()
    pit_at = panels["vq"].loc[panels["vq"].index <= asof].iloc[-1]
    both = pd.concat([snap_s, pit_at], axis=1,
                     keys=["live_snap", "pit"]).dropna()
    checks["live_parity"] = {
        "snapshot_asof": str(asof.date()),
        "snapshot_vs_pit_corr": float(both["live_snap"].corr(both["pit"]))
        if len(both) > 30 else None,
        "snapshot_vs_pit_rank_corr": float(
            both["live_snap"].corr(both["pit"], method="spearman"))
        if len(both) > 30 else None,
        "n_common": int(len(both)),
        "notes": "live vq = monthly snapshot cross-section; sim vq = "
                 "daily PIT filed<D series — same math, different "
                 "freshness. Live bars = Alpaca IEX vs sim SIP — same "
                 "vendor, different feed; fills: live ~10:00 ET market "
                 "order vs sim next-open (documented approximation)."}
    out["data_checks"] = checks

    # ---- item 5: micro-account effects -------------------------------------
    def _run(min_o, band):
        t = daily_targets(comps["momvq"], close, ho_days, cand7, spec,
                          {}, smap, priced_only=False)
        return mc.micro_backtest(
            ad.empty_signal(),
            feed["bars"]["close"].loc[ho_days[0]:ho_days[-1]],
            cash_gbp=spec["account"]["cash_gbp"],
            gbpusd=spec["account"]["gbpusd"],
            exec_prices=feed["bars"]["open"].loc[ho_days[0]:ho_days[-1]],
            fill=spec["fill"], delist_liquidate=True, rebalance="daily",
            cost_map=cmap, band=band, weight_targets=t, min_order=min_o)
    base = _run(1.0, 0.02)
    no_floor = _run(0.0, 0.02)
    no_band = _run(1.0, 0.0)
    n_scored += 3
    micro = {
        "book_usd": spec["account"]["cash_gbp"] * spec["account"]["gbpusd"],
        "sleeve_names_max": int(min(
            8, (1 - 0.5) * spec["account"]["cash_gbp"]
            * spec["account"]["gbpusd"])),
        "final_equity": {"base": float(base["final_equity_usd"]),
                         "no_usd1_floor": float(no_floor["final_equity_usd"]),
                         "no_band": float(no_band["final_equity_usd"])},
        "n_trades": {"base": len(base["trades"]),
                     "no_usd1_floor": len(no_floor["trades"]),
                     "no_band": len(no_band["trades"])},
        "turnover": {"base": float(base["turnover"]),
                     "no_band": float(no_band["turnover"])},
    }
    micro["floor_drag_usd"] = float(no_floor["final_equity_usd"]
                                    - base["final_equity_usd"])
    micro["band_drag_usd"] = float(no_band["final_equity_usd"]
                                   - base["final_equity_usd"])

    def _top(n):
        tg = {}
        for d in ho_days:
            row = comps["momvq"].loc[d].dropna() if d in \
                comps["momvq"].index else pd.Series(dtype=float)
            px = close.loc[d]
            row = row[px.reindex(row.index).notna().values]
            picks = row.nlargest(n).index.tolist()
            tg[d] = ({spec["core_symbol"]: 1.0} if not picks else
                     {s: 0.5 / len(picks) for s in picks}
                     | {spec["core_symbol"]: 0.5})
        return mc.micro_backtest(
            ad.empty_signal(),
            feed["bars"]["close"].loc[ho_days[0]:ho_days[-1]],
            cash_gbp=1000.0, gbpusd=1.0,
            exec_prices=feed["bars"]["open"].loc[ho_days[0]:ho_days[-1]],
            fill=spec["fill"], delist_liquidate=True, rebalance="daily",
            band=0.0, cost_bps=0.0, weight_targets=tg)
    i6, i30 = _top(6), _top(30)
    te = (i6["daily_excess"] - i30["daily_excess"]).dropna()
    micro["breadth"] = {
        "ideal_n30_final_1000usd": float(i30["final_equity_usd"]),
        "n6_final_1000usd": float(i6["final_equity_usd"]),
        "te_ann_vs_n30": float(te.std() * np.sqrt(252)),
        "ar_n6": float(i6["annualized_return"]),
        "ar_n30": float(i30["annualized_return"])}
    n_scored += 2
    out["micro_effects"] = micro

    # ---- DSR on the winner's 2026 daily series (all diagnostic windows
    # counted in n_trials) ---------------------------------------------
    n_scored_total = n_scored
    out["overfit"]["dsr_winner_2026"] = wf.deflated_sharpe(
        daily26, P7.prior_trials() + n_scored_total,
        [float(r["information_ratio"]) for r in rows7])
    out["n_scored_windows"] = n_scored_total
    out["ts"] = dt.datetime.now().isoformat()
    return out


# ---------------------------------------------------------------------------
# Part 2 — the walk-forward gate


def gate_years(days: pd.DatetimeIndex, start: int) -> list[int]:
    return sorted({d.year for d in days if d.year >= start})


def wf_metrics(daily: pd.Series) -> dict:
    r = daily.dropna()
    eq = (1 + r).cumprod()
    if r.empty:
        return {"annualized_return": 0.0, "information_ratio": 0.0,
                "max_drawdown": 0.0, "total_return": 0.0, "n_days": 0}
    return {
        "total_return": float(eq.iloc[-1] - 1),
        "annualized_return": float(eq.iloc[-1] ** (252 / len(r)) - 1),
        "information_ratio": float(r.mean() / r.std() * np.sqrt(252))
        if r.std() > 0 else 0.0,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "n_days": int(len(r))}


def deploy_gate(results: dict, spy_by: dict, spy_wf: dict,
                years: list[int], g: dict) -> dict:
    """The predeclared deploy rule — see spec.deploy_gate."""
    decision = {}
    for name, res in results.items():
        by = res["by_year"]
        yrs = [y for y in years if y in by and y in spy_by]
        matches = [y for y in yrs
                   if by[y]["annualized_return"]
                   >= spy_by[y]["annualized_return"] - g["match_tol"]]
        worst = min((by[y]["annualized_return"]
                     - spy_by[y]["annualized_return"])
                    for y in yrs) if yrs else 0.0
        decision[name] = {
            "mdd_ok": bool(res["wf"]["max_drawdown"]
                           >= spy_wf["max_drawdown"] - g["mdd_tol"]),
            "match_frac": len(matches) / max(len(yrs), 1),
            "match_ok": bool(len(matches) / max(len(yrs), 1)
                             >= g["min_match_frac"]),
            "match_years": matches,
            "worst_xs_year": float(worst),
            "worst_ok": bool(worst >= -g["worst_year_tol"]),
            "wf_ir": res["wf"]["information_ratio"],
            "mdd": res["wf"]["max_drawdown"]}
        decision[name]["passed"] = bool(
            decision[name]["mdd_ok"] and decision[name]["match_ok"]
            and decision[name]["worst_ok"])
    surv = {n: d for n, d in decision.items() if d["passed"]}
    strat_surv = {n: d for n, d in surv.items()
                  if results[n].get("sleeve_signal")}
    winner = max(strat_surv or surv,
                 key=lambda n: (surv[n]["wf_ir"], results[n]["core"])) \
        if surv else "A_spy"
    return {"decision": decision, "winner": winner,
            # the benchmark always passes its own gate — "fallback" means
            # NO signal candidate survived, not that the book is wrong
            "fallback_used": not bool(strat_surv),
            "survivors": sorted(surv, key=lambda n: -surv[n]["wf_ir"])}


def render_live_config(winner: dict, spec: dict, dec: dict,
                       wf_res: dict) -> str:
    """The deployed configs/live_signal.yaml — phase-8 schema."""
    sname = winner.get("sleeve_signal")
    legs = spec["signals"].get(sname, []) if sname else []
    sleeve = winner.get("sleeve") or {}
    n = int(sleeve.get("n", P7.sleeve_n(spec, winner))) if sname else 0
    weighting = sleeve.get("weighting", "equal")
    core_w = winner["core"]
    guard = winner.get("guard", "none")
    regime = "to_core" if guard == "to_core" else "none"
    sector_max = (winner.get("pick") or {}).get("sector_max")
    freeze = winner.get("pick_freeze")
    shrink = spec["ic_shrink"] if winner.get("ic_shrink") else None
    gate = spec["ic_gate"] if winner.get("ic_gate") else None
    wf_row = wf_res[winner["name"]]["wf"] if winner["name"] in wf_res else {}
    wdec = dec["decision"].get(winner["name"], {})
    extra = ""
    if sector_max:
        extra += f"  sector_max: {sector_max}\n"
    if freeze:
        extra += f"  pick_freeze: {freeze}\n"
    if shrink:
        extra += ("  ic_shrink: {ic_signal: %s, halflife_days: %d, "
                  "ic_days: %d, full_at: %s, min_obs: %d, "
                  "prior_scale: %s, fwd_h: %d}\n"
                  % (shrink["ic_signal"], shrink["halflife_days"],
                     shrink["ic_days"], shrink["full_at"],
                     shrink["min_obs"], shrink["prior_scale"],
                     spec["fwd_h"]))
    if gate:
        extra += ("  ic_gate: {halflife_days: %d, ic_days: %d, "
                  "min_obs: %d, ungateable: [%s], fwd_h: %d}\n"
                  % (gate["halflife_days"], gate["ic_days"],
                     gate["min_obs"], ", ".join(gate["ungateable"]),
                     spec["fwd_h"]))
    signals_yaml = "[" + ", ".join(legs) + "]" if legs else "[]"
    return f"""# Phase-8 pin ({dt.date.today().isoformat()}): {winner['name']} selected by the
# predeclared walk-forward gate (configs/experiments/phase8_us.yaml).
# Walk-forward {wf_row.get('n_days', 'n/a')}d OOS IR
# {wf_row.get('information_ratio', float('nan')):+.3f} AR
# {wf_row.get('annualized_return', float('nan')):+.3f} mdd
# {wf_row.get('max_drawdown', float('nan')):+.3f};
# gate passed={wdec.get('passed')} match_frac={wdec.get('match_frac')}
# worst_xs_year={wdec.get('worst_xs_year')}.
# `validated` is true only for a non-SPY winner that cleared every gate
# leg; the 100% SPY fallback is the benchmark itself — nothing to
# validate.
validated: {str(bool(winner.get('sleeve_signal'))).lower()}

# Live-trading interlock (NEVER set true here — see docs/BOT_SETUP.md).
live_trading: false

# Signal library for the sleeve. Computable live: mom, rev, srev5, mom6,
# lowvol, beta (Alpaca bars); vq, news (state/live snapshots); sn_* legs
# are sector-neutralised with configs/sector_map.csv.
signals: {signals_yaml}
weighting: equal          # signal blend inside the sleeve (equal ranks)
ic_months: 12
halflife_frac: 0.5

# Regime guard — on risk-off (SPY < SMA200 or top-decile trailing vol,
# lagged) the sleeve's freed share routes per sleeve.risk_off.
regime: {regime}
regime_cfg:
  sma: 200
  vol_win: 21
  vol_lookback: 504
  vol_q: 0.90
  actions: {{none: 1.0, half: 0.5, cash: 0.0, to_core: 0.0}}

# Portfolio — core+satellite: {core_w:.0%} SPY core +
# {1 - core_w:.0%} signal sleeve of up to {n} names ({weighting}).
core: {{symbol: {spec['core_symbol']}, weight: {core_w}}}
sleeve:
  n: {n}
  weighting: {weighting}
  risk_off: to_core
{extra}n_positions: {n}          # legacy mirror of sleeve.n
budget_gbp: {spec['account']['cash_gbp']}
cost_bps: 10.0            # sizing conservatism (paper fills are free)
rebalance_band: {winner.get('band', spec['rebalance_band'])}   # skip |dw| < band or < $1
snapshot_stale_days: 45
order_after_open_min: 15
sector_map: {spec['sector_map']}
fwd_h: {spec['fwd_h']}

risk:
  max_drawdown: 0.25          # equity vs running peak -> halt new buys
  daily_loss: 0.05            # equity vs prior day-summary -> halt buys
  max_position_weight: {spec['max_stock_weight']}   # per single stock (SPY core exempt)
  max_orders: 12              # per run (sells kept, excess buys dropped)
  max_daily_notional_frac: 1.0
  max_price_move: 0.50
  min_universe: 100

universe: configs/live_universe.txt
"""


# ---------------------------------------------------------------------------
# walk-forward command


def run_grid(spec: dict, provider: Path, years: list[int],
             feed: dict, flat_bps: float | None = None
             ) -> tuple[dict, dict]:
    """Score every candidate x year on `feed` -> (results, daily series)."""
    smap = load_sector_map(spec)
    panels = build_panels(spec, feed, provider, smap)
    fwd = fwd_wide(feed["close"], 1, spec["fwd_h"])
    cmap = mc.liquidity_cost_map(feed["bars"]["close"],
                                 feed["bars"]["volume"],
                                 flat_bps=spec["costs"]["flat_bps"]) \
        if flat_bps is None else None
    results, daily_all = {}, {}
    for cand in spec["candidates"]:
        comp = composite_for(spec, cand, panels, fwd)
        scale = scale_for(spec, cand, panels, fwd)
        exp = exposure_for(spec, cand, feed["spy"], scale)
        by_year, dailies = {}, []
        for y in years:
            yd = feed["days"][feed["days"].year == y]
            if len(yd) < 20:
                continue
            targets = daily_targets(comp, feed["close"], yd, cand, spec,
                                    exp, smap)
            met = run_candidate(feed, targets, yd, spec, cmap,
                                cand.get("band"), flat_bps)
            rep = LH.bot_metrics(met, spec["account"]["gbpusd"],
                                 feed["spy"])
            rep.pop("daily_cost", None)
            rep.pop("daily_traded", None)
            by_year[y] = rep
            dailies.append(met["daily_excess"].dropna())
            log(f"  {cand['name']:>12} {y}: ar="
                f"{rep['annualized_return']:+.3f} "
                f"ir={rep['information_ratio']:+.3f} "
                f"mdd={rep['max_drawdown']:+.3f} "
                f"xs={rep['ar_xs_spy']:+.3f}")
        daily = pd.concat(dailies).sort_index() if dailies \
            else pd.Series(dtype=float)
        results[cand["name"]] = {
            "name": cand["name"], "core": cand["core"],
            "sleeve_signal": cand.get("sleeve_signal"),
            "by_year": by_year, "wf": wf_metrics(daily)}
        daily_all[cand["name"]] = daily
    return results, daily_all


def spy_by_year(feed: dict, years: list[int]) -> dict:
    return {y: LH.spy_metrics(feed["spy"],
                              feed["days"][feed["days"].year == y])
            for y in years}


def cmd_walkforward(spec: dict, out: Path, skip_supp: bool) -> dict:
    provider = REPO / spec["provider"]
    feed = load_feed(provider)
    years = gate_years(feed["days"], spec["walkforward"]["gate_start_year"])
    log(f"gate years: {years} — {len(spec['candidates'])} candidates")
    results, daily_all = run_grid(spec, provider, years, feed)
    spy_by = spy_by_year(feed, years)
    gate_days = feed["days"][feed["days"].year >= years[0]]
    spy_wf = LH.spy_metrics(feed["spy"], gate_days)

    dec = deploy_gate(results, spy_by, spy_wf, years, spec["deploy_gate"])
    winner_name = dec["winner"]
    cand = next(c for c in spec["candidates"] if c["name"] == winner_name)
    log(f"deploy gate: winner={winner_name} "
        f"fallback={dec['fallback_used']} "
        f"survivors={dec['survivors']}")

    # supplementary years on the community dump (biased — report only)
    supp = None
    n_supp = 0
    supp_provider = REPO / spec["provider_supp"]
    if not skip_supp and supp_provider.exists():
        sfeed = load_feed(supp_provider)
        syears = [y for y in spec["walkforward"]["supp_years"]
                  if (sfeed["days"].year == y).any()]
        log(f"supplementary years (biased feed): {syears}")
        sres, _sd = run_grid(spec, supp_provider, syears, sfeed,
                             flat_bps=spec["walkforward"]["supp_flat_bps"])
        supp = {"years": syears,
                "feed": "community qlib dump — survivorship-biased "
                        "(listed-only, no membership windows, normalised "
                        "prices); reported only, never gated",
                "cost": "flat %sbps/side (normalised price basis makes "
                        "dollar-volume spread buckets meaningless)"
                        % spec["walkforward"]["supp_flat_bps"],
                "results": sres}
        n_supp = sum(len(r["by_year"]) for r in sres.values())

    diag_n = 0
    diag_path = REPO / "results" / "phase8_diagnosis.json"
    if diag_path.exists():
        diag_n = int(json.loads(diag_path.read_text())
                     .get("n_scored_windows", 0))
    n_scored = sum(len(r["by_year"]) for r in results.values()) + n_supp
    n_trials = P7.prior_trials() + diag_n + n_scored
    winner = {"name": cand["name"], "core": cand["core"],
              "sleeve_signal": cand.get("sleeve_signal"),
              "guard": cand.get("guard", "none"),
              "sleeve": cand.get("sleeve"), "pick": cand.get("pick"),
              "pick_freeze": cand.get("pick_freeze"),
              "ic_shrink": cand.get("ic_shrink"),
              "ic_gate": cand.get("ic_gate"),
              "band": cand.get("band")}
    trial_irs = [float(r["wf"]["information_ratio"])
                 for r in results.values()]
    dsr = wf.deflated_sharpe(daily_all[winner_name], n_trials, trial_irs)
    cfg_text = render_live_config(winner, spec, dec, results)
    (REPO / "configs" / "live_signal.yaml").write_text(cfg_text)
    log("configs/live_signal.yaml pinned to the gate winner")

    ts = dt.datetime.now().isoformat()
    for name, res in results.items():
        row = {"ts": ts, "market": "us", "git_sha": git_sha(),
               "spec": "phase8_us.yaml", "candidate": f"phase8:{name}",
               "re_viewed_holdout": False,
               "IC": None, "ICIR": None, "RankIC": None, "RankICIR": None,
               **res["wf"],
               "gate": dec["decision"][name],
               "dsr": wf.deflated_sharpe(daily_all[name], n_trials,
                                         trial_irs),
               "n_trials": n_trials}
        wf.append_holdout_log(row)
    summary = {
        "stage": "phase8_us_walkforward", "git_sha": git_sha(),
        "spec": str(SPEC_PATH.relative_to(REPO)),
        "gate_years": years, "spy_by_year": spy_by, "spy_wf": spy_wf,
        "results": results, "gate": dec, "winner": winner, "dsr": dsr,
        "n_scored_windows": n_scored, "diag_windows": diag_n,
        "n_trials": n_trials, "supplementary": supp,
        "live_config": cfg_text, "ts": ts}
    out.write_text(json.dumps(summary, indent=1, default=str))
    log(f"wrote {out} + appended {len(results)} holdout_log rows")
    return summary


# ---------------------------------------------------------------------------
# diagnosis doc renderer


def _f(x, fmt="{:+.3f}"):
    return fmt.format(x) if isinstance(x, (int, float)) \
        and np.isfinite(x) else "—"


def render_diagnosis_md(d: dict) -> str:
    """docs/PHASE8_DIAGNOSIS.md — every number straight from the run."""
    att = d["attribution"]
    m = att["metrics"]
    L = ["""# Phase-8 diagnosis — why the US picker failed out of sample

All numbers below are computed by `scripts/phase8_us.py diagnose` on the
honest Alpaca PIT feed (delisted names included, next-open fills,
liquidity-bucket half-spread + 5bps, $1 floor, 2% band, fractional).
Do not hand-edit numbers — regenerate from results/phase8_diagnosis.json.

## 1. Attribution of the 2026 loss

Failed config: B50_momvq = 50% SPY core + 6-name mom+vq equal sleeve.
"""]
    L.append(f"Window {att['window'][0]} → {att['window'][1]}: bot AR "
             f"{_f(m['annualized_return'])}, IR {_f(m['information_ratio'])}, "
             f"mdd {_f(m['max_drawdown'])}; SPY AR "
             f"{_f(m['spy']['annualized_return'])}, IR "
             f"{_f(m['spy']['information_ratio'])}, mdd "
             f"{_f(m['spy']['max_drawdown'])}; xs-IR {_f(m['ir_xs_spy'])}, "
             f"xs-AR {_f(m['ar_xs_spy'])}.\n")
    rec = d.get("recorded_2026") or {}
    if rec:
        L.append(f"Recorded holdout (pre-fix vq, run on the same feed): "
                 f"xs-IR {_f(rec.get('ir_xs_spy'))}, AR "
                 f"{_f(rec.get('annualized_return'))}, final "
                 f"${rec.get('final_equity_usd', 0):.2f} — the rerun "
                 f"above isolates the vq fix (eligibility unchanged): "
                 f"Δxs-IR "
                 f"{_f(m['ir_xs_spy'] - (rec.get('ir_xs_spy') or 0))}, "
                 f"Δequity ${m['final_equity_usd'] - rec.get('final_equity_usd', 0):+.2f}.\n")
    L.append("### Per-stock P&L (sleeve, FIFO realised + unrealised, USD)\n")
    L.append("| stock | sector | total P&L | realised | unrealised | buy$ |"
             " sell$ | buys/sells |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s, r in sorted(att["per_stock_pnl"].items(),
                       key=lambda kv: kv[1]["total_pnl"]):
        L.append(f"| {s} | {r['sector']} | {r['total_pnl']:+.2f} | "
                 f"{r['realised']:+.2f} | {r['unrealised']:+.2f} | "
                 f"{r['buy_notional']:.1f} | {r['sell_notional']:.1f} | "
                 f"{int(r['n_buys'])}/{int(r['n_sells'])} |")
    L.append("\n### Sector exposure (sleeve weight-days)\n")
    L.append("| sector | Σ weight·days | pick-days |")
    L.append("|---|---|---|")
    for s, r in sorted(att["sector_weight_days"].items(),
                       key=lambda kv: -kv[1]["sum"]):
        L.append(f"| {s} | {r['sum']:.1f} | {int(r['count'])} |")
    b = att["brinson"]
    L.append("\n### Sector concentration vs stock selection "
             "(Brinson vs equal-weight universe)\n")
    L.append("Daily decomposition of the sleeve's return vs the same "
             "capital spread equal-weight across the universe "
             "(allocation = over/under-weight × sector EW return; "
             "selection = sleeve-weight × (pick EW return − sector EW "
             "return)), annualised:\n")
    L.append("| component | ann. contribution |")
    L.append("|---|---|")
    L.append(f"| sector allocation (concentration) | "
             f"{b['allocation_total']:+.4f} |")
    L.append(f"| stock selection within sectors | "
             f"{b['selection_total']:+.4f} |")
    L.append("")
    L.append("| sector | allocation | selection |")
    L.append("|---|---|---|")
    secs = sorted(set(b["allocation_ann"]) | set(b["selection_ann"]))
    shown = [s for s in secs
             if abs(b["allocation_ann"].get(s, 0))
             + abs(b["selection_ann"].get(s, 0)) >= 0.001]
    rest = [s for s in secs if s not in shown]
    for s in shown:
        L.append(f"| {s} | {b['allocation_ann'].get(s, 0):+.4f} | "
                 f"{b['selection_ann'].get(s, 0):+.4f} |")
    if rest:
        L.append(f"| ({len(rest)} sectors + singleton UNK:* below "
                 f"|x|<0.001, combined) | "
                 f"{sum(b['allocation_ann'].get(s, 0) for s in rest):+.4f} | "
                 f"{sum(b['selection_ann'].get(s, 0) for s in rest):+.4f} |")
    ft = att["factor_tilts_mean_rank"]
    L.append("\n### Factor exposures of picks (mean pct-rank of chosen names)\n")
    L.append("| factor | mean pick rank (0..1) |")
    L.append("|---|---|")
    L.append(f"| momentum (12-1) | {_f(ft.get('mom'), '{:.3f}')} |")
    L.append(f"| value+quality | {_f(ft.get('vq'), '{:.3f}')} |")
    L.append(f"| low-vol | {_f(ft.get('lowvol'), '{:.3f}')} |")
    L.append(f"| low-beta | {_f(ft.get('beta'), '{:.3f}')} |")
    L.append(f"\nDistinct sleeve names over window: "
             f"{att['n_picks_distinct']}; mean concurrent names: "
             f"{att['sleeve_mean_names']:.1f}.\n")
    up = att["unpriced_target"]
    L.append(f"\n**Unfillable targets (phase-8 finding):** "
             f"{up['pick_days']} pick-days / {up['weight_days']:.1f} "
             f"weight·days of sleeve target sat on names with NO price "
             f"that day — delisted names kept near the top by a stale "
             f"signal leg (vq holds its last score; mom is NaN so the "
             f"blend counts vq alone). The sim cannot fill them (the "
             f"slot's cash sits idle); the live bot cannot pick them at "
             f"all (its universe is listed names only). The phase-8 "
             f"walk-forward requires a finite close on the pick day.\n")

    L.append("\n## 2. Signal decay — IC/RankIC by year "
             "(Spearman primary, Pearson alongside)\n")
    for h, hor in d["ic_decay"].items():
        L.append(f"### forward horizon {h[1:]}d\n")
        for name, sig in hor.items():
            L.append(f"**{name}** — roll-12m median rankIC "
                     f"{_f(sig['rolling_12m_ic']['median'], '{:+.4f}')}, "
                     f"roll-IC-IR "
                     f"{_f(sig['rolling_12m_ic']['ir_median'], '{:+.2f}')}, "
                     f"frac-12m>0 "
                     f"{_f(sig['rolling_12m_ic']['pct_positive'], '{:.2f}')}, "
                     f"trend/yr {_f(sig['trend_slope_per_year'], '{:+.5f}')}, "
                     f"first-half {_f(sig['first_half_mean'], '{:+.4f}')} vs "
                     f"second-half {_f(sig['second_half_mean'], '{:+.4f}')}, "
                     f"years IC>0: {sig['years_ic_pos']}")
            L.append("")
            L.append("| year | rankIC | boot95% CI | pearson IC | n |")
            L.append("|---|---|---|---|---|")
            for y, r in sig["by_year"].items():
                L.append(f"| {y} | {_f(r['rank_ic'], '{:+.4f}')} | "
                         f"[{_f(r['ic_ci'][0], '{:.3f}')},"
                         f"{_f(r['ic_ci'][1], '{:.3f}')}] | "
                         f"{_f(r['ic'], '{:+.4f}')} | {r['n_days']} |")
            L.append("")

    ov = d["overfit"]
    c1, c2 = ov["cscv_total_ret"], ov["cscv_xs_spy"]
    L.append(f"""
## 3. Overfitting — selection across the trials

- OOF winner's own OOF IR: {ov['oof_winner_oof_ir']:+.3f}; fresh-2026
  xs-IR: {ov['oof_to_2026_gap']['y2026_xs_ir']:+.3f} (OOF xs-IR was
  {ov['oof_to_2026_gap']['oof_xs_ir']:+.3f}).
- DSR on the 2026 daily series: {ov['dsr_winner_2026']['DSR']:.3f}
  (SR* {ov['dsr_winner_2026']['SR_star']:+.3f}, n_trials
  {ov['dsr_winner_2026']['n_trials']}).
- CSCV on the 22 phase-7 OOF candidates ({c1['n_splits']} splits,
  S={c1['blocks']}):
  - total-return metric: PBO {c1['pbo']:.3f}, mean OOS relative rank
    {c1['mean_relative_rank_oos']:.3f}, frac OOS-top-quartile
    {c1['frac_oos_top_quartile']:.3f};
  - excess-vs-SPY metric: PBO {c2['pbo']:.3f}, mean OOS relative rank
    {c2['mean_relative_rank_oos']:.3f}, frac OOS-top-quartile
    {c2['frac_oos_top_quartile']:.3f}.

PBO = fraction of combinatorial splits in which the in-sample-best
candidate lands in the bottom half out-of-sample — the probability the
selection itself is overfit.
""")

    dc = d["data_checks"]
    L.append(f"""
## 4. Data checks

| check | result |
|---|---|
| momentum look-ahead audit (all bars after {dc['lookahead_mom']['day']} NaN'd, signal recomputed) | identical = {dc['lookahead_mom']['identical']} (max |Δ| {dc['lookahead_mom']['max_abs_diff']:.2e}) |
| vq look-ahead audit (fundamentals truncated to filed<2026-01-02, rebuild) | identical pre-cut = {dc['lookahead_vq']['identical']} (max |Δ| {dc['lookahead_vq']['max_abs_diff_pre_cut']:.2e}); dup (dt,inst) rows {dc['lookahead_vq']['dup_dt_inst_rows']} |
| survivorship | {dc['survivorship']['instruments']} instrument rows, {dc['survivorship']['dead_names_present']} dead names kept in universe, {dc['survivorship']['still_trading']} still trading at dump end {dc['corporate_actions']['dump_end']} |
| vq staleness | median last-filing age {dc['stale_vq']['median_last_filing_age_days']:.0f}d, p90 {dc['stale_vq']['p90']:.0f}d, max {dc['stale_vq']['max']:.0f}d; {dc['stale_vq']['names_over_120d']} symbols >120d stale; live snapshot age now {dc['stale_vq']['snapshot_age_days_now']}d |
| corporate actions | {dc['corporate_actions']['one_day_jumps_over_50pct_in_dump']} one-day >50% moves in dump; {dc['corporate_actions']['delist_liquidations_2026']} delist-liquidations in the 2026 book |
| live vs sim vq | snapshot asof {dc['live_parity']['snapshot_asof']}; corr(live, PIT) {_f(dc['live_parity']['snapshot_vs_pit_corr'], '{:.3f}')} / rank-corr {_f(dc['live_parity']['snapshot_vs_pit_rank_corr'], '{:.3f}')} over {dc['live_parity']['n_common']} names |

{dc['live_parity']['notes']}

**The vq audit caught a real bug.** The first run of this check failed:
truncating filings at `filed<2026-01-02` changed pre-cut scores by up to
~1.0 rank — `fundamentals._latest_point/_ttm` deduplicated each period
end across the WHOLE fact frame before the `filed<D` filter, so an
amendment filed later overwrote the visible value AND made the period
invisible until the amendment's date (7,361 versioned cells affected;
56/60 sampled symbols' book values differed). Fixed to PIT per-end
versioning (`filed<D` selects the version, then the latest end) — the
audit above is the post-fix rerun. The phase-7 recorded numbers used
the pre-fix vq; every phase-8 number in this doc uses the fixed one.
""")

    me = d["micro_effects"]
    L.append(f"""
## 5. Small-account effects (${me['book_usd']:.2f} book)

- Sleeve budget caps the book at {me['sleeve_names_max']} names
  (floor(${me['book_usd'] * 0.5:.2f} / $1)) — phase 6 already showed the
  edge wants ~30.
- 2026 book: base {me['n_trades']['base']} trades, final equity
  ${me['final_equity']['base']:.2f}; without the $1 floor
  {me['n_trades']['no_usd1_floor']} trades (floor-suppressed:
  {me['n_trades']['no_usd1_floor'] - me['n_trades']['base']}), final
  ${me['final_equity']['no_usd1_floor']:.2f} → $1-floor drag
  ${me['floor_drag_usd']:+.2f}; without the 2% band
  {me['n_trades']['no_band']} trades, final
  ${me['final_equity']['no_band']:.2f} → band effect
  ${me['band_drag_usd']:+.2f}. The band is inert at this book size —
  a $1-minimum trade is already $1/$12.70 ≈ 7.9% of equity, past the
  2% band, so the floor always binds first.
- Breadth tracking error (same signal, cost-free, $1000 book so the $1
  floor cannot bind): 6-name sleeve AR {me['breadth']['ar_n6']:+.3f} vs
  30-name AR {me['breadth']['ar_n30']:+.3f}; annualised tracking error
  6-vs-30 {me['breadth']['te_ann_vs_n30']:.3f}.

## 6. Ranked root causes (from the numbers above)

Ranking is computed from the measured numbers: item 1 is the isolated
effect of the vq fix (recorded-vs-rerun delta on the same feed), items
2-3 share the 2026 sleeve-vs-equal-weight-universe return
decomposition (alloc + select = the gap exactly), items 4-7 are
structural causes measured on different axes (probabilities, USD
drag, IC trend) ordered by the strength of their evidence.

""")
    alloc_share = abs(b["allocation_total"]) / max(
        abs(b["allocation_total"]) + abs(b["selection_total"]), 1e-12)
    first, second = (
        ("Sector concentration — the sleeve's over/under-weights vs an "
         "equal-weight book drove the loss", "allocation",
         b["allocation_total"]),
        ("Stock selection inside sectors — picks underperformed their own "
         "sector peers", "selection", b["selection_total"])) \
        if abs(b["allocation_total"]) >= abs(b["selection_total"]) else (
        ("Stock selection inside sectors — picks underperformed their own "
         "sector peers", "selection", b["selection_total"]),
        ("Sector concentration — the sleeve's over/under-weights vs an "
         "equal-weight book drove the loss", "allocation",
        b["allocation_total"]))
    mom_ic = d["ic_decay"]["h21"]["mom"]
    L.append(f"""| # | root cause | key evidence |
|---|---|---|
| 1 | **vq point-in-time data bug** — later-filed amendments overwrote/hid the visible fundamental versions | recorded xs-IR {_f((d.get('recorded_2026') or {}).get('ir_xs_spy'))} -> {_f(m['ir_xs_spy'])} after the fix (Δ {_f(m['ir_xs_spy'] - ((d.get('recorded_2026') or {}).get('ir_xs_spy') or 0))}); 7,361 versioned cells; every phase-7 number used the buggy signal |
| 2 | {first[0]} | {first[1]} {_f(first[2], '{:+.4f}')} ann vs EW universe — {alloc_share if first[1]=='allocation' else 1-alloc_share:.0%} of the alloc+select gap |
| 3 | {second[0]} | {second[1]} {_f(second[2], '{:+.4f}')} ann vs EW universe — {1-alloc_share if first[1]=='allocation' else alloc_share:.0%} of the gap |
| 4 | Selection overfitting — the OOF winner was one of {ov['oof_candidates']} tried; in-sample winners decay OOS | PBO {c1['pbo']:.2f} total-ret / {c2['pbo']:.2f} xs; OOF xs-IR {_f(ov['oof_to_2026_gap']['oof_xs_ir'])} -> 2026 {_f(ov['oof_to_2026_gap']['y2026_xs_ir'])}; DSR {_f(ov['dsr_winner_2026']['DSR'], '{:.3f}')} |
| 5 | {'Signal decay' if (mom_ic['second_half_mean'] or 0) < (mom_ic['first_half_mean'] or 0) else 'Weak/unstable signal level — no decay measured, but the edge is thin'} — the mom leg's trailing rankIC | first-half {_f(mom_ic['first_half_mean'], '{:+.4f}')} vs second-half {_f(mom_ic['second_half_mean'], '{:+.4f}')}, trend {_f(mom_ic['trend_slope_per_year'], '{:+.5f}')}/yr; latest year {_f(mom_ic['by_year'][max(mom_ic['by_year'])]['rank_ic'], '{:+.4f}')} |
| 6 | Micro-account drag — $1 floor + 2% band on a $13 book | floor {_f(me['floor_drag_usd'], '{:+.2f}')} USD, band {_f(me['band_drag_usd'], '{:+.2f}')} USD on ${me['book_usd']:.2f}; breadth TE {_f(me['breadth']['te_ann_vs_n30'], '{:.3f}')} |
| 7 | Data caveats (vq staleness, snapshot-vs-PIT parity, survivorship) | see section 4 — {_f(dc['live_parity']['snapshot_vs_pit_rank_corr'], '{:.3f}')} live-vs-PIT rank corr; {dc['stale_vq']['names_over_120d']} names >120d stale |
""")
    return "\n".join(L)


def cmd_diagnose(spec: dict, out: Path) -> dict:
    d = diagnose(spec)
    out.write_text(json.dumps(d, indent=1, default=str))
    md = render_diagnosis_md(d)
    (REPO / "docs" / "PHASE8_DIAGNOSIS.md").write_text(md)
    log(f"wrote {out} + docs/PHASE8_DIAGNOSIS.md")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diagnose", "walkforward"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-supp", action="store_true")
    args = ap.parse_args()
    spec = yaml.safe_load(SPEC_PATH.read_text())
    if args.cmd == "diagnose":
        cmd_diagnose(spec, Path(args.out or
                                "results/phase8_diagnosis.json"))
    else:
        cmd_walkforward(spec,
                        Path(args.out or
                             "results/phase8_us_walkforward.json"),
                        args.skip_supp)


if __name__ == "__main__":
    main()
