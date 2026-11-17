#!/usr/bin/env python3
"""Phase-8 AUDIT — independent verification of the B50_momvq walk-forward.

    uv run python scripts/phase8_audit.py [--out results/phase8_audit.json]

Prompt-driven audit (owner's checklist):

  1. return arithmetic — per-year sleeve-only vs core-only decomposition
     rebuilt from the exact trade ledger (fills + delist payouts), check
     for double counting, executed weight sums >1 (leverage), cash
     earning returns, dividends counted twice.
  2. look-ahead — dump the exact features behind 20 seeded rebalance
     dates (mom input dates, vq facts with their `filed` dates, universe
     membership, the actual fill price) and assert all are knowable
     before the fill; re-verify the per-end-version PIT fix directly
     against amended facts.
  3. survivorship — dead names in the universe with real final returns.
  4. picks sanity — why the live sleeve is still APA/FTI/VLO/MPC/OGN/LYB;
     diff live-snapshot vs fixed-PIT vq ranks at the latest feed day.
  5. DSR — recompute at n_trials 392, plus the excess-vs-SPY variant.
  6. robustness — costs x2/x3, next-close fills, sleeve n=5 and n=8.
  7. community dump disagreement — data diagnostics on the overlap years.

Writes results/phase8_audit.json + docs/PHASE8_AUDIT.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import live_holdout as LH
import phase7_us as P7
import phase8_us as P8

from quant_research import adaptive as ad
from quant_research import events as ev
from quant_research import fundamentals as F
from quant_research import micro as mc
from quant_research import walkforward as wf

SPEC = yaml.safe_load((REPO / "configs/experiments/phase8_us.yaml").read_text())
PROVIDER = REPO / SPEC["provider"]
YEARS = list(range(SPEC["walkforward"]["gate_start_year"], 2027))
CORE = SPEC["core_symbol"]
CAND = {"name": "B50_momvq", "core": 0.50, "sleeve_signal": "momvq",
        "guard": "none"}
STORED = json.loads((REPO / "results/phase8_us_walkforward.json").read_text())
SEED = 20261117


def log(msg: str) -> None:
    print(f"[audit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# book reconstruction from the trade ledger (audit item 1)


def reconstruct(trades: list[dict], days: pd.DatetimeIndex,
                close: pd.DataFrame, cash0: float) -> dict:
    """Replay fills -> daily positions, cash, equity, per-name PnL.

    Mirrors micro_backtest accounting exactly: fills on their ledger date,
    EOD marks at close (unpriced name marks 0), cash earns nothing.
    PnL_i,t = mark(q_t*close_t) - mark(q_{t-1}*close_{t-1}) - (buys - sells
    - liq payouts of i on t): each name's contribution incl. the
    fill->close intraday move and the delist payout."""
    by_day: dict = {}
    for t in trades:
        by_day.setdefault(t["date"], []).append(t)
    qty: dict[str, float] = {}
    cash = cash0
    rows = []
    prev_marks: dict[str, float] = {}
    for d in days:
        marks = close.loc[d]
        day_pnl: dict[str, float] = {}
        for t in by_day.get(d, []):
            s = t["sym"]
            if t["side"] == "buy":
                qty[s] = qty.get(s, 0.0) + t["qty"]
                cash -= t["notional"]
                day_pnl[s] = day_pnl.get(s, 0.0) - t["notional"]
            else:
                qty[s] = qty.get(s, 0.0) - t["qty"]
                cash += t["notional"]
                day_pnl[s] = day_pnl.get(s, 0.0) + t["notional"]
            if abs(qty.get(s, 0.0)) < 1e-9:
                qty.pop(s, None)
        cur_marks = {s: (float(marks.get(s)) if np.isfinite(marks.get(s))
                         else 0.0) * q for s, q in qty.items()}
        for s, m in cur_marks.items():
            day_pnl[s] = day_pnl.get(s, 0.0) + m - prev_marks.get(s, 0.0)
        for s, pm in prev_marks.items():
            if s not in cur_marks:
                # exited today entirely OR unpriced (engine marks 0)
                day_pnl[s] = day_pnl.get(s, 0.0) - pm
        equity = cash + sum(cur_marks.values())
        wsum = sum(cur_marks.values()) / equity if equity > 0 else np.nan
        rows.append({"date": d, "equity": equity, "cash": cash,
                     "wsum": wsum,
                     "pnl": day_pnl, "qty": dict(qty),
                     "marks": cur_marks})
        prev_marks = cur_marks
    return rows


def per_year_decomp(rows: list[dict], spy: pd.Series) -> dict:
    """Sleeve-only vs core-only contributions per year + sanity totals."""
    out = {}
    for y in YEARS:
        yr = [r for r in rows if r["date"].year == y]
        if not yr:
            continue
        sleeve_ann, core_ann, spy_ann, tot_ann = [], [], [], []
        wmax, cmin, unpriced_days = 0.0, np.inf, 0
        sleeve_cap = []
        for i, r in enumerate(yr):
            e0 = yr[i - 1]["equity"] if i else 10.0 * SPEC["account"]["gbpusd"]
            pnl = r["pnl"]
            spnl = sum(v for s, v in pnl.items() if s != CORE)
            cpnl = pnl.get(CORE, 0.0)
            sleeve_ann.append(spnl / e0)
            core_ann.append(cpnl / e0)
            tot_ann.append(sum(pnl.values()) / e0)
            spy_ann.append(spy.pct_change().get(r["date"], np.nan))
            wmax = max(wmax, r["wsum"])
            cmin = min(cmin, r["cash"])
            scap = sum(v for s, v in r["marks"].items() if s != CORE)
            sleeve_cap.append(scap / e0)
            n_unpriced = sum(1 for s, q in r["qty"].items() if s != CORE
                             and r["marks"].get(s, 0.0) == 0.0)
            unpriced_days += n_unpriced
        sa = np.array(sleeve_ann)
        ca = np.array(core_ann)
        ta = np.array(tot_ann)
        spy_a = np.nan_to_num(np.array(spy_ann), nan=0.0)
        scap = np.array(sleeve_cap)
        out[y] = {
            # contribution to book return, annualised (mean daily x 252)
            "sleeve_contrib_ann": float(sa.mean() * 252),
            "core_contrib_ann": float(ca.mean() * 252),
            "book_ann_decomp": float(ta.mean() * 252),
            "spy_ann": float(spy_a.mean() * 252),
            # standalone sleeve AR: sleeve pnl / sleeve capital employed
            "sleeve_only_ann": float(np.nanmean(
                sa / np.where(scap > 1e-9, scap, np.nan)) * 252),
            "max_executed_wsum": float(wmax),
            "min_cash_usd": float(cmin),
            "unpriced_hold_days": int(unpriced_days),
            "mean_sleeve_frac": float(np.nanmean(scap)),
        }
    return out


# ---------------------------------------------------------------------------
# look-ahead dump (audit item 2)


def mom_inputs(close: pd.DataFrame, s: str, d: pd.Timestamp) -> dict:
    px = close[s].dropna()
    i = px.index.searchsorted(d)
    if i >= len(px) or px.index[i] != d:
        i -= 1                       # last bar <= d
    num = px.iloc[i - 21] if i - 21 >= 0 else np.nan
    num_d = px.index[i - 21] if i - 21 >= 0 else None
    den = px.iloc[i - 252] if i - 252 >= 0 else np.nan
    den_d = px.index[i - 252] if i - 252 >= 0 else None
    return {"num_date": str(num_d.date()) if num_d is not None else None,
            "den_date": str(den_d.date()) if den_d is not None else None,
            "mom": float(num / den - 1) if np.isfinite(num) and
            np.isfinite(den) and den != 0 else None,
            "latest_bar_used": str(px.index[i].date()),
            "ok": (num_d is None or num_d < d) and
                  (den_d is None or den_d < d)}


def vq_visible(fs: pd.DataFrame, sh: pd.DataFrame, d: pd.Timestamp) -> dict:
    """The exact fact rows vq_features would read at day d (PIT per-end
    versioning — replicates _latest_point/_ttm selection)."""
    d64 = np.datetime64(d.date())
    out = {}
    bk = fs[fs["group"] == "book"].sort_values(["end", "filed"])
    chosen = []
    for end, g in bk.groupby("end"):
        filed = g["filed"].values.astype("datetime64[D]")
        pos = np.searchsorted(filed, d64, side="left") - 1
        if pos >= 0:
            r = g.iloc[pos]
            chosen.append({"end": str(r["end"].date()),
                           "filed": str(r["filed"].date()),
                           "val": float(r["val"]), "form": r["form"]})
    out["book_visible"] = chosen[-1] if chosen else None
    ni = fs[fs["group"] == "ni"].copy()
    ni["span"] = (ni["end"] - ni["start"]).dt.days
    ni = ni[ni["span"] <= F.MAX_QUARTER_DAYS].sort_values(["end", "filed"])
    qvis = []
    for end, g in ni.groupby("end"):
        filed = g["filed"].values.astype("datetime64[D]")
        pos = np.searchsorted(filed, d64, side="left") - 1
        if pos >= 0:
            r = g.iloc[pos]
            qvis.append({"end": str(r["end"].date()),
                         "filed": str(r["filed"].date()),
                         "val": float(r["val"]), "form": r["form"],
                         "n_versions": len(g)})
    out["ni_quarters_visible"] = qvis[-4:]
    shr = sh.sort_values("filed") if sh is not None else pd.DataFrame()
    if len(shr):
        filed = shr["filed"].values.astype("datetime64[D]")
        pos = np.searchsorted(filed, d64, side="left") - 1
        if pos >= 0:
            r = shr.iloc[pos]
            out["shares"] = {"filed": str(r["filed"].date()),
                             "shares": float(r.get("shares",
                                                  r.get("val", np.nan)))}
    out["all_filed_before_d"] = all(
        pd.Timestamp(x["filed"]) < d for x in
        ([out["book_visible"]] if out["book_visible"] else [])
        + out["ni_quarters_visible"]
        + ([out["shares"]] if out.get("shares") else []))
    return out


def amendment_check(fund: pd.DataFrame, close: pd.DataFrame,
                    days: pd.DatetimeIndex, n_check: int = 30) -> dict:
    """For facts whose period end has >1 filed versions, assert the day-d
    visible value equals the latest version with filed < d — the PIT fix
    re-verified at the fact level."""
    tested, failures = 0, []
    multi = (fund[fund["group"].isin(["book", "ni"])]
             .groupby(["symbol", "end", "tag"])
             .filter(lambda g: g["filed"].nunique() > 1))
    rng = np.random.default_rng(SEED)
    syms = rng.choice(multi["symbol"].unique(),
                      size=min(n_check, multi["symbol"].nunique()),
                      replace=False)
    for s in syms:
        fs = multi[multi["symbol"] == s]
        for (end, tag), g in fs.groupby(["end", "tag"]):
            g = g.sort_values("filed")
            filed = g["filed"].values.astype("datetime64[D]")
            vals = g["val"].values
            for v_i in range(1, len(g)):
                # a day between filing v_{i-1} and v_i: the earlier
                # version must still be the visible one
                probe_days = days[(days >= pd.Timestamp(filed[v_i - 1]))
                                  & (days < pd.Timestamp(filed[v_i]))]
                if not len(probe_days):
                    continue
                d = probe_days[len(probe_days) // 2]
                d64 = np.datetime64(d.date())
                pos = np.searchsorted(filed, d64, side="left") - 1
                visible = vals[pos] if pos >= 0 else np.nan
                tested += 1
                if not (pos >= 0 and visible == vals[v_i - 1]):
                    failures.append({"sym": s, "end": str(end.date()),
                                     "tag": tag,
                                     "day": str(d.date()),
                                     "want": float(vals[v_i - 1]),
                                     "got": float(visible)
                                     if np.isfinite(visible) else None})
                if tested >= n_check:
                    break
            if tested >= n_check:
                break
        if tested >= n_check:
            break
    return {"n_tested": tested, "failures": failures}


def lookahead_dump(feed: dict, comp: pd.DataFrame, inst: pd.DataFrame,
                   fund_by: dict, shr_by: dict, n_dates: int = 20) -> dict:
    """20 seeded signal days: every input behind the day's picks + fill."""
    close = feed["close"]
    ox = feed["bars"]["open"]["open"].unstack("instrument").sort_index()
    days = feed["days"]
    rng = np.random.default_rng(SEED)
    picks_days = sorted(pd.Timestamp(x) for x in rng.choice(
        days[(days >= "2017-01-01") & (days <= "2026-10-31")].values,
        size=n_dates, replace=False))
    mem = {}
    for r in inst.itertuples():
        mem.setdefault(r.instrument, []).append((pd.Timestamp(r.start),
                                                 pd.Timestamp(r.end)))
    recs, violations = [], []
    for d in picks_days:
        i = days.get_loc(d)
        fill_day = days[i + 1] if i + 1 < len(days) else None
        row = comp.loc[d].dropna()
        priced = close.loc[d]
        row = row[priced.reindex(row.index).notna().values]
        picks = row.nlargest(6).index.tolist()
        rec = {"signal_day": str(d.date()), "fill_day": str(fill_day.date())
               if fill_day is not None else None, "picks": {}}
        for s in picks:
            mi = mom_inputs(close, s, d)
            vv = vq_visible(fund_by.get(s, pd.DataFrame()),
                            shr_by.get(s), d)
            mwin = [f"{a.date()}..{b.date()}" for a, b in mem.get(s, [])
                    if a <= d <= b]
            rec["picks"][s] = {
                "mom": mi, "vq": vv,
                "membership_window": mwin[0] if mwin else None,
                "fill_open": float(ox.loc[fill_day, s])
                if fill_day is not None and s in ox.columns
                and np.isfinite(ox.loc[fill_day, s]) else None}
            if not mi["ok"] or not vv["all_filed_before_d"]:
                violations.append({"day": str(d.date()), "sym": s})
        recs.append(rec)
    return {"n_dates": n_dates, "records": recs, "violations": violations}


# ---------------------------------------------------------------------------
# survivorship (audit item 3)


def survivorship(feed: dict, inst: pd.DataFrame, trades_all: list) -> dict:
    close = feed["close"]
    last_day = feed["days"][-1]
    dead = inst[pd.to_datetime(inst["end"]) < last_day]["instrument"].unique()
    r = close.pct_change()
    finals = []
    for s in dead:
        px = close[s].dropna()
        if len(px) < 2:
            continue
        tail = px.iloc[-1] / px.iloc[-min(len(px), 61)] - 1
        finals.append({"sym": s, "last_close": float(px.iloc[-1]),
                       "last_day": str(px.index[-1].date()),
                       "tail_60d_ret": float(tail),
                       "min_daily_move": float(r[s].min())})
    df = pd.DataFrame(finals)
    liq = [t for t in trades_all if t.get("reason") == "delist-liquidate"]
    return {
        "dead_names": len(dead),
        "dead_with_prices": len(df),
        "ended_below_1usd": int((df["last_close"] < 1.0).sum()),
        "ended_below_5usd": int((df["last_close"] < 5.0).sum()),
        "tail60_down_over_50pct": int((df["tail_60d_ret"] < -0.5).sum()),
        "tail60_down_over_80pct": int((df["tail_60d_ret"] < -0.8).sum()),
        "worst_tails": df.nsmallest(8, "tail_60d_ret")
        .to_dict("records"),
        "worst_daily_moves": df.nsmallest(8, "min_daily_move")
        [["sym", "min_daily_move"]].to_dict("records"),
        "delist_liquidations_in_sims": len(liq),
        "liq_examples": [{k: t[k] for k in ("date", "sym", "price",
                                           "notional")} for t in liq[:10]],
    }


# ---------------------------------------------------------------------------
# live-vs-PIT vq diff (audit item 4)


def live_vs_pit(feed: dict, panels: dict, close: pd.DataFrame) -> dict:
    snap = pd.read_parquet(REPO / "state/live/vq_snapshot.parquet")
    snap_s = snap.set_index(snap["symbol"].map(ev.canon))["score"]
    asof = pd.to_datetime(snap["asof"]).max()
    last = panels["vq"].dropna(how="all").index[-1]
    pit = panels["vq"].loc[last]
    both = pd.concat({"snap": snap_s, "pit": pit}, axis=1).dropna()
    rank_ic = both["snap"].corr(both["pit"], method="spearman")
    # today's composite under each vq source (mom identical)
    mom_r = panels["mom"].loc[last].rank(pct=True)
    live_rank = snap_s.reindex(mom_r.index).rank(pct=True)
    pit_rank = pit.reindex(mom_r.index).rank(pct=True)
    priced = close.loc[last].notna()
    comp_live = (0.5 * mom_r + 0.5 * live_rank)[priced]
    comp_pit = (0.5 * mom_r + 0.5 * pit_rank)[priced]
    live_names = ["APA", "FTI", "VLO", "MPC", "OGN", "LYB"]
    return {
        "snapshot_asof": str(asof.date()),
        "pit_day": str(last.date()),
        "rank_corr": float(rank_ic),
        "top6_snapshot_vq": comp_live.nlargest(6).index.tolist(),
        "top6_pit_vq": comp_pit.nlargest(6).index.tolist(),
        "live_sleeve": live_names,
        "live_names_pit_vq_rank": {s: float(pit_rank.get(s, np.nan))
                                   for s in live_names},
        "live_names_pit_comp_rank": {s: float(comp_pit.rank(pct=True)
                                              .get(s, np.nan))
                                     for s in live_names},
        "n_names": len(both),
    }


# ---------------------------------------------------------------------------
# robustness (audit item 6)


def yearly_run(comp, feed, spec, cand, exp, smap, cmap,
               targets_shift=False, fill=None):
    """B50 over all gate years -> per-year metrics + concat daily."""
    by_year, dailies = {}, []
    for y in YEARS:
        yd = feed["days"][feed["days"].year == y]
        if len(yd) < 20:
            continue
        targets = P8.daily_targets(comp, feed["close"], yd, cand, spec,
                                   exp, smap)
        if targets_shift:
            t2 = {}
            ds = list(yd)
            for i, d in enumerate(ds[:-1]):
                t2[ds[i + 1]] = targets[d]
            targets = t2
        cl = feed["bars"]["close"].loc[yd[0]:yd[-1]]
        ox = feed["bars"]["open"].loc[yd[0]:yd[-1]]
        met = mc.micro_backtest(
            ad.empty_signal(), cl,
            cash_gbp=spec["account"]["cash_gbp"],
            gbpusd=spec["account"]["gbpusd"],
            exec_prices=ox,          # ignored under fill='close'
            fill=fill or spec["fill"],
            delist_liquidate=spec["delist_liquidate"], rebalance="daily",
            cost_map=cmap, cost_bps=spec["costs"]["flat_bps"],
            band=spec["rebalance_band"],
            min_order=float(spec.get("min_order_usd", mc.MIN_ORDER_USD)),
            weight_targets=targets)
        rep = LH.bot_metrics(met, spec["account"]["gbpusd"], feed["spy"])
        by_year[y] = {"ar": rep["annualized_return"],
                      "ir": rep["information_ratio"],
                      "mdd": rep["max_drawdown"],
                      "xs_ar": rep["ar_xs_spy"],
                      "final": rep["final_equity_usd"],
                      "trades": met["trades"]}
        dailies.append(met["daily_excess"].dropna())
    daily = pd.concat(dailies).sort_index()
    return {"by_year": by_year, "wf": P8.wf_metrics(daily), "daily": daily}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase8_audit.json")
    ap.add_argument("--quick", action="store_true",
                    help="skip the 5 robustness re-runs")
    args = ap.parse_args()
    t0 = time.time()
    log("loading honest feed")
    feed = P8.load_feed(PROVIDER)
    smap = P8.load_sector_map(SPEC)
    log("building panels (mom, vq — B50 legs only)")
    panels = {}
    panels["mom"] = LH.bars_score(feed["close"], "mom") \
        .unstack("instrument")
    inst = ev.load_instruments(PROVIDER, "sp500")
    from quant_research import signals6 as s6
    sigs = s6.build_signals(
        PROVIDER, inst["instrument"].unique().tolist(), feed["cal"], ["vq"],
        fundamentals_path=REPO / SPEC["fundamentals"],
        feats_path=REPO / "data/events/features_us.parquet",
        news_dir=REPO / "results", sector_map=None,
        close=feed["close"], volume=feed["close"])
    panels["vq"] = sigs["vq"].unstack("instrument") \
        .reindex(feed["close"].index)
    comp = P7.blend(panels, ["mom", "vq"])
    cmap = mc.liquidity_cost_map(feed["bars"]["close"],
                                 feed["bars"]["volume"],
                                 flat_bps=SPEC["costs"]["flat_bps"])
    spy = feed["spy"]
    spy_by = {y: LH.spy_metrics(spy, feed["days"][feed["days"].year == y])
              for y in YEARS}
    out = {"stage": "phase8_audit", "git_sha": P8.git_sha(),
           "ts": dt.datetime.now().isoformat()}

    # ---- item 1: return arithmetic ---------------------------------------
    log("item 1: rerun B50 per year + ledger reconstruction")
    rows_all, by_year_rep, trades_all, daily_b50 = [], {}, [], []
    for y in YEARS:
        yd = feed["days"][feed["days"].year == y]
        if len(yd) < 20:
            continue
        targets = P8.daily_targets(comp, feed["close"], yd, CAND, SPEC,
                                   {}, smap)
        met = P8.run_candidate(feed, targets, yd, SPEC, cmap, None)
        rep = LH.bot_metrics(met, SPEC["account"]["gbpusd"], spy)
        by_year_rep[y] = rep
        trades_all += met["trades"]
        daily_b50.append(met["daily_excess"].dropna())
        rows_all += reconstruct(met["trades"], yd, feed["close"],
                                10.0 * SPEC["account"]["gbpusd"])
        stored_ar = STORED["results"]["B50_momvq"]["by_year"][str(y)][
            "annualized_return"]
        log(f"  {y}: ar={rep['annualized_return']:+.3f} "
            f"(stored {stored_ar:+.3f})")
    daily_b50 = pd.concat(daily_b50).sort_index()
    decomp = per_year_decomp(rows_all, spy)
    reproduce = {str(y): {
        "audit_ar": by_year_rep[y]["annualized_return"],
        "stored_ar": STORED["results"]["B50_momvq"]["by_year"][str(y)]
        ["annualized_return"],
        "audit_ir": by_year_rep[y]["information_ratio"],
        "stored_ir": STORED["results"]["B50_momvq"]["by_year"][str(y)]
        ["information_ratio"],
        "match": abs(by_year_rep[y]["annualized_return"]
                     - STORED["results"]["B50_momvq"]["by_year"][str(y)]
                     ["annualized_return"]) < 5e-3}
        for y in by_year_rep}
    equities = pd.Series({r["date"]: r["equity"] for r in rows_all})
    eq_rebuilt = equities.groupby(equities.index.year).last()
    out["item1_return_arithmetic"] = {
        "reproduce_stored": reproduce,
        "all_match_0p5pt": all(r["match"] for r in reproduce.values()),
        "decomp": decomp,
        "rebuilt_year_end_equity": {str(k): float(v)
                                    for k, v in eq_rebuilt.items()},
        "min_cash_any_day": float(min(r["cash"] for r in rows_all)),
        "max_executed_weight_sum": float(max(r["wsum"] for r in rows_all)),
        "trade_reasons": sorted({t.get("reason", "rebalance")
                                 for t in trades_all}),
        "n_trades": len(trades_all),
        "cash_yield_in_code": False,
        "dividend_stream_in_code": False,
        "note": "returns are pure adjusted-price moves; cash earns 0; no "
                "dividend leg exists to double-count (adjustment lives "
                "inside the price series once)."}

    # ---- item 2: look-ahead ----------------------------------------------
    log("item 2: look-ahead dump at 20 seeded dates + amendment re-check")
    fund = pd.read_parquet(REPO / SPEC["fundamentals"])
    fund["symbol"] = fund["symbol"].map(ev.canon)
    for c in ("start", "end", "filed"):
        fund[c] = pd.to_datetime(fund[c])
    shares = pd.read_parquet(REPO / "data/events/shares_out.parquet")
    shares["filed"] = pd.to_datetime(shares["filed"])
    shares["symbol"] = shares["symbol"].map(ev.canon)
    shr_by = {s: g for s, g in shares.groupby("symbol")}
    fund_by = {s: g for s, g in fund.groupby("symbol")}
    out["item2_lookahead"] = lookahead_dump(feed, comp, inst, fund_by,
                                            shr_by)
    out["item2_lookahead"]["amendment_check"] = amendment_check(
        fund, feed["close"], feed["days"])

    # ---- item 3: survivorship --------------------------------------------
    log("item 3: survivorship")
    out["item3_survivorship"] = survivorship(feed, inst, trades_all)

    # ---- item 4: live sleeve sanity --------------------------------------
    log("item 4: live vs PIT vq diff")
    out["item4_picks"] = live_vs_pit(feed, panels, feed["close"])

    # ---- item 5: DSR ------------------------------------------------------
    log("item 5: DSR recomputes")
    trial_irs = [float(r["wf"]["information_ratio"])
                 for r in STORED["results"].values()]
    xs_daily = (daily_b50 - spy.pct_change().reindex(
        daily_b50.index)).dropna()
    # stored JSON lacks wf-level xs-IR; estimate per candidate from its
    # per-year xs-AR weighted by year length (documented approximation)
    trial_xs_ir = []
    for r in STORED["results"].values():
        ys = [v for v in r["by_year"].values() if "ir_xs_spy" in v]
        if ys:
            trial_xs_ir.append(float(np.average(
                [v["ir_xs_spy"] for v in ys],
                weights=[v.get("n_days", 1) or 1 for v in ys])))
    out["item5_dsr"] = {
        "stored": STORED["dsr"],
        "recompute_total_ret": wf.deflated_sharpe(
            daily_b50, STORED["n_trials"], trial_irs),
        "recompute_excess": wf.deflated_sharpe(
            xs_daily, STORED["n_trials"], trial_xs_ir or trial_irs),
        "note": "DSR on TOTAL returns tests SR>max(SR) — a 50% SPY book "
                "gets SPY's own IR 0.865 for free; the excess-vs-SPY "
                "variant is the honest question."}

    # ---- item 6: robustness ----------------------------------------------
    if not args.quick:
        log("item 6: robustness re-runs (costs x2/x3, next-close, n5, n8)")
        rob = {}
        for scale in (2.0, 3.0):
            cm2 = {k: v * scale for k, v in cmap.items()}
            log(f"  costs x{scale:g}")
            rob[f"costs_x{scale:g}"] = yearly_run(
                comp, feed, SPEC, CAND, {}, smap, cm2)
        log("  next-day close fills")
        rob["fill_next_close"] = yearly_run(
            comp, feed, SPEC, CAND, {}, smap, cmap,
            targets_shift=True, fill="close")
        for n in (5, 8):
            log(f"  sleeve n={n}")
            cand_n = dict(CAND, sleeve={"n": n, "weighting": "equal"})
            rob[f"sleeve_n{n}"] = yearly_run(
                comp, feed, SPEC, cand_n, {}, smap, cmap)
        out["item6_robustness"] = {
            k: {"wf": v["wf"],
                "by_year": {str(y): {kk: vv for kk, vv in r.items()
                                     if kk != "trades"}
                            for y, r in v["by_year"].items()},
                "spy_ar_by_year": {str(y): spy_by[y]["annualized_return"]
                                   for y in v["by_year"]}}
            for k, v in rob.items()}
        # the 2024 spike: which names carried it
        log("  2024 attribution — per-name PnL from the ledger")
        rows24 = [r for r in rows_all if r["date"].year == 2024]
        pnl24: dict[str, float] = {}
        for r in rows24:
            for s, v in r["pnl"].items():
                if s != CORE:
                    pnl24[s] = pnl24.get(s, 0.0) + v
        out["item6_robustness"]["pnl_2024_by_name"] = dict(
            sorted(pnl24.items(), key=lambda kv: kv[1]))

    # ---- item 7: community dump ------------------------------------------
    log("item 7: community-feed diagnostics")
    supp_p = REPO / SPEC["provider_supp"]
    item7 = {"provider": str(SPEC["provider_supp"])}
    if supp_p.exists():
        sinst = ev.load_instruments(supp_p, "sp500")
        scal = ev.load_calendar(supp_p)
        item7["n_inst"] = int(sinst["instrument"].nunique())
        spans = (pd.to_datetime(sinst["end"]) - pd.to_datetime(
            sinst["start"])).dt.days
        item7["membership_windows_per_name"] = float(
            sinst.groupby("instrument").size().mean())
        item7["median_span_days"] = float(spans.median())
        # SPY identity check on overlap
        sspy = pd.Series(ev.load_field(supp_p, "SPY", scal, "close"),
                         index=scal).dropna()
        ovl = pd.concat({"community": sspy, "honest": spy},
                        axis=1).dropna()
        def _span_ret(s, a, b):
            s = s.dropna()
            lo, hi = s.loc[:a], s.loc[:b]
            if not len(s) or not len(lo) or not len(hi):
                return None
            return float(hi.iloc[-1] / lo.iloc[-1] - 1)
        item7["spy_overlap"] = {
            "n_days": len(ovl),
            "ret_corr": float(ovl.pct_change().corr().iloc[0, 1]),
            "community_total_2017_2020": _span_ret(
                sspy, "2016-12-30", "2020-12-31"),
            "honest_total_2017_2020": _span_ret(
                spy, "2016-12-30", "2020-12-31")}
        # unadjusted-split / bad-print evidence: |1-day move| > 40%
        sbars = mc.load_bars(supp_p, "sp500", ("close",), masked=False)
        sclose = sbars["close"]["close"].unstack("instrument").sort_index()
        moves = sclose.loc["2017":"2020"].pct_change()
        item7["one_day_moves_over_40pct_2017_2020"] = int(
            (moves.abs() > 0.4).sum().sum())
        hmoves = feed["close"].loc["2017":"2020"].pct_change()
        item7["honest_one_day_moves_over_40pct_2017_2020"] = int(
            (hmoves.abs() > 0.4).sum().sum())
        common = [s for s in feed["close"].columns if s in sclose.columns]
        item7["names_in_both_feeds"] = len(common)
        item7["honest_names"] = int(feed["close"].shape[1])
        item7["community_names"] = int(sclose.shape[1])
        # pick divergence for one seeded 2019 day on each feed's own
        # composite (mom only — vq has no community-side fundamentals)
        d = pd.Timestamp("2019-06-03")
        for tag, cl in (("honest", feed["close"]),
                        ("community", sclose)):
            if d in cl.index:
                m = LH.bars_score(cl.loc[:d], "mom").xs(
                    d, level="datetime").dropna()
                m = m[cl.loc[d].reindex(m.index).notna().values]
                item7[f"top6_mom_{tag}"] = m.nlargest(6).index.tolist()
    else:
        item7["error"] = "community dump absent"
    out["item7_community"] = item7

    out["runtime_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    log(f"wrote {args.out} ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
