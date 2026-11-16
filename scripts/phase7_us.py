#!/usr/bin/env python3
"""Phase-7 US live-bot — core+satellite select + one-shot 2026 holdout.

    uv run python scripts/phase7_us.py select
    uv run python scripts/phase7_us.py holdout

select : score every predeclared candidate (configs/experiments/
         phase7_us.yaml) on the pre-2025 OOF window only, apply the
         predeclared selection rule, write results/phase7_us_select.json
         and pin configs/live_signal.yaml to the winner — all committed
         BEFORE the 2026 window is scored.

holdout: re-score every candidate on the re-viewed 2025 window (seen in
         phases 2/4/5/6 — reported, never tuned on), then score ONLY the
         selected candidate plus the SPY baseline on the FRESH 2026
         window, exactly once. Applies the match gate, appends
         results/holdout_log.jsonl, flips `validated` iff it passes.

Sim spec = what the Alpaca bot actually does: next-open fills, bucketed
half-spread + flat bps per side, $1 notional floor, 2% rebalance band,
fractional rounding, one order per symbol per day (PDT by construction),
delist liquidation. The same coresat.build_targets + micro.weights_step
the live bot calls — research/production parity by construction.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import live_holdout as LH  # noqa: E402
from quant_research import adaptive as ad  # noqa: E402
from quant_research import coresat as CS  # noqa: E402
from quant_research import events as ev  # noqa: E402
from quant_research import micro as mc  # noqa: E402
from quant_research import signals6 as s6  # noqa: E402
from quant_research import walkforward as wf  # noqa: E402

SPEC_PATH = REPO / "configs" / "experiments" / "phase7_us.yaml"


def log(msg: str) -> None:
    print(f"[phase7 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          check=False, cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


# ---------------------------------------------------------------------------
# data


def load_books(spec: dict) -> dict:
    """All-listed universe bars (delisted names included) + SPY appended as
    a tradable row — the honest counterpart of live_universe.txt."""
    provider = REPO / spec["provider"]
    cal = ev.load_calendar(provider)
    bars = mc.load_bars(provider, "sp500", ("close", "open", "volume"),
                        masked=False)
    spy_rows = {}
    for f in ("close", "open", "volume"):
        arr = ev.load_field(provider, "SPY", cal, f)
        spy_rows[f] = pd.DataFrame(
            {f: arr}, index=cal).assign(instrument="SPY") \
            .set_index("instrument", append=True)
        bars[f] = pd.concat([bars[f], spy_rows[f]]).sort_index()
    spy = pd.Series(spy_rows["close"]["close"].values,
                    index=cal).dropna().replace(0, np.nan).dropna()
    days = pd.DatetimeIndex(
        bars["close"].index.get_level_values("datetime").unique()
        .sort_values())
    return {"bars": bars, "spy": spy, "cal": cal, "days": days}


def window_days(days: pd.DatetimeIndex, a: str, b: str) -> pd.DatetimeIndex:
    return days[(days >= pd.Timestamp(a)) & (days <= pd.Timestamp(b))]


# ---------------------------------------------------------------------------
# signals — every leg computable by the live bot (parity: LV.* math on the
# same bars; vq = the PIT ancestor of state/live/vq_snapshot.parquet)


def lowvol_wide(close: pd.DataFrame) -> pd.DataFrame:
    """-std(daily returns, last 20) per symbol — LV.lowvol semantics."""
    out = {}
    for s in close.columns:
        px = close[s].dropna()
        if len(px) < 21:
            continue
        out[s] = -px.pct_change().rolling(20).std() \
            .reindex(close.index).ffill()
    return pd.DataFrame(out)


def beta_wide(close: pd.DataFrame) -> pd.DataFrame:
    """-60d beta vs the equal-weight universe mean — LV.lowbeta semantics."""
    mkt = close.pct_change().mean(axis=1)
    out = {}
    for s in close.columns:
        px = close[s].dropna()
        if len(px) < 61:
            continue
        r = px.pct_change()
        b = -r.rolling(60).cov(mkt) / mkt.rolling(60).var()
        out[s] = b.reindex(close.index).ffill()
    return pd.DataFrame(out)


def signal_panels(spec: dict, close: pd.DataFrame,
                  needed: set[str]) -> dict[str, pd.DataFrame]:
    """day x symbol score frames for each requested leg."""
    out = {}
    if "mom" in needed:
        out["mom"] = LH.bars_score(close, "mom").unstack("instrument")
    if "lowvol" in needed:
        out["lowvol"] = lowvol_wide(close)
    if "beta" in needed:
        out["beta"] = beta_wide(close)
    if "vq" in needed:
        provider = REPO / spec["provider"]
        cal = ev.load_calendar(provider)
        inst = ev.load_instruments(provider, "sp500")
        sigs = s6.build_signals(
            provider, inst["instrument"].unique().tolist(), cal, ["vq"],
            fundamentals_path=REPO / spec["fundamentals"],
            feats_path=REPO / "data/events/features_us.parquet",
            news_dir=REPO / "results", sector_map=None,
            close=close, volume=close)   # volume unused for vq only
        out["vq"] = sigs["vq"].unstack("instrument")
    return {k: v.reindex(close.index) for k, v in out.items()}


def blend(panels: dict[str, pd.DataFrame], legs: list[str]) -> pd.DataFrame:
    """Equal-weight per-day pct-rank blend — LV.final_scores semantics:
    ranks computed over each signal's own coverage, NaN propagates, so a
    name missing any leg is dropped (intersection)."""
    acc = None
    for leg in legs:
        r = panels[leg].rank(axis=1, pct=True)
        acc = r if acc is None else acc + r     # NaN + x = NaN (intersect)
    return acc / len(legs) if acc is not None else pd.DataFrame()


# ---------------------------------------------------------------------------
# candidates


def sleeve_n(spec: dict, cand: dict) -> int:
    """n = min(8, floor(sleeve_budget_usd / $1)) — every order clears the
    Alpaca minimum on the micro book."""
    budget = spec["account"]["cash_gbp"] * spec["account"]["gbpusd"]
    sleeve_budget = (1.0 - cand["core"]) * budget
    return int(min(8, max(0, int(sleeve_budget / spec["min_order_usd"]))))


def daily_targets(comp: pd.DataFrame, close: pd.DataFrame,
                  days: pd.DatetimeIndex, cand: dict,
                  spec: dict, exposure: dict) -> dict:
    """{sig_day: {sym: weight}} — the map the bot would trade next open.

    Picks are restricted to names with at least one close <= d — the same
    `prices` convention the live bot uses (last available bar), so a name
    never traded can't appear in the book."""
    sleeve = cand.get("sleeve") or {}
    n = int(sleeve.get("n", sleeve_n(spec, cand)))
    weighting = sleeve.get("weighting", "equal")
    risk_off = "to_core" if cand.get("guard") == "to_core" else "cash"
    first_seen = close.apply(lambda c: c.first_valid_index())
    out = {}
    for d in days:
        row = comp.loc[d].dropna() if d in comp.index else \
            pd.Series(dtype=float)
        eligible = row.index[
            [first_seen.get(s) is not None
             and first_seen.get(s) <= d for s in row.index]]
        picks = row.loc[eligible].nlargest(n).index.tolist()
        iv = CS.invvol(close, picks, d) if weighting == "invvol" else None
        out[d] = CS.build_targets(
            picks, cand["core"], invvol_w=iv,
            exposure=float(exposure.get(d, 1.0)), risk_off=risk_off,
            core_sym=spec["core_symbol"],
            cap=float(spec["max_stock_weight"]))
    return out


def regime_map(spec: dict, spy: pd.Series, guard: str) -> dict:
    """Per-day sleeve exposure; to_core -> 0.0 on risk-off days."""
    if guard != "to_core":
        return {}
    e = ad.regime_exposure(spy, sma=200, vol_win=21, vol_lookback=504,
                           vol_q=0.90, action=0.0)
    return {d: float(v) for d, v in e.items()}


def run_candidate(bars: dict, targets: dict, days: pd.DatetimeIndex,
                  spec: dict) -> dict:
    """micro_backtest over `days` with the phase-7 honest spec."""
    cl = bars["close"].loc[days[0]:days[-1]]
    ox = bars["open"].loc[days[0]:days[-1]]
    vl = bars["volume"].loc[days[0]:days[-1]]
    cost_map = mc.liquidity_cost_map(cl, vl,
                                     flat_bps=spec["costs"]["flat_bps"])
    return mc.micro_backtest(
        ad.empty_signal(), cl, cash_gbp=spec["account"]["cash_gbp"],
        gbpusd=spec["account"]["gbpusd"], exec_prices=ox, fill="next_open",
        delist_liquidate=spec["delist_liquidate"], rebalance="daily",
        cost_map=cost_map, band=spec["rebalance_band"],
        weight_targets=targets)


def breakeven_bps(met: dict, spy: pd.Series) -> dict:
    """Approximate cost sensitivity: the per-side cost (all-in) at which
    the candidate's arithmetic annualised return = SPY's, derived from
    the traded/cost series of the SAME run (no extra scoring)."""
    r = met["daily_excess"].dropna()
    spy_ann = float(spy.pct_change().reindex(r.index).mean() * 252)
    net_ann = float(r.mean() * 252)
    traded = met["daily_traded"].reindex(r.index)
    mu_traded = float(traded.mean())
    if mu_traded <= 0:
        return {"breakeven_extra_bps": np.inf, "breakeven_total_bps": np.inf,
                "avg_cost_bps": 0.0}
    avg_cost = float(met["daily_cost"].reindex(r.index).mean() /
                     mu_traded * 1e4)
    extra = max(0.0, (net_ann - spy_ann) / (252 * mu_traded) * 1e4)
    return {"breakeven_extra_bps": extra,
            "breakeven_total_bps": avg_cost + extra,
            "avg_cost_bps": avg_cost}


def score_window(cand: dict, comp: pd.DataFrame, books: dict,
                 days: pd.DatetimeIndex, spec: dict) -> tuple[dict, pd.Series]:
    """One candidate on one window -> (report row, daily net returns)."""
    exposure = regime_map(spec, books["spy"], cand.get("guard"))
    targets = daily_targets(comp, books["close"], days, cand, spec,
                            exposure)
    met = run_candidate(books["bars"], targets, days, spec)
    rep = LH.bot_metrics(met, spec["account"]["gbpusd"], books["spy"])
    daily = met["daily_excess"].dropna()
    roll = LH.rolling_xs_ir(daily, books["spy"])
    rep.update({
        "rolling_12m": roll,
        "roll12_win": roll.get("pct_positive"),
        "breakeven": breakeven_bps(met, books["spy"]),
        "by_year": LH.per_year_table(daily, books["spy"]),
        "n_targets_days": int(sum(bool(t) for t in targets.values())),
    })
    return rep, daily


def spy_row(books: dict, days: pd.DatetimeIndex) -> dict:
    return LH.spy_metrics(books["spy"], days)


# ---------------------------------------------------------------------------
# selection + config wiring


def select_winner(rows: list[dict], spy_oof: dict,
                  rule: dict) -> tuple[dict, dict]:
    """The predeclared rule: OOF mdd no deeper than SPY's AND >=50%
    rolling-12m OOF wins vs SPY -> highest OOF IR; else candidate A."""
    surv, dropped = [], {}
    for r in rows:
        if r["name"] == "A_spy":
            continue
        if r.get("max_drawdown", -1) < spy_oof["max_drawdown"]:
            dropped[r["name"]] = (f"mdd {r['max_drawdown']:.3f} deeper "
                                  f"than SPY {spy_oof['max_drawdown']:.3f}")
            continue
        if (r.get("roll12_win") or 0.0) < rule["rolling_12m_min_win_rate"]:
            dropped[r["name"]] = (f"roll12 win "
                                  f"{r.get('roll12_win'):.2f} < "
                                  f"{rule['rolling_12m_min_win_rate']:.2f}")
            continue
        surv.append(r)
    surv.sort(key=lambda r: (r["information_ratio"], r["core"]),
              reverse=True)
    winner = surv[0] if surv else next(r for r in rows
                                       if r["name"] == "A_spy")
    return winner, {"survivors": [r["name"] for r in surv],
                    "dropped": dropped,
                    "fallback_used": not bool(surv)}


def render_live_config(winner: dict, spec: dict, sel: dict) -> str:
    """The deployed configs/live_signal.yaml — full documented rewrite
    (the core+satellite schema differs from the legacy top-5 file)."""
    sname = winner.get("sleeve_signal")
    legs = spec["signals"].get(sname, []) if sname else []
    sleeve = winner.get("sleeve") or {}
    n = int(sleeve.get("n", sleeve_n(spec, winner)))
    weighting = sleeve.get("weighting", "equal")
    core_w = winner["core"]
    guard = winner.get("guard", "none")
    regime = "to_core" if guard == "to_core" else "none"
    return f"""# Phase-7 pin ({dt.date.today().isoformat()}): {winner['name']} selected
# by the predeclared OOF rule (configs/experiments/phase7_us.yaml).
# OOF 2018-2024 IR {winner['information_ratio']:+.3f} ret
# {winner['annualized_return']:+.3f} mdd {winner['max_drawdown']:+.3f};
# roll-12m win {winner.get('roll12_win')}; survivors {sel['survivors']}.
# `validated` flips true ONLY if the single 2026 holdout look clears the
# match gate (see docs/BOT_SETUP.md).
validated: false

# Live-trading interlock (NEVER set true here — see docs/BOT_SETUP.md).
# Going live additionally needs the LIVE_TRADING_ACK repo secret set to
# I_UNDERSTAND_LIVE_RISK, a non-paper ALPACA_BASE_URL + live keys, and the
# `live` GitHub Environment approval click. With this false every path
# stays paper-only.
live_trading: false

# Signal library for the sleeve. Computable live: mom, rev, srev5, mom6,
# lowvol, beta (Alpaca bars); vq, news (state/live snapshots).
signals: {legs}
weighting: equal          # signal blend inside the sleeve (equal ranks)
ic_months: 12
halflife_frac: 0.5

# Regime guard — on risk-off (SPY < SMA200 or top-decile trailing vol,
# lagged) the sleeve's freed share routes per sleeve.risk_off. to_core:
# sleeve -> SPY, never full cash.
regime: {regime}
regime_cfg:
  sma: 200
  vol_win: 21
  vol_lookback: 504
  vol_q: 0.90
  actions: {{none: 1.0, half: 0.5, cash: 0.0, to_core: 0.0}}

# Portfolio — core+satellite (Phase-7): {core_w:.0%} SPY core +
# {1 - core_w:.0%} signal sleeve of up to {n} names ({weighting}).
core: {{symbol: {spec['core_symbol']}, weight: {core_w}}}
sleeve: {{n: {n}, weighting: {weighting}, risk_off: to_core}}
n_positions: {n}          # legacy mirror of sleeve.n
budget_gbp: {spec['account']['cash_gbp']}
cost_bps: 10.0            # sizing conservatism (paper fills are free)
rebalance_band: {spec['rebalance_band']}   # skip |dw| < band or < $1
snapshot_stale_days: 45
order_after_open_min: 15

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
# commands


def build_all(spec: dict) -> tuple[dict, dict, dict]:
    """books (incl. wide `close`), per-signal wide panels, per-sleeve
    composite frames."""
    books = load_books(spec)
    books["close"] = books["bars"]["close"]["close"].unstack("instrument") \
        .sort_index()
    needed = {leg for legs in spec["signals"].values() for leg in legs}
    panels = signal_panels(spec, books["close"], needed)
    log(f"signals built: { {k: int(v.notna().sum().sum()) for k, v in panels.items()} }")
    comps = {name: blend(panels, legs)
             for name, legs in spec["signals"].items()}
    return books, panels, comps


def eval_candidates(spec: dict, books: dict, comps: dict,
                    days: pd.DatetimeIndex) -> tuple[list[dict], dict]:
    rows, daily = [], {}
    for cand in spec["candidates"]:
        comp = comps[cand["sleeve_signal"]] if cand["sleeve_signal"] \
            else pd.DataFrame()
        rep, dr = score_window(cand, comp, books, days, spec)
        rows.append({"name": cand["name"], "core": cand["core"],
                     "sleeve_signal": cand["sleeve_signal"],
                     "guard": cand.get("guard", "none"), **rep})
        daily[cand["name"]] = dr
        log(f"  {cand['name']:>10}: ar={rep['annualized_return']:+.3f} "
            f"ir={rep['information_ratio']:+.3f} "
            f"mdd={rep['max_drawdown']:+.3f} "
            f"roll12={rep.get('roll12_win')} to={rep['turnover']:.4f}")
    return rows, daily


def prior_trials() -> int:
    log_path = REPO / "results" / "holdout_log.jsonl"
    best = 0
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                v = json.loads(line).get("n_trials") or 0
                best = max(best, int(v))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return best


def cmd_select(spec: dict, out: Path) -> dict:
    books, _panels, comps = build_all(spec)
    oof_days = window_days(books["days"], *spec["oof"])
    log(f"OOF {oof_days[0].date()}..{oof_days[-1].date()} "
        f"({len(oof_days)}d) — {len(spec['candidates'])} candidates")
    rows, _daily = eval_candidates(spec, books, comps, oof_days)
    spy_oof = spy_row(books, oof_days)
    log(f"SPY OOF: ar={spy_oof['annualized_return']:+.3f} "
        f"ir={spy_oof['information_ratio']:+.3f} "
        f"mdd={spy_oof['max_drawdown']:+.3f}")
    winner, sel = select_winner(rows, spy_oof, spec["selection"])
    log(f"selection: winner={winner['name']} "
        f"(fallback={sel['fallback_used']}) survivors={sel['survivors']}")
    cand = next(c for c in spec["candidates"]
                if c["name"] == winner["name"])
    cfg_text = render_live_config({**winner, "sleeve": cand.get("sleeve")},
                                  spec, sel)
    (REPO / "configs" / "live_signal.yaml").write_text(cfg_text)
    log("configs/live_signal.yaml pinned to the winner")
    summary = {
        "stage": "phase7_us_select", "git_sha": git_sha(),
        "spec": str(SPEC_PATH.relative_to(REPO)),
        "oof_window": [str(oof_days[0].date()), str(oof_days[-1].date())],
        "n_oof_evals": len(rows), "prior_trials": prior_trials(),
        "spy_oof": spy_oof, "rows": rows, "winner": winner,
        "selection": sel, "live_config": cfg_text,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    log(f"wrote {out}")
    return summary


def _log_row(ts: str, cand_name: str, rep: dict, dsr: dict, n_trials: int,
             fresh: bool) -> dict:
    return {"ts": ts, "market": "us", "git_sha": git_sha(),
            "spec": "phase7_us.yaml", "candidate": f"phase7:{cand_name}",
            "re_viewed_holdout": not fresh,
            "IC": None, "ICIR": None, "RankIC": None, "RankICIR": None,
            **{k: v for k, v in rep.items()
               if k not in ("rolling_12m", "breakeven", "by_year",
                            "n_targets_days")},
            "dsr": dsr, "n_trials": n_trials}


def cmd_holdout(spec: dict, select_path: Path, out: Path) -> dict:
    sel = json.loads(select_path.read_text())
    winner = sel["winner"]
    if not winner:
        sys.exit("no winner in select — nothing to score")
    books, _panels, comps = build_all(spec)
    rev_days = window_days(books["days"], *spec["reviewed"])
    ho_days = window_days(books["days"], *spec["holdout"])
    ho_days = ho_days[ho_days <= books["days"][-1]]
    if len(ho_days) < 20:
        sys.exit(f"holdout window too thin: {len(ho_days)} trading days "
                 "— refresh the data dump before taking the single look")
    log(f"re-viewed 2025 {rev_days[0].date()}..{rev_days[-1].date()} | "
        f"FRESH holdout {ho_days[0].date()}..{ho_days[-1].date()} "
        f"({len(ho_days)}d) — ONE LOOK, winner={winner['name']}")

    # every candidate on the re-viewed 2025 window (already seen — report
    # only); the fresh 2026 window gets the winner + the SPY baseline ONLY.
    rev_rows, rev_daily = eval_candidates(spec, books, comps, rev_days)
    ho_names = {winner["name"], "A_spy"}
    ho_rows, ho_daily = [], {}
    for cand in spec["candidates"]:
        if cand["name"] not in ho_names:
            continue
        comp = comps[cand["sleeve_signal"]] if cand["sleeve_signal"] \
            else pd.DataFrame()
        rep, dr = score_window(cand, comp, books, ho_days, spec)
        ho_rows.append({"name": cand["name"], "core": cand["core"],
                        "sleeve_signal": cand["sleeve_signal"],
                        "guard": cand.get("guard", "none"), **rep})
        ho_daily[cand["name"]] = dr
        log(f"  [2026] {cand['name']:>10}: ar={rep['annualized_return']:+.3f} "
            f"ir={rep['information_ratio']:+.3f} "
            f"mdd={rep['max_drawdown']:+.3f}")
    spy_rev = spy_row(books, rev_days)
    spy_ho = spy_row(books, ho_days)

    n_trials = (prior_trials() + sel["n_oof_evals"] + len(rev_rows)
                + len(ho_rows))
    trial_irs = [float(r["information_ratio"]) for r in sel["rows"]
                 if r.get("information_ratio") is not None]
    ts = dt.datetime.now().isoformat()
    for r in rev_rows:
        rep_full = {k: v for k, v in r.items()
                    if k not in ("name", "core", "sleeve_signal", "guard")}
        dsr = wf.deflated_sharpe(rev_daily[r["name"]], n_trials, trial_irs)
        wf.append_holdout_log(_log_row(ts, r["name"], rep_full, dsr,
                                     n_trials, fresh=False))
    for r in ho_rows:
        rep_full = {k: v for k, v in r.items()
                    if k not in ("name", "core", "sleeve_signal", "guard")}
        dsr = wf.deflated_sharpe(ho_daily[r["name"]], n_trials, trial_irs)
        r["dsr"] = dsr
        wf.append_holdout_log(_log_row(ts, r["name"], rep_full, dsr,
                                       n_trials, fresh=True))

    # match gate — the single 2026 look
    g = spec["match_gate"]
    w = next(r for r in ho_rows if r["name"] == winner["name"])
    matched = (w["information_ratio"] >= spy_ho["information_ratio"]
               - g["ir_tol"]) or \
              (w["annualized_return"] >= spy_ho["annualized_return"]
               - g["ret_tol"])
    gate = {"matches_or_beats": bool(matched),
            "dd_no_worse": bool(w["max_drawdown"] >=
                                spy_ho["max_drawdown"] - g["mdd_tol"])}
    passed = all(gate.values())
    verdict = "MATCH GATE PASSED — deploy as validated" if passed else \
        "MATCH GATE FAILED — " + ", ".join(k for k, v in gate.items()
                                          if not v)
    log(f"gate {gate} -> {verdict} | DSR {w['dsr']['DSR']:.3f} "
        f"n_trials {n_trials}")

    if passed:
        cfg_path = REPO / "configs" / "live_signal.yaml"
        txt = cfg_path.read_text()
        txt = txt.replace("validated: false", "validated: true", 1)
        txt = txt.replace(
            "# `validated` flips true ONLY if the single 2026 holdout "
            "look clears the\n# match gate (see docs/BOT_SETUP.md).",
            "# `validated` flipped true — the single 2026 holdout look "
            f"cleared the\n# match gate: IR {w['information_ratio']:+.3f}"
            f" vs SPY {spy_ho['information_ratio']:+.3f}, mdd "
            f"{w['max_drawdown']:+.3f} vs {spy_ho['max_drawdown']:+.3f}.")
        cfg_path.write_text(txt)
        log("configs/live_signal.yaml -> validated: true")

    summary = {
        "stage": "phase7_us_holdout (fresh 2026, single look)",
        "git_sha": git_sha(), "winner": winner,
        "n_trials": n_trials,
        "reviewed_window": [str(rev_days[0].date()),
                            str(rev_days[-1].date())],
        "holdout_window": [str(ho_days[0].date()),
                           str(ho_days[-1].date())],
        "spy_reviewed": spy_rev, "spy_holdout": spy_ho,
        "reviewed_2025": rev_rows, "holdout_2026": ho_rows,
        "gate": gate, "verdict": verdict,
        "dsr": w["dsr"], "ts": ts}
    out.write_text(json.dumps(summary, indent=1))
    log(f"wrote {out} + appended {len(rev_rows) + len(ho_rows)} rows to "
        "holdout_log.jsonl")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "holdout"])
    ap.add_argument("--select", default="results/phase7_us_select.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    spec = yaml.safe_load(SPEC_PATH.read_text())
    if args.cmd == "select":
        cmd_select(spec, Path(args.out or "results/phase7_us_select.json"))
    else:
        cmd_holdout(spec, Path(args.select),
                    Path(args.out or "results/phase7_us_holdout.json"))


if __name__ == "__main__":
    main()
