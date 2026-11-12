"""Live-bot holdout — the paper bot's exact pipeline scored on 2025+.

    uv run python scripts/live_holdout.py \
        --out results/live_holdout_us.json

What this is: configs/live_signal.yaml (the pinned paper-bot config) run as
a monthly walk-forward — signal selection/weighting decided with data
through 2024-12-31 only, then the book trades 2025-01-01 -> latest bar,
re-deriving weights at each month boundary exactly like the live run does.
Costs 10 bps/side on the Phase-3 MicroTop5 £10 fractional simulator.

Pipeline parity with scripts/paper_trade.py (by construction — the same
functions are called):

  mom   LV.momentum math (close[t-21]/close[t-252]-1 on each symbol's
        available closes <= day) computed vectorised over the whole grid —
        bit-identical to LV.bars_signals, verified in tests.
  vq    the point-in-time ancestor of state/live/vq_snapshot.parquet:
        mean per-day rank of the store's vq_bm/vq_ep/vq_roe columns (XBRL
        facts keyed filed < D). The live snapshot is just this series'
        latest cross-section.
  news  optional PIT daily FinBERT signal (nw.daily_signal) — the ancestor
        of news_snapshot.parquet; included only with --with-news when the
        Kaggle results branch has landed.
  wts   config weighting re-derived at each month boundary using only days
        < month start ('equal' => constant 1/k, same as live; 'ewic' uses
        trailing rank-IC vs LABEL0 and falls back to equal when a
        snapshot-source signal is in the mix — the same rule the bot uses).
  guard ad.regime_exposure on SPY — identical shift-1 lagged math to the
        bot's LV.regime_state.
  book  mc.micro_backtest — the same MicroTop5 simulator; GBPUSD static
        1.27 like the research spec (live fetches the rate at run time).

HONESTY: the 2025+ window was already viewed in phases 2/3 — this is a
RE-VIEWED holdout and is logged as such. The gate for `validated: true`
stays: positive return AND beats SPY on IR AND DSR > 0.5. Never tune on
this result.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from quant_research import adaptive as ad  # noqa: E402
from quant_research import micro as mc  # noqa: E402


def log(msg: str) -> None:
    print(f"[live-ho {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          check=False, cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


# ---------------------------------------------------------------------------
# signals — PIT series on the (datetime, instrument) grid


def closes_wide(provider: Path, instruments: str) -> pd.DataFrame:
    """qlib dump -> wide date x symbol closes (the Alpaca bars the bot sees)."""
    long = mc.load_closes(provider, instruments)
    return long["close"].unstack("instrument").sort_index()


def bars_score(closes: pd.DataFrame, name: str) -> pd.Series:
    """(datetime, instrument) mom/rev over all days — LV.momentum semantics.

    LV.momentum uses the last-22nd / last-253rd *available* close <= day,
    i.e. positional on the dropna'd series — so we dropna per column, then
    shift, which is identical to calling LV.momentum on every day.
    """
    if name == "mom":
        num, den = 21, 252
    elif name == "rev":
        num, den = 0, 21
    else:
        raise ValueError(name)
    out = {}
    for s in closes.columns:
        px = closes[s].dropna()
        if len(px) < den + 1:
            continue
        v = px.shift(num) / px.shift(den) - 1.0
        if name == "rev":
            v = -v
        # LV.momentum(closes[s], d) reads px.loc[:d] — on a no-bar day the
        # score is the last bar's value, so ffill over the shared calendar
        out[s] = v.reindex(closes.index).ffill()
    df = pd.DataFrame(out).sort_index()
    long = df.stack().rename(name)
    long.index.names = ["datetime", "instrument"]
    return long


def vq_score(store: pd.DataFrame) -> pd.Series:
    """PIT vq composite — mean per-day pct rank of bm/ep/roe (filed < D)."""
    parts = [store[c].swaplevel(0, 1).sort_index()
             .groupby(level="datetime").rank(pct=True)
             for c in ("vq_bm", "vq_ep", "vq_roe")]
    return (sum(parts) / len(parts)).rename("vq")


def monthly_weights(cfg: dict, sigs: dict[str, pd.Series],
                    label: pd.Series | None,
                    months: list[tuple]) -> list[dict[str, float]]:
    """Weight dict per month block — the same rule compute_signals applies.

    'equal' => 1/k over the signals live at that month (all of them here).
    'ewic'  => EW rank-IC over trailing ic_months of days < m0; a
               snapshot-source signal (vq/news) in the mix => equal, the
               bot's documented fallback (live can't score their IC).
    """
    weighting = cfg.get("weighting", "equal")
    snap_src = {"vq", "news"}
    if weighting == "ewic" and label is not None and \
            not any(s in snap_src for s in sigs):
        hist = ad.ic_history(sigs, label)
        hl = cfg["ic_months"] * 21 * cfg.get("halflife_frac", 0.5)
        return [{s: float(w.get(s, 0.0)) for s in sigs}
                for w in (ad.ew_ic_weights(hist, m0, cfg["ic_months"], hl,
                                           list(sigs)) for m0, _ in months)]
    if weighting == "ewic":
        log("ewic: snapshot-source signal in mix — equal weights (live rule)")
    return [{s: 1.0 / len(sigs) for s in sigs} for _ in months]


def composite(sigs: dict[str, pd.Series],
              weights: list[dict[str, float]],
              months: list[tuple]) -> pd.Series:
    """Per-day weighted-rank composite — LV.final_scores semantics.

    Each signal is pct-ranked over its own coverage that day; the composite
    sums w*rank and DROPS symbols missing any positively-weighted signal
    (intersection, exactly like `sum(parts).dropna()` in final_scores).
    """
    ranked = {}
    for s, v in sigs.items():
        r = v.groupby(level="datetime").rank(pct=True)
        ranked[s] = r.unstack("instrument")          # day x sym ranks
    parts = []
    for (m0, m1), w in zip(months, weights):
        pos = [s for s, wt in w.items() if wt > 0]
        if not pos:
            continue                                  # flat month — no score
        acc = None
        for s in pos:
            r = ranked[s].loc[m0:m1]
            acc = r * w[s] if acc is None else acc.add(r * w[s])
        parts.append(acc)
    if not parts:
        return pd.Series(dtype=float, name="score")
    wide = pd.concat(parts)
    # stack() drops NaN cells -> per-(day,symbol) intersection across the
    # weighted signals, exactly like sum(parts).dropna() in final_scores
    long = wide.stack().rename("score")
    long.index.names = ["datetime", "instrument"]
    return long.sort_index()


# ---------------------------------------------------------------------------
# metrics


def spy_metrics(spy: pd.Series, days: pd.DatetimeIndex) -> dict:
    """SPY buy-and-hold over exactly the bot's equity-curve days."""
    r = spy.pct_change().reindex(days).dropna()
    eq = (1 + r).cumprod()
    if r.empty:
        return {"total_return": 0.0, "annualized_return": 0.0,
                "information_ratio": 0.0, "max_drawdown": 0.0,
                "hit_rate": 0.0, "n_days": 0}
    return {"total_return": float(eq.iloc[-1] - 1),
            "annualized_return": float(eq.iloc[-1] ** (252 / len(r)) - 1),
            "information_ratio": float(r.mean() / r.std() * np.sqrt(252))
            if r.std() > 0 else 0.0,
            "max_drawdown": float((eq / eq.cummax() - 1).min()),
            "hit_rate": float((r > 0).mean()),
            "n_days": len(r)}


def roundtrips(trades: list[dict]) -> dict:
    """FIFO lot-matching on the trade ledger -> round-trip win stats."""
    open_lots: dict[str, list[list]] = {}
    closed, wins = 0, 0
    pnl = 0.0
    for t in trades:
        if t["side"] == "buy":
            open_lots.setdefault(t["sym"], []).append(
                [t["qty"], t["price"]])
        else:
            qty = t["qty"]
            lots = open_lots.get(t["sym"], [])
            while qty > 1e-12 and lots:
                take = min(qty, lots[0][0])
                pnl += take * (t["price"] - lots[0][1])
                closed += 1
                wins += t["price"] > lots[0][1]
                lots[0][0] -= take
                qty -= take
                if lots[0][0] <= 1e-12:
                    lots.pop(0)
    return {"round_trips": closed, "win_rate": wins / closed if closed else 0.0,
            "realised_pnl_usd": round(pnl, 4)}


def bot_metrics(met: dict, gbpusd: float, spy: pd.Series) -> dict:
    """Extend micro_backtest's dict with the owner's report fields."""
    r = met["daily_excess"].dropna()
    eq_days = r.index
    total = met["final_equity_usd"] / (10.0 * gbpusd) - 1.0
    spy_m = spy_metrics(spy, pd.DatetimeIndex(eq_days))
    xs = (r - spy.pct_change().reindex(r.index)).dropna()
    out = {k: v for k, v in met.items()
           if k not in ("daily_excess", "trades")}
    out.update({
        "total_return": float(total),
        "final_gbp": float(met["final_equity_usd"] / gbpusd),
        "hit_rate": float((r > 0).mean()) if len(r) else 0.0,
        "ir_xs_spy": float(xs.mean() / xs.std() * np.sqrt(252))
        if len(xs) > 1 and xs.std() > 0 else 0.0,
        "ar_xs_spy": float(xs.mean() * 252) if len(xs) else 0.0,
        "spy": spy_m,
        **roundtrips(met["trades"])})
    return out


# ---------------------------------------------------------------------------
# run


def run_window(cfg: dict, sigs: dict[str, pd.Series],
               label: pd.Series | None, closes_long: pd.DataFrame,
               spy: pd.Series, days: pd.DatetimeIndex,
               regime: str, micro_kw: dict) -> tuple[dict, dict]:
    """One regime variant over `days`; returns (metrics, weight_ledger).

    Signals are restricted to `closes_long`'s index — the point-in-time
    eligible universe (membership-windowed, optionally live-list filtered)
    — so per-day ranks/weights are computed over tradable names only."""
    elig = closes_long.index
    sigs = {s: v[v.index.isin(elig)] for s, v in sigs.items()}
    months = ad.month_blocks(days)
    wlist = monthly_weights(cfg, sigs, label, months)
    ledger = {str(m0.date()): w for (m0, _), w in zip(months, wlist)}
    score = composite(sigs, wlist, months)
    exposure = None
    if regime != "none":
        rc = cfg["regime_cfg"]
        exp = ad.regime_exposure(
            spy, sma=rc["sma"], vol_win=rc["vol_win"],
            vol_lookback=rc["vol_lookback"], vol_q=rc["vol_q"],
            action=rc["actions"][regime])
        exposure = {d: float(e) for d, e in exp.items()}
    # sim iterates every priced day in `closes` — slice to the window or
    # thousands of flat pre-window days would dilute every metric
    cl = closes_long.loc[days[0]:days[-1]]
    kw = dict(micro_kw)
    ex = kw.pop("exec_long", None)
    ox = ex.loc[days[0]:days[-1]] if ex is not None else None
    met = mc.micro_backtest(score, cl, exposure=exposure,
                            exec_prices=ox, **kw)
    return met, ledger


# ---------------------------------------------------------------------------
# robustness (REPORTED ONLY — never used to pick the config)


def per_year_table(daily: pd.Series, spy: pd.Series) -> dict:
    """Calendar-year breakdown of an equity-return series vs SPY."""
    out = {}
    spyr = spy.pct_change()
    for y, r in daily.groupby(daily.index.year):
        eq = (1 + r).cumprod()
        xs = (r - spyr.reindex(r.index)).dropna()
        out[int(y)] = {
            "total": float(eq.iloc[-1] - 1),
            "ir": float(r.mean() / r.std() * np.sqrt(252))
            if r.std() > 0 else 0.0,
            "max_dd": float((eq / eq.cummax() - 1).min()),
            "xs_ann": float(xs.mean() * 252) if len(xs) else 0.0,
            "n_days": int(len(r))}
    return out


def rolling_xs_ir(daily: pd.Series, spy: pd.Series,
                  win: int = 252) -> dict:
    """Rolling `win`-day IR of (bot - SPY) daily returns -> summary stats."""
    xs = (daily - spy.pct_change().reindex(daily.index)).dropna()
    if len(xs) < win:
        return {"window": win, "n_windows": 0}
    roll = xs.rolling(win)
    ir = roll.mean() / roll.std() * np.sqrt(252)
    ir = ir.dropna()
    return {"window": win, "n_windows": int(len(ir)),
            "min": float(ir.min()), "median": float(ir.median()),
            "latest": float(ir.iloc[-1]),
            "pct_positive": float((ir > 0).mean())}


def bootstrap_xs(daily: pd.Series, spy: pd.Series, block: int = 21,
                 n_boot: int = 2000, seed: int = 0) -> dict:
    """Stationary block bootstrap on (bot - SPY) daily returns -> 95% CI
    of the annualized mean excess return."""
    xs = (daily - spy.pct_change().reindex(daily.index)).dropna().values
    rng = np.random.default_rng(seed)
    if len(xs) < block * 2:
        return {"block": block, "n_boot": n_boot}
    n_blocks = int(np.ceil(len(xs) / block))
    starts = rng.integers(0, len(xs) - block + 1, size=(n_boot, n_blocks))
    # gather blocks -> (n_boot, n_blocks*block), truncate to len(xs)
    idx = starts[..., None] + np.arange(block)
    samples = xs[idx].reshape(n_boot, -1)[:, :len(xs)]
    ann = samples.mean(axis=1) * 252
    return {"block": block, "n_boot": n_boot,
            "xs_ann_mean": float(ann.mean()),
            "ci_lo": float(np.percentile(ann, 2.5)),
            "ci_hi": float(np.percentile(ann, 97.5)),
            "pct_positive": float((ann > 0).mean())}


def _filter_universe(bars: dict, keep: set | None) -> dict:
    """Restrict (datetime, instrument) frames to `keep` symbols (None=all)."""
    if keep is None:
        return bars
    return {f: df[df.index.get_level_values("instrument").isin(keep)]
            for f, df in bars.items()}


def eval_variant(cfg: dict, sigs: dict[str, pd.Series],
                 label: pd.Series | None, bars: dict, spy: pd.Series,
                 days: pd.DatetimeIndex, spec: dict,
                 base_kw: dict) -> tuple[dict, pd.Series, dict]:
    """One spec -> (report, daily_excess, weight ledger).

    spec keys: universe(None=PIT full dump), cost('flat'|'liquidity'),
    flat_bps, fill, delist, rebalance, n, regime."""
    closes_long = bars["close"]
    opens_long = bars["open"]
    micro_kw = {**base_kw, "n": spec.get("n") or base_kw["n"],
                "fill": spec.get("fill", "close"),
                "delist_liquidate": bool(spec.get("delist")),
                "rebalance": spec.get("rebalance", "daily"),
                "exec_long": opens_long}
    if spec.get("cost") == "liquidity":
        micro_kw["cost_map"] = mc.liquidity_cost_map(
            closes_long, bars["volume"], flat_bps=spec.get("flat_bps", 10.0))
    else:
        micro_kw["cost_bps"] = spec.get("flat_bps", base_kw["cost_bps"])
    met, ledger = run_window(cfg, sigs, label, closes_long, spy, days,
                             spec.get("regime", cfg["regime"]), micro_kw)
    rep = bot_metrics(met, base_kw["gbpusd"], spy)
    return rep, met["daily_excess"], ledger


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO / "configs" /
                                           "live_signal.yaml"))
    p.add_argument("--provider-uri", default=str(REPO / "data" / "qlib" /
                                               "us_alpaca"))
    p.add_argument("--instruments", default="sp500")
    p.add_argument("--store", default=str(REPO / "data" / "features4" /
                                          "store_us.parquet"))
    p.add_argument("--news", default=str(REPO / "results" / "news_sentiment" /
                                         "daily_sentiment.parquet"))
    p.add_argument("--with-news", action="store_true",
                   help="include the news signal (needs --news parquet)")
    p.add_argument("--eval-start", default="2025-01-01")
    p.add_argument("--eval-end", default="")
    p.add_argument("--oof-end", default="2024-12-31",
                   help="per-year robustness table covers data_start..oof_end")
    p.add_argument("--insample-start", default="2023-01-01",
                   help="context-only in-sample window start (train side)")
    p.add_argument("--gbpusd", type=float, default=mc.GBPUSD)
    p.add_argument("--cost-sweep", default="0,10,25,50",
                   help="flat per-side bps sensitivity grid")
    p.add_argument("--no-robustness", action="store_true",
                   help="skip the report-only robustness block")
    p.add_argument("--prior-trials", type=int, default=-1,
                   help="override DSR trial count (default: spec "
                        "prior_trials_us + select n_candidates + 2 prior "
                        "livebot rows + variants scored here)")
    p.add_argument("--no-log", action="store_true",
                   help="skip appending results/holdout_log.jsonl")
    p.add_argument("--out", default=str(REPO / "results" /
                                        "live_holdout_us.json"))
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    log(f"=== live-bot holdout config={args.config} signals={cfg['signals']} "
        f"weighting={cfg['weighting']} regime={cfg['regime']} ===")

    from quant_research import events as ev
    live_uni = {ev.canon(l) for l in
                (REPO / cfg["universe"]).read_text().splitlines()
                if l.strip() and not l.startswith("#")}

    # -- market data (PIT dump: membership windows, delisted names included) ------
    provider = Path(args.provider_uri)
    # `bars` is membership-windowed: tradable/eligible universe per day.
    bars = mc.load_bars(provider, args.instruments,
                        ("close", "open", "volume"), masked=True)
    # `full_close` is the full data span per symbol — signal math may read
    # pre-membership history (a new member still has 252d of prices).
    full_close = mc.load_bars(provider, args.instruments, ("close",),
                              masked=False)["close"]
    cal = ev.load_calendar(provider)
    spy = pd.Series(ev.load_close(provider, "SPY", cal), index=cal) \
        .dropna().replace(0, np.nan).dropna()
    all_days = pd.DatetimeIndex(
        bars["close"].index.get_level_values("datetime").unique()
        .sort_values())
    e0 = pd.Timestamp(args.eval_start)
    e1 = pd.Timestamp(args.eval_end) if args.eval_end else all_days.max()
    eval_days = pd.DatetimeIndex(all_days[(all_days >= e0) &
                                          (all_days <= e1)])
    is_days = pd.DatetimeIndex(all_days[(all_days >= args.insample_start) &
                                        (all_days < e0)])
    oof_days = pd.DatetimeIndex(all_days[(all_days <=
                                          pd.Timestamp(args.oof_end))])
    if eval_days.empty:
        sys.exit(f"no bars in eval window {args.eval_start}..{args.eval_end}")
    log(f"bars through {all_days.max().date()}; eval {eval_days[0].date()}"
        f"..{eval_days[-1].date()} ({len(eval_days)}d); "
        f"oof {oof_days[0].date()}..{oof_days[-1].date()} "
        f"({len(oof_days)}d)")

    # -- signals (PIT values; eligibility applied per-variant via the mask) ------
    wide = full_close["close"].unstack("instrument").sort_index()
    sigs: dict[str, pd.Series] = {}
    notes: list[str] = []
    store = None
    need_store = any(s in ("vq", "news") for s in cfg["signals"]) or \
        cfg.get("weighting") == "ewic"
    if need_store:
        cols = ["vq_bm", "vq_ep", "vq_roe", "LABEL0"]
        store = pd.read_parquet(
            args.store, columns=[c for c in cols])
        log(f"store cols loaded: {store.shape}")
    for s in cfg["signals"]:
        if s in ("mom", "rev"):
            sigs[s] = bars_score(wide, s)
        elif s == "vq":
            sigs[s] = vq_score(store)
        elif s == "news":
            if args.with_news and Path(args.news).exists():
                from quant_research import news as nw
                daily = pd.read_parquet(args.news)
                sigs[s] = nw.daily_signal(daily, store).rename("news")
            else:
                notes.append("news: skipped (no --with-news / file absent)")
        else:
            notes.append(f"{s}: not live-computable — dropped (bot rule)")
    sigs = {s: v.dropna() for s, v in sigs.items() if not v.dropna().empty}
    log(f"signals live: {list(sigs)} | " + "; ".join(notes))
    if not sigs:
        sys.exit("no usable signals")

    label = None
    if cfg.get("weighting") == "ewic":
        label = store["LABEL0"].swaplevel(0, 1).sort_index()

    # -- variant grid ---------------------------------------------------------------
    # PRIMARY = the honest spec: PIT universe (delisted names included),
    # liquidity-bucket half-spread + 10bps, next-open fills, delist
    # liquidation. Every row below is a scored trial and counts in n_trials.
    base_kw = {"n": int(cfg["n_positions"]),
               "cash_gbp": float(cfg["budget_gbp"]),
               "gbpusd": args.gbpusd,
               "cost_bps": float(cfg["cost_bps"])}
    HONEST = {"cost": "liquidity", "flat_bps": 10.0, "fill": "next_open",
              "delist": True, "rebalance": "daily"}
    LEGACY = {"cost": "flat", "flat_bps": float(cfg["cost_bps"]),
              "fill": "close", "delist": False, "rebalance": "daily"}

    variants: list[tuple[str, dict, set | None]] = [
        # survivorship before/after: identical spec, only the universe changes
        ("survivors_only(legacy)", {**LEGACY}, live_uni),
        ("pit(legacy)", {**LEGACY}, None),
        # the honest primary + regime-off comparator
        ("honest", {**HONEST, "regime": cfg["regime"]}, None),
        ("honest_regimeoff", {**HONEST, "regime": "none"}, None),
    ]
    # flat-cost sensitivity on the honest spec
    for bps in [float(x) for x in args.cost_sweep.split(",") if x.strip()]:
        variants.append((f"flat{int(bps)}bps",
                         {**HONEST, "cost": "flat", "flat_bps": bps}, None))
    if not args.no_robustness:
        variants += [
            ("robust_top10", {**HONEST, "n": 10}, None),
            ("robust_top20", {**HONEST, "n": 20}, None),
            ("robust_weekly", {**HONEST, "rebalance": "weekly"}, None),
            ("robust_monthly", {**HONEST, "rebalance": "monthly"}, None),
        ]

    results, ledgers, daily = {}, {}, {}
    for name, spec, uni in variants:
        vb = _filter_universe(bars, uni)
        rep, dr, led = eval_variant(cfg, sigs, label, vb, spy, eval_days,
                                    spec, base_kw)
        results[name] = rep
        ledgers[name] = led
        daily[name] = dr
        log(f"{name:>22}: tot={rep['total_return']:+.3f} "
            f"ar={rep['annualized_return']:+.3f} "
            f"ir={rep['information_ratio']:+.3f} "
            f"mdd={rep['max_drawdown']:+.3f} "
            f"ir_xs={rep['ir_xs_spy']:+.3f} to={rep['turnover']:.3f} "
            f"final=£{rep['final_gbp']:.2f}")

    # in-sample context (train side — NOT a holdout scoring, no tuning use)
    insample = {}
    for rg in {cfg["regime"], "none"}:
        spec = {**HONEST, "regime": rg}
        rep, _, _ = eval_variant(cfg, sigs, label, bars, spy, is_days,
                                 spec, base_kw)
        insample[f"regime:{rg}"] = rep
        log(f"  [in-sample ctx] regime:{rg} tot={rep['total_return']:+.3f} "
            f"ir={rep['information_ratio']:+.3f}")

    # -- robustness (reported, NEVER used to pick) ---------------------------------
    robust = {}
    if not args.no_robustness:
        # per-year OOF table on the honest spec (pre-2025 data only)
        _, dr_oof, _ = eval_variant(cfg, sigs, label, bars, spy,
                                    oof_days, HONEST, base_kw)
        robust["per_year_oof"] = per_year_table(dr_oof, spy)
        log("  [oof] " + " ".join(
            f"{y}:{v['ir']:+.2f}" for y, v in robust["per_year_oof"].items()))
        robust["rolling_12m_xs_ir"] = rolling_xs_ir(daily["honest"], spy)
        robust["bootstrap_xs"] = bootstrap_xs(daily["honest"], spy)
        log(f"  [robust] rolling12m xs-IR med="
            f"{robust['rolling_12m_xs_ir'].get('median'):+.2f} | boot xs "
            f"CI=[{robust['bootstrap_xs']['ci_lo']:+.3f},"
            f"{robust['bootstrap_xs']['ci_hi']:+.3f}]")

    # -- DSR + honest log ------------------------------------------------------------
    from quant_research import walkforward as wf
    sel_json = REPO / "results" / "phase4_us_select.json"
    n_sel, sel_irs = 0, []
    if sel_json.exists():
        d = json.loads(sel_json.read_text())
        n_sel = d["n_candidates"]
        sel_irs = [c["information_ratio"] for c in d["candidates"].values()]
    spec_path = REPO / "configs/experiments/walkforward_us_adaptive.yaml"
    prior = 0
    if spec_path.exists():
        prior = int(yaml.safe_load(spec_path.read_text())
                    .get("prior_trials_us", 0))
    # honest trial count: every prior scored attempt (66 US phases 2/3 +
    # 27 select + 2 earlier livebot rows) plus every variant scored above
    n_prior_livebot = 2
    n_trials = (args.prior_trials if args.prior_trials > 0
                else prior + n_sel + n_prior_livebot + len(variants))
    for name, rep in results.items():
        dsr = wf.deflated_sharpe(daily[name], n_trials, sel_irs)
        rep["dsr"] = dsr
        log(f"{name:>22} DSR={dsr['DSR']:.3f} (SR*={dsr['SR_star']:.3f}, "
            f"n_trials={n_trials})")

    # -- verdict ----------------------------------------------------------------------
    primary = results["honest"]
    gate = {"positive": primary["total_return"] > 0,
            "beats_spy_ir": primary["information_ratio"] >
            primary["spy"]["information_ratio"],
            "dsr_gt_0.5": primary["dsr"]["DSR"] > 0.5}
    verdict = "VALIDATION GATE PASSED" if all(gate.values()) else \
        "VALIDATION GATE FAILED — " + \
        ", ".join(k for k, v in gate.items() if not v)
    log(f"verdict: {verdict}")

    # -- persist ------------------------------------------------------------------------
    ts = dt.datetime.now().isoformat()
    summary = {
        "market": "us", "git_sha": git_sha(), "mode": "live_bot_holdout",
        "re_viewed_holdout": True,
        "note": "2025+ window previously viewed in phases 2/3 — this is a "
                "re-reviewed holdout of the pinned live-bot pipeline; no "
                "tuning permitted on this result",
        "config": {"path": args.config, "signals": cfg["signals"],
                   "weighting": cfg["weighting"], "regime": cfg["regime"],
                   "n_positions": cfg["n_positions"],
                   "budget_gbp": cfg["budget_gbp"],
                   "cost_bps": cfg["cost_bps"], "gbpusd": args.gbpusd},
        "with_news": bool(args.with_news and "news" in sigs),
        "eval_window": [str(eval_days[0].date()), str(eval_days[-1].date())],
        "insample_window": [str(is_days[0].date()), str(is_days[-1].date())],
        "honest_spec": {"universe": "pit (membership windows, delisted "
                        "names included)", "cost": "liquidity half-spread "
                        "buckets + 10bps flat", "fill": "next_open",
                        "delist_liquidate": True},
        "variant_specs": {n: {**s, "universe": "live" if u is not None
                              else "pit"} for n, s, u in variants},
        "n_trials": n_trials, "results": results,
        "insample_context": insample,
        "robustness": robust,
        "weight_ledgers": ledgers, "gate": gate, "verdict": verdict,
        "finished": ts}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=str))
    log(f"wrote {out}")

    if not args.no_log:
        for name, rep in results.items():
            row = {"ts": ts, "market": "us", "git_sha": git_sha(),
                   "spec": Path(args.config).name,
                   "candidate": f"livebot:{'+'.join(sigs)}@micro5-"
                                f"{cfg['weighting']}-{name.replace(':', '')}",
                   "re_viewed_holdout": True,
                   "IC": None, "ICIR": None, "RankIC": None,
                   "RankICIR": None,
                   **{k: v for k, v in rep.items() if k != "dsr"},
                   "dsr": rep["dsr"], "n_trials": n_trials}
            wf.append_holdout_log(row)
        log(f"appended {len(results)} rows to results/holdout_log.jsonl")

    print(json.dumps({"verdict": verdict, "gate": gate,
                      "primary": {k: primary.get(k) for k in
                                  ("total_return", "annualized_return",
                                   "information_ratio", "max_drawdown",
                                   "turnover", "hit_rate", "final_gbp")},
                      "spy": primary["spy"],
                      "dsr": primary["dsr"]["DSR"],
                      "n_trials": n_trials}, indent=1))


if __name__ == "__main__":
    main()
