"""Phase-4 adaptive pipeline driver.

    # selection — monthly walk-forward over 2021-2024 ONLY:
    uv run python scripts/phase4_run.py --mode select

    # holdout — chosen process scored once over 2025-01 -> latest:
    uv run python scripts/phase4_run.py --mode holdout \
        --config '{"train_months":24,"ic_months":6,"weighting":"ewic","regime":"half"}'

    # smoke — plumbing check, 3 months x one window (runner-side, not the Mac):
    uv run python scripts/phase4_run.py --mode smoke

Inputs (produced by earlier jobs — never fetched/trained on the owner's Mac):
  data/features4/store_us.parquet                  `store` job / phase4_build.py
  results/news_sentiment/daily_sentiment.parquet   Kaggle FinBERT run
  data/qlib/us_alpaca                              us-qlib-dump artifact

Monthly model preds checkpoint to results/oof/us_adaptive/ so a reclaimed
runner resumes where it left off.
"""
import argparse
import datetime as dt
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quant_research import adaptive as ad
from quant_research import events as ev
from quant_research import micro as mc
from quant_research import walkforward as wf

REPO = Path(__file__).resolve().parents[1]
RUN_KEY = "us_adaptive"


def log(msg: str) -> None:
    print(f"[phase4 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False,
                          cwd=REPO, capture_output=True, text=True).stdout.strip()


def load_spec() -> dict:
    import yaml
    p = REPO / "configs/experiments/walkforward_us_adaptive.yaml"
    spec = yaml.safe_load(p.read_text())
    spec["_spec_name"] = p.stem
    return spec


# ---------------------------------------------------------------------------
# inputs


def load_inputs(spec: dict, columns: list[str] | None = None):
    store_path = REPO / spec["store"]
    store = pd.read_parquet(store_path, columns=columns)
    mpath = store_path.with_suffix(".manifest.json")
    meta = json.loads(mpath.read_text()) if mpath.exists() else {}
    news_path = REPO / spec["news_daily"]
    news_daily = pd.read_parquet(news_path) if news_path.exists() else None
    if news_daily is None:
        log("WARNING: no news sentiment file — 'news' signal all-NaN")
    return store, meta, news_daily


def spy_close(spec: dict) -> pd.Series:
    provider = REPO / spec["provider_uri"]
    cal = ev.load_calendar(provider)
    close = ev.load_close(provider, "SPY", cal)
    return pd.Series(close, index=cal, name="SPY").dropna()


# ---------------------------------------------------------------------------
# signal assembly (shared by select + holdout)


def assemble_signals(store: pd.DataFrame, meta: dict, news_daily,
                     pred_months: list, windows: list[int],
                     threads: int) -> tuple:
    """static signals + monthly-retrained model preds per window length.

    Returns ({W: {sig: (datetime,instrument) series}}, label series)."""
    cache = wf.oof_dir(RUN_KEY)
    static = ad.static_signals(store, news_daily)
    a158 = [c for c in meta.get("a158_cols", []) if c in store.columns]
    evc = [c for c in meta.get("event_cols", []) if c in store.columns]
    X = {
        "gbdt": store[a158].swaplevel(0, 1).sort_index(),
        "ev": store[evc].swaplevel(0, 1).sort_index(),
    }
    y = store["LABEL0"].swaplevel(0, 1).sort_index()
    out = {}
    for w in windows:
        sigs = dict(static)
        for name in ("gbdt", "ev"):
            log(f"monthly preds {name} w={w}m over {len(pred_months)} months")
            sigs[name] = ad.model_signal_series(
                X[name], y, name, pred_months, w, cache, threads, log)
        out[w] = sigs
    return out, y


def eval_exposure(spec: dict, regime: str) -> pd.Series | None:
    """Precomputed per-day exposure for a regime variant (SPY-only data)."""
    if regime == "none":
        return None
    rc = spec["regime"]
    spy = spy_close(spec)
    return ad.regime_exposure(spy, sma=rc["sma"], vol_win=rc["vol_win"],
                              vol_lookback=rc["vol_lookback"],
                              vol_q=rc["vol_q"],
                              action=rc["action"][regime])


def xs_metrics(daily: pd.Series, spy: pd.Series) -> dict:
    """Excess vs SPY buy-and-hold + SPY standalone stats on the same days."""
    spy_r = spy.pct_change().reindex(daily.index)
    x = (daily - spy_r).dropna()
    spy_r = spy_r.dropna()

    def sharpe(r):
        return float(r.mean() / r.std() * np.sqrt(252)) \
            if len(r) > 1 and r.std() > 0 else 0.0
    spy_eq = (1 + spy_r).cumprod()
    return {"ir_xs_spy": sharpe(x),
            "ar_xs_spy": float(x.mean() * 252) if len(x) else 0.0,
            "spy_ir": sharpe(spy_r),
            "spy_ar": float((1 + spy_r).prod() ** (252 / max(len(spy_r), 1)) - 1)
            if len(spy_r) else 0.0,
            "spy_mdd": float((spy_eq / spy_eq.cummax() - 1).min())
            if len(spy_eq) else 0.0}


def backtest_one(name, score, ledger, signals, label, months, exposure,
                 closes, micro_kw, spy, hist, candidates, ledgers):
    """Micro sim + signal metrics for one candidate; writes into the dicts."""
    met = mc.micro_backtest(score, closes, exposure=exposure, **micro_kw)
    sig = ad.signal_metrics(score, label)
    res = {**sig, **{k: v for k, v in met.items() if k != "daily_excess"}}
    res.update(xs_metrics(met["daily_excess"], spy))
    candidates[name] = res
    ledgers[name] = ledger
    log(f"  {name}: ric={sig['RankIC']} ir={res['information_ratio']:.4f} "
        f"ar={res['annualized_return']:.4f} mdd={res['max_drawdown']:.4f} "
        f"ir_xs={res['ir_xs_spy']:.4f}")
    return res


# ---------------------------------------------------------------------------
# selection run (2021-2024 ONLY)


def run_select(spec: dict, args) -> dict:
    store, meta, news_daily = load_inputs(spec)
    grid = spec["grid"]
    windows = sorted(set(grid["train_months"]) | {36})
    e0, e1 = (pd.Timestamp(spec["eval_window"][0]),
              pd.Timestamp(spec["eval_window"][1]))
    max_ic = max(grid["ic_months"])
    hist_start = e0 - pd.DateOffset(months=max_ic + 1)

    days = store.index.get_level_values("datetime").unique().sort_values()
    pred_days = days[(days >= hist_start) & (days <= e1)]
    pred_months = ad.month_blocks(pd.DatetimeIndex(pred_days))
    eval_days = pd.DatetimeIndex(days[(days >= e0) & (days <= e1)])
    eval_months = ad.month_blocks(eval_days)
    log(f"pred months {pred_months[0][0].date()}..{pred_months[-1][1].date()} "
        f"({len(pred_months)}), eval months {len(eval_months)}")

    sig_by_w, label = assemble_signals(store, meta, news_daily, pred_months,
                                       windows, args.threads)

    provider = REPO / spec["provider_uri"]
    closes = mc.load_closes(provider, spec["instruments"])
    closes = closes.loc[e0:e1]
    spy = spy_close(spec)
    micro_kw = spec["micro"]
    exposures = {r: eval_exposure(spec, r) for r in set(grid["regime"])}

    # ranked + IC history are weighting/regime-independent: once per W
    ranked_by_w, hist_by_w = {}, {}
    for w, sigs in sig_by_w.items():
        ranked_by_w[w] = pd.DataFrame({s: ad.cs_rank(v)
                                       for s, v in sigs.items()})
        hist_by_w[w] = ad.ic_history(sigs, label)

    candidates, ledgers = {}, {}

    def score_combo(name, w, weighting, ic_months, regime):
        hl = ic_months * 21 * spec.get("halflife_frac", 0.5)
        score, ledger = ad.combined_score(
            ranked_by_w[w], hist_by_w[w], eval_months, weighting,
            ic_months, hl, list(sig_by_w[w]))
        backtest_one(name, score, ledger, sig_by_w[w], label, eval_months,
                     exposures[regime], closes, micro_kw, spy, hist_by_w[w],
                     candidates, ledgers)

    # --- grid: ewic (W x H x R), equal (W x R, H deduped) -------------------
    for w, h, r in itertools.product(grid["train_months"], grid["ic_months"],
                                     grid["regime"]):
        log(f"candidate ewic-w{w}-h{h}-r{r}")
        score_combo(f"ewic-w{w}-h{h}-r{r}", w, "ewic", h, r)
    for w, r in itertools.product(grid["train_months"], grid["regime"]):
        log(f"candidate equal-w{w}-r{r}")
        score_combo(f"equal-w{w}-r{r}", w, "equal", grid["ic_months"][0], r)

    # --- comparators: single signals (regime none) ---------------------------
    for w in windows:
        for sig in ("gbdt", "ev"):
            log(f"candidate single:{sig}@w{w}")
            score_combo(f"single:{sig}@w{w}", w, f"single:{sig}",
                        grid["ic_months"][0], "none")
    for sig in ad.STATIC_SIGNALS:
        log(f"candidate single:{sig}")
        score_combo(f"single:{sig}", windows[0], f"single:{sig}",
                    grid["ic_months"][0], "none")

    # --- frozen GBDT comparator (train once on 36m pre-eval, no retrain) -----
    a158 = [c for c in meta.get("a158_cols", []) if c in store.columns]
    X158 = store[a158].swaplevel(0, 1).sort_index()
    fz = ad.frozen_signal_series(X158, label, e0, eval_days, 36, args.threads)
    fz_sigs = {**sig_by_w[windows[0]], ad.FROZEN_SIGNAL: fz}
    fz_ranked = pd.DataFrame({s: ad.cs_rank(v) for s, v in fz_sigs.items()})
    fz_hist = ad.ic_history(fz_sigs, label)
    log(f"candidate single:{ad.FROZEN_SIGNAL}@w36")
    score, ledger = ad.combined_score(fz_ranked, fz_hist, eval_months,
                                    f"single:{ad.FROZEN_SIGNAL}",
                                    grid["ic_months"][0], 1,
                                    list(fz_sigs))
    backtest_one(f"single:{ad.FROZEN_SIGNAL}@w36", score, ledger, fz_sigs,
                 label, eval_months, None, closes, micro_kw, spy, fz_hist,
                 candidates, ledgers)

    deployable = {k: v for k, v in candidates.items()
                  if k.startswith(("ewic", "equal"))}
    best = max(deployable.items(),
               key=lambda kv: kv[1]["information_ratio"])
    return {"market": "us", "git_sha": git_sha(), "mode": "select",
            "eval_window": spec["eval_window"],
            "n_candidates": len(candidates), "candidates": candidates,
            "best_deployable": best[0], "weight_ledgers": ledgers,
            "finished": dt.datetime.now().isoformat()}


# ---------------------------------------------------------------------------
# holdout run (chosen process, scored once)


def run_holdout(spec: dict, args) -> dict:
    cfg = json.loads(args.config) if args.config else spec.get("chosen")
    if not cfg:
        sys.exit("holdout needs --config '{...}' or spec 'chosen:' set after "
                 "the select run")
    store, meta, news_daily = load_inputs(spec)
    w, h = int(cfg["train_months"]), int(cfg["ic_months"])
    weighting, regime = cfg["weighting"], cfg["regime"]
    h0, h1 = (pd.Timestamp(spec["holdout"][0]),
              pd.Timestamp(spec["holdout"][1]))
    hist_start = h0 - pd.DateOffset(months=h + 1)

    days = store.index.get_level_values("datetime").unique().sort_values()
    pred_days = days[(days >= hist_start) & (days <= h1)]
    pred_months = ad.month_blocks(pd.DatetimeIndex(pred_days))
    eval_months = ad.month_blocks(pd.DatetimeIndex(
        days[(days >= h0) & (days <= h1)]))
    log(f"holdout {h0.date()}..{h1.date()}: {len(eval_months)} months, "
        f"preds from {pred_months[0][0].date()}")

    sig_by_w, label = assemble_signals(store, meta, news_daily, pred_months,
                                       [w], args.threads)
    signals = sig_by_w[w]
    ranked = pd.DataFrame({s: ad.cs_rank(v) for s, v in signals.items()})
    hist = ad.ic_history(signals, label)

    provider = REPO / spec["provider_uri"]
    closes = mc.load_closes(provider, spec["instruments"])
    closes = closes.loc[h0:h1]
    spy = spy_close(spec)
    exp = eval_exposure(spec, regime)
    hl = h * 21 * spec.get("halflife_frac", 0.5)
    score, ledger = ad.combined_score(ranked, hist, eval_months, weighting,
                                    h, hl, list(signals))
    met = mc.micro_backtest(score, closes, exposure=exp, **spec["micro"])
    sig_m = ad.signal_metrics(score, label)
    res = {**sig_m, **{k: v for k, v in met.items() if k != "daily_excess"}}
    res.update(xs_metrics(met["daily_excess"], spy))

    # honest trial accounting: every prior US trial + this phase's selection
    # candidates + this single holdout scoring.
    sel_json = REPO / "results" / "phase4_us_select.json"
    n_sel = (json.loads(sel_json.read_text())["n_candidates"]
             if sel_json.exists() else args.n_select)
    n_trials = int(spec.get("prior_trials_us", 0)) + n_sel + 1
    dsr = wf.deflated_sharpe(met["daily_excess"], n_trials,
                             [c["information_ratio"] for c in
                              json.loads(sel_json.read_text())["candidates"]
                              .values()] if sel_json.exists() else [])
    res["dsr"] = dsr["DSR"]
    res["dsr_details"] = dsr
    res["n_trials"] = n_trials
    res["weight_ledger"] = ledger
    wf.append_holdout_log({"ts": dt.datetime.now().isoformat(), "market": "us",
                           "git_sha": git_sha(),
                           "spec": spec["_spec_name"] + ".yaml",
                           "candidate": f"phase4:{json.dumps(cfg)}",
                           **{k: v for k, v in res.items()
                              if k not in ("weight_ledger",)}})
    log(f"HOLDOUT phase4 {cfg}: IR {res['information_ratio']:.4f} "
        f"IRxs {res['ir_xs_spy']:.4f} (SPY {res['spy_ir']:.4f}) "
        f"DSR {dsr['DSR']:.3f} n_trials={n_trials}")
    return {"market": "us", "git_sha": git_sha(), "mode": "holdout",
            "config": cfg, "holdout": spec["holdout"], "metrics": res,
            "finished": dt.datetime.now().isoformat()}


# ---------------------------------------------------------------------------
# preds — one signal x one window's monthly retrains (matrix-job unit)


def run_preds(spec: dict, args) -> dict:
    """Compute+checkpoint monthly preds for --signals a,b --window W.

    Writes results/oof/us_adaptive/{sig}__w{W}__YYYY-MM.pkl — the select job
    (or a rerun) resumes from these. Only the columns each model needs are
    loaded from the store."""
    meta_path = (REPO / spec["store"]).with_suffix(".manifest.json")
    meta = json.loads(meta_path.read_text())
    w = int(args.window)
    e0, e1 = (pd.Timestamp(spec["eval_window"][0]),
              pd.Timestamp(spec["eval_window"][1]))
    max_ic = int(args.hist_months)
    hist_start = e0 - pd.DateOffset(months=max_ic + 1)

    wanted = [s.strip() for s in args.signals.split(",")]
    colmap = {"gbdt": meta["a158_cols"], "ev": meta["event_cols"]}
    cols = sorted({c for s in wanted for c in colmap[s]} | {"LABEL0"})
    store = pd.read_parquet(REPO / spec["store"], columns=cols)
    days = store.index.get_level_values("datetime").unique().sort_values()
    pred_days = pd.DatetimeIndex(days[(days >= hist_start) & (days <= e1)])
    pred_months = ad.month_blocks(pred_days)
    log(f"preds {wanted} w={w}: {len(pred_months)} months "
        f"{pred_months[0][0].date()}..{pred_months[-1][1].date()}")

    cache = wf.oof_dir(RUN_KEY)
    y = store["LABEL0"].swaplevel(0, 1).sort_index()
    for sig in wanted:
        X = store[colmap[sig]].swaplevel(0, 1).sort_index()
        ad.model_signal_series(X, y, sig, pred_months, w, cache,
                               args.threads, log)
    return {"mode": "preds", "signals": wanted, "window": w,
            "months": len(pred_months)}


# ---------------------------------------------------------------------------
# smoke — three months, one window, plumbing check (runner-side)


def run_smoke(spec: dict, args) -> dict:
    spec = dict(spec)
    spec["eval_window"] = ["2024-10-01", "2024-12-31"]
    spec["grid"] = {"train_months": [12], "ic_months": [3],
                    "weighting": ["ewic"], "regime": ["none", "half"]}
    out = run_select(spec, args)
    print(json.dumps({"n_candidates": out["n_candidates"],
                      "best": out["best_deployable"]}, indent=1))
    for k, v in out["candidates"].items():
        print(f"  {k:30s} ir={v['information_ratio']:+.4f} "
              f"ar={v['annualized_return']:+.4f} ric={v['RankIC']}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["select", "holdout", "smoke", "preds"])
    p.add_argument("--config", default=None, help="chosen config JSON")
    p.add_argument("--signals", default="gbdt",
                   help="preds mode: comma list (gbdt,ev)")
    p.add_argument("--window", type=int, default=24,
                   help="preds mode: train window months")
    p.add_argument("--hist-months", type=int, default=13,
                   help="preds mode: history margin before eval start")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--n-select", type=int, default=0,
                   help="selection candidate count for DSR n_trials")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    spec = load_spec()
    summary = {"select": run_select, "holdout": run_holdout,
               "smoke": run_smoke, "preds": run_preds}[args.mode](spec, args)
    if args.mode in ("smoke", "preds"):     # preds output = the pkl cache
        return
    out = args.out or (REPO / "results" /
                       ("phase4_us_select.json" if args.mode == "select"
                        else "phase4_us.json"))
    out.write_text(json.dumps(summary, indent=1, default=str))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
