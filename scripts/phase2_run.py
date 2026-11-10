"""Phase-2 walk-forward pipeline driver.

    # validation-side run (OOF folds; never touches the holdout):
    python scripts/phase2_run.py --market cn --models lgb,xgb,de,lgb360

    # optional Optuna tuning of LightGBM over the last --tune-folds folds:
    python scripts/phase2_run.py --market cn --tune-trials 25

    # holdout scoring — runs ONCE for the named candidate:
    python scripts/phase2_run.py --market cn --holdout ridge_stack@topk

Checkpointed: per-fold predictions land in results/oof/<market>/*.pkl so a
rerun resumes where a timed-out job left off. GPU models (transformer, hist)
run through the same path on a GPU box; their pkls merge into the ensemble
when present.
"""
import argparse
import datetime as dt
import itertools
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quant_research import walkforward as wf

REPO = Path(__file__).resolve().parents[1]
TUNE_JSON = "tune_lgbm_study.json"


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, cwd=REPO,
                          capture_output=True, text=True).stdout.strip()


def models_for_spec(spec: dict, requested: list[str] | None,
                    market: str) -> dict:
    reg = dict(spec["models"])
    tuned_json = wf.oof_dir(market) / TUNE_JSON
    if tuned_json.exists():
        tuned = dict(reg["lgb"])
        tuned["kwargs"] = {
            **tuned["kwargs"],
            **json.loads(tuned_json.read_text())["best_params"]}
        reg["lgb_tuned"] = tuned
    if requested is None:
        return {k: v for k, v in reg.items() if not v.get("requires_gpu")}
    out = {}
    for m in requested:
        if m not in reg:
            sys.exit(f"unknown model {m}; registry: {sorted(reg)}")
        out[m] = reg[m]
    return out


def ensure_concept_assets(spec: dict) -> None:
    assets = REPO / "configs/benchmarks/assets"
    idx = assets / f"qlib_{spec['instruments']}_stock_index.npy"
    c2 = assets / f"qlib_{spec['instruments']}_stock2concept.npy"
    if idx.exists() and c2.exists():
        return
    wf.log("building concept assets for HIST")
    subprocess.run([sys.executable, "scripts/build_us_concepts.py",
                    "--instruments",
                    str(REPO / spec["provider_uri"] / "instruments"
                        / f"{spec['instruments']}.txt"),
                    "--out-dir", str(assets)], cwd=REPO, check=True)


def run_folds(spec: dict, models: dict, folds: list[wf.Fold],
              holdout_fold: wf.Fold, threads: int, market: str,
              include_holdout: bool) -> dict:
    """Train each model on each fold; return {model: (OOF pred, OOF label)}."""
    all_folds = folds + ([holdout_fold] if include_holdout else [])
    needed_fs = sorted({cfg["feature_set"] for cfg in models.values()})
    for fold in all_folds:
        for fs in needed_fs:
            todo = [m for m, c in models.items() if c["feature_set"] == fs
                    and wf.load_fold_pred(market, m, fold.name) is None]
            if not todo:
                continue
            t0 = time.time()
            handler = wf.build_handler(spec, fs, fold)
            wf.log(f"handler {fs} {fold.name} ready ({time.time()-t0:.0f}s)")
            label = handler.fetch(
                slice(pd.Timestamp(fold.eval[0]), pd.Timestamp(fold.eval[1])),
                col_set="label", data_key="infer").iloc[:, 0]
            for m in todo:
                cfg = models[m]
                if cfg.get("needs_concept_assets"):
                    ensure_concept_assets(spec)
                ts_step = cfg.get("step_len")
                ds = wf.FoldDataset(handler, {"train": fold.train,
                                              "valid": fold.valid,
                                              "test": fold.eval},
                                    ts_step=ts_step)
                t0 = time.time()
                pred = wf.train_predict(cfg, ds, threads)
                wf.save_fold_pred(market, m, fold.name, pred, label)
                wf.log(f"  {m} {fold.name} done ({time.time()-t0:.0f}s)")

    oof = {}
    for m in models:
        parts, labels = [], []
        for fold in folds:
            got = wf.load_fold_pred(market, m, fold.name)
            if got is None:
                wf.log(f"WARN missing {m} {fold.name} — skipped in OOF")
                continue
            parts.append(got[0])
            labels.append(got[1])
        if parts:
            oof[m] = (pd.concat(parts).sort_index(), pd.concat(labels).sort_index())
    return oof


def tune_lgbm(spec: dict, folds: list[wf.Fold], trials: int, tune_folds: int,
              threads: int, market: str) -> dict:
    """Optuna over the LAST `tune_folds` OOF folds; returns study summary."""
    import lightgbm as lgb
    import optuna

    use = folds[-tune_folds:]
    prepared = []
    for fold in use:
        handler = wf.build_handler(spec, "alpha158", fold)
        ds = wf.FoldDataset(handler, {"train": fold.train, "valid": fold.valid,
                                      "test": fold.eval})
        dtrain, dvalid = ds.prepare(["train", "valid"],
                                    col_set=["feature", "label"], data_key="learn")
        deval = ds.prepare("test", col_set=["feature", "label"], data_key="infer")
        prepared.append((dtrain, dvalid, deval))
        wf.log(f"tune fold {fold.name} prepared")

    def objective(trial):
        params = {
            "objective": "mse", "verbosity": -1, "num_threads": threads,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 512),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-2, 1000, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 1000, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
        }
        ics = []
        for dtrain, dvalid, deval in prepared:
            m = lgb.train(
                params, lgb.Dataset(dtrain["feature"], dtrain["label"].iloc[:, 0]),
                num_boost_round=1000,
                valid_sets=[lgb.Dataset(dvalid["feature"], dvalid["label"].iloc[:, 0])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
            pred = pd.Series(m.predict(deval["feature"]), index=deval.index)
            ics.append(wf.signal_metrics(pred, deval["label"].iloc[:, 0])["IC"] or 0)
        return float(np.mean(ics))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=trials)
    out = {"best_mean_ic": study.best_value, "best_params": study.best_params,
           "n_trials": len(study.trials), "folds": [f.name for f in use]}
    (wf.oof_dir(market) / TUNE_JSON).write_text(json.dumps(out, indent=2))
    wf.log(f"tuning done: IC {study.best_value:.4f} over {len(study.trials)} trials")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True, choices=["cn", "us"])
    p.add_argument("--models", default=None,
                   help="comma list; default = all non-GPU models in spec")
    p.add_argument("--tune-trials", type=int, default=0)
    p.add_argument("--tune-folds", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--instruments", default=None,
                   help="override spec instruments (debug/smoke runs only)")
    p.add_argument("--holdout", default=None,
                   help='score "signal@strategy" once on the holdout')
    p.add_argument("--force", action="store_true",
                   help="retrain even if fold pkls exist")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    spec = wf.load_spec(args.market)
    if args.instruments:
        spec["instruments"] = args.instruments
    wf.qlib_init(str(REPO / spec["provider_uri"]), spec["region"])
    folds, holdout_fold = wf.make_folds(spec)
    requested = args.models.split(",") if args.models else None
    models = models_for_spec(spec, requested, args.market)
    if args.force:
        for f in wf.oof_dir(args.market).glob("*__*.pkl"):
            f.unlink()

    summary = {"market": args.market, "git_sha": git_sha(),
               "folds": {f.name: vars(f) for f in folds},
               "holdout_fold": vars(holdout_fold),
               "threads": args.threads, "started": dt.datetime.now().isoformat()}

    # ---- optional tuning --------------------------------------------------
    n_trials = 0
    if args.tune_trials > 0:
        study = tune_lgbm(spec, folds, args.tune_trials, args.tune_folds,
                          args.threads, args.market)
        n_trials += study["n_trials"]
        summary["tuning"] = study

    # ---- OOF training ------------------------------------------------------
    oof = run_folds(spec, models, folds, holdout_fold, args.threads,
                    args.market, include_holdout=args.holdout is not None)

    sig_metrics = {}
    oof_preds = {}
    oof_label = None
    for m, (pred, lab) in oof.items():
        oof_preds[m] = pred
        oof_label = lab
        sig_metrics[m] = wf.signal_metrics(pred, lab)
    summary["oof_signal_metrics"] = sig_metrics

    # ---- ensembles ---------------------------------------------------------
    ens_preds = {}
    if len(oof_preds) >= 2:
        ens_preds["rank_avg"] = wf.rank_average(oof_preds)
        meta = wf.fit_ridge(oof_preds, oof_label, spec["ensemble"]["ridge_alpha"])
        ens_preds["ridge_stack"] = wf.apply_ridge(meta, oof_preds)
        summary["ridge"] = {
            "alpha": spec["ensemble"]["ridge_alpha"],
            "coef": dict(zip(oof_preds, meta.coef_.tolist())),
            "intercept": float(meta.intercept_)}
        with open(wf.oof_dir(args.market) / "ridge_meta.pkl", "wb") as f:
            pickle.dump(meta, f)
    for name, pred in ens_preds.items():
        sig_metrics[name] = wf.signal_metrics(pred, oof_label)
    summary["oof_signal_metrics"] = sig_metrics

    # ---- candidate table: signal x strategy on OOF -------------------------
    strat = spec["strategies"]
    idx = next(iter(oof.values()))[1].index.get_level_values("datetime")
    oof_window = (str(idx.min().date()), str(idx.max().date()))
    candidates = {}
    for sname, skey in itertools.product(list(sig_metrics), list(strat)):
        pred = ens_preds[sname] if sname in ens_preds else oof_preds[sname]
        try:
            r = wf.backtest_signal(pred, spec, strat[skey], oof_window)
        except Exception as e:  # noqa: BLE001  solver/backtest failure — log, keep going
            wf.log(f"backtest {sname}@{skey} FAILED: {e}")
            r = {"annualized_return": None, "information_ratio": None,
                 "max_drawdown": None, "turnover": None, "n_days": 0}
        candidates[f"{sname}@{skey}"] = {
            **r, **sig_metrics[sname]}
        n_trials += 1
    summary["candidates"] = candidates
    summary["oof_window"] = oof_window
    valid = [c for c in candidates.items() if c[1]["information_ratio"] is not None]
    summary["best_oof_candidate"] = \
        max(valid, key=lambda kv: kv[1]["information_ratio"])[0] if valid else None
    trial_irs = [c["information_ratio"] for _, c in valid]

    # ---- holdout scoring (once per candidate) ------------------------------
    if args.holdout:
        sname, skey = args.holdout.split("@")
        hp, hlab = {}, None
        for m in models:
            got = wf.load_fold_pred(args.market, m, "holdout")
            if got is not None:
                hp[m], hlab = got[0], got[1]
        if sname in ("rank_avg", "ridge_stack"):
            missing = set(oof_preds) - set(hp)
            if missing:
                sys.exit(f"holdout preds missing for {missing} (needed by {sname})")
            if sname == "rank_avg":
                hsig = wf.rank_average(hp)
            else:
                with open(wf.oof_dir(args.market) / "ridge_meta.pkl", "rb") as f:
                    meta = pickle.load(f)
                hsig = wf.apply_ridge(meta, hp)
        else:
            if sname not in hp:
                sys.exit(f"holdout pred missing for {sname}")
            hsig = hp[sname]
        hmetrics = wf.signal_metrics(hsig, hlab)
        n_trials += 1
        r = wf.backtest_signal(hsig, spec, strat[skey], tuple(spec["holdout"]))
        n_trials += 1
        dsr = wf.deflated_sharpe(r["daily_excess"], n_trials, trial_irs)
        holdout_res = {"candidate": args.holdout, **hmetrics,
                       **{k: v for k, v in r.items() if k != "daily_excess"},
                       "dsr": dsr}
        summary["holdout"] = holdout_res
        summary["n_trials"] = n_trials
        wf.append_holdout_log({"ts": dt.datetime.now().isoformat(),
                               "market": args.market, "git_sha": git_sha(),
                               "spec": f"walkforward_{args.market}.yaml",
                               **holdout_res})
        wf.log(f"HOLDOUT {args.holdout}: IR {holdout_res['information_ratio']:.4f} "
               f"DSR {dsr['DSR']:.3f} (n_trials={n_trials})")
    else:
        summary["n_trials"] = n_trials

    out = args.out or REPO / "results" / f"phase2_{args.market}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    wf.log(f"wrote {out}")


if __name__ == "__main__":
    main()
