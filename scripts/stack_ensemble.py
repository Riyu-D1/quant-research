"""Stacked ensemble over out-of-fold predictions (Phase 2).

Discovers every model's OOF preds in results/oof/<market>/ (GPU preds drop in
automatically once present), then:
  * rank-average of per-day cross-sectional percentile ranks
  * ridge meta-model fit on OOF preds vs OOF labels only
and backtests each stacked signal on the OOF window with both portfolio
strategies. Writes results/oof/<market>/stack_metrics.json. No retraining;
run after phase2_run.py (or after GPU preds have been merged in).

    python scripts/stack_ensemble.py --market cn
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quant_research import walkforward as wf

REPO = Path(__file__).resolve().parents[1]


def discover_models(market: str) -> dict[str, list[str]]:
    """{model: [fold names with saved preds]} from the oof dir."""
    d = wf.oof_dir(market)
    out: dict[str, list[str]] = {}
    for p in sorted(d.glob("*__fold*.pkl")):
        model, fold = p.stem.split("__", 1)
        out.setdefault(model, []).append(fold)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True, choices=["cn", "us"])
    p.add_argument("--strategy", default=None,
                   help="limit backtests to one strategy (default: all)")
    args = p.parse_args()

    spec = wf.load_spec(args.market)
    wf.qlib_init(str(REPO / spec["provider_uri"]), spec["region"])
    folds, _ = wf.make_folds(spec)

    have = discover_models(args.market)
    wf.log(f"components found: { {k: sorted(v) for k, v in have.items()} }")
    preds, lab_parts = {}, []
    for m in have:
        parts = []
        for f in folds:
            got = wf.load_fold_pred(args.market, m, f.name)
            if got is not None:
                parts.append(got[0])
        if parts:
            preds[m] = pd.concat(parts).sort_index()
    if len(preds) < 2:
        sys.exit("need >=2 component models with OOF preds")
    # OOF label (identical across components)
    ref = next(iter(preds))
    for f in folds:
        got = wf.load_fold_pred(args.market, ref, f.name)
        if got is not None:
            lab_parts.append(got[1])
    label = pd.concat(lab_parts).sort_index()

    sigs = {"rank_avg": wf.rank_average(preds)}
    meta = wf.fit_ridge(preds, label, spec["ensemble"]["ridge_alpha"])
    sigs["ridge_stack"] = wf.apply_ridge(meta, preds)

    idx = label.index.get_level_values("datetime")
    window = (str(idx.min().date()), str(idx.max().date()))
    strategies = spec["strategies"]
    if args.strategy:
        strategies = {args.strategy: strategies[args.strategy]}

    out = {"market": args.market, "components": sorted(preds),
           "ridge": {"alpha": spec["ensemble"]["ridge_alpha"],
                     "coef": dict(zip(preds, meta.coef_.tolist())),
                     "intercept": float(meta.intercept_)},
           "oof_window": window, "signals": {}}
    for sname, sig in sigs.items():
        out["signals"][sname] = {"signal": wf.signal_metrics(sig, label)}
        for k, scfg in strategies.items():
            try:
                r = wf.backtest_signal(sig, spec, scfg, window)
                out["signals"][sname][k] = {
                    kk: vv for kk, vv in r.items() if kk != "daily_excess"}
            except Exception as e:  # noqa: BLE001
                out["signals"][sname][k] = {"error": str(e)}
    (wf.oof_dir(args.market) / "stack_metrics.json").write_text(
        json.dumps(out, indent=2, default=str))
    with open(wf.oof_dir(args.market) / "ridge_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
