"""Optuna tuning for LightGBM over Phase-2 walk-forward folds.

Tunes on OOF folds only — the holdout is never touched. The best params are
written to results/oof/<market>/tune_lgbm_study.json; phase2_run.py picks them
up as the `lgb_tuned` model on subsequent runs. Trial count is recorded there
for the deflated-Sharpe accounting.

    python scripts/tune_lgbm.py --market cn --trials 25 [--tune-folds 2]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quant_research import walkforward as wf
from scripts.phase2_run import tune_lgbm

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True, choices=["cn", "us"])
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--tune-folds", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    spec = wf.load_spec(args.market)
    wf.qlib_init(str(REPO / spec["provider_uri"]), spec["region"])
    folds, _ = wf.make_folds(spec)
    out = tune_lgbm(spec, folds, args.trials, args.tune_folds,
                    args.threads, args.market)
    print(out)


if __name__ == "__main__":
    main()
