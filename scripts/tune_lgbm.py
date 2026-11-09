"""Optuna hyperparameter tuning for LightGBM on the Alpha158/CSI300 task.

Tunes on the VALIDATION window only (2015-2016). The test window is never
touched by the study; after tuning, the best params are written to a runnable
benchmark config (default configs/experiments/lightgbm_tuned_cn.yaml) that can
be run through scripts/run_benchmark.py for an honest test/backtest eval.

    python scripts/tune_lgbm.py --trials 40
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.config import C
from qlib.data.dataset import DatasetH
from qlib.utils import init_instance_by_config

REPO = Path(__file__).resolve().parents[1]
BASE_CFG = REPO / "configs/benchmarks/lightgbm_alpha158_cn.yaml"


def ic_by_day(pred: pd.Series, label: pd.Series) -> float:
    df = pd.concat({"p": pred, "y": label}, axis=1).dropna()
    return df.groupby(level="datetime").apply(
        lambda g: g["p"].corr(g["y"])).mean()


def build_dataset(cfg: dict) -> DatasetH:
    return init_instance_by_config(cfg["task"]["dataset"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--provider-uri", type=Path, default=REPO / "data/qlib/cn_data")
    p.add_argument("--write-config", type=Path,
                   default=REPO / "configs/experiments/lightgbm_tuned_cn.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(BASE_CFG.read_text())
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(REPO / "mlruns")
    qlib.init(provider_uri=str(args.provider_uri), region="cn",
              exp_manager=exp_manager)

    dataset: DatasetH = build_dataset(cfg)
    dtrain = dataset.prepare("train", col_set=["feature", "label"],
                             data_key=qlib.data.dataset.handler.DataHandlerLP.DK_L)
    dvalid = dataset.prepare("valid", col_set=["feature", "label"],
                             data_key=qlib.data.dataset.handler.DataHandlerLP.DK_I)
    Xv, yv = dvalid["feature"], dvalid["label"].iloc[:, 0]

    import lightgbm as lgb
    import optuna

    def objective(trial):
        params = {
            "objective": "mse", "verbosity": -1, "num_threads": 20,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 512),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-2, 1000, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 1000, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
        }
        m = lgb.train(params, lgb.Dataset(dtrain["feature"], dtrain["label"].iloc[:, 0]),
                      num_boost_round=1000,
                      valid_sets=[lgb.Dataset(Xv, yv)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = pd.Series(m.predict(Xv), index=Xv.index)
        return ic_by_day(pred, yv)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=args.trials)
    print("best valid IC:", study.best_value, "params:", study.best_params)

    out = cfg.copy()
    mk = out["task"]["model"]["kwargs"]
    mk.update(study.best_params)
    args.write_config.write_text(yaml.safe_dump(out, sort_keys=False))
    (REPO / "results/runs/tune_lgbm_study.json").write_text(json.dumps({
        "best_valid_ic": study.best_value, "best_params": study.best_params,
        "n_trials": len(study.trials)}, indent=2))
    print(f"wrote {args.write_config}")


if __name__ == "__main__":
    main()
