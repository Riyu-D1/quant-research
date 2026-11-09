"""Stacked ensemble: ridge meta-model over component pred.pkl signals.

Reads component recorder ids from results/runs/*.json, loads their saved
pred/label pickles from mlruns, fits the meta-learner on the validation
window only, then evaluates on the untouched test window and backtests.

    python scripts/stack_ensemble.py --config configs/experiments/stacked_ensemble_cn.yaml
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.backtest import backtest as normal_backtest
from qlib.config import C

REPO = Path(__file__).resolve().parents[1]


def find_recorder_dir(experiment: str, recorder_id: str) -> Path:
    for meta in (REPO / "mlruns").glob("*/meta.yaml"):
        d = yaml.safe_load(meta.read_text())
        if d.get("name") == experiment:
            rd = meta.parent / recorder_id
            if rd.exists():
                return rd
    raise FileNotFoundError(f"{experiment}/{recorder_id}")


def load_pred_label(run_json: Path):
    d = json.loads(run_json.read_text())
    rd = find_recorder_dir(d["experiment"], d["recorder_id"])
    pred = pd.read_pickle(rd / "artifacts/pred.pkl")
    label = pd.read_pickle(rd / "artifacts/label.pkl")
    return pred.iloc[:, 0], label.iloc[:, 0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--provider-uri", type=Path, default=REPO / "data/qlib/cn_data")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(REPO / "mlruns")
    qlib.init(provider_uri=str(args.provider_uri), region="cn",
              exp_manager=exp_manager)

    preds, label = {}, None
    for name, rj in cfg["components"].items():
        s, lab = load_pred_label(REPO / rj)
        preds[name] = s
        label = lab if label is None else lab
    P = pd.DataFrame(preds)
    y = label

    v0, v1 = [pd.Timestamp(x) for x in cfg["valid_window"]]
    t0, t1 = [pd.Timestamp(x) for x in cfg["test_window"]]
    dt_idx = P.index.get_level_values("datetime")
    vm = (dt_idx >= v0) & (dt_idx <= v1)
    tm = (dt_idx >= t0) & (dt_idx <= t1)

    Xv = P[vm].dropna()
    yv = y.reindex(Xv.index)
    keep = yv.notna()
    from sklearn.linear_model import Ridge
    meta = Ridge(alpha=cfg["meta_learner"]["alpha"])
    meta.fit(Xv[keep], yv[keep])

    Xt = P[tm]
    stacked = pd.Series(meta.predict(Xt.fillna(0)), index=Xt.index, name="score")
    Path(cfg["out_pred"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["out_pred"], "wb") as f:
        pickle.dump(stacked.to_frame(), f)

    ic = pd.concat({"p": stacked, "y": y.reindex(Xt.index)}, axis=1).dropna()
    daily_ic = ic.groupby(level="datetime").apply(lambda g: g["p"].corr(g["y"]))
    rank_ic = ic.groupby(level="datetime").apply(
        lambda g: g["p"].corr(g["y"], method="spearman"))

    strategy_cfg = dict(cfg["strategy"])
    strategy_cfg.setdefault("kwargs", {})["signal"] = stacked.to_frame()
    portfolio_metric, _ = normal_backtest(
        executor={
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        strategy=strategy_cfg,
        **cfg["backtest"],
    )
    # portfolio_metric: {"1day": (report_df, positions_obj)}
    df = portfolio_metric["1day"][0]
    excess = df["return"] - df["bench"] - df["cost"]
    ann = excess.mean() * 252
    ir = excess.mean() / excess.std() * np.sqrt(252)
    cummax = (1 + excess).cumprod().cummax()
    mdd = (((1 + excess).cumprod() / cummax) - 1).min()
    out = {
        "metrics": {"IC": daily_ic.mean(), "ICIR": daily_ic.mean() / daily_ic.std(),
                    "Rank IC": rank_ic.mean(), "Rank ICIR": rank_ic.mean() / rank_ic.std(),
                    "1day.excess_return_with_cost.annualized_return": ann,
                    "1day.excess_return_with_cost.information_ratio": ir,
                    "1day.excess_return_with_cost.max_drawdown": mdd},
        "turnover": float(df["turnover"].mean()) if "turnover" in df else None,
        "config": str(args.config),
        "components": cfg["components"],
    }
    Path(cfg["out"]).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()
