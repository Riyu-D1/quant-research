"""Run one qlib benchmark workflow config and dump all metrics to JSON.

Mirrors qlib/cli/run.py:workflow() but keeps the recorder so we can pull
metrics + artifacts. Relative paths inside model kwargs are resolved against
the repo root so vendored configs can point at configs/benchmarks/assets/.
"""
import argparse
import json
import time
from pathlib import Path

import qlib
from qlib.config import C
from qlib.model.trainer import task_train
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[1]
# model kwargs whose values are local files to resolve to absolute paths
FILE_KWARGS = ("model_path", "stock2concept", "stock_index")


def load_config(config_path: Path) -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open(config_path) as f:
        config = yaml.load(f)
    return config


def resolve_paths(config: dict, provider_uri: Path | None) -> dict:
    if provider_uri is not None:
        config["qlib_init"]["provider_uri"] = str(provider_uri.resolve())
    model_kwargs = config.get("task", {}).get("model", {}).get("kwargs", {})
    for key in FILE_KWARGS:
        if key in model_kwargs and model_kwargs[key] is not None:
            p = Path(model_kwargs[key])
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            model_kwargs[key] = str(p)
    return config


def extract_results(recorder, report_path: str) -> dict:
    metrics = dict(recorder.list_metrics())
    report = recorder.load_object(report_path)
    turnover = float(report["turnover"].mean()) if "turnover" in report else None
    n_days = len(report)
    return {"metrics": metrics, "turnover": turnover, "backtest_days": n_days}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--provider-uri", type=Path, default=None,
                   help="override qlib_init.provider_uri (default: config value)")
    p.add_argument("--experiment", default="benchmarks")
    args = p.parse_args()

    config = resolve_paths(load_config(args.config), args.provider_uri)

    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str((REPO_ROOT / "mlruns").resolve())
    qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    t0 = time.time()
    recorder = task_train(config.get("task"), experiment_name=args.experiment)
    elapsed_min = (time.time() - t0) / 60

    # report file name depends on analysis_freq; standard benchmark uses 1day
    report_name = "portfolio_analysis/report_normal_1day.pkl"
    try:
        results = extract_results(recorder, report_name)
    except Exception:  # noqa: BLE001
        # fall back: list artifacts for debugging, metrics still saved
        results = {"metrics": dict(recorder.list_metrics()), "turnover": None,
                   "backtest_days": None,
                   "artifacts": recorder.list_artifacts()}

    results.update({
        "config": str(args.config),
        "experiment": args.experiment,
        "recorder_id": recorder.info["id"],
        "provider_uri": config["qlib_init"]["provider_uri"],
        "train_minutes": round(elapsed_min, 2),
        "model_class": config["task"]["model"]["class"],
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n=== saved {args.out} ({elapsed_min:.1f} min) ===")
    print(json.dumps(results["metrics"], indent=2, default=str))


if __name__ == "__main__":
    main()
