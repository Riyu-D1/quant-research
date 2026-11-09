"""End-to-end cloud-GPU benchmark driver.

    python scripts/gpu_run.py --models transformer_cn hist_cn
    make gpu-run MODELS="transformer_cn hist_cn"

Steps per invocation:
  1. Download qlib CN/US community data if missing (free, no key).
  2. Run each requested model via scripts/run_benchmark.py (sequential).
  3. Rebuild results/LEADERBOARD.md.
  4. Append one row per run to EXPERIMENTS.md.

Model names come from the REGISTRY below; `--list` prints them.
"""
import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV_PY = REPO / ".venv" / "bin" / "python"
PY = str(VENV_PY if VENV_PY.exists() else Path(sys.executable))

CN = REPO / "data/qlib/cn_data"
US = REPO / "data/qlib/us_data"

# name -> (config, provider_dir, experiment, notes)
REGISTRY = {
    # Phase 1 baselines
    "lightgbm_cn":      ("configs/benchmarks/lightgbm_alpha158_cn.yaml", CN, "benchmarks", "Alpha158 csi300"),
    "xgboost_cn":       ("configs/benchmarks/xgboost_alpha158_cn.yaml", CN, "benchmarks", "Alpha158 csi300"),
    "doubleensemble_cn":("configs/benchmarks/doubleensemble_alpha158_cn.yaml", CN, "benchmarks", "Alpha158 csi300"),
    "transformer_cn":   ("configs/benchmarks/transformer_alpha158_cn.yaml", CN, "benchmarks", "Alpha158 csi300"),
    "hist_cn":          ("configs/benchmarks/hist_alpha360_cn.yaml", CN, "benchmarks", "Alpha360 csi300"),
    "lightgbm_us":      ("configs/benchmarks/lightgbm_alpha158_us.yaml", US, "benchmarks_us", "Alpha158 sp500"),
    "xgboost_us":       ("configs/benchmarks/xgboost_alpha158_us.yaml", US, "benchmarks_us", "Alpha158 sp500"),
    "doubleensemble_us":("configs/benchmarks/doubleensemble_alpha158_us.yaml", US, "benchmarks_us", "Alpha158 sp500"),
    "transformer_us":   ("configs/benchmarks/transformer_alpha158_us.yaml", US, "benchmarks_us", "Alpha158 sp500"),
    # Phase 2 experiments (do not run until baselines complete)
    "lgb_combined_cn":  ("configs/experiments/lightgbm_alpha158_360_cn.yaml", CN, "phase2", "Alpha158+Alpha360 concat"),
    "lgb_tuned_cn":     ("scripts/tune_lgbm.py", CN, "phase2", "Optuna-tuned LightGBM"),
    "stacked_cn":       ("scripts/stack_ensemble.py --config configs/experiments/stacked_ensemble_cn.yaml", CN, "phase2", "stacked ensemble"),
    "mvo_cn":           ("configs/experiments/mvo_turnover_cn.yaml", CN, "phase2", "turnover-penalised MVO"),
}


def ensure_data(provider: Path) -> None:
    if (provider / "calendars/day.txt").exists():
        return
    region = "us" if "us" in provider.name else "cn"
    print(f"[gpu-run] downloading qlib {region} data -> {provider}")
    provider.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, "scripts/download_data.py", "--target-dir", str(provider),
                    "--region", region], cwd=REPO, check=True)


def run_one(name: str) -> Path:
    entry = REGISTRY[name]
    config, provider, experiment, _ = entry
    ensure_data(provider)
    out = REPO / "results/runs" / f"{name}.json"
    cmd = config if config.startswith("scripts/") else \
        f"scripts/run_benchmark.py --config {config} --provider-uri {provider} " \
        f"--experiment {experiment} --out {out}"
    print(f"[gpu-run] === {name}: {cmd}")
    subprocess.run(f"{PY} {cmd}", shell=True, cwd=REPO, check=True)
    return out


def rebuild_table() -> None:
    subprocess.run([PY, "scripts/make_table.py", "--runs", "results/runs",
                    "--published", "results/published_leaderboard.csv",
                    "--out", "results/LEADERBOARD.md"], cwd=REPO, check=False)


def append_experiments(results: dict[str, Path]) -> None:
    exp_md = REPO / "EXPERIMENTS.md"
    text = exp_md.read_text() if exp_md.exists() else "# Experiments\n\n| ID | Date | Hyp # | Change | Data | IC | ICIR | Rank IC | AR | IR | MDD | TO | Verdict |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    ids = [int(m.group(1)) for m in re.finditer(r"EX-(\d+)", text)]
    n = max(ids, default=0)
    rows = []
    for name, out in results.items():
        if not out.exists():
            continue
        d = json.loads(out.read_text())
        m = d.get("metrics", {})
        def g(k, m=m): return f"{m[k]:.4f}" if isinstance(m.get(k), (int, float)) else "—"
        n += 1
        notes = REGISTRY[name][3]
        rows.append(
            f"| EX-{n:03d} | {dt.date.today()} | — | {name} (GPU run) | {notes} | "
            f"{g('IC')} | {g('ICIR')} | {g('Rank IC')} | "
            f"{g('1day.excess_return_with_cost.annualized_return')} | "
            f"{g('1day.excess_return_with_cost.information_ratio')} | "
            f"{g('1day.excess_return_with_cost.max_drawdown')} | "
            f"{(format(d['turnover'], '.4f') if isinstance(d.get('turnover'), float) else '—')} | "
            f"auto-logged by gpu_run |")
    if rows:
        exp_md.write_text(text.rstrip() + "\n" + "\n".join(rows) + "\n")
        print(f"[gpu-run] appended {len(rows)} row(s) to EXPERIMENTS.md")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=False, default=None,
                   help="registry names to run (default: --list to see them)")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list or not args.models:
        for k, v in REGISTRY.items():
            print(f"  {k:20s} {v[0]}")
        return
    models = [m for a in args.models for m in re.split(r"[\s,]+", a) if m]
    unknown = [m for m in models if m not in REGISTRY]
    if unknown:
        sys.exit(f"unknown model(s) {unknown}; run --list")

    results = {}
    failures = []
    for name in models:
        try:
            results[name] = run_one(name)
        except subprocess.CalledProcessError as e:
            failures.append((name, e.returncode))
            print(f"[gpu-run] FAILED {name} (exit {e.returncode}) — continuing")
    rebuild_table()
    append_experiments(results)
    if failures:
        sys.exit(f"failed: {failures}")
    print("[gpu-run] all done")


if __name__ == "__main__":
    main()
