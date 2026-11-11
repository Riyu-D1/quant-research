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
US = REPO / "data/qlib/us_alpaca"   # Alpaca-built dump (us-alpaca-data workflow artifact)

# name -> (config, provider_dir, experiment, notes)
REGISTRY = {
    # Phase 1 baselines
    "lightgbm_cn":      ("configs/benchmarks/lightgbm_alpha158_cn.yaml", CN, "benchmarks", "Alpha158 csi300"),
    "xgboost_cn":       ("configs/benchmarks/xgboost_alpha158_cn.yaml", CN, "benchmarks", "Alpha158 csi300"),
    "doubleensemble_cn":("configs/benchmarks/doubleensemble_alpha158_cn.yaml", CN, "benchmarks", "Alpha158 csi300"),
    "transformer_cn":   ("configs/benchmarks/transformer_alpha158_cn.yaml", CN, "benchmarks", "Alpha158 csi300"),
    "hist_cn":          ("configs/benchmarks/hist_alpha360_cn.yaml", CN, "benchmarks", "Alpha360 csi300"),
    "lightgbm_us":      ("configs/benchmarks/lightgbm_alpha158_us.yaml", US, "benchmarks_us", "Alpha158 sp500 PIT"),
    "xgboost_us":       ("configs/benchmarks/xgboost_alpha158_us.yaml", US, "benchmarks_us", "Alpha158 sp500 PIT"),
    "doubleensemble_us":("configs/benchmarks/doubleensemble_alpha158_us.yaml", US, "benchmarks_us", "Alpha158 sp500 PIT"),
    "transformer_us":   ("configs/benchmarks/transformer_alpha158_us.yaml", US, "benchmarks_us", "Alpha158 sp500 PIT"),
    "hist_us":          ("configs/benchmarks/hist_alpha360_us.yaml", US, "benchmarks_us", "Alpha360 sp500 PIT"),
    # Phase 2 — walk-forward pipeline (scripts/phase2_run.py). The main entry
    # runs OOF folds + tuning + ensemble + TopkDropout-vs-MVO comparison.
    # Holdout scoring is a separate, deliberate step: append
    # `--holdout <signal>@<strategy>` (e.g. via MODELS="phase2_cn" then edit).
    "phase2_cn":        ("scripts/phase2_run.py --market cn --tune-trials 25", CN, "phase2", "walk-forward pipeline cn"),
    "phase2_us":        ("scripts/phase2_run.py --market us --tune-trials 25", US, "phase2", "walk-forward pipeline us"),
    "lgb_tuned_cn":     ("scripts/tune_lgbm.py --market cn --trials 25", CN, "phase2", "Optuna-tuned LightGBM"),
    "lgb_tuned_us":     ("scripts/tune_lgbm.py --market us --trials 25", US, "phase2", "Optuna-tuned LightGBM"),
    "stacked_cn":       ("scripts/stack_ensemble.py --market cn", CN, "phase2", "stacked ensemble"),
    "stacked_us":       ("scripts/stack_ensemble.py --market us", US, "phase2", "stacked ensemble"),
    # GPU OOF preds for the stacker (run on Kaggle; merge results/oof pkls)
    "transformer_cn_wf":("scripts/phase2_run.py --market cn --models transformer", CN, "phase2", "Transformer OOF preds"),
    "hist_cn_wf":       ("scripts/phase2_run.py --market cn --models hist", CN, "phase2", "HIST OOF preds"),
    "transformer_us_wf":("scripts/phase2_run.py --market us --models transformer", US, "phase2", "Transformer OOF preds"),
    "hist_us_wf":       ("scripts/phase2_run.py --market us --models hist", US, "phase2", "HIST OOF preds"),
    # Holdout-fold preds only (no scoring) — needed when a stack that includes
    # a GPU model wins validation; the CI holdout run then scores once.
    "hist_cn_ho":       ("scripts/phase2_run.py --market cn --models hist --train-holdout", CN, "phase2", "HIST holdout-fold preds"),
    "transformer_cn_ho":("scripts/phase2_run.py --market cn --models transformer --train-holdout", CN, "phase2", "Transformer holdout-fold preds"),
    # Phase 4 — FinBERT news sentiment on GPU (downloads the phase4-news
    # artifact itself; output lands under results/news_sentiment/ which the
    # PUSH_RESULTS=1 commit picks up).
    "phase4_news_score":("scripts/phase4_score_news.py", CN, "phase4", "FinBERT news sentiment scoring", False),
}


def ensure_data(provider: Path) -> None:
    if (provider / "calendars/day.txt").exists():
        return
    if provider == US:
        sys.exit(
            f"[gpu-run] US dump missing at {provider}.\n"
            "  It is built by the `us-alpaca-data` GitHub Actions workflow\n"
            "  (Actions -> us-alpaca-data -> Run workflow) and stored as the\n"
            "  `us-qlib-dump` artifact. Fetch + unpack it with:\n"
            "    gh run download -R Riyu-D1/quant-research -n us-qlib-dump \\\n"
            "        -D data/qlib/us_alpaca\n"
            "  (on Kaggle/Colab the notebook cell does this via GH_TOKEN —\n"
            "   no Alpaca keys needed)")
    region = "us" if "us" in provider.name else "cn"
    print(f"[gpu-run] downloading qlib {region} data -> {provider}")
    provider.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, "scripts/download_data.py", "--target-dir", str(provider),
                    "--region", region], cwd=REPO, check=True)


def ensure_us_concepts() -> None:
    """HIST needs stock_index/stock2concept matching the dump's universe."""
    assets = REPO / "configs/benchmarks/assets"
    if (assets / "qlib_sp500_stock_index.npy").exists() and \
            (assets / "qlib_sp500_stock2concept.npy").exists():
        return
    print("[gpu-run] building sp500 concept assets for HIST")
    subprocess.run([PY, "scripts/build_us_concepts.py", "--instruments",
                    str(US / "instruments" / "sp500.txt"),
                    "--out-dir", str(assets)], cwd=REPO, check=True)


def run_one(name: str) -> Path:
    entry = REGISTRY[name]
    config, provider, experiment = entry[0], entry[1], entry[2]
    needs_data = entry[4] if len(entry) > 4 else True
    if needs_data:
        ensure_data(provider)
    if name == "hist_us":
        ensure_us_concepts()
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
