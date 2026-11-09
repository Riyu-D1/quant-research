"""Build results/LEADERBOARD.md: our runs vs published qlib benchmark numbers."""
import argparse
import json
from pathlib import Path

import pandas as pd

WANT = [
    ("ic", "IC"),
    ("icir", "ICIR"),
    ("rank_ic", "Rank IC"),
    ("rank_icir", "Rank ICIR"),
    ("annualized_return", "excess_return_with_cost.annualized_return"),
    ("information_ratio", "excess_return_with_cost.information_ratio"),
    ("max_drawdown", "excess_return_with_cost.max_drawdown"),
]


def get_metric(metrics: dict, suffix: str):
    for k, v in metrics.items():
        if k == suffix or k.endswith("." + suffix):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def fmt(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.{nd}f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--published", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    pub = pd.read_csv(args.published)
    pub_key = {f"{r.model}|{r.dataset}|{r.region}": r for r in pub.itertuples()}

    rows = []
    for f in sorted(args.runs.glob("*.json")):
        run = json.loads(f.read_text())
        m = run.get("metrics", {})
        stem = f.stem  # e.g. lightgbm_alpha158_cn
        parts = stem.split("_")
        region = parts[-1]
        dataset = parts[-2]
        model = "_".join(parts[:-2])
        rec = {"run": stem, "model": model, "dataset": dataset, "region": region,
               "train_min": run.get("train_minutes"),
               "turnover": run.get("turnover")}
        for col, suffix in WANT:
            rec[col] = get_metric(m, suffix)
        rec["published"] = pub_key.get(f"{model.capitalize()}|{dataset}|{region}")
        rows.append(rec)

    lines = ["# Leaderboard — our runs vs published qlib benchmarks",
             "",
             "Backtest: test period 2017-01-01→2020-08-01, TopkDropout(50,5),",
             "open_cost 5bps / close_cost 15bps, deal at close. Returns are",
             "excess vs benchmark *with* cost. Turnover = mean daily portfolio turnover.",
             ""]
    for r in rows:
        pub = r["published"]
        lines.append(f"## {r['run']}  ({r['train_min']} min, {r['backtest_days' if 'backtest_days' in r else 'n/a']} days)")
        hdr = "| metric | ours | published |"
        lines += [hdr, "|---|---|---|"]
        pub_vals = {} if pub is None else pub._asdict()
        for col, _ in WANT:
            pv = pub_vals.get(col) if pub_vals else None
            lines.append(f"| {col} | {fmt(r[col])} | {fmt(pv)} |")
        lines.append(f"| turnover | {fmt(r['turnover'],4)} | — |")
        lines.append("")
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
