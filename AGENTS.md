# Agent notes

- Env: `uv` + Python 3.11; local venv at `.venv`. Tests: `.venv/bin/python -m pytest tests/ -q`
  (all synthetic, ~2 min, safe on the Mac).
- Heavy work (backtests, data pulls, FinBERT, retrains) runs on GitHub Actions only — never on the Mac.
  Phase workflows dispatch via `workflow_dispatch` once on master; `phase9.yml` also runs from a
  push to `phase9/improve` whose commit message carries `[p9:news]`, `[p9:wf]` or `[p9:retrain]`;
  `phase10.yml` likewise from `phase10/alpha` with `[p10:data]`, `[p10:dev]` or `[p10:holdout]`
  (the holdout job refuses a second run once `results/phase10_holdout.json` exists);
  `phase11.yml` from `phase11/hybrid` with `[p11:data]`, `[p11:baseline]`, `[p11:text]`,
  `[p11:llm]`, `[p11:dev]` or `[p11:holdout]` (holdout refuses once `results/phase11/holdout.json` exists).
- CI: `.github/workflows/pytest.yml` runs the suite on every PR and on push to master;
  `tests/test_workflows.py` lints workflows for per-page `--paginate` reductions.
- Backtest universe (phase 11): rank only names that were S&P 500 members on the signal day
  (`src/quant_research/pit.py`, `results/phase11/sp500_membership.csv`); the bar is `B50_honest`.
  The pre-phase-11 B50 numbers used every ever-member and overstate the edge.
- Artifact lookups in workflows: `--paginate -q '.artifacts[] | select(...) | .id' | sort -n | tail -1`
  (a `[...] | max` filter prints one value per page and breaks).
- Paper only: never set `live_trading: true` or `validated: true` without strong out-of-sample evidence;
  new candidates go in a predeclared `configs/experiments/phaseN_us.yaml` committed before any run, and
  every scored window counts toward `n_trials` in `results/holdout_log.jsonl`.
