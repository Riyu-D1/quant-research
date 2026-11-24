# Agent notes

- Env: `uv` + Python 3.11; local venv at `.venv`. Tests: `.venv/bin/python -m pytest tests/ -q`
  (all synthetic, ~2 min, safe on the Mac).
- Heavy work (backtests, data pulls, FinBERT, retrains) runs on GitHub Actions only — never on the Mac.
  Phase workflows dispatch via `workflow_dispatch` once on master; `phase9.yml` also runs from a
  push to `phase9/improve` whose commit message carries `[p9:news]`, `[p9:wf]` or `[p9:retrain]`.
- Artifact lookups in workflows: `--paginate -q '.artifacts[] | select(...) | .id' | sort -n | tail -1`
  (a `[...] | max` filter prints one value per page and breaks).
- Paper only: never set `live_trading: true` or `validated: true` without strong out-of-sample evidence;
  new candidates go in a predeclared `configs/experiments/phaseN_us.yaml` committed before any run, and
  every scored window counts toward `n_trials` in `results/holdout_log.jsonl`.
