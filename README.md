# quant-research

Quant research + paper-trading system. Goal: a strategy that beats the public
Microsoft Qlib benchmark leaderboard on **risk-adjusted, out-of-sample** metrics,
then paper-trades it via NautilusTrader + Alpaca. **No real money, ever.**

## Status

Phase 1 — baseline reproduction of the Qlib benchmarks. See
`results/LEADERBOARD.md` for the table and `EXPERIMENTS.md` for the attempt log.

## Layout

    configs/benchmarks/   Qlib workflow configs (vendored from microsoft/qlib,
                          provider_uri repointed to ./data). Provenance in
                          configs/PROVENANCE.md
    data/qlib/            qlib-format dumps (gitignored; see `make data-cn`)
    results/              published_leaderboard.csv + per-run JSON + LEADERBOARD.md
    scripts/              run_benchmark.py, make_table.py, data_info.py
    src/quant_research/   shared code (metrics, runner helpers)
    third_party/qlib/     pinned clone of microsoft/qlib for reference (gitignored)

## Setup

    make setup            # uv sync on Python 3.11 (managed by uv)
    make data-cn          # ~1-2 GB download, CSI300 dump
    make data-us          # US dump (nasdaq100/sp500/dow30 instruments)

## Run a benchmark

    make bench-cn-lightgbm        # one model
    make bench-cn                 # all CN models sequentially
    make table                    # rebuild results/LEADERBOARD.md

## Cloud GPU

Neural-model training runs on a cloud GPU — see **CLOUD.md** (Kaggle / Colab /
Docker GPU VM; nothing is provisioned automatically).

    make gpu-run MODELS="transformer_cn hist_cn"   # downloads data, runs, logs EXPERIMENTS.md
    make gpu-run-list                              # all registered model names
    make docker-gpu-build                          # CUDA image (Dockerfile.gpu)

## Docker

    make docker-build && make docker-shell

## Rules for honest results (project policy)

- Walk-forward validation; a final untouched hold-out evaluated once, at the end.
- Realistic costs: commission + spread/slippage; zero look-ahead in features.
- Survivorship-bias-free universe where the data allows; documented where not.
- Every experiment logged in EXPERIMENTS.md, including failures.
- Deflated Sharpe ratio for the final strategy.

## Phase gates

1. **Baseline (this phase):** reproduce Qlib LightGBM / XGBoost / DoubleEnsemble /
   Transformer / HIST on Alpha158, CSI300 + US data. Report actual vs published.
2. Beat the baseline — one hypothesis per branch/PR, keep only what improves
   out-of-sample information ratio without worsening max drawdown.
3. Port winner to NautilusTrader, verify parity with Qlib backtest, run on
   Alpaca **paper** account behind a kill switch + daily loss limit.
   Live trading requires explicit user approval and has not been given.
