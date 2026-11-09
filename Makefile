SHELL := /bin/bash
DATA_DIR := $(CURDIR)/data/qlib
CN_DATA := $(DATA_DIR)/cn_data
US_DATA := $(DATA_DIR)/us_data

.PHONY: setup lock data-cn data-us bench-cn bench-us table test lint docker-build docker-shell clean

setup:
	uv sync

lock:
	uv lock

# --- Data -------------------------------------------------------------------
# Qlib community data dumps (free, no key). CN dump contains CSI300/500/800/100
# instrument files with historical constituent windows; US dump contains
# nasdaq100/sp500/dow30 etc.
data-cn:
	uv run python -m qlib.run.get_data qlib_data \
		--target_dir $(CN_DATA) --region cn --interval 1d

data-us:
	uv run python -m qlib.run.get_data qlib_data \
		--target_dir $(US_DATA) --region us --interval 1d

data-info:
	uv run python scripts/data_info.py --cn $(CN_DATA) --us $(US_DATA)

# --- Benchmarks ---------------------------------------------------------------
# Each run writes one JSON record into results/runs/<model>_<region>.json
bench-cn-lightgbm:
	uv run python scripts/run_benchmark.py --config configs/benchmarks/lightgbm_alpha158_cn.yaml --out results/runs/lightgbm_alpha158_cn.json

bench-cn-xgboost:
	uv run python scripts/run_benchmark.py --config configs/benchmarks/xgboost_alpha158_cn.yaml --out results/runs/xgboost_alpha158_cn.json

bench-cn-doubleensemble:
	uv run python scripts/run_benchmark.py --config configs/benchmarks/doubleensemble_alpha158_cn.yaml --out results/runs/doubleensemble_alpha158_cn.json

bench-cn-transformer:
	uv run python scripts/run_benchmark.py --config configs/benchmarks/transformer_alpha158_cn.yaml --out results/runs/transformer_alpha158_cn.json

bench-cn-hist:
	uv run python scripts/run_benchmark.py --config configs/benchmarks/hist_alpha158_cn.yaml --out results/runs/hist_alpha158_cn.json

bench-cn: bench-cn-lightgbm bench-cn-xgboost bench-cn-doubleensemble bench-cn-transformer bench-cn-hist

bench-us-lightgbm:
	uv run python scripts/run_benchmark.py --config configs/benchmarks/lightgbm_alpha158_us.yaml --provider-uri $(US_DATA) --out results/runs/lightgbm_alpha158_us.json

bench-us-xgboost:
	uv run python scripts/run_benchmark.py --config configs/benchmarks/xgboost_alpha158_us.yaml --provider-uri $(US_DATA) --out results/runs/xgboost_alpha158_us.json

bench-us-doubleensemble:
	uv run python scripts/run_benchmark.py --config configs/benchmarks/doubleensemble_alpha158_us.yaml --provider-uri $(US_DATA) --out results/runs/doubleensemble_alpha158_us.json

bench-us-transformer:
	uv run python scripts/run_benchmark.py --config configs/benchmarks/transformer_alpha158_us.yaml --provider-uri $(US_DATA) --out results/runs/transformer_alpha158_us.json

bench-us: bench-us-lightgbm bench-us-xgboost bench-us-doubleensemble bench-us-transformer

table:
	uv run python scripts/make_table.py --runs results/runs --published results/published_leaderboard.csv --out results/LEADERBOARD.md

# --- Hygiene ------------------------------------------------------------------
test:
	uv run pytest tests -q

lint:
	uv run ruff check src scripts tests

docker-build:
	docker build -t quant-research .

docker-shell:
	docker run --rm -it -v $(CURDIR)/data:/app/data quant-research bash

clean:
	rm -rf results/runs/*.json
