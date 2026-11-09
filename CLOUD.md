# Running benchmarks on cloud GPUs

Phase-1 neural models (Transformer, HIST) need a GPU. Everything here is
**run-by-you**: no account has been or will be created without approval.

The driver is `scripts/gpu_run.py` — it downloads missing qlib data, runs the
models you name sequentially, rebuilds `results/LEADERBOARD.md`, and appends
each finished run to `EXPERIMENTS.md`.

```
python scripts/gpu_run.py --list            # all registered model names
make gpu-run MODELS="transformer_cn hist_cn"
```

## Option A — Kaggle (default, free)

* GPU: 2×T4 or 1×P100, ~30 h/week quota, 12 h per session.
* Steps: new notebook → *Settings → Accelerator: GPU* → *Add-ons → Secrets* add
  `GH_TOKEN` (a GitHub PAT with repo access) → *File → Import* the notebook
  `notebooks/qlib_gpu_run.ipynb` from this repo (or paste its cells) → Run all.
* Data + deps: handled inside the notebook (uv sync, qlib download ~1–2 GB).
* Results: set env `PUSH_RESULTS=1` to push `results/` to a dated branch, or
  download `results/runs/*.json` from the Output pane.

## Option B — Colab (free tier)

* GPU: T4, ~4 h sessions (disconnects kill state — run ONE model per session).
* Steps: new notebook → *Runtime → Change runtime type → T4* → paste the same
  `notebooks/qlib_gpu_run.ipynb` cells → set `GH_TOKEN` in the Secrets pane → run.

## Option C — any GPU VM with Docker

```
docker build -f Dockerfile.gpu -t quant-research:gpu .
docker run --gpus all --rm -it \
    -v $PWD/data:/app/data \
    -v $PWD/results:/app/results \
    -v $PWD/mlruns:/app/mlruns \
    -v $PWD/EXPERIMENTS.md:/app/EXPERIMENTS.md \
    quant-research:gpu bash
cd /app && python scripts/gpu_run.py --models transformer_cn hist_cn
```

The mounts persist the qlib dump, `results/runs/*.json`, mlflow recorders and
the EXPERIMENTS.md log back to the host — without them `--rm` throws the run
away. Do NOT mount `$PWD` over `/app`: it would shadow the image's `.venv`.

(Needs the NVIDIA container toolkit on the host. `Dockerfile.gpu` is a CUDA 12.x
runtime; the locked torch 2.6.0 wheel bundles CUDA 12.4 libs — host driver
>= 550 required. The macOS OpenMP patches are NOT needed — Linux wheels ship
one consistent runtime.)

## Estimated runtime per model (single T4/P100, csi300 Alpha158, 2008–2020)

| Model | Data init | Train+backtest | Total |
|---|---|---|---|
| lightgbm_cn | ~8–15 min | ~5–10 min | **~20 min** |
| xgboost_cn | ~8–15 min | ~5 min | **~20 min** |
| doubleensemble_cn | ~8–15 min | ~15–30 min | **~30–45 min** |
| transformer_cn | ~10–15 min | ~60–120 min (200 epochs max, early stop 20) | **~1.5–2.5 h** |
| hist_cn (Alpha360) | ~10–20 min | ~90–180 min | **~2–3.5 h** |

US variants run on sp500 (larger universe): add ~50–100% to data init.
GBDTs are CPU-bound — GPU only helps Transformer/HIST.

## CPU profile for the local Mac

`configs/benchmarks/` now holds upstream defaults (threads 20, GPU on).
The Mac-local profile (threads 10, `GPU: -1`) lives in `configs/benchmarks/cpu/`:

```
make bench-cn-transformer PROFILE=cpu
```
