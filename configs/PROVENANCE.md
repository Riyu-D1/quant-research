# Config provenance

Vendored from microsoft/qlib @ `54355232463878d2eebb91fe0ee5fa7fa1f5976c` (tag v0.9.8),
`examples/benchmarks/`. Installed runtime is `pyqlib==0.9.7` (latest on PyPI).

Differences vs upstream configs — mechanical only, no strategy/hyperparameter changes:

| File | Change |
|------|--------|
| all | `provider_uri` left as upstream; `scripts/run_benchmark.py --provider-uri` overrides to `./data/qlib/*_data` at runtime |
| lightgbm | `num_threads` 20 → 10 (this machine has 12 hw threads) |
| xgboost | `nthread` 20 → 10 |
| doubleensemble | `num_threads` 20 → 10 |
| transformer | `n_jobs` 20 → 10 |
| hist_alpha360 | `GPU` 0 → -1 (no CUDA); `model_path`/`stock2concept`/`stock_index` repointed to `configs/benchmarks/assets/` |

## Assets

- `model_lstm_csi300.pkl` — copied verbatim from `examples/benchmarks/LSTM/` (HIST's required pretrained base).
- `qlib_csi300_stock_index.json` — copied verbatim from `examples/benchmarks/HIST/` (v0.9.8 format).
- `qlib_csi300_stock_index.npy` — **generated** from the JSON above (`np.save` of the dict as object array) because pyqlib 0.9.7 loads stock_index via `np.load(..., allow_pickle=True).item()`. Same 735 instrument→row mappings; deterministic conversion.
- `qlib_csi300_stock2concept.npy` — downloaded from the URL hard-coded in `qlib/contrib/model/pytorch_hist.py` (`github.com/SunsetWolf/qlib_dataset/releases/download/v0/qlib_csi300_stock2concept.npy`), shape (735, 4635).

## macOS x86_64 OpenMP fix (required to run torch models)

pyqlib eagerly imports lightgbm + xgboost (which link `libomp.dylib`) alongside
torch (which bundles `libiomp5.dylib`). Two OpenMP runtimes in one process
segfault inside `__kmp_suspend_64` on Darwin 24 (SIGSEGV in
`pytorch_transformer_ts` / any parallel ATen op). Even after unifying on torch's
libiomp5, its bundled version (torch 2.2.2) still crashes in the same place —
the old Intel OMP suspend path is broken on macOS 15.

Applied fix (venv-local, reversed by `uv sync --reinstall`):

1. Every Mach-O in site-packages referencing `libomp.dylib` retargeted to
   `@rpath/libiomp5.dylib` via `install_name_tool -change ... -add_rpath
   <site-packages>/torch/lib` (74 files: sklearn exts, lib_lightgbm, libxgboost).
2. `torch/lib/libiomp5.dylib` and `functorch/.dylibs/libiomp5.dylib` replaced
   with Homebrew `libomp` 23.1.3 (current LLVM OpenMP, fixes the Darwin-24
   suspend crash) with `install_name_tool -id @rpath/libiomp5.dylib`.
3. The two copies are **hardlinked** (same inode) so dyld loads exactly one
   image — two files on different paths are NOT deduplicated.

This affects only *which* OpenMP runtime executes; not numerics.
`KMP_DUPLICATE_LIB_OK` / `KMP_BLOCKTIME` / `OMP_NUM_THREADS=1` were tried first
and were insufficient (still crashed/hung).

## Data

`data/qlib/cn_data` and `data/qlib/us_data`: qlib community dumps, `v2`, 1d bars,
from `qlib.tests.data.GetData` (github release mirror of the original
qlibpublic blob storage). **Published leaderboard numbers were produced on the
v1 dump; v2 differs (Yahoo-sourced), so exact-match reproduction is not expected.**

US configs are derived (no upstream US benchmarks exist): same task structure
with `market`/`benchmark` swapped for the US universe.

## Config profiles (2026-11-09, GPU migration)

`configs/benchmarks/*.yaml` now carry **upstream defaults**: threads 20 and
`GPU: 0` (Transformer/HIST). The former local values (threads 10, `GPU: -1`)
live unchanged in `configs/benchmarks/cpu/` for the Intel Mac — select with
`make ... PROFILE=cpu`. On Linux/CUDA none of the macOS OpenMP patching below
is needed (linux wheels ship one consistent runtime; torch auto-selects cuda).

## Cloud assets

`configs/benchmarks/assets/` is committed to the repo (incl. the 27 MB
`stock2concept.npy`, ~1 MB packed — it's sparse) so clones get every HIST
input without extra downloads.
