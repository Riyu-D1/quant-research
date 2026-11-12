# Phase 4 — adaptive US strategy (monthly retrain + signal weighting)

The Phase-3 critique that motivates this phase: every prior US model was
trained once and **frozen** for ~2 years of holdout. Real desks retrain
often and re-weight signals. Phase 4 asks whether an *adaptive process* —
monthly retraining plus trailing-performance signal weighting plus a
predeclared regime guard — does better than the frozen Phase-2/3 process
on the same feed, honestly scored.

## Honesty contract

- **No-lookahead everywhere.** A signal value on trading day `D` uses only
  data known before `D`:
  - price/fundamental features are point-in-time (`Alpha158` on the Alpaca
    qlib dump; XBRL facts key on `filed < D`; shares-out key on `filed < D`);
  - news headlines key on publish calendar date — an article is visible on
    the first trading day **strictly after** its publish date (overnight /
    pre-market alpha deliberately discarded, documented and conservative);
  - the regime guard reads SPY closes strictly before `D`
    (`regime_exposure` shifts all inputs by one day).
- **Month boundaries are airtight.** Month `m`'s GBDTs train on rows with
  `datetime < first_trading_day(m)` only; month `m`'s signal weights use
  only rank ICs from days `< first_trading_day(m)`. Verified by corruption
  tests in `tests/test_phase4.py` (mutating eval-window labels or
  post-window features cannot change that month's predictions/weights).
- **Selection is 2021-2024 only**, month-by-month walk-forward; the
  full candidate grid + comparators were predeclared in
  `configs/experiments/walkforward_us_adaptive.yaml` before any sim ran.
- **Holdout scored once.** The chosen process (its monthly retrains
  included) is simulated over 2025-01 → latest exactly once and appended
  to `results/holdout_log.jsonl` with the git SHA. The US holdout was
  viewed in Phases 2 and 3 — `prior_trials_us: 66` (37 + 29) is added to
  the selection candidate count for the honest DSR `n_trials`.
- **Nothing heavy on the owner's Mac.** Fetch/train/eval all run on
  GitHub Actions (`phase4-adaptive` workflow) or Kaggle GPU (FinBERT
  scoring, `MODELS="phase4_news_score"` in `notebooks/qlib_gpu_run.ipynb`).

## Signal library (six, each an independent score)

| Signal | What | Data |
|---|---|---|
| `gbdt` | Alpha158 LightGBM, retrained monthly on a rolling 24- or 36-month window, per-day CSRankNorm labels, early stop on last 42 train days | qlib dump |
| `ev` | Phase-3 insider/13F/congress disclosure features, same monthly LightGBM | `features_us.parquet` |
| `mom` | `close[t-21]/close[t-252] - 1` (12-1 momentum) | closes |
| `rev` | `-(close[t]/close[t-21] - 1)` (1-month reversal) | closes |
| `vq` | mean of per-day ranks of `vq_bm` (book/mcap), `vq_ep` (NI-TTM/mcap), `vq_roe` (NI-TTM/book) — all `filed < D` | SEC companyfacts + shares_out |
| `news` | per-symbol EW mean of FinBERT (pos−neg) headline scores, 5-trading-day halflife, publish date < day | Alpaca `/v1beta1/news` + ProsusAI/finbert on Kaggle |

## Adaptive weighting

Each month `m`, every signal's daily rank IC is exponentially-weighted-
averaged over the trailing `ic_months` (grid: 6 or 12; halflife =
`ic_months × 0.5 × 21` days). Weights are floored at 0 — losing signals
switch off — and L1-normalised; an all-nonpositive month yields all-zero
weights (book sits in cash). Comparators: `equal` (1/N every month),
`single:<sig>` (one signal alone, incl. the Phase-2-style `gbdt_frozen`
train-once comparator).

## Regime guard (predeclared)

On each day `D`, measured on SPY closes strictly before `D`:

    cut exposure to `action` when
        SPY < SMA200   OR   21d realised vol > 90th pct of its trailing 504d

Grid over `action`: `none` (off) / `half` (0.5) / `cash` (0.0) —
implemented as a per-day exposure multiplier in the MicroTop5 simulator.

## Portfolio

Phase-3 `MicroTop5`: £10 ≈ $12.70, fractional shares, long-only, max 5
names, 10 bps/side cost, daily rebalance (equal-weight top-5).

## Pipeline (all remote)

1. `phase4.yml job=fetch` — Alpaca news headlines (monthly pages, cached,
   resumable) + SEC companyfacts for ~745 CIKs → artifacts `phase4-news`,
   `phase4-inputs`.
2. `phase4.yml job=store` — one heavy pass: Alpha158 raw (no processors)
   + event cols + mom/rev + v/q + label → `data/features4/store_us.parquet`
   (float32) → artifact `phase4-store`. Peak ~7-8 GB with swap.
3. Kaggle: `MODELS="phase4_news_score"`, `PUSH_RESULTS=1` — FinBERT over
   unique texts → `results/news_sentiment/daily_sentiment.parquet` on a
   `results/kaggle-*` branch.
4. `phase4.yml job=preds` (matrix `signals × window`) — monthly-retrained
   preds → `results/oof/us_adaptive/{sig}__w{W}__YYYY-MM.pkl` committed to
   this branch (resume-safe).
5. `phase4.yml job=select` — 2021-2024 monthly walk-forward over the grid
   → `results/phase4_us_select.json`.
6. `phase4.yml job=holdout --config '<chosen json>'` — scored once →
   `results/phase4_us.json` + `holdout_log.jsonl`.

## Paper-trade gate

Paper trading (`scripts/paper_trade.py`, Alpaca paper endpoint only with
a hard live-endpoint guard, Notion Trade Journal, dispatch-only workflow)
is built **only if** the adaptive holdout is positive AND beats SPY
buy-and-hold on IR AND DSR > 0.5. Otherwise none is written.

## Results

**Select (2021-2024, run 43141704337):** 27 candidates —
`results/phase4_us_select.json`. Best deployable `equal-w36-rcash`; every
candidate's IR was negative to flat on OOF — no deployable adaptive config
cleared bar on the selection window.

**Live-bot holdout (RE-VIEWED, run 43323965312):** the pinned paper-bot
pipeline (`configs/live_signal.yaml`: mom+vq equal, regime guard cash,
MicroTop5 £10, 10 bps) scored once on 2025-01-02 → 2026-11-06 by
`scripts/live_holdout.py` — signals/weights decided through 2024-12-31
only, weights re-derived monthly exactly like the live run. This window
was already viewed in phases 2/3 — **re-viewed holdout, no tuning
permitted on it.**

| variant | total | ann. | IR | IRxs SPY | max DD | turnover | hit | final £ | DSR |
|---|---|---|---|---|---|---|---|---|---|
| regime:cash (pinned) | +39.2% | +19.8% | +0.977 | +0.236 | -17.3% | 0.097 | 46.4% | £13.92 | 0.415 |
| regime:none (guard off) | +62.2% | +30.2% | +1.189 | +0.746 | -25.7% | 0.107 | 54.9% | £16.22 | 0.527 |
| SPY buy & hold | +28.2% | +14.5% | +0.876 | — | -18.8% | — | — | — | — |

In-sample context (2023-2024, not a holdout): bot IR 1.37/1.52 < SPY 1.87
— the OOS SPY-beat is *not* predicted by the in-sample window.

**Gate verdict (first pass, legacy spec):** positive ✓, beats SPY on IR ✓,
DSR > 0.5 ✗ (0.415, n_trials = 95) → gate FAILED.

**Phase-5 honest rescore (run 43329820435, `job=honest_holdout` →
`results/honest_holdout_us.json`):** the same pinned pipeline re-scored
with a corrected universe, costs, and fills. What changed:

- **Universe — the real surprise.** `live_universe.txt` already contains
  the 2025-exited names (WBA, JNPR, PARA, DFS, BWA…), so the deployed
  universe was never survivors-biased on the membership axis. The old
  score's inflation came from a subtler bug: signals were computed on
  *membership-masked* bars, so any mid-window index joiner (COIN +1030d,
  DASH +1076d, CVNA +2176d of real history at join) was invisible to
  12-1 momentum for its first ~252 member-days — the hot additions of
  2025 could never be picked. The honest loader gives each name its full
  listed history (exactly what the live bot sees on Alpaca), and the
  eligibility mask is the per-day bar frame.
- **Costs** — cap-bucket half-spread by dollar volume (1/5/20 bps
  mega/mid/illiquid) + 10 bps flat, per-name per-day.
- **Fills** — next-day open instead of signal-close (no same-close
  look-ahead).
- **Delisted holds** — liquidate at last close instead of zombifying.

| variant | total | ann. | IR | xsIR vs SPY | maxDD | turnover | hit | DSR |
|---|---|---|---|---|---|---|---|---|
| survivors_only (legacy fills/costs) | +26.8% | +13.8% | +0.725 | +0.008 | -21.3% | 0.096 | 46.4% | 0.280 |
| listed (legacy) — the bot's universe | +26.8% | +13.8% | +0.725 | +0.008 | -21.3% | 0.096 | 46.4% | 0.280 |
| pit_index (legacy; S&P members only) | +23.8% | +12.4% | +0.664 | -0.050 | -21.2% | 0.093 | 46.2% | 0.253 |
| **honest (primary)** | **+20.6%** | **+10.7%** | **+0.593** | **-0.119** | **-21.6%** | 0.096 | 46.4% | **0.223** |
| honest_pit (honest on index members) | +23.8% | +12.3% | +0.657 | -0.052 | -21.2% | 0.093 | 46.4% | 0.250 |
| honest regime off | +34.4% | +17.5% | +0.766 | +0.216 | -26.6% | 0.106 | 54.2% | 0.298 |
| SPY buy & hold | +28.2% | +14.5% | +0.876 | — | -18.8% | — | — | — |

`survivors_only` ≡ `listed` here — dead names are already in the universe
file, so there was no survivorship delta to remove; the revision is from
signal-history eligibility + costs + fills. **Cost fragility is severe**
(flat-bps sweep on the honest spec):

| cost/side | total | IR |
|---|---|---|
| 0 bps | +34.4% | +0.878 |
| 10 bps | +22.5% | +0.635 |
| 25 bps | +8.3% | +0.312 |
| 50 bps | -11.6% | -0.207 |

**Robustness (reported, never used to pick):**
- Per-year OOF IR: 2016 ~0 (startup), 2017 +1.65, 2018 +0.97, 2019 +0.66
  (xs -0.17), 2020 +1.18, 2021 +1.35, **2022 -1.68** (bear), 2023 +1.62,
  2024 +1.42 — good years exist but the sign isn't stable.
- Rolling-12m excess IR vs SPY: median **-0.25**, only **40.6%** of
  windows positive (latest +0.99 — recent strength is not the norm).
- Block-bootstrap annualised excess vs SPY: mean -4.9%, 95% CI
  **[-30.7%, +22.1%]** — includes zero.
- Top-N: top-10 +18.6%/IR 0.64 (similar); **top-20 is infeasible at £10**
  ($0.66/position < Alpaca's $1 minimum → flat book — noted, not a bug).
- Rebalance: daily +20.6% >> weekly +4.6% / monthly +8.1% — the edge is
  churn-dependent → fragile.

**Gate verdict (honest):** positive ✓ (+20.6%), beats SPY on IR ✗
(0.593 < 0.876), DSR > 0.5 ✗ (0.223, n_trials = 66 prior + 27 select +
2 prior livebot + 14 scored here = 109) → **VALIDATION GATE FAILED on two
of three legs. `validated` stays `false`.** Plainly: after honest
universe/costs/fills the bot *loses* to SPY on this re-viewed window —
the earlier +39.2% result was inflated by masked-signal history and
free/instant fills, and should not be quoted.
