# Phase 9: research and improve B50_momvq

Spec (predeclared, committed before any run):
`configs/experiments/phase9_us.yaml` · harness: `scripts/phase9_us.py` ·
FinBERT scorer: `scripts/phase9_news.py` · workflow:
`.github/workflows/phase9.yml` · tests: `tests/test_phase9.py` ·
results: `results/phase9_us_walkforward.json` · literature scan:
[`docs/PHASE9_LIT_REVIEW.md`](PHASE9_LIT_REVIEW.md).

## Verdict: keep B50_momvq. Nothing beat it out of sample.

- **Adopted: none.** None of the 14 paper-book alternatives (plus 2 news
  variants) beat B50_momvq on walk-forward AR, active return or DSR.
  Every active-vs-B50 return is negative, and every active DSR is 0.00.
- **The paper config is unchanged.** `configs/paper_signal.yaml` and
  `configs/live_signal.yaml` are untouched. `validated: false`,
  `live_trading: false`, and the live cron stays commented.
- **Correction to the phase-8 audit.** The "sleeve_n=8 → AR +7.7%,
  below SPY" result was a **$1-order-floor artifact** of the $12.70
  micro book (0.5 × $12.70 / 8 = $0.79 per name, so no sleeve order could
  ever fill). Phase 9 reproduces it exactly (B50_n8_micro: +7.7%, 11
  trades in ten years). On the ~$100k paper book, n=8 earns **+27.1%**.
  Breadth is not a cliff. The return falls smoothly as names are added
  (n 6 → 8 → 10 → 15 → 20: +29.6 → 27.1 → 26.2 → 24.3 → 20.7%), which
  says the edge sits in the very top of the ranking.
- **The adaptive model is built and runs, but in shadow mode only.** It
  has weekly retraining on Actions, monthly IC weights and LightGBM
  refits, a vol overlay, and a drift monitor that falls back to B50 and
  then SPY. Walk-forward it loses to B50 (+22.9% vs +29.6%). The safety
  net did its job: the fallback cut the raw ensemble's drawdown from
  −41.9% to −34.0% and lifted AR from +21.0% to +22.9%. As of 2026-11-23
  the live retrain sits in `fallback_b50`, so its picks are the B50
  picks.
- **The edge is still unproven.** B50's excess-vs-SPY DSR is **0.14**
  at 574 trials (phase-8 audit: 0.36 at 392). The hurdle rises with
  every trial run, phase 9's included. The paper cron remains the only
  clean out-of-sample test.

## What was run (all GitHub Actions, nothing heavy on the Mac)

| run | job | what |
|---|---|---|
| 45400120074 | news ×8 shards + walkforward | FinBERT on 454,028 unique universe headlines (790,014 symbol rows, 100% scored, ~5 CPU-h over 8 runners). First grid pass, before news existed. |
| 45402836702 | retrain (`[p9:retrain-refresh]`) | Full weekly job: rebuilt the Alpaca dump to 2026-11-23, retrained, wrote `state/phase9/`. |
| 45409245921 | walkforward | **Final scored run** (deterministic re-run of the same predeclared grid plus the 2 news rows; the script replaces its own log rows instead of appending twice). |

Trial accounting: n_trials = **574** = 392 (prior running total) + 160
(16 candidates × 10 years) + 10 (phase-8 code-path parity re-run) + 12
(2 news candidates × 6 years). Discarded candidates count too. Every DSR
below uses n = 574.

Reproduction: on the same CI inputs, `B50_micro` equals the phase-8
code path bit-for-bit (`code_parity: true`). Against the numbers stored
in phase 8, 5 of 10 years are identical and 5 differ by ≤0.6pt (largest:
2018, +3.9 vs +3.4%). Walk-forward AR is 27.9% vs the stored 27.8%. No
`phase8.yml` runs exist, so phase 8 ran outside CI on an older local
`data/events` vintage. The gap is input drift, not code.

## Results (gate years 2017-2026, yearly episodes, paper book ≈ $100k)

Costs are liquidity-bucket half-spread + 5 bps per side, with next-open
fills, a 2% band, a 20% single-name cap and delist liquidation. Sharpe
is computed on daily total returns. The DSR columns use n = 574.

| candidate | idea | AR | Sharpe | MDD | xs-SPY AR | DSR xs | active vs B50 (95% block-boot) | DSR active | yrs > B50 | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| A_spy | benchmark | +14.8% | 0.86 | −33.8% | 0.0% | — | — | — | 1/10 | pass |
| **B50_momvq** | **current** | **+29.6%** | **1.24** | **−32.3%** | **+13.1%** | **0.14** | — | — | — | **pass** |
| B50_micro | $12.70 anchor | +27.9% | 1.25 | −31.6% | +11.5% | 0.06 | −1.7% [−4.9, +1.1] | 0.00 | 5/10 | pass |
| B50_n8_micro | audit n=8 anchor | +7.7% | 0.90 | −17.3% | −7.6% | 0.00 | −20.8% | 0.00 | 0/10 | fail |
| B50_n8 | 1 breadth | +27.1% | 1.20 | −34.8% | +11.0% | 0.10 | −2.2% [−4.1, −0.2] | 0.00 | 4/10 | pass |
| B50_n10 | 1 breadth | +26.2% | 1.19 | −33.8% | +10.1% | 0.11 | −3.0% [−5.6, −0.2] | 0.00 | 1/10 | pass |
| B50_n15 | 1 breadth | +24.3% | 1.15 | −34.8% | +8.4% | 0.09 | −4.7% [−8.2, −1.1] | 0.00 | 3/10 | pass |
| B50_n20 | 1 breadth | +20.7% | 1.02 | −35.9% | +5.4% | 0.01 | −7.7% [−11.6, −3.7] | 0.00 | 1/10 | fail |
| R50_n15_ivol | 1 risk-parity | +21.2% | 1.07 | −35.7% | +5.8% | 0.02 | −7.4% [−11.9, −2.8] | 0.00 | 3/10 | pass |
| V50_vol | 2 vol-managed | +27.2% | 1.19 | −34.1% | +11.1% | 0.10 | −2.0% [−3.5, −0.8] | 0.00 | 1/10 | pass |
| T50_trendvol | 2 trend + vol | +25.4% | 1.15 | −33.9% | +9.5% | 0.05 | −3.6% [−5.9, −1.5] | 0.00 | 1/10 | pass |
| IC50_adapt | 3 online IC weights | +21.7% | 1.04 | −35.6% | +6.4% | 0.00 | −6.7% [−12.1, −1.6] | 0.00 | 1/10 | fail |
| GB50_lgbm | 4 purged LightGBM | +24.1% | 0.98 | −40.8% | +9.3% | 0.00 | −3.8% [−11.6, +3.2] | 0.00 | 5/10 | fail |
| EN50_ens | 3+4 ensemble | +21.0% | 0.94 | −41.9% | +6.3% | 0.00 | −6.8% [−12.8, −0.8] | 0.00 | 2/10 | fail |
| AD50_full | **end model** | +22.9% | 1.03 | −34.0% | +7.7% | 0.01 | −5.4% [−9.4, −1.8] | 0.00 | 1/10 | pass |
| AD50_n10 | end model, n=10 | +21.8% | 1.03 | −36.2% | +6.5% | 0.01 | −6.7% [−10.7, −3.1] | 0.00 | 1/10 | fail |

The SPY walk-forward over the same days: AR +14.9%, Sharpe 0.87, MDD
−33.8%. The B50 total-return DSR is 0.996 (SR* 0.38). That number is
not informative, because a half-SPY book clears it for free (see
phase-8 audit item 5). The excess-vs-SPY DSR, 0.14 with SR* 1.55, is the
honest one.

### Per-year AR (%)

| year | SPY | B50 | n8 | n10 | n15 | V50_vol | IC50 | GB50 | AD50 |
|---|---|---|---|---|---|---|---|---|---|
| 2017 | +21.8 | +37.8 | +34.0 | +33.2 | +31.2 | +37.8 | +33.8 | +37.8 | +36.7 |
| 2018 | −5.0 | +4.1 | +0.5 | −0.4 | −2.3 | +0.9 | −0.8 | +11.0 | −0.8 |
| 2019 | +31.1 | +29.3 | +31.9 | +32.7 | +34.4 | +28.8 | +19.4 | +31.5 | +27.6 |
| 2020 | +18.4 | +32.9 | +28.8 | +29.9 | +24.0 | +22.5 | +17.0 | +63.0 | +30.9 |
| 2021 | +28.6 | +32.1 | +32.2 | +30.8 | +32.6 | +32.1 | +44.0 | +16.6 | +34.6 |
| 2022 | −18.2 | −1.9 | −7.4 | −8.3 | −11.1 | −4.4 | −10.7 | −29.6 | −17.4 |
| 2023 | +26.4 | +46.5 | +47.6 | +46.3 | +40.0 | +46.7 | +36.4 | +46.0 | +39.1 |
| 2024 | +24.9 | +84.8 | +72.3 | +70.7 | +67.2 | +79.9 | +61.6 | +33.2 | +55.0 |
| 2025 | +17.9 | +34.6 | +39.0 | +31.4 | +27.2 | +32.1 | +25.4 | +34.7 | +30.1 |
| 2026* | +13.3 | +12.7 | +9.1 | +12.3 | +15.9 | +12.7 | +6.1 | +22.8 | +9.8 |

\*2026 runs to 2026-11-13, the dump end used by the grid.

### Idea 5: FinBERT headline sentiment (news years 2021-2026 only)

| candidate | AR | Sharpe | MDD | xs-SPY AR | active vs B50 | DSR active | yrs > B50 |
|---|---|---|---|---|---|---|---|
| B50_momvq (same days) | +32.6% | 1.34 | −26.8% | +15.8% | — | — | — |
| NW50_momvqnews (⅓ mom, ⅓ vq, ⅓ news) | +22.4% | 1.18 | −24.1% | +6.9% | −8.9% | 0.00 | 0/6 |
| NWAD50_full (ens + news, vol, drift) | +25.7% | 1.15 | −26.7% | +10.2% | −5.6% | 0.00 | 1/6 |

Adding the news leg shallows the drawdown a little and costs about 6-9
points of AR a year. That matches the literature (Chen, Kelly & Xiu;
Lopez-Lira & Tang): headline sentiment is a 1-3 day effect, strongest in
small caps. Mixed into a slow S&P 500 sleeve, it mostly adds noise to
the momentum ranking.

### Reading the results, idea by idea

1. **Breadth / risk-weighting.** This is the main robustness finding.
   Wider sleeves give up return steadily and gain diversification. The
   median share of yearly sleeve P&L from the top 3 names falls from
   **73%** (n=6) to 53% (n=10) and 43% (n=15). For 2024, top-3 share is
   66% (CVNA, APP, DECK) at n=6 and 37% at n=15. B50_n15 and R50_n15_ivol
   are the only variants within 0.5pt of SPY or better in **all 10
   years** (B50_n15's worst year is +2.6pt vs SPY; B50's worst is
   −1.9pt). It still gives up about
   5 points a year vs B50, and the bootstrap CI of that gap excludes zero.
   That is a different risk preference, not an improvement. Inverse-vol
   weighting (R50) is worse than equal weight at the same breadth,
   because it tilts away from the high-vol winners that carry the edge.
2. **Vol-managed sleeve / trend regime.** Both are slightly worse than
   B50 in most years (V50 −2.0pt/yr, CI excludes zero). They don't reduce
   MDD, because the worst drawdowns come from SPY-wide selloffs in the 50%
   core. This is consistent with Cederburg et al. (2020): real-time
   vol-management rarely beats the unmanaged portfolio.
3. **Online IC weights (factor momentum).** The weights do adapt: mom's
   weight goes 0.40 → 0.26 → 0.25 → 0.67 (2024) → 0.71 (2025) → 0.38
   (2026), and rev, mom6 and lowvol rotate in and out. But they chase
   last year's winning leg into its reversal (2019 +19.4% vs B50 +29.3%;
   2024 +61.6% vs +84.8%). The trailing-IC estimate is too noisy at
   252 days to beat a fixed 50/50.
4. **LightGBM (monthly expanding refit, 27-day purge+embargo, purged CV
   each January).** It has the widest year-to-year swings: 2020 +63%,
   2018 +11%, but 2022 −29.6% and MDD −40.8%. It beats B50 in 5/10
   years and its bootstrap CI is the only paper-book one that includes
   zero, but the
   mean is −3.8pt/yr and turnover is 3.5× B50's. Purged CV picked the
   largest grid point (31 leaves, 400 trees) every year from 2021, and
   the CV IC was ≤0.03 in every year, about zero in 2026. This matches
   phases 2-7: no robust ML edge in this S&P 500 feed.
5. **FinBERT news.** See above. It is clean (no LLM look-ahead), but it
   is the wrong horizon for this book.

LLM agents (TradingAgents, FinMem, FinAgent) and LLM factor mining
(Alpha-GPT and similar) were **not** tested. No paper reports credible
out-of-sample results. The one long-horizon re-test (FINSABER, KDD 2026)
is negative. And any present-day LLM was trained on the 2017-2026 test
years, so no honest backtest of it is possible here. See the literature
scan, §4 and §6.

## Adaptive design (the end model, AD50_full)

The design is built so it can adapt, with B50 and then SPY as the
safety floor. Everything below is re-estimated inside the walk-forward
exactly as the live retrain does it:

| component | how it adapts | when | causality |
|---|---|---|---|
| Online factor weights | `w = ½·prior(mom ½, vq ½) + ½·max(0, EW rank-IC)/Σ`. The EW mean uses a 126d halflife over 252 usable days, across mom, vq, mom6, lowvol, beta and rev | first trading day of each month | an IC is usable only once its 21d forward window has closed (shifted 22 days on the full grid); <126 obs or all ≤0 → prior |
| LightGBM | expanding-window refit on 10 factor ranks; label = rank of the 21d next-open forward return | monthly; hyper-parameters by purged 3-fold blocked CV each January | train rows need `d ≤ pos(m) − 22 − 5` (label purge + embargo); <252 labelled days → B50 composite (2017-02/2018) |
| Ensemble | mean of the IC-blend and GBM ranks; names need mom & vq (B50 parity) | daily | — |
| Regime / vol overlay | sleeve exposure = clip(504d median σ / 63d σ, 0, 1) of the shadow sleeve; freed share → SPY, never levered | daily | trailing realised returns only |
| Drift monitor + fallback | state machine below | daily | inputs known at close t govern t's signal |

**Fallback behaviour** (`drift_modes`, unit-tested):

- `adaptive → fallback_b50` when the adaptive shadow sleeve trails the
  B50 shadow sleeve by more than 5% over 126 days, **or** the model's
  realised EW rank-IC (63d halflife) is below 0.
- `fallback_b50 → adaptive` when both recover: lag ≥ 0 and IC > 0.
- Any mode `→ fallback_spy` (sleeve to core, 100% SPY) when the B50
  sleeve itself trails SPY by more than 15% over 252 days. It recovers
  once that gap is back above −5%.
- Every switch holds for at least 21 days.

The shadow sleeves are cost-free equal-weight top-n portfolios formed
at yesterday's close. They keep running in every mode, so the monitor
can see a recovery.

Observed in the walk-forward: AD50 spent 81% of 2017 in `adaptive`,
51-56% in 2018-19, 9-40% in 2020-24, 0% in 2025 and 57% in 2026. It
made 33 switches in ten years and never needed `fallback_spy`. Net: it
fixed most of the raw ensemble's damage (MDD −41.9% → −34.0%, gate fail
→ pass) but did not catch up with B50. When the adaptive model stops
adding value, the fallback means it mostly *becomes* B50 plus some
switching cost.

**Live retrain (shadow).** `phase9.yml` job `retrain` runs every
**Monday 06:00 UTC** once the file is on master, and on dispatch. Each
run does four things:

1. Rebuilds the Alpaca PIT dump (same steps as `us-alpaca-data`, about
   8 minutes).
2. Recomputes the IC weights, the trailing 18 months of monthly GBM
   refits, the overlay and the drift state.
3. Commits `state/phase9/adaptive.json` and appends
   `state/phase9/drift_log.jsonl`.
4. Uploads `phase9-qlib-dump` (14-day retention; deliberately not
   `us-qlib-dump`, so other workflows don't pick it up unreviewed).

The first run (2026-11-23 data) gave `mode: fallback_b50`, lag −5.5%,
IC −0.06, picks SNDK/APA/LYB/VLO/MPC/PSX (identical to B50), and
`live_effect: shadow`. **No config reads this file.** Wiring it into
`paper_trade.py` is deliberately left out until a candidate passes the
adoption rule.

## Adoption rule (predeclared) and why nothing passed

A candidate replaces B50 only if all four hold:

1. It passes the phase-8 SPY gate.
2. Active-vs-B50 mean > 0 **and** DSR(active, n = 574) ≥ 0.95.
3. It beats B50's AR in at least 60% of years.
4. Its excess-vs-SPY DSR is at least B50's.

Every candidate fails leg 2 at its first step: all active means are
negative. The breadth, vol-overlay and AD50 variants pass the SPY gate
but not the B50 comparison. So B50_momvq stays, as the brief requires.

## Caveats and data limits

- 2025 and 2026 have been viewed in earlier phases. Every year here is a
  re-viewed walk-forward window, not a fresh holdout
  (`re_viewed_holdout: true` in the log).
- Fundamentals and shares come from the `phase4-inputs` and
  `phase3-events` artifacts (SEC pulls of 2026-11-11). vq for the newest
  quarter is therefore stale in the retrain until those are refreshed
  (`phase4.yml job=fetch`). At the default 90-day retention the
  artifacts expire around 2027-02. A
  refresh should be added before then, or the retrain will fail loudly.
- News runs from 2020-09 (the phase-4 fetch window) and is headline-only
  (FinBERT, max 64 tokens, int8). The run took about 5 CPU-hours across
  8 Actions shards. The Kaggle GPU alternative is in place but was not
  needed. If it is wanted (e.g. headline+summary at 128 tokens), set
  `MODELS="phase4_news_score"` with `PUSH_RESULTS=1` in
  `notebooks/qlib_gpu_run.ipynb`.
- The yearly-episode structure (the book starts flat each 1 January)
  follows phase 8 for comparability. Adaptive state (IC history, GBM
  training rows, drift monitor) is continuous across years, as it would
  be live.

## Ops notes found on the way (not fixed here)

- `phase6.yml`, `phase7.yml` and `phase8.yml` artifact lookups use
  `--paginate -q '[...] | max'`, which prints one max per page. It broke
  the first phase-9 run once the repo passed one page of artifacts
  ("accepts 1 arg(s), received 2"). `phase9.yml` uses
  `... | sort -n | tail -1`, and the older workflows need the same fix
  before their next dispatch.
- The `claude` CLI on the Mac returns `Invalid API key · Please run
  /login`. Run `claude` then `/login` once to use Claude Max as a
  research assistant.
- Actions minutes for this phase: about 5 CPU-h for FinBERT, 2 × ~12 min
  walk-forward, ~10 min retrain. The weekly retrain will add about
  10-15 min a week after merge.

## What would change the verdict

- **Paper out-of-sample evidence.** If the paper book keeps beating SPY,
  the excess DSR rises without new trials. That is the cheapest
  evidence, and the cron is already collecting it.
- **If a smoother book is preferred over maximum return,** B50_n15 is
  the documented alternative: 10/10 years ≥ SPY−0.5pt and a top-3 share
  of 43%, but about −4.7pt/yr vs B50. Switching to it is a risk-preference
  decision for Riyansh, not an out-of-sample win, so it was not made
  here.
- **Next text idea with credible OOS evidence:** "Lazy Prices" 10-K/10-Q
  change signals from EDGAR (monthly horizon, free, no LLM look-ahead).
  It needs a full-text EDGAR pull on Actions.
