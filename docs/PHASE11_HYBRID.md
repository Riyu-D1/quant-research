# Phase 11: honest PIT baseline + hybrid LLM / algorithmic bot

Spec (predeclared: part 1 before the baseline run, part 2 before any
candidate, text or LLM job, plus one pre-dev parser amendment):
`configs/experiments/phase11_us.yaml` · harness: `scripts/phase11_us.py` ·
text layer: `scripts/phase11_text.py` · factor proposals:
`scripts/phase11_llm.py` · formula language: `src/quant_research/factor_dsl.py` ·
membership: `src/quant_research/pit.py` · workflow: `.github/workflows/phase11.yml` ·
tests: `tests/test_phase11.py` · literature: [`docs/PHASE11_LIT.md`](PHASE11_LIT.md) ·
results: `results/phase11/`.

## Verdict: keep B50_momvq. Nothing passed the adoption rule.

- **Adopted: none.** The adoption rule had two legs. A candidate had to
  beat honest B50's holdout Sharpe, and its full-sample DSR had to be at
  least 0.95 at n_trials = 1,033. No finalist cleared the DSR leg. The
  best DSR was QV50's at 0.43.
- **The paper and live configs are unchanged.** `validated: false` and
  `live_trading: false` stay as they are. No leverage was used.
- **The backtest look-ahead is fixed, and B50's edge mostly goes with it.**
  On point-in-time S&P 500 members, B50 earned +19.6%/yr in dev
  (2017 to 2025-11-21) with Sharpe 0.82. The old universe gave +31.4%
  with Sharpe 1.19, and SPY gave +14.8% with Sharpe 0.73. Honest B50's
  excess over SPY is +4.9%/yr, and its excess-vs-SPY DSR is 0.17 over
  dev + holdout. Honest B50 is the bar from now on.
- **The best new book is QV50, a QVAL-style value/quality sleeve.** It
  holds 50% SPY plus 20 names ranked on EBIT/TEV, GP/A and ROA.
  - Holdout: Sharpe 0.93 vs 0.44 for honest B50, MDD −8.5% vs −16.9%,
    +$15.6k vs +$11.2k on the $100k book. It beat 7 of the 10 proxies;
    VLUE, QVAL and DBMF still beat it.
  - Dev: Sharpe 0.85 vs 0.82, but AR +18.0% vs +19.6%.
  - It is not adopted. The DSR is 0.43, one year of holdout Sharpe has
    a standard error of about 1, and value's 2026 run was known before
    this design (see Caveats).
- **The LLM layer added nothing out of sample.**
  - None of the 16 LLM-proposed factors passed the yearly screen in any
    year (2017-2026). LF50 is therefore identical to B50.
  - The ChronoBERT "Lazy Prices" text filter lowered dev Sharpe from
    0.82 to 0.76.
- **The learning layer did what it was built to do: it protected the
  downside.** It did not add alpha.
  - HY_full (Hedge weights over the 5 sleeves, with stress reset and B50
    fallback) had a dev MDD of −31.5% vs −33.7%. Its Sharpe was 0.78 vs
    0.82.
  - On the holdout its Sharpe was 0.34. It spent 68% of 2026 in fallback
    (100% B50), so it paid switching costs without catching QV50's year.

## What was run (all GitHub Actions)

| run | job | what |
|---|---|---|
| 45505697747 | data `[p11:data]` | PIT membership (760 spells / 751 symbols, 503 members at 2026-11-23). Alpaca dump rebuilt from that universe (SIP, 750/752 symbols with bars). One SEC companyfacts pull: 744 issuers with facts, shares for 672. Artifact `phase11-inputs` (90 days). |
| 45507642664 | baseline `[p11:baseline]` | A_spy, B50_honest, B50_ever on dev. Every B50_honest pick written to `results/phase11/b50_honest_picks.csv`. |
| 45509003409 | text + llm `[p11:text] [p11:llm]` | 6,858 10-K Item 1A sections (685 issuers, 2015 to 2026), embedded with ChronoBERT-2016 on 4 CPU shards (about 3 h each). Qwen2.5-1.5B-Instruct factor proposals committed. The merge job first failed on account billing. It was re-run after the repo went public. |
| 45618514746 | dev `[p11:dev]` | 8 candidates × 9 yearly episodes + 10 proxies. Holdout sliced off before any signal was built. Runtime 122 s. |
| 45619828827 | holdout `[p11:holdout]` | The finalists QV50, LF50 and HY_full, with B50_honest, SPY and the proxies, scored **once** on 2025-11-24 to 2026-11-23. The script refuses a second run. |

**Trial accounting** (`results/holdout_log.jsonl`):
- Start: 769, the running total after phase 10.
- Baseline: +27 windows (3 books × 9 years), giving 796.
- Dev: +72 windows (8 candidates × 9 years), giving 868.
- Screen: +144 (16 LLM factors screened in each of 9 dev years), giving 1,012.
- Holdout: +5 windows and +16 screens for 2026, giving **1,033**.
- The proxies are benchmarks the brief imposed and were never selected,
  so they are not counted.

No GPU was needed, so there is no Kaggle MODELS string for this phase.

## Step 1: the point-in-time universe

**How it is built** (`pit.py`):
- Base: the fja05680/sp500 membership spells file, pinned to commit
  `a2430f2a` (its 2026-09-07 update).
- Carried forward to 2026-11-23 with the last Wikipedia revision of each
  day. A membership change counts only once it holds for 5 consecutive
  daily snapshots.
- 14 changes after the pin were applied, for example BE, ILMN and P in;
  TTD, TAP and BLDR out.
- On 2026-09-01, the pinned open set and the Wikipedia table agree
  exactly (`results/phase11/membership_audit.json`).
- Date added vs the pinned spell starts: they agree for 154 of the 169
  names added since 2016. The 15 that differ are mostly Wikipedia
  showing an older, pre-spinoff or ticker-reuse date. The pinned file is
  kept.

**Rule:** a name is pickable on signal day d only if start ≤ d < end.
The mom and vq ranks are taken among that day's members only.

**Test:** `tests/test_phase11.py` checks every committed B50_honest pick
(13,416 pick-rows) against the membership on its signal day, and it
passes with 0 violations. The synthetic tests also show two things. A
top-scoring name is first picked on its inclusion day. A ranking over
all ever-members is caught by the same check.

The old universe made 4,261 non-member pick-days. The largest were
DECK (703), SMCI (619), CVNA (491), ANET (377), APP (342) and TPL (340).

| book (dev 2017 to 2025-11-21) | P&L on $100k | AR | Sharpe | MDD | xs-SPY AR | DSR (n=796) |
|---|---|---|---|---|---|---|
| A_spy | $238,332 | +14.8% | 0.73 | −33.8% | 0.0% | 0.24 |
| **B50_honest** | **$383,481** | **+19.5%** | **0.81** | **−33.8%** | **+4.9%** | **0.32** (xs-SPY 0.00) |
| B50_ever (old universe) | $1,014,437 | +31.4% | 1.19 | −32.2% | +14.6% | 0.74 |

**Honest B50 on the holdout:** AR +11.3%, Sharpe 0.44, MDD −16.9%,
+2.9%/yr over SPY, $11,168 on $100k. Over dev + holdout: AR +18.7%,
Sharpe 0.78, DSR 0.35, excess-vs-SPY DSR 0.17 (n = 1,033).

The dev-stage numbers for B50_honest (below) differ from the step-1 run
by at most 0.23 pt in any year (+19.6% vs +19.5% AR). Between the two
runs the vq panels moved from `P9.load_vq_parts` to the shared
`fund_panels` loader. The values are the same, but the column order
differs, so equal composite scores are tie-broken differently. Every
phase-11 comparison below uses the dev-stage run.

**Pagination fix:** the `phase3-events` lookups in `phase4.yml` (2) and
`paper_trade.yml` (1) now use `--paginate ... | sort -n | tail -1` and
fail loudly when nothing matches. `tests/test_workflows.py` now fails
on any unpaginated artifact-list call.

## Step 3: the hybrid design

**The LLM never places a trade.** It produces text features and formula
proposals. The algorithm tests them, weights them and sizes the book
through the same simulator and caps.

**(a) Algorithmic core.** Every sleeve holds 50% SPY plus equal-weight
picks among PIT members, with a 20% name cap and a 2% band.
- **B50_honest:** ½ mom + ½ vq, 6 names.
- **QV50:** ½ rank(EBIT/TEV) + ½ mean rank(GP/A, ROA), 20 names. EBIT
  is `OperatingIncomeLoss` TTM. TEV is market cap + long-term debt −
  cash. All facts are SEC XBRL, used only once filed (`filed < d`).
- **BQ50:** ⅓ mom + ⅓ vq + ⅓ QV, 6 names.

**(b) LLM layer**, with free, local, pinned models:
- **Text:** the Item 1A risk-factor section of each 10-K, embedded with
  `manelalab/chrono-bert-v1-20161231`. TX50 drops names whose cosine
  similarity to their previous 10-K is in the day's bottom 20% (Cohen,
  Malloy & Nguyen 2020). A score is live from the day after filing for
  400 days. Coverage averaged 89% of members.
- **Hypotheses:** `Qwen/Qwen2.5-1.5B-Instruct` received the formula
  language and 6 themes. It never saw a ticker, a date or a return.
  - `factor_dsl` parses its text against a causal whitelist with no
    `eval`. Rejected: future shifts, unknown fields, windows over 252,
    more than 30 nodes.
  - Of 36 lines, 16 distinct formulas were kept and 20 rejected (prose,
    `^`, unknown fields such as `roce`, windows of 275 and 365).
  - Each January the algorithm screens every factor on month-start
    rank-IC vs the 21-day next-open return over the trailing 5 years,
    using only samples whose forward window closed before the decision.
    A factor needs t > 2 and correlation with the B50 composite below
    0.7. LF50 = ½ B50 + ½ mean rank of the factors accepted that year.

**(c) Learning layer.** HY_algo uses the first 3 sleeves; HY_full uses
all 5.
1. **Walk-forward retraining on a schedule.** The LLM-factor screen is
   re-run every January. The sleeve weights are re-estimated on the
   first trading day of every month.
2. **Weights shift toward what has been working.** Each sleeve has a
   cost-free shadow return (top-n at yesterday's close). Every month the
   weights become w ∝ exp(4 × trailing 126-day log return), shrunk 50%
   toward equal weights. This is a Hedge / exponentiated-gradient update,
   and the shrinkage reflects the forecast-combination puzzle.
3. **Regime detection.** When SPY is below its 200-day average AND
   21-day vol is above its trailing 80th percentile (10-day minimum
   hold), the weights reset to equal. This avoids chasing recent winners
   right after a crash (Daniel & Moskowitz 2016).
4. **Drift monitor and fallback.** If the hybrid's shadow return trails
   B50_honest's shadow by more than 5% over 126 days, the book switches
   to 100% B50_honest. It switches back once the gap is ≥ 0, and every
   switch holds at least 21 days.
5. The final mix of sleeve books trades through the phase-8 micro sim.
   `state/phase11/hybrid.json` records the end-of-sample state. On
   2026-11-23 the mode was stress → equal weights, with no LLM factors
   live and `live_effect: shadow`. No config reads it.

**How it behaved in dev** (`results/phase11/dev.json` → `meta`):
- Fallback to B50 was active, for HY_full / HY_algo: 57% / 71% of 2020,
  28% / 37% of 2022, 34% / 25% of 2023, 0% / 40% of 2024 and 92% / 95%
  of 2025.
- Stress resets took 48% of 2022 and 24% of 2020.
- Outside fallback the Hedge weights stayed close to equal (each sleeve
  between about 0.18 and 0.38), because of the 50% shrinkage.

### LLM look-ahead: what was done
- **Text scores:** these use ChronoBERT with a 2016-12-31 knowledge
  cutoff. Every signal date is 2017-01-03 or later, so the model cannot
  know any outcome it predicts. One vintage is used throughout.
- **Factor proposals:** Qwen2.5 was released 2024-09, so it has seen
  2017-2023. Its dev results are flagged `llm_dev_contaminated: true`
  in the log. Only the holdout, which falls after its release, tests
  them cleanly. That makes no difference here, because none of the
  factors passed the screen in any year, including the January 2026
  screen that applies to the holdout.
- **The researcher:** the brief quoted the phase-10 holdout returns
  (VLUE +42%, QVAL +29.5%) before this design existed. A value tilt that
  wins on this holdout is therefore weak evidence, and the full-sample
  DSR is the binding test.

## Step 4: results

### Dev walk-forward (2017-01-03 to 2025-11-21, holdout sealed, DSR at n_trials = 1,012)

Sharpe is computed on daily returns in excess of BIL. Costs are the
liquidity-bucket half-spread + 5 bps per side, with next-open fills.
The book is about $100k, and each yearly episode starts flat.

| book | idea | P&L on $100k | AR | Sharpe | MDD | xs-SPY AR | turnover/day | DSR |
|---|---|---|---|---|---|---|---|---|
| A_spy | benchmark | $238,332 | +14.8% | 0.73 | −33.8% | +0.0% | 0.002 | 0.34 |
| **B50_honest** | **honest B50 (the bar)** | **$384,542** | **+19.6%** | **0.82** | **−33.7%** | **+4.9%** | 0.043 | **0.45** |
| QV50 | (a) EBIT/TEV + GP/A + ROA, 20 names | $330,407 | +18.0% | 0.85 | −32.3% | +2.9% | 0.012 | 0.48 |
| BQ50 | (a) B50 + ⅓ QV tilt | $370,072 | +19.1% | 0.81 | −34.6% | +4.5% | 0.033 | 0.44 |
| LF50 | (b) B50 + screened LLM factors (none accepted, so = B50) | $384,542 | +19.6% | 0.82 | −33.7% | +4.9% | 0.043 | 0.45 |
| TX50 | (b) B50 minus the Lazy-Prices bottom 20% | $321,569 | +17.7% | 0.76 | −34.2% | +3.2% | 0.040 | 0.38 |
| HY_algo | (c) Hedge over B50/QV50/BQ50 | $293,269 | +16.8% | 0.77 | −30.4% | +2.0% | 0.030 | 0.39 |
| HY_full | (c) Hedge over all 5 sleeves (end model) | $309,823 | +17.3% | 0.78 | −31.5% | +2.6% | 0.029 | 0.40 |
| X_MTUM | proxy | $259,919 | +15.6% | 0.69 | −34.1% | +1.3% | — | — |
| X_QUAL | proxy | $222,003 | +14.1% | 0.69 | −34.1% | −0.5% | — | — |
| X_USMV | proxy | $146,695 | +10.8% | 0.61 | −33.1% | −4.1% | — | — |
| X_VLUE | proxy | $116,839 | +9.2% | 0.43 | −39.5% | −4.6% | — | — |
| X_DBMF | proxy, from 2019-05-09 | $45,031 | +5.9% | 0.31 | −20.0% | −9.8% | — | — |
| X_KMLM | proxy, from 2020-12-03 | $25,519 | +4.7% | 0.18 | −27.9% | −9.5% | — | — |
| X_QMOM | proxy | $168,543 | +11.8% | 0.47 | −40.5% | −0.5% | — | — |
| X_QVAL | proxy | $119,915 | +9.3% | 0.40 | −51.2% | −3.7% | — | — |
| X_SPY | proxy | $238,332 | +14.8% | 0.73 | −33.8% | +0.0% | — | — |
| X_6040 | proxy (60/40 SPY/AGG) | $127,980 | +9.8% | 0.68 | −21.9% | −5.4% | — | — |

The DSR hurdle is SR* = 0.86. The DSR values here are higher than in
the step-1 table because the trial-Sharpe dispersion now includes the
tightly clustered phase-11 books.

The short-history proxies were also scored on their own windows:
- From 2019-05-09 (DBMF): DBMF 0.31 / −20.0%; B50_honest 0.82 / −33.7%;
  QV50 0.79 / −32.3%.
- From 2020-12-03 (KMLM): KMLM 0.18 / −27.9%; B50_honest 0.78 / −23.5%;
  QV50 0.79 / −22.7%.

**Finalists (predeclared rule):** the top 2 by dev Sharpe with dev MDD
≥ B50_honest − 2pt, plus HY_full. That gave QV50, LF50 and HY_full.
LF50 got in because it ties B50_honest exactly. Since it is the same
book, it cannot beat B50 on the holdout.

### Per-year total return, dev (%)

| year | SPY | B50_honest | QV50 | BQ50 | TX50 | HY_algo | HY_full | B50_ever |
|---|---|---|---|---|---|---|---|---|
| 2017 | +20.5 | +26.8 | +23.3 | +35.3 | +23.2 | +24.2 | +26.3 | +36.8 |
| 2018 | −5.7 | −3.7 | −2.1 | −4.4 | −5.0 | −4.2 | −4.7 | +4.6 |
| 2019 | +32.0 | +28.4 | +37.9 | +30.5 | +29.4 | +25.5 | +28.0 | +30.5 |
| 2020 | +18.6 | +33.6 | +25.9 | +25.4 | +27.5 | +32.4 | +29.2 | +43.6 |
| 2021 | +30.7 | +19.8 | +34.7 | +31.5 | +30.5 | +24.9 | +25.0 | +27.0 |
| 2022 | −19.0 | −10.2 | −15.7 | −19.0 | −12.7 | −19.8 | −19.1 | −5.3 |
| 2023 | +25.9 | +33.7 | +30.0 | +34.3 | +21.9 | +31.1 | +32.3 | +44.2 |
| 2024 | +26.1 | +34.2 | +24.8 | +37.1 | +34.9 | +27.0 | +29.8 | +84.9 |
| 2025* | +13.1 | +19.7 | +11.5 | +13.4 | +16.5 | +19.2 | +18.5 | +29.6 |

\*2025 runs to 2025-11-21. The rest of 2025 is inside the holdout.

### Holdout: scored once (2025-11-24 to 2026-11-23, 251 days)

"Full" means dev + holdout. The full DSRs use **n_trials = 1,033**
(SR* = 0.91). The proxies' full columns are on their own windows.

| book | holdout P&L ($100k) | AR | Sharpe | MDD | xs-SPY AR | turnover/day | full AR | full Sharpe | full MDD | full DSR | full DSR xs-SPY |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_spy | $9,131 | +9.2% | 0.42 | −11.5% | 0.0% | 0.002 | +14.2% | 0.70 | −33.8% | 0.26 | — |
| **B50_honest** | **$11,168** | **+11.3%** | **0.44** | **−16.9%** | **+2.9%** | 0.046 | **+18.7%** | **0.78** | **−33.7%** | **0.35** | **0.17** |
| QV50 (finalist) | **$15,629** | **+15.8%** | **0.93** | **−8.5%** | +5.5% | 0.016 | +17.7% | 0.85 | −32.3% | 0.43 | 0.31 |
| LF50 (finalist, = B50) | $11,168 | +11.3% | 0.44 | −16.9% | +2.9% | 0.046 | +18.7% | 0.78 | −33.7% | 0.35 | 0.17 |
| HY_full (finalist, end model) | $8,771 | +8.8% | 0.34 | −15.7% | +0.5% | 0.044 | +16.4% | 0.74 | −31.5% | 0.30 | 0.06 |
| X_MTUM | $6,439 | +6.5% | 0.24 | −30.0% | +1.1% | — | +14.6% | 0.62 | −34.1% | — | — |
| X_QUAL | $12,223 | +12.3% | 0.65 | −9.0% | +2.6% | — | +14.0% | 0.68 | −34.1% | — | — |
| X_USMV | $8,577 | +8.6% | 0.53 | −6.5% | −1.2% | — | +10.5% | 0.60 | −33.1% | — | — |
| X_VLUE | $41,562 | +42.0% | 1.50 | −16.6% | +27.6% | — | +12.1% | 0.55 | −39.5% | — | — |
| X_DBMF | $15,828 | +16.0% | 0.95 | −6.7% | +5.6% | — | +7.2% | 0.39 | −20.0% | — | — |
| X_KMLM | $13,979 | +14.1% | 0.75 | −9.6% | +4.2% | — | +6.2% | 0.28 | −27.9% | — | — |
| X_QMOM | $1,983 | +2.0% | 0.08 | −23.9% | −4.3% | — | +10.8% | 0.43 | −40.5% | — | — |
| X_QVAL | $29,275 | +29.5% | 1.63 | −7.9% | +16.9% | — | +11.2% | 0.48 | −51.2% | — | — |
| X_SPY | $9,131 | +9.2% | 0.42 | −11.5% | 0.0% | — | +14.2% | 0.70 | −33.8% | — | — |
| X_6040 | $5,437 | +5.5% | 0.23 | −6.9% | −4.2% | — | +9.3% | 0.64 | −21.9% | — | — |

### Adoption checks (predeclared)

| leg | QV50 | LF50 | HY_full |
|---|---|---|---|
| 1. holdout Sharpe > B50_honest (0.44) | ✓ 0.93 | ✗ 0.44 (same book) | ✗ 0.34 |
| 2. full DSR ≥ 0.95 and ≥ B50_honest (0.35) | ✗ 0.43 | ✗ 0.35 | ✗ 0.30 |
| (info) proxies beaten on holdout Sharpe | 7/10 | 4/10 | 3/10 |
| **adopt** | **no** | **no** | **no** |

Honest B50 beats 4 of 10 proxies on holdout Sharpe (MTUM, QMOM, SPY,
60/40). So the goal of beating most quant/factor ETFs out of sample is
**not** met by the current book, and it is met by QV50 only on this one
year.

## Reading the results

1. **The fixed universe is the main finding.**
   - About two-thirds of B50's backtest excess over SPY came from picking
     future index members: +14.6%/yr falls to +4.9%/yr.
   - The 2024 "monster year" shrinks from +84.9% to +34.2%.
   - With 1,033 trials behind it, an honest +4.9%/yr excess with
     Sharpe 0.82 vs 0.73 is indistinguishable from luck (excess DSR 0.17).
2. **Value/quality is the one idea with a clean out-of-sample case.**
   - QV50 has lower turnover (0.012/day vs 0.043), shallower drawdowns
     and the best dev Sharpe.
   - It gives up about 1.6 pt/yr of AR in dev, mostly in 2020 and 2024,
     the momentum years.
   - Its holdout win came in a year when every value ETF won, so it is
     the textbook value premium, not a new edge.
   - Mixing it into B50 (BQ50) diluted both legs (dev Sharpe 0.81).
3. **LLM factors: no edge.**
   - The 1.5B model's ideas were mostly ratios of existing fields.
   - Trailing rank-ICs ranged from −0.06 to +0.10. The only t > 2 cases
     in the right direction were L13 (`ts_mean(bm, 10)`, t = 3.4) and
     L16 (t = 2.7) in January 2017, on 9-10 monthly samples, below the
     12-sample minimum. No factor reached t > 2 again in 2018-2026; the
     best was 1.7 in the January 2026 screen.
   - The screen rejecting all of them is the correct outcome. Allowing
     them in would have added 144+ trials of noise to the book.
4. **ChronoBERT Lazy Prices: it hurt.**
   - In the paper, the effect lives in small and mid caps and in
     long-short portfolios.
   - Here, cutting the most-changed fifth of S&P 500 filers from a
     6-name momentum sleeve removed some winners: 2023 fell from +33.7%
     to +21.9%.
5. **Learning layer.** It is an insurance policy, not an alpha source.
   - Fallback and stress resets cut dev MDD by 2-3 pt.
   - Hedge weights chase 6-month winners, and 6-month sleeve
     leadership did not persist (the forecast-combination puzzle again).

## Caveats

- **Re-viewed holdout.** Phase 10 already scored B50, SPY and the
  proxies on this window. It is fresh only for QV50, LF50 and HY_full.
  The value tilt was designed after the phase-10 ETF holdout returns
  were known.
- **One year of holdout** gives a Sharpe standard error of about ±1. The
  holdout can veto a candidate, but it cannot prove one.
- **Membership data.** The pinned file plus Wikipedia snapshots are free
  sources with known small date errors, and 15 "Date added" mismatches
  are listed in the audit. Six symbols have no SEC CIK (CBS, FRC, SBNY,
  VIAB, VIAC, WYND), so they lack vq/QV scores and are never picked.
  CCEP and WYND have no Alpaca bars.
- **Text coverage.** 554 10-Ks had no extractable Item 1A (for example
  filers that incorporate it by reference to an annual-report exhibit, or
  whose section has no recognisable end heading). Those firm-years have
  no text score and are never excluded by TX50.
- **The live bot is unchanged.** Its universe is already today's
  members, so it never had this look-ahead. Only the backtest
  overstated what to expect.
- **No live shadow cron was added for the hybrid.** A weekly shadow run
  would need the dump, SEC facts, new 10-Ks and embeddings each week.
  Since nothing won, I did not add that recurring Actions cost.
  `state/phase11/hybrid.json` is a one-off end-of-sample snapshot.

## Ops notes

- **Actions billing.** On 2026-11-25 every new job failed to start with
  "recent account payments have failed or your spending limit needs to
  be increased". That included 4 scheduled `paper-trade` runs
  (02:59-05:37 UTC). The 4 × 3 h embedding shards were the last large
  user. The repo was made **public** on 2026-11-25 at the owner's
  request, so standard runners are free. Before the switch, the full
  git history was scanned for keys, tokens and private keys, and none
  were found. Alpaca and Notion keys live only in Actions secrets.
- **Artifact expiry.**
  - `phase11-inputs` and `phase11-text-emb`: about 2027-02-22.
  - `phase11-text` and the embedding shards: 2026-12-02.
  - Re-running dev needs the first two. The holdout cannot be re-run.

## What would change the verdict

- **Paper out-of-sample time.** On an honest backtest, B50's case rests
  on +4.9%/yr. Months of live paper P&L against SPY are now the cheapest
  evidence.
- **QV50 as a risk-preference switch.** It has a similar dev Sharpe,
  shallower drawdowns, a quarter of the turnover and a strong holdout.
  It costs about 1.6 pt/yr in momentum years. Switching to it is a
  decision for Riyansh, not an out-of-sample win under the rule.
- **A text signal at the right size.** Lazy Prices on a wider,
  long-short or mid-cap universe is where the paper found the effect.
  It does not fit a long-only 6-name S&P 500 sleeve.
