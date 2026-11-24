# Phase 10: hunt for a higher-benchmarking quant model

Spec (predeclared, committed before any run, plus one pre-run amendment):
`configs/experiments/phase10_us.yaml` · harness: `scripts/phase10_us.py` ·
building blocks: `src/quant_research/multiasset.py` · workflow:
`.github/workflows/phase10.yml` · tests: `tests/test_phase10.py` ·
results: `results/phase10_dev.json`, `results/phase10_holdout.json`,
`results/phase10/dev_daily.csv`, `results/phase10/*_coverage.json`.

## Verdict: keep B50_momvq. Nothing passed the adoption rule.

- **Adopted: none.** The three dev finalists were OV1_b50_crash,
  PF3_b50_tr50 and PF2_b50_tr30, picked by the predeclared rule. All
  three lost to B50 on **holdout Sharpe**. None reached the **DSR ≥ 0.95**
  bar at n = 769 trials. None beat a majority of the investable proxies
  on the holdout.
- **The paper and live configs are unchanged.** `configs/paper_signal.yaml`
  and `configs/live_signal.yaml` are untouched. `validated: false` and
  `live_trading: false` stay as they are, and the live cron stays
  commented. No leverage or leveraged product is adopted.
- **The best risk-reducer is PF3 (50% B50 + 50% multi-asset trend).**
  On the holdout it had about the same Sharpe as B50 (0.43 vs 0.45) with
  **half the drawdown** (−8.5% vs −16.9%). Over dev + holdout its MDD
  was −17.4% vs −32.3%, but AR was 16.5% vs 28.4%. That is a
  risk-preference option for Riyansh. It is not an out-of-sample win,
  so it was not adopted (see "What would change the verdict").
- **New finding: B50's backtest has a look-ahead in its universe.** The
  walk-forward universe is every name that was *ever* in the S&P 500
  over 2016-2026. So a 2024 decision could pick names that only joined
  the index later. SMCI joined on 2024-03-18, APP on 2025-09-22 and CVNA
  on 2025-12-22, and those are the three names that carried 2024. If
  picks are restricted to names that were members **on the signal day**
  (B50_pit, the same book otherwise), dev AR falls from **+30.4% to
  +19.2%**, Sharpe from 1.16 to 0.79 (SPY 0.73), and 2024 from +83.7%
  to +29.1%. The excess-vs-SPY DSR over dev + holdout falls from 0.41
  to 0.01. Most of B50's headline edge over SPY comes from this
  membership look-ahead. Live trading is not affected (today's universe
  is today's list), but the backtest overstates what to expect.
- **B50 on the holdout:** AR +11.6%, Sharpe 0.45, MDD −16.9%, vs SPY
  +9.2%, 0.42, −11.5%. It beat SPY on Sharpe but not on drawdown, and it
  beat only 4 of the 10 investable proxies on Sharpe (MTUM, QMOM, SPY,
  60/40). QUAL, USMV, VLUE, DBMF, KMLM and QVAL all had better holdout
  Sharpes. Value had an exceptional year (VLUE +42%, QVAL +29.5%).

## What was run (all GitHub Actions, nothing heavy on the Mac)

| run | job | what |
|---|---|---|
| 45427955640 | data (`[p10:data]`) | Rebuilt the Alpaca PIT dump through `sample_end` 2026-11-23 (SIP, 744/746 symbols with bars). Fetched 28 ETFs (adjustment=all, SIP, none missing). Artifact `phase10-inputs` (90-day retention). |
| 45441623367 | dev (`[p10:dev]`) | 21 candidates × 9 yearly episodes (2017 … 2025-11-21), with the holdout sliced off before any signal was built, plus 10 proxies. Finalists picked by the predeclared rule. Runtime 221 s. |
| 45442758014 | holdout (`[p10:holdout]`) | The 3 frozen finalists + B50, B50_pit and SPY scored **once** on 2025-11-24 … 2026-11-23 (251 days), plus the 10 proxies. The script refuses a second run. |

**Trial accounting:** n_trials = **769** = 574 (running total through
phase 9) + 189 dev windows (21 candidates × 9 years, including B50, SPY,
B50_pit, discarded candidates and the leverage sensitivities) + 6
holdout windows. The proxies are benchmarks the brief imposed and were
never selected, so they are not counted as trials. The dev table uses
n = 763 and the holdout/full-sample DSRs use n = 769.

The `claude` CLI still returns `Invalid API key · Please run /login`,
so it was skipped. No GPU was needed, so there is no Kaggle MODELS
string for this phase.

## 1. Literature shortlist: which ideas have credible post-publication evidence?

Grades follow `docs/PHASE9_LIT_REVIEW.md`:
- **A**: credible out-of-sample or post-publication evidence over long
  samples or many markets, with realistic costs.
- **B**: mixed evidence.
- **C**: in-sample or narrow evidence only.

| family | key references | grade | note for this book |
|---|---|---|---|
| Multi-asset time-series momentum / trend | Moskowitz, Ooi & Pedersen (2012, JFE); Hurst, Ooi & Pedersen (2017, JPM) "A century of evidence on trend-following"; Hamill, Rattray & Van Hemert (2016) "crisis alpha"; critique: Huang, Li, Wang & Zhou (2020, JFE) "Time-series momentum: is it there?" | **A/B** | The long-sample evidence is strong, and 2022 was a live out-of-sample win. There was a weak decade in 2012-2019. The headline results use **levered futures** on 50+ markets with shorting. A long-only, unlevered ETF version on 8 assets keeps the crisis hedging but drops most of the return. |
| Dual momentum / GEM | Antonacci (2014) *Dual Momentum Investing*; Faber (2007, updated 2013) "A quantitative approach to tactical asset allocation" | **B** | This is practitioner work with little independent OOS testing. It whipsaws in V-shaped selloffs (2018 Q4, 2020, 2025). Faber's own post-2007 update shows lower drawdowns than buy-and-hold but similar or lower returns. |
| Cross-asset risk parity + vol target | Maillard, Roncalli & Teiletche (2010); Asness, Frazzini & Pedersen (2012, FAJ); Roncalli (2013) | **B** | It works when the stock-bond correlation is negative. Without leverage it is bond-heavy and low-return. 2022 (stocks and bonds falling together) was its worst year on record. |
| Sector / industry momentum | Moskowitz & Grinblatt (1999, JF); Asness, Moskowitz & Pedersen (2013, JF) | **B** | Industry momentum explains a large share of stock momentum. Long-only ETF rotation has weaker post-publication evidence. |
| Low volatility / residual vol | Ang, Hodrick, Xing & Zhang (2006, JF); Blitz & van Vliet (2007); Frazzini & Pedersen (2014, JFE) BAB; critique: Novy-Marx (2016) | **A/B** | It works in long samples and many markets as a *defensive* factor with lower beta. In a 2017-2026 bull market a low-beta book trails on raw return. Gross profitability / QMJ was **not** tested: the repo's XBRL pull has no `GrossProfit`/`Assets` tags, so it would need a new SEC pull (flagged below). |
| Short-term reversal | Jegadeesh (1990, JF); Lehmann (1990, QJE); costs: Avramov, Chordia & Goyal (2006, JF); de Groot, Huij & Zhou (2012, JBF) | **B** (gross) / **C** (net, retail) | The gross effect is strong, but it lives in small and illiquid names. In large caps it survives costs only with very careful construction. Expect costs to eat it. |
| PEAD | Bernard & Thomas (1989); decay: Martineau (2022, CFR) "Rest in peace post-earnings announcement drift" | **B → C** | It was strong historically. Martineau finds it has largely disappeared in large caps since the mid-2000s. Free data only gives 10-Q **filing** dates, not announcement dates, which makes the signal later and weaker. |
| Portfolio of strategies | Markowitz; Asness, Moskowitz & Pedersen (2013) value + momentum; trend as a crisis diversifier (Hamill et al. 2016) | **A** (as theory) | Uncorrelated sleeves can raise Sharpe and cut MDD even when the new sleeve has lower AR. It is the most plausible route to a risk-adjusted win here. |
| Drawdown / crash overlays | Faber (2007); Daniel & Moskowitz (2016, JFE); Moreira & Muir (2017, JF); critique: Cederburg et al. (2020, JFE) | **B** | This is a risk overlay, not alpha. Real-time vol and trend timing mostly cuts tail losses at the cost of whipsaw. Phase 9 tested sleeve-only versions routed to SPY. Phase 10 tests a *whole-book* de-risk into T-bills. |
| Investable proxies (ADDENDUM) | MTUM, QMOM (momentum); QUAL (quality); USMV (min vol); VLUE, QVAL (value); DBMF (SG CTA replication); KMLM (Mount Lucas trend); 60/40 SPY/AGG | — | These are free benchmarks. DBMF trades from 2019-05-08 and KMLM from 2020-12-02 on Alpaca, so they are scored only on overlapping windows (stated below). |

**Shortlist tested** (all predeclared in the spec, with nothing added
after data was seen):
- TS1 12m TSMOM, TS2 1/3/12m trend blend, TS3 Faber GTAA-5
- DM1 GEM, DM2 dual-momentum top-3
- RP1 ERC, RP2 ERC + trend filter
- SR1/SR2 sector top-3 (relative / + absolute filter)
- LV1 50 lowest residual-vol names
- ST1 5-day reversal sleeve, PE1 SUE sleeve
- PF1 70/30 B50 + RP1, PF2 70/30 B50 + TS2, PF3 50/50 B50 + TS2
- OV1 B50 + crash overlay
- LX1/LX2 1.5× leverage sensitivities

## 2. Holdout protocol (pre-registered)

- **Sample:** 2016-01-04 … 2026-11-23, fixed in the spec, and the data
  job rebuilds the dump with `--end 2026-11-23`.
- **Holdout:** **2025-11-24 … 2026-11-23** (the last 12 months, 251
  trading days). `dev` slices the S&P feed, the ETF bars, the SPY
  series and the calendar to days before 2025-11-24 *before* any signal
  is built, and it asserts that cut. `holdout` reads the finalists from
  `phase10_dev.json`, scores them once, and exits if
  `phase10_holdout.json` already exists. Nothing was changed between
  the dev and holdout runs (both are empty commits on the same code).
- **Dev:** yearly episodes 2017-2025, with the book starting flat each
  1 January (the phase 8/9 convention). Look-backs are continuous. The
  holdout is one episode that starts flat on 2025-11-24.
- **One simulator for everything:** the phase-8 micro sim on explicit
  weight targets.
  - Book: ~$100k paper. Fills at the next open.
  - Costs: liquidity-bucket half-spread plus 5 bps per side.
  - Trading rules: $1 floor, 2% band, delist liquidation.
  - The ETF bars are appended to the feed the same way SPY is.
  - Long-only cash account. There is no shorting, no CFDs and no
    leveraged ETFs. "Cash" is BIL (1-3m T-bills), which pays its own
    spread. Single stocks are capped at 20%; ETFs are funds and exempt,
    like the SPY core.
- **Metrics:**
  - Sharpe is computed on daily returns **in excess of BIL**.
  - P&L is in $ on the $100k book, compounded over the window. Dev P&L
    chains the yearly episodes.
  - DSR (Bailey & López de Prado) uses the cumulative n_trials, with the
    trial-Sharpe dispersion taken from the phase-10 candidates.
- **Finalist rule (dev only):**
  - Exclude SPY, B50, diagnostics and leverage sensitivities.
  - Keep candidates whose dev MDD is no worse than B50's dev MDD
    (−32.3%) minus 2pt. LV1 (−37.5%), PE1 (−35.8%) and ST1 (−49.3%)
    failed this.
  - Take the top 3 by dev Sharpe.
- **Adoption rule:** all six legs must hold.
  1. Holdout Sharpe beats both B50's and SPY's.
  2. Holdout MDD and full-sample MDD are no worse than B50's minus 2pt.
  3. Dev Sharpe is at least B50's.
  4. Full-sample DSR is at least 0.95 and at least B50's DSR at the
     same n.
  5. Holdout Sharpe beats at least 6 of the 10 proxies.
  6. It is not a leverage sensitivity.

## 3. Results

### Dev walk-forward (2017-01-03 … 2025-11-21, holdout sealed, n_trials = 763)

| book | idea | P&L on $100k | AR | Sharpe | MDD | xs-SPY AR | turnover/day | DSR |
|---|---|---|---|---|---|---|---|---|
| A_spy | benchmark | $238,332 | +14.8% | 0.73 | −33.8% | 0.0% | 0.002 | 0.23 |
| **B50_momvq** | **current** | **$945,293** | **+30.4%** | **1.16** | **−32.3%** | **+13.9%** | 0.039 | **0.70** |
| B50_pit | B50, PIT members (diagnostic) | $373,433 | +19.2% | 0.79 | −36.3% | +4.7% | 0.042 | 0.30 |
| TS1_tsmom12 | 1 TSMOM 12m | $50,928 | +4.8% | 0.48 | −9.0% | −10.6% | 0.008 | 0.07 |
| TS2_trend3h | 1 trend 1/3/12m | $42,793 | +4.1% | 0.47 | −5.9% | −11.3% | 0.012 | 0.07 |
| TS3_gtaa5 | 1 Faber GTAA-5 | $46,105 | +4.4% | 0.31 | −17.4% | −10.8% | 0.009 | 0.03 |
| DM1_gem | 1 GEM | $107,566 | +8.6% | 0.45 | −33.8% | −5.8% | 0.010 | 0.06 |
| DM2_dual_top3 | 1 dual mom top-3 | $141,961 | +10.5% | 0.73 | −21.9% | −4.8% | 0.009 | 0.24 |
| RP1_erc | 2 ERC, 10% vol | $61,728 | +5.6% | 0.45 | −18.8% | −9.7% | 0.003 | 0.06 |
| RP2_erc_trend | 2 ERC + trend | $47,079 | +4.5% | 0.45 | −8.4% | −10.9% | 0.007 | 0.06 |
| SR1_sector3 | 3 sector top-3 | $199,604 | +13.2% | 0.63 | −31.5% | −1.2% | 0.011 | 0.16 |
| SR2_sector3_abs | 3 + abs filter | $173,929 | +12.1% | 0.59 | −31.5% | −2.3% | 0.012 | 0.13 |
| LV1_resvol50 | 4 residual low-vol | $182,789 | +12.5% | 0.64 | −37.5% | −2.2% | 0.007 | 0.17 |
| ST1_rev5_50 | 5 5d reversal | $91,233 | +7.6% | 0.34 | −49.3% | −5.4% | **0.216** | 0.03 |
| PE1_sue_50 | 5 PEAD (SUE) | $190,593 | +12.8% | 0.63 | −35.8% | −1.7% | 0.016 | 0.15 |
| PF1_b50_rp30 | 6 70% B50 + 30% RP1 | $514,273 | +22.8% | 1.11 | −26.7% | +6.7% | 0.028 | 0.66 |
| PF2_b50_tr30 | 6 70% B50 + 30% TS2 | $510,358 | +22.7% | 1.17 | −23.3% | +6.5% | 0.030 | 0.72 |
| PF3_b50_tr50 | 6 50% B50 + 50% TS2 | $315,044 | +17.5% | 1.18 | **−17.4%** | +1.5% | 0.025 | 0.72 |
| OV1_b50_crash | 7 crash overlay | $767,135 | +27.7% | **1.23** | −28.9% | +11.0% | 0.040 | **0.77** |
| ⚠ LX1_erc_x15 | **LEVERAGE 1.5× (sensitivity)** | $64,538 | +5.8% | 0.38 | −24.4% | −9.3% | 0.004 | 0.04 |
| ⚠ LX2_trend_x15 | **LEVERAGE 1.5× (sensitivity)** | $47,507 | +4.5% | 0.44 | −8.8% | −10.9% | 0.015 | 0.06 |
| X_MTUM | proxy | $259,919 | +15.6% | 0.69 | −34.1% | +1.3% | — | — |
| X_QUAL | proxy | $222,003 | +14.1% | 0.69 | −34.1% | −0.5% | — | — |
| X_USMV | proxy | $146,695 | +10.8% | 0.61 | −33.1% | −4.1% | — | — |
| X_VLUE | proxy | $116,839 | +9.2% | 0.43 | −39.5% | −4.6% | — | — |
| X_DBMF | proxy, from 2019-05-09 | $45,031 | +5.9% | 0.31 | −20.0% | −9.8% | — | — |
| X_KMLM | proxy, from 2020-12-03 | $25,519 | +4.7% | 0.18 | −27.9% | −9.5% | — | — |
| X_QMOM | proxy | $168,543 | +11.8% | 0.47 | −40.5% | −0.5% | — | — |
| X_QVAL | proxy | $119,915 | +9.3% | 0.40 | −51.2% | −3.7% | — | — |
| X_SPY | proxy | $238,332 | +14.8% | 0.73 | −33.8% | 0.0% | — | — |
| X_6040 | proxy (60/40 SPY/AGG) | $127,980 | +9.8% | 0.68 | −21.9% | −5.4% | — | — |

On the short-history proxies' own windows, B50 and the finalists still
lead on Sharpe:
- From 2019-05-09 (DBMF): DBMF 0.31 / −20.0%; B50 1.15 / −32.3%; PF3
  1.19 / −17.4%.
- From 2020-12-03 (KMLM): KMLM 0.18 / −27.9%; B50 1.26 / −26.8%; PF3
  1.26 / −15.7%.

### Per-year total return, dev (%)

| year | SPY | B50 | B50_pit | OV1 | PF2 | PF3 | TS2 | RP1 | DM2 | SR1 | LV1 | PE1 | ST1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2017 | +20.5 | +37.4 | +28.2 | +37.4 | +27.8 | +21.4 | +6.6 | +11.4 | +20.3 | +13.6 | +18.9 | +22.9 | +5.1 |
| 2018 | −5.7 | +3.7 | −5.4 | +3.6 | +2.0 | +0.9 | −2.5 | −3.9 | −13.7 | −6.5 | −3.9 | −12.7 | −12.3 |
| 2019 | +32.0 | +28.9 | +28.6 | +25.2 | +22.1 | +17.0 | +5.6 | +15.2 | +11.4 | +26.9 | +32.7 | +32.8 | +27.2 |
| 2020 | +18.6 | +32.9 | +28.2 | +25.9 | +25.3 | +19.6 | +5.8 | +7.2 | +18.2 | +14.8 | +12.1 | +11.4 | +41.4 |
| 2021 | +30.7 | +31.6 | +29.1 | +31.6 | +22.6 | +17.0 | +3.6 | +5.3 | +17.9 | +20.7 | +29.0 | +35.4 | +15.0 |
| 2022 | −19.0 | −1.6 | −10.4 | −1.8 | −0.7 | +0.8 | −0.6 | −12.8 | −2.8 | +11.8 | −3.4 | −18.2 | −24.7 |
| 2023 | +25.9 | +46.1 | +36.1 | +46.1 | +32.0 | +24.1 | +3.9 | +7.4 | +4.3 | +10.1 | +8.8 | +21.1 | +14.5 |
| 2024 | +26.1 | +83.7 | +29.1 | +83.7 | +57.7 | +43.0 | +5.1 | +8.5 | +20.2 | +17.1 | +14.0 | +21.4 | +15.4 |
| 2025* | +13.1 | +23.2 | +16.4 | +11.4 | +20.6 | +15.7 | +9.3 | +14.3 | +23.1 | +11.3 | +7.6 | +12.4 | +0.9 |

\*2025 runs to 2025-11-21. The rest of 2025 is inside the holdout.

### Holdout: scored once (2025-11-24 … 2026-11-23, 251 days)

This table also gives the full sample (dev + holdout), with DSR at
**n_trials = 769**.

| book | holdout P&L ($100k) | holdout AR | holdout Sharpe | holdout MDD | turnover/day | xs-SPY AR | full AR | full Sharpe | full MDD | full DSR |
|---|---|---|---|---|---|---|---|---|---|---|
| A_spy | $9,131 | +9.2% | 0.42 | −11.5% | 0.002 | 0.0% | +14.2% | 0.70 | −33.8% | 0.59 |
| **B50_momvq** | **$11,506** | **+11.6%** | **0.45** | **−16.9%** | 0.048 | +3.3% | **+28.4%** | **1.09** | −32.3% | **0.93** |
| B50_pit (diagnostic) | $11,008 | +11.1% | 0.43 | −16.9% | 0.048 | +2.8% | +18.4% | 0.76 | −36.3% | 0.66 |
| OV1_b50_crash (finalist) | $6,699 | +6.8% | 0.24 | −15.8% | 0.053 | −1.3% | +25.4% | 1.13 | −28.9% | 0.94 |
| PF3_b50_tr50 (finalist) | $8,374 | +8.4% | 0.43 | **−8.5%** | 0.030 | −1.1% | +16.5% | 1.10 | **−17.4%** | 0.93 |
| PF2_b50_tr30 (finalist) | $9,580 | +9.7% | 0.43 | −12.2% | 0.037 | +0.5% | +21.3% | 1.10 | −23.3% | 0.93 |
| X_MTUM | $6,439 | +6.5% | 0.24 | −30.0% | — | +1.1% | | | | |
| X_QUAL | $12,223 | +12.3% | 0.65 | −9.0% | — | +2.6% | | | | |
| X_USMV | $8,577 | +8.6% | 0.53 | −6.5% | — | −1.2% | | | | |
| X_VLUE | $41,562 | +42.0% | 1.50 | −16.6% | — | +27.6% | | | | |
| X_DBMF | $15,828 | +16.0% | 0.95 | −6.7% | — | +5.6% | | | | |
| X_KMLM | $13,979 | +14.1% | 0.75 | −9.6% | — | +4.2% | | | | |
| X_QMOM | $1,983 | +2.0% | 0.08 | −23.9% | — | −4.3% | | | | |
| X_QVAL | $29,275 | +29.5% | 1.63 | −7.9% | — | +16.9% | | | | |
| X_SPY | $9,131 | +9.2% | 0.42 | −11.5% | — | 0.0% | | | | |
| X_6040 | $5,437 | +5.5% | 0.23 | −6.9% | — | −4.2% | | | | |

Other full-sample DSRs at n = 769:
- **Excess vs SPY:** B50 0.41, OV1 0.10, PF2 0.04, PF3 0.00, B50_pit 0.01.
- **Active vs B50:** all ≤ 0.0005.

The total-return DSR hurdle is SR* = 0.62 (excess of BIL). Each
holdout Sharpe comes from one year of data, so its standard error is
about ±1.0. The holdout can veto a candidate, but it cannot prove one.

### Adoption checks (predeclared rule)

| leg | OV1_b50_crash | PF3_b50_tr50 | PF2_b50_tr30 |
|---|---|---|---|
| 1. holdout Sharpe > B50 (0.45) | ✗ 0.24 | ✗ 0.43 | ✗ 0.43 |
| 1. holdout Sharpe > SPY (0.42) | ✗ | ✓ | ✓ |
| 2. holdout MDD ≥ B50 − 2pt (−18.9%) | ✓ −15.8% | ✓ −8.5% | ✓ −12.2% |
| 2. full MDD ≥ B50 − 2pt (−34.3%) | ✓ | ✓ | ✓ |
| 3. dev Sharpe ≥ B50 (1.16) | ✓ 1.23 | ✓ 1.18 | ✓ 1.17 |
| 4. full DSR ≥ 0.95 and ≥ B50 (0.926) | ✗ 0.940 | ✗ 0.932 | ✗ 0.931 |
| 5. ≥ 6/10 proxies on holdout Sharpe | ✗ 3/10 | ✗ 4/10 | ✗ 4/10 |
| **adopt** | **no** | **no** | **no** |

B50 itself beats only 4/10 proxies on holdout Sharpe. So the ADDENDUM
goal (beat most quant/factor proxies on risk-adjusted out-of-sample
P&L) is **not** met by the current book either, on this one year.

### Leverage sensitivity (FLAGGED, not adoptable)

LX1 and LX2 allow gross exposure up to 1.5× under the same 10% vol
target. Borrowed exposure is financed at BIL + 1.5%/yr. Over dev:
- LX1 (ERC) averaged 0.9-1.5× gross. It *lowered* the Sharpe (0.45 →
  0.38) and deepened MDD (−18.8% → −24.4%), because 2022 hit levered
  bonds and stocks together.
- LX2 (trend) rarely levers: its signal-weighted gross averaged 0.2-0.8.
  Its result is within noise of TS2's.

Leverage does not rescue the unlevered multi-asset sleeves, and per the
brief and the ADDENDUM it would not be adopted anyway. No CFDs and no
leveraged ETFs were designed or tested.

## 4. Reading the results, family by family

1. **Trend / TSMOM / dual momentum.**
   - The long-only, unlevered ETF versions do what the literature
     predicts. Drawdowns are tiny (TS2 −5.9%, RP2 −8.4%) and returns
     are T-bill-plus (4-5% AR). With a 10% vol target, no leverage and
     switched-off assets parked in T-bills, the books hold roughly 25-85%
     cash. In 2022, when SPY fell −19%, TS2 lost
     only −0.6%.
   - On their own they are not competitive with an equity book. Their
     value is as a **diversifier**, which is why PF2/PF3 reached the
     final.
   - GEM whipsawed in 2018, 2020 and 2022.
   - DM2 (top-3 dual momentum) was the best stand-alone ETF strategy
     (Sharpe 0.73, MDD −21.9%). It tied SPY's Sharpe on lower AR.
2. **Risk parity.** Unlevered ERC is bond-heavy. 2022 (−12.8%) was its
   worst year, consistent with the stock-bond correlation flip. The
   trend filter cut MDD from −18.8% to −8.4% at the same Sharpe.
3. **Sector rotation.** SR1 earned +11.8% in 2022 (energy), but its
   Sharpe of 0.63 is below SPY's. The absolute filter did not help.
4. **Low residual vol (PIT universe).** Sharpe was 0.64, below SPY. Its
   MDD of −37.5% is deeper than SPY's, because the 2020 crash hit
   low-vol names too. It failed the finalist MDD screen.
5. **Short-term reversal.** Costs ate it, as expected. Turnover was
   0.216/day (5.5× B50's), with Sharpe 0.34 and MDD −49%.
   **Rejected.**
   **PEAD (SUE at 10-Q filing, PIT universe)** reached Sharpe 0.63 with
   MDD −35.8%, which is not better than SPY. That matches Martineau
   (2022): large-cap PEAD is largely gone, and filing dates lag
   announcements. **Rejected.**
6. **Portfolio of strategies.** This is the best risk-adjusted family.
   - Adding the trend sleeve cut B50's drawdown roughly in proportion:
     −32.3% → −23.3% (30% trend) → −17.4% (50% trend). Sharpe held up
     or rose slightly in dev (1.16 → 1.17/1.18).
   - On the holdout the Sharpe was 0.43 vs 0.45. The diversifier kept
     its drawdown promise (−8.5% vs −16.9%) but did not add return.
   - The ERC diversifier (PF1) was weaker than trend in dev (Sharpe
     1.11).
7. **Crash overlay (risk overlay, not alpha).**
   - In dev, OV1 de-risked half the book into T-bills when SPY was below
     its 200-day SMA *and* 21-day vol was in the top quintile. That
     helped in 2020 (shallower drawdown) and was the top dev Sharpe
     (1.23).
   - It cost money in the 2025 tariff V-rebound (+11.4% vs +23.2%) and
     again in the holdout (Sharpe 0.24 vs 0.45). This is the classic
     whipsaw that Cederburg et al. describe.

## 5. Caveats

- **B50's universe look-ahead** (above) also inflates the dev numbers of
  every B50-based book: OV1, PF1-3, and the "B50" column itself. The
  stand-alone ETF strategies and the PIT single-name books (LV1, ST1,
  PE1) do not have this bias. So the comparison in dev is tilted
  *toward* B50 and its combos. In the holdout the bias is small, since
  APP was a member by then and only CVNA joined during it. The fix, if
  wanted:
  - Score the paper book's backtests on PIT membership (B50_pit).
  - Consider whether the paper bot's `live_universe.txt` should hold
    only current members. It already does, apart from the demoted names
    flagged in phase 8.
- B50, B50_pit and SPY over 2025-2026 are **re-viewed** windows: phase 9
  scored them, and the holdout_log rows say so. The holdout is fresh
  only for the new phase-10 candidates. I had not seen 2026 ETF data
  before the holdout run.
- One year of holdout is short, so the Sharpe standard error is about
  1. The DSR at n = 769 is the stricter test, and no candidate clears
  0.95.
- **Value/quality artifacts:** B50 (vq leg) and PE1 read the
  `phase4-inputs` SEC fundamentals (pulled 2026-11-11). Those expire
  around **2027-02**, and so does the `phase10-inputs` artifact (90-day
  retention). Refresh with `phase4.yml job=fetch` before then, or the
  B50 retrain and any phase-10 re-run will fail loudly.
- **ETF history is Alpaca SIP from 2016,** so 12-month look-backs first
  bind in early 2017. Trend has no pre-2016 test here. The long-sample
  trend evidence is from the literature, not this repo.
- **Simplifications:**
  - PEAD uses the first 10-Q filing date and seasonal net-income
    surprises. There are no Q4 events, because 10-Ks rarely carry
    3-month facts.
  - Sharpe uses BIL as the risk-free rate.
  - Uninvested cash earns 0 in the sim (books that want cash hold BIL).
  - The ETF strategies are long-only, so there are no ETF shorts and no
    borrow costs to model.
- Gross profitability / QMJ was not tested. It needs `GrossProfit`,
  `CostOfRevenue` and `Assets` XBRL tags that the phase-4 pull did not
  fetch. That is a new SEC pull on Actions.

## 6. Required engineering (this PR)

- **`.github/workflows/pytest.yml`:** runs the test suite on every pull
  request, on push to `master`, and on dispatch. It uses the locked uv
  environment and skips the heavy optional stacks, as the phase
  workflows do.
- **Pagination fix in `phase6.yml`, `phase7.yml` and `phase8.yml`.**
  The old artifact lookup `--paginate -q '[...] | max'` printed one
  value per page. It now streams ids as
  `... | .id' | sort -n | tail -1`, and a lookup that matches nothing
  now fails with `artifact X not found`. `tests/test_workflows.py`
  checks every workflow for a per-page reduction under `--paginate`.
  The test fails on the old phase-6/7/8 lines and passes on the new
  ones.
- **Flagged, not changed:** `phase4.yml` (2 lookups) and
  `paper_trade.yml` (1 lookup, `phase3-events`) call
  `gh api .../artifacts` **without** `--paginate`. They only see the
  newest 30 artifacts, so they will fail once 30 newer artifacts exist.
  The paper bot's workflow was left alone in this PR because it is
  production. The fix is the same one-line change.

## 7. What would change the verdict

- **Paper out-of-sample evidence** is still the cheapest evidence. B50's
  full-sample DSR is 0.93 at n = 769, and that rises without new trials
  if the paper book keeps earning.
- **If a smoother book is preferred:** PF3 (50% B50 + 50% TS2
  multi-asset trend, unlevered, all ETFs/stocks, no shorting) is the
  documented option.
  - Dev Sharpe was 1.18 vs 1.16 and full-sample MDD −17.4% vs −32.3%.
  - Holdout Sharpe was 0.43 vs 0.45 and holdout MDD −8.5% vs −16.9%.
  - The cost is about 12 points/yr of AR over the full sample.
  - Switching is a risk-preference decision for Riyansh, not an
    out-of-sample win, so it was not made here.
- **Honest expectations:** on a PIT-membership universe, B50's backtest
  edge over SPY falls to about +4.7%/yr in dev (Sharpe 0.79 vs 0.73).
  Any future comparison should use B50_pit as the like-for-like anchor.
- **The next idea with credible OOS evidence** is still "Lazy Prices"
  10-K text changes (phase-9 note). A gross-profitability / QMJ leg
  from a fuller XBRL pull is next after that.
