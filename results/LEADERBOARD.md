<!-- BEGIN AUTO-GENERATED (make_table.py) -->
# Leaderboard — our runs vs published qlib benchmarks

Backtest: test period 2017-01-01→2020-08-01, TopkDropout(50,5),
open_cost 5bps / close_cost 15bps, deal at close. Returns are
excess vs benchmark *with* cost. Turnover = mean daily portfolio turnover.

## doubleensemble_alpha158_cn  (46.43 min, 871 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0506 | 0.0521 |
| icir | 0.4145 | 0.4223 |
| rank_ic | 0.0495 | 0.0502 |
| rank_icir | 0.4081 | 0.4117 |
| annualized_return | 0.1085 | 0.1158 |
| information_ratio | 1.2773 | 1.3432 |
| max_drawdown | -0.0631 | -0.0920 |
| turnover | 0.1936 | — |

## doubleensemble_alpha158_us  (122.91 min, 901 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0070 | — |
| icir | 0.0600 | — |
| rank_ic | 0.0079 | — |
| rank_icir | 0.0701 | — |
| annualized_return | -0.0048 | — |
| information_ratio | -0.0376 | — |
| max_drawdown | -0.2229 | — |
| turnover | 0.1190 | — |

## hist_alpha360_cn  (35.68 min, 871 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0528 | 0.0522 |
| icir | 0.3636 | 0.3530 |
| rank_ic | 0.0651 | 0.0667 |
| rank_icir | 0.4595 | 0.4576 |
| annualized_return | 0.0964 | 0.0987 |
| information_ratio | 1.3604 | 1.3726 |
| max_drawdown | -0.0650 | -0.0681 |
| turnover | 0.1598 | — |

## lightgbm_alpha158_cn  (191.38 min, 871 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0468 | 0.0448 |
| icir | 0.3816 | 0.3660 |
| rank_ic | 0.0490 | 0.0469 |
| rank_icir | 0.4067 | 0.3877 |
| annualized_return | 0.0807 | 0.0901 |
| information_ratio | 0.9145 | 1.0164 |
| max_drawdown | -0.0861 | -0.1038 |
| turnover | 0.1911 | — |

## lightgbm_alpha158_us  (16.99 min, 901 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0066 | — |
| icir | 0.0634 | — |
| rank_ic | 0.0047 | — |
| rank_icir | 0.0453 | — |
| annualized_return | -0.0035 | — |
| information_ratio | -0.0286 | — |
| max_drawdown | -0.2610 | — |
| turnover | 0.1340 | — |

## transformer_alpha158_cn  (14.45 min, 871 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0317 | 0.0264 |
| icir | 0.2538 | 0.2053 |
| rank_ic | 0.0435 | 0.0407 |
| rank_icir | 0.3626 | 0.3273 |
| annualized_return | 0.0323 | 0.0273 |
| information_ratio | 0.4738 | 0.3970 |
| max_drawdown | -0.0968 | -0.1101 |
| turnover | 0.1841 | — |

## transformer_alpha158_us  (17.58 min, 961 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0027 | — |
| icir | 0.0228 | — |
| rank_ic | 0.0064 | — |
| rank_icir | 0.0551 | — |
| annualized_return | -0.1053 | — |
| information_ratio | -1.1419 | — |
| max_drawdown | -0.5225 | — |
| turnover | 0.0531 | — |

## xgboost_alpha158_cn  (4.97 min, 871 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0500 | 0.0498 |
| icir | 0.3752 | 0.3779 |
| rank_ic | 0.0484 | 0.0505 |
| rank_icir | 0.3917 | 0.4131 |
| annualized_return | 0.0620 | 0.0780 |
| information_ratio | 0.7459 | 0.9070 |
| max_drawdown | -0.1149 | -0.1168 |
| turnover | 0.1867 | — |

## xgboost_alpha158_us  (14.52 min, 901 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0022 | — |
| icir | 0.0183 | — |
| rank_ic | 0.0028 | — |
| rank_icir | 0.0181 | — |
| annualized_return | 0.0017 | — |
| information_ratio | 0.0123 | — |
| max_drawdown | -0.3117 | — |
| turnover | 0.1781 | — |
<!-- END AUTO-GENERATED (make_table.py) -->

---

## Phase 2 — walk-forward holdout scores (single scoring, untouched window)

Different period than above, so not apples-to-apples: CN holdout is
2018-08-01→2020-08-01 (486 days); US holdout is 2025-01-01→2026-11 (459 days).
Each candidate below was scored exactly once; selection used OOF folds only.
Costs identical to Phase 1 (CN 5/15bps+5CNY min; US 10bps/side). DSR = deflated
Sharpe ratio over all recorded trials.

## phase2_rank_avg_ens_cn  (rank-avg lgb+xgb+de+hist @ TopkDropout(50,5))
| metric | ours | published ref |
|---|---|---|
| ic | 0.0355 | DoubleEnsemble 0.0521 |
| icir | 0.2723 | 0.4223 |
| rank_ic | 0.0546 | 0.0502 |
| annualized_return | 0.0878 | 0.1158 |
| information_ratio | 1.2191 | **1.3432** |
| max_drawdown | -0.0733 | -0.0920 |
| turnover | 0.1924 | — |
| deflated_sharpe | 0.438 (SR* 1.33) | — |
| n_trials | 39 (25 Optuna + 12 OOF candidates + 2 holdout) | — |
| verdict | DOES NOT beat published DE (1.2191 < 1.3432) — OOF IR 3.02 was regime-inflated | |

## phase2_ridge_stack_us  (ridge stack lgb+xgb+de @ TopkDropout(50,5))
| metric | ours |
|---|---|
| ic | 0.0001 |
| icir | 0.0011 |
| rank_ic | -0.00002 |
| annualized_return | -0.0691 |
| information_ratio | -0.7345 |
| max_drawdown | -0.2098 |
| turnover | 0.1415 |
| deflated_sharpe | 0.0001 (SR* 2.09) |
| n_trials | 37 (25 Optuna + 12 OOF candidates) |
| verdict | no harvestable alpha in the US Alpaca/PIT feed for the GBDT stack — consistent OOF (-0.266) → holdout (-0.735) |

---

## Phase 3 — top-trader disclosure features (US)

## phase3_rule_cluster_hold20_us  (insider cluster-buy ≥2 in 30d, hold 20d @ MicroTop5 £10≈$12.70 acct)
| metric | ours |
|---|---|
| ic | -0.0207 |
| icir | -0.0386 |
| rank_ic | -0.0290 |
| annualized_return | -0.1091 |
| information_ratio | -0.1408 |
| max_drawdown | -0.4731 |
| turnover | 0.0986 |
| deflated_sharpe | 0.061 (SR* 0.96) |
| n_trials | 29 (27 OOF candidates + 2 holdout) |
| verdict | insider/13F/congress features add NO harvestable alpha over the Alpaca feed — OOF winner (+0.244 on ~1% coverage) was noise; holdout gate FAILED → no paper trading |

---

## Phase 6 — benchmark chase (branch `phase6/benchmarks`)

US grid scored against SPY (not a qlib baseline — the phase-2 finding is
that the US GBDT stack has no alpha on this feed, so honest comparators
are SPY itself + realistic cost/fill discipline: 10bps/side, next-open
fills, PIT sp500 membership, delistings handled).

## phase6_us_vq_top30_h5  (value+quality composite, top-30, 5d hold)
| metric | ours | SPY |
|---|---|---|
| rank_ic (OOF 18-24) | -0.006 | — |
| annualized_return | 0.2404 | 0.1483 |
| information_ratio | **1.1671** | 0.8673 |
| max_drawdown | -0.2053 | -0.1875 |
| deflated_sharpe | 0.878 | — |
| n_trials | 77 (76 OOF + 1 holdout) | — |
| verdict | BROAD GATE PASSES (positive + beats SPY IR + DSR>0.5) at top-30; **deployable top5_h20 shape FAILS** (IR +0.559 < SPY +0.867) — edge lives in diversification a 5-name micro account can't hold. `validated` stays false |

OOF selection (76 predeclared trials, 2018-2024): `vq/top30_h5` won on
net IR +0.815. Selection, gate, and the top-5 feasibility check were all
predeclared in `configs/experiments/phase6_us.yaml` before the single
holdout look (2025-01→2026-11). Runner: `scripts/phase6_us.py`, run
43397877113 → `results/phase6_us_holdout.json`.

## phase6_ridge_stack_topk_cn  (9-model ridge stack @ TopkDropout(50,5), holdout re-viewed)
| metric | ours | best published |
|---|---|---|
| ic | 0.0414 | HIST α360 0.0522 / DE α158 0.0521 |
| icir | 0.2767 | HIST 0.3530 / DE 0.4223 |
| rank_ic | 0.0576 | HIST 0.0667 / **DE 0.0502 (beaten)** |
| rank_icir | 0.3933 | HIST 0.4576 / DE 0.4117 |
| annualized_return | 0.0991 | DE 0.1158 / HIST 0.0987 |
| information_ratio | 1.2231 | HIST 1.3726 / **DE 1.3432** |
| max_drawdown | -0.0615 | HIST -0.0681 / DE -0.0920 |
| turnover | 0.1911 | — |
| deflated_sharpe | 0.281 (SR* 1.63) | — |
| n_trials | 56 | — |
| verdict | DOES NOT beat published on IR (1.2231 < 1.3432); RankIC 0.0576 tops DE's 0.0502. OOF IR 3.44 was regime-inflated — same pattern as EX-023 (3.02→1.22). +5 models on the phase-2 stack bought ΔIR +0.004 |

Scored once on the re-viewed holdout 2018-08-01→2020-07-31 (486d; phase-2
holdout — second look, logged). Candidate chosen as best OOF IR after
re-computing the ensemble rows over all 8 base models (3.440); the run
fit ridge on 9 incl. `lgb_tuned` (holdout coef ~5e-5, immaterial). Run
43468264714 → `results/phase2_cn.json`, `results/holdout_log.jsonl`.

- **CN OOF** (run 43430705177): all 4 new CPU models delivered fold
  preds. OOF table (IR, topk): ridge_stack 3.440, ewic_avg 3.081, xgb
  2.822, lgb 2.779, lgb5 2.748, cat 2.742, rank_avg 2.704, lgb_tuned
  2.547 (30-trial tune UNDERPERFORMS base lgb), de 2.493, lrn 2.456,
  hist 2.323, lgb_rank 0.944. CatBoost needed `bootstrap_type: MVS`
  (Poisson is GPU-only).
- **GPU (owner-run, one Kaggle session)**: `MODELS =
  'tcn_cn_wf alstm_cn_wf tra_cn_wf igmtf_cn_wf tcn_us_wf'` — fills
  `results/oof/{cn,us}/` incl. holdout-fold preds via --train-holdout.

## phase6_ridge_stack13_topk_cn  (13-model ridge stack incl. tcn/alstm/tra/igmtf @ TopkDropout(50,5), holdout re-viewed)
| metric | ours | best published |
|---|---|---|
| ic | 0.0409 | HIST α360 0.0522 / DE α158 0.0521 |
| icir | 0.2758 | HIST 0.3530 / DE 0.4223 |
| rank_ic | 0.0563 | HIST 0.0667 / **DE 0.0502 (beaten)** |
| rank_icir | 0.3909 | HIST 0.4576 / DE 0.4117 |
| annualized_return | 0.0915 | DE 0.1158 / HIST 0.0987 |
| information_ratio | 1.1780 | HIST 1.3726 / **DE 1.3432** |
| max_drawdown | -0.0725 | HIST -0.0681 / DE -0.0920 |
| turnover | 0.1932 | — |
| deflated_sharpe | 0.130 (SR* 1.97) | — |
| n_trials | 64 | — |
| verdict | DOES NOT beat published on IR (1.1780 < 1.3432 < 1.3726) and is WORSE than the 9-model stack it extends (1.2231, ΔIR -0.045). GPU preds added noise, not lift; RankIC 0.0563 still tops DE's 0.0502. OOF IR 3.14 again regime-inflated (EX-023/EX-033 pattern) |

Scored once on the re-viewed holdout 2018-08-01→2020-07-31 (486d; third
look, logged). Candidate predeclared in
`configs/experiments/phase6_gpu_score.yaml`: OOF-IR argmax over
{ridge_stack@topk, tcn, alstm, tra, igmtf @topk} = ridge_stack@topk
(OOF IR 3.136; GPU solos 2.52/2.47/1.92/0.29). Kaggle v8 preds
(scriptVersionId 370197588, branch results/kaggle-wf-2026-11-16-1650).
Runs 43934368427 (cn-oof) + 43938177196 (cn-score) →
`results/phase2_cn.json`, `results/holdout_log.jsonl`.

---

## Phase 7 — US core+satellite live bot (branch `phase7/us-live`)

The deployable-shape question from phase 6, asked directly: can a ~£10
Alpaca book (fractional shares, $1 order floor, 2% rebalance band,
≤1 order/symbol/day, 20% stock cap w/ SPY overflow, bucketed
half-spread+5bps, next-open fills) match or beat SPY? 22 predeclared
core/satellite candidates (A: 100% SPY; B: 50/70/85% SPY core + ≤8-name
sleeve × {vq, mom+vq, ens_def}; C: B + sleeve→core regime guard; D:
sleeve-only top-8 invvol). Selection on OOF 2018–2024 ONLY; the fresh
2026 window scored once on the winner + SPY baseline.

## phase7_B50_momvq  (50% SPY core + 6-name mom+vq sleeve, equal wt)
| metric | OOF 18–24 | re-viewed 2025 | fresh 2026 | SPY 2026 |
|---|---|---|---|---|
| annualized_return | 0.179 | 0.247 | 0.009 | 0.133 |
| information_ratio | 0.943 | 1.037 | 0.137 | 0.965 |
| max_drawdown | -0.319 | -0.246 | -0.159 | -0.089 |
| excess-IR vs SPY | +0.440 | +0.646 | -1.187 | — |
| rolling-12m win rate | 0.608 | — | — | — |
| turnover (daily) | 0.030 | 0.022 | 0.028 | — |
| verdict | **OOF winner** (2/22 survived: this + C50_momvq IR 0.874) | beat SPY on all legs | **MATCH GATE FAILED — does not match or beat SPY** | — |

Selection (runs 43979807942 select / 43981216061 holdout →
`results/phase7_us_select.json`, `phase7_us_holdout.json`,
`holdout_log.jsonl` n_trials 155): predeclared rule = drop OOF mdd
deeper than SPY (-33.8%) or <50% rolling-12m wins, then best OOF IR.
2026 gate (IR ≥ SPY-0.05 or AR ≥ SPY-0.5pt, and mdd ≤ SPY+2pt) failed
on all legs; DSR 0.422. `validated` stays **false** — the deployed
`configs/live_signal.yaml` is B50_momvq, paper-only, honestly flagged.
If all signal candidates had failed, A_spy would have deployed by the
same rule.

---

## Phase 8 — Diagnose the 2026 failure + strict yearly walk-forward

Part 1 diagnosis (`docs/PHASE8_DIAGNOSIS.md`,
`results/phase8_diagnosis.json`) found the recorded phase-7 failure was
dominated by a **point-in-time fundamentals/vq bug**: the fact frame
deduplicated each report period before the `filed < D` cutoff, so later
amendments overwrote AND hid the visible version (7,361 cells; 56/60
sampled symbols differ). Rerun of the failed 2026 config on identical
feed+eligibility with fixed vq: excess-IR **+0.023 vs recorded -1.187**
(Δ +1.21 — the single largest measured cause). Sector concentration cost
-4.1pt/yr; dead-name targets wasted 272 pick-days; the $1 floor makes
the 2% band inert on a £10 book; OOF→fresh IR gap was selection noise
(CSCV PBO 0.25 total-ret / 0.00 excess).

Part 2 — strict expanding yearly walk-forward on the honest Alpaca PIT
feed (gate years 2017–2026; it only starts 2016-01-04), 11 predeclared
candidates (`configs/experiments/phase8_us.yaml`, committed before any
run). Deploy rule: per-year AR within 0.5pt of SPY in ≥60% of years AND
worst year ≥ -5pt AND OOS mdd ≤ SPY mdd +2pt → highest walk-forward IR;
else 100% SPY.

## phase8_B50_momvq  (same shape as phase 7, fixed pipeline)
| metric | wf OOS 17–26 | worst gate year | fresh 2026 | SPY wf |
|---|---|---|---|---|
| annualized_return | 0.278 | 2019: -0.012 xs | 0.114 | 0.149 |
| information_ratio | 1.250 | — | 0.683 | 0.865 |
| max_drawdown | -0.316 | — | -0.143 | -0.338 |
| match years | 8/10 (≥ -0.5pt) | worst xs -1.9pt | xs -0.9pt | — |
| verdict | **GATE PASSED — deployed, `validated: true`** | — | matched SPY | — |

Every enhancement candidate failed the worst-year leg (each fixed one
window while breaking another): C50_momvq -7.3pt, G50_gate -5.7pt,
N35_n8 -6.0pt, M50_month -11.2pt, S50_sn -12.0pt, S50_sec1 -13.1pt,
K50_ic -13.2pt, S50_sec2 -19.1pt, S50_sec2_ic -22.4pt. DSR 0.998
(n_trials 392 incl. every scored window). The gate picked the plain
phase-7 shape — the mechanisms stay in the code but none deploys.

**Supplementary years 2012–2020 (biased community dump, never gated):**
B50 wf IR +0.400, xs negative 2016–2020 — the survivorship-biased feed
does **not** corroborate; the only supporting evidence is the honest
feed. Caveat: gate years 2018–2024 overlap the window B50's *shape* was
phase-7-selected on (with the buggy signal, so fixed-vq scores are new
information, but the reuse is disclosed). `live_trading: false` always.
