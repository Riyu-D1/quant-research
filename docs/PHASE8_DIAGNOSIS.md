# Phase-8 diagnosis — why the US picker failed out of sample

All numbers below are computed by `scripts/phase8_us.py diagnose` on the
honest Alpaca PIT feed (delisted names included, next-open fills,
liquidity-bucket half-spread + 5bps, $1 floor, 2% band, fractional).
Do not hand-edit numbers — regenerate from results/phase8_diagnosis.json.

## 1. Attribution of the 2026 loss

Failed config: B50_momvq = 50% SPY core + 6-name mom+vq equal sleeve.

Window 2026-01-02 → 2026-11-13: bot AR +0.129, IR +0.788, mdd -0.159; SPY AR +0.132, IR +0.952, mdd -0.089; xs-IR +0.023, xs-AR +0.002.

Recorded holdout (pre-fix vq, run on the same feed): xs-IR -1.187, AR +0.009, final $12.80 — the rerun above isolates the vq fix (eligibility unchanged): Δxs-IR +1.211, Δequity $+1.30.

### Per-stock P&L (sleeve, FIFO realised + unrealised, USD)

| stock | sector | total P&L | realised | unrealised | buy$ | sell$ | buys/sells |
|---|---|---|---|---|---|---|---|
| EXPE | Consumer Discretionary | -0.46 | -0.46 | +0.00 | 13.1 | 12.6 | 12/12 |
| EIX | Utilities | -0.30 | -0.30 | +0.00 | 4.6 | 4.3 | 4/4 |
| CVNA | Consumer Discretionary | -0.19 | -0.19 | +0.00 | 3.1 | 2.9 | 3/3 |
| APP | Communication Services | -0.17 | -0.17 | +0.00 | 5.4 | 5.2 | 5/5 |
| RL | Consumer Discretionary | -0.12 | -0.12 | +0.00 | 7.5 | 7.3 | 7/7 |
| ABNB | Consumer Discretionary | -0.05 | -0.05 | +0.00 | 1.1 | 1.1 | 1/1 |
| SPG | Real Estate | -0.05 | -0.05 | +0.00 | 5.8 | 5.7 | 5/5 |
| PSX | Energy | -0.05 | +0.00 | -0.05 | 1.2 | 0.0 | 1/0 |
| SNDK | Information Technology | -0.05 | -0.05 | +0.00 | 1.1 | 1.1 | 1/1 |
| VLO | Energy | -0.00 | +0.02 | -0.02 | 2.4 | 1.2 | 2/1 |
| GOOG | Communication Services | +0.01 | +0.01 | +0.00 | 5.4 | 5.5 | 5/5 |
| DVN | Energy | +0.05 | +0.05 | +0.00 | 2.1 | 2.2 | 2/2 |
| GM | Consumer Discretionary | +0.06 | +0.06 | +0.00 | 13.1 | 13.2 | 12/12 |
| DAL | Industrials | +0.11 | +0.11 | +0.00 | 3.4 | 3.5 | 3/3 |
| NEM | Materials | +0.15 | +0.15 | +0.00 | 4.4 | 4.6 | 4/4 |
| LYB | Materials | +0.27 | +0.16 | +0.11 | 6.6 | 5.7 | 6/5 |
| APA | Energy | +0.28 | +0.00 | +0.28 | 1.0 | 0.0 | 1/0 |
| MPC | Energy | +0.33 | +0.27 | +0.06 | 2.3 | 1.4 | 2/1 |
| KLAC | Information Technology | +0.89 | +0.89 | +0.00 | 1.1 | 1.9 | 1/2 |

### Sector exposure (sleeve weight-days)

| sector | Σ weight·days | pick-days |
|---|---|---|
| Consumer Discretionary | 28.6 | 343 |
| Communication Services | 21.0 | 252 |
| Energy | 17.5 | 210 |
| Information Technology | 12.6 | 151 |
| UNK:CELG | 12.5 | 150 |
| UNK:CSRA | 5.1 | 61 |
| Materials | 3.7 | 44 |
| UNK:BCR | 3.6 | 43 |
| Real Estate | 1.6 | 19 |
| Utilities | 1.2 | 15 |
| UNK:LLTC | 0.8 | 9 |
| Industrials | 0.7 | 8 |
| UNK:XRX | 0.5 | 6 |
| UNK:MXIM | 0.2 | 2 |
| UNK:PXD | 0.1 | 1 |

### Sector concentration vs stock selection (Brinson vs equal-weight universe)

Daily decomposition of the sleeve's return vs the same capital spread equal-weight across the universe (allocation = over/under-weight × sector EW return; selection = sleeve-weight × (pick EW return − sector EW return)), annualised:

| component | ann. contribution |
|---|---|
| sector allocation (concentration) | -0.0412 |
| stock selection within sectors | -0.0009 |

| sector | allocation | selection |
|---|---|---|
| Communication Services | -0.0036 | -0.0083 |
| Consumer Discretionary | -0.0190 | -0.0391 |
| Consumer Staples | -0.0049 | +0.0000 |
| Energy | -0.0013 | +0.0268 |
| Financials | -0.0070 | +0.0000 |
| Health Care | -0.0124 | +0.0000 |
| Industrials | -0.0034 | +0.0080 |
| Information Technology | +0.0111 | +0.0100 |
| Materials | +0.0041 | +0.0030 |
| Real Estate | +0.0012 | -0.0013 |
| Utilities | -0.0065 | +0.0002 |
| (241 sectors + singleton UNK:* below |x|<0.001, combined) | +0.0007 | +0.0000 |

### Factor exposures of picks (mean pct-rank of chosen names)

| factor | mean pick rank (0..1) |
|---|---|
| momentum (12-1) | 0.918 |
| value+quality | 0.949 |
| low-vol | 0.435 |
| low-beta | 0.507 |

Distinct sleeve names over window: 29; mean concurrent names: 6.0.


**Unfillable targets (phase-8 finding):** 272 pick-days / 22.7 weight·days of sleeve target sat on names with NO price that day — delisted names kept near the top by a stale signal leg (vq holds its last score; mom is NaN so the blend counts vq alone). The sim cannot fill them (the slot's cash sits idle); the live bot cannot pick them at all (its universe is listed names only). The phase-8 walk-forward requires a finite close on the pick day.


## 2. Signal decay — IC/RankIC by year (Spearman primary, Pearson alongside)

### forward horizon 5d

**mom** — roll-12m median rankIC +0.0254, roll-IC-IR +1.70, frac-12m>0 0.77, trend/yr +0.00092, first-half +0.0147 vs second-half +0.0273, years IC>0: 9

| year | rankIC | boot95% CI | pearson IC | n |
|---|---|---|---|---|
| 2017 | +0.0232 | [-0.017,0.066] | +0.0224 | 251 |
| 2018 | +0.0221 | [-0.016,0.064] | +0.0173 | 251 |
| 2019 | +0.0237 | [-0.040,0.085] | +0.0294 | 252 |
| 2020 | +0.0339 | [-0.045,0.103] | +0.0553 | 253 |
| 2021 | -0.0293 | [-0.072,0.013] | -0.0232 | 252 |
| 2022 | +0.0309 | [0.000,0.082] | +0.0357 | 251 |
| 2023 | +0.0009 | [-0.029,0.063] | +0.0050 | 250 |
| 2024 | +0.0565 | [0.008,0.090] | +0.0651 | 252 |
| 2025 | +0.0220 | [-0.026,0.062] | +0.0348 | 250 |
| 2026 | +0.0263 | [-0.035,0.091] | +0.0428 | 213 |

**vq** — roll-12m median rankIC +0.0004, roll-IC-IR +0.06, frac-12m>0 0.46, trend/yr +0.00380, first-half -0.0145 vs second-half +0.0081, years IC>0: 5

| year | rankIC | boot95% CI | pearson IC | n |
|---|---|---|---|---|
| 2017 | -0.0114 | [-0.038,0.022] | -0.0155 | 251 |
| 2018 | -0.0206 | [-0.044,-0.006] | -0.0235 | 251 |
| 2019 | -0.0194 | [-0.051,0.023] | -0.0133 | 252 |
| 2020 | -0.0273 | [-0.053,0.004] | -0.0305 | 253 |
| 2021 | +0.0062 | [-0.033,0.031] | +0.0083 | 252 |
| 2022 | +0.0226 | [-0.011,0.041] | +0.0232 | 251 |
| 2023 | -0.0105 | [-0.038,0.013] | -0.0086 | 250 |
| 2024 | +0.0040 | [-0.019,0.034] | +0.0019 | 252 |
| 2025 | +0.0108 | [-0.011,0.035] | -0.0008 | 250 |
| 2026 | +0.0135 | [-0.019,0.045] | -0.0008 | 213 |

**momvq** — roll-12m median rankIC +0.0127, roll-IC-IR +1.23, frac-12m>0 0.68, trend/yr +0.00363, first-half -0.0004 vs second-half +0.0251, years IC>0: 6

| year | rankIC | boot95% CI | pearson IC | n |
|---|---|---|---|---|
| 2017 | +0.0084 | [-0.016,0.039] | +0.0056 | 251 |
| 2018 | -0.0007 | [-0.033,0.026] | -0.0016 | 251 |
| 2019 | -0.0004 | [-0.024,0.032] | +0.0065 | 252 |
| 2020 | +0.0044 | [-0.043,0.048] | +0.0045 | 253 |
| 2021 | -0.0137 | [-0.050,0.013] | -0.0134 | 252 |
| 2022 | +0.0355 | [0.001,0.076] | +0.0357 | 251 |
| 2023 | -0.0076 | [-0.031,0.037] | -0.0095 | 250 |
| 2024 | +0.0436 | [0.008,0.070] | +0.0428 | 252 |
| 2025 | +0.0240 | [-0.006,0.049] | +0.0161 | 250 |
| 2026 | +0.0298 | [-0.006,0.070] | +0.0238 | 213 |

**ens_def** — roll-12m median rankIC -0.0027, roll-IC-IR -0.18, frac-12m>0 0.41, trend/yr +0.00011, first-half -0.0012 vs second-half +0.0021, years IC>0: 5

| year | rankIC | boot95% CI | pearson IC | n |
|---|---|---|---|---|
| 2017 | -0.0037 | [-0.040,0.042] | -0.0102 | 251 |
| 2018 | +0.0050 | [-0.020,0.047] | -0.0027 | 251 |
| 2019 | -0.0084 | [-0.074,0.068] | -0.0086 | 252 |
| 2020 | +0.0062 | [-0.079,0.086] | +0.0018 | 253 |
| 2021 | -0.0049 | [-0.068,0.052] | -0.0076 | 252 |
| 2022 | +0.0404 | [-0.019,0.097] | +0.0390 | 251 |
| 2023 | -0.0412 | [-0.103,0.040] | -0.0433 | 250 |
| 2024 | +0.0181 | [-0.022,0.046] | +0.0022 | 252 |
| 2025 | -0.0213 | [-0.093,0.043] | -0.0314 | 250 |
| 2026 | +0.0147 | [-0.036,0.060] | +0.0006 | 213 |

### forward horizon 21d

**mom** — roll-12m median rankIC +0.0230, roll-IC-IR +1.73, frac-12m>0 0.67, trend/yr +0.00216, first-half +0.0018 vs second-half +0.0359, years IC>0: 9

| year | rankIC | boot95% CI | pearson IC | n |
|---|---|---|---|---|
| 2017 | +0.0391 | [-0.032,0.100] | +0.0437 | 251 |
| 2018 | +0.0089 | [-0.055,0.075] | +0.0213 | 251 |
| 2019 | +0.0178 | [-0.090,0.112] | +0.0258 | 252 |
| 2020 | +0.0108 | [-0.117,0.116] | +0.0754 | 253 |
| 2021 | -0.0675 | [-0.143,0.016] | -0.0530 | 252 |
| 2022 | +0.0344 | [-0.021,0.143] | +0.0536 | 251 |
| 2023 | +0.0284 | [-0.027,0.089] | +0.0315 | 250 |
| 2024 | +0.0832 | [0.007,0.125] | +0.1275 | 252 |
| 2025 | +0.0058 | [-0.068,0.054] | +0.0654 | 250 |
| 2026 | +0.0275 | [-0.068,0.158] | +0.0662 | 197 |

**vq** — roll-12m median rankIC +0.0007, roll-IC-IR +0.11, frac-12m>0 0.47, trend/yr +0.00645, first-half -0.0174 vs second-half +0.0140, years IC>0: 5

| year | rankIC | boot95% CI | pearson IC | n |
|---|---|---|---|---|
| 2017 | -0.0183 | [-0.061,0.037] | -0.0242 | 251 |
| 2018 | -0.0436 | [-0.079,-0.014] | -0.0482 | 251 |
| 2019 | -0.0218 | [-0.069,0.050] | -0.0136 | 252 |
| 2020 | -0.0302 | [-0.077,0.020] | -0.0460 | 253 |
| 2021 | +0.0268 | [-0.039,0.078] | +0.0307 | 252 |
| 2022 | +0.0326 | [-0.015,0.077] | +0.0341 | 251 |
| 2023 | -0.0185 | [-0.050,0.023] | -0.0155 | 250 |
| 2024 | +0.0030 | [-0.034,0.051] | +0.0045 | 252 |
| 2025 | +0.0245 | [-0.004,0.067] | -0.0097 | 250 |
| 2026 | +0.0286 | [-0.034,0.074] | +0.0014 | 197 |

**momvq** — roll-12m median rankIC +0.0128, roll-IC-IR +1.36, frac-12m>0 0.52, trend/yr +0.00674, first-half -0.0143 vs second-half +0.0346, years IC>0: 6

| year | rankIC | boot95% CI | pearson IC | n |
|---|---|---|---|---|
| 2017 | +0.0152 | [-0.029,0.058] | +0.0165 | 251 |
| 2018 | -0.0328 | [-0.085,0.011] | -0.0195 | 251 |
| 2019 | -0.0118 | [-0.052,0.035] | -0.0016 | 252 |
| 2020 | -0.0159 | [-0.098,0.049] | -0.0165 | 253 |
| 2021 | -0.0263 | [-0.088,0.031] | -0.0170 | 252 |
| 2022 | +0.0440 | [-0.008,0.125] | +0.0524 | 251 |
| 2023 | +0.0056 | [-0.039,0.054] | +0.0043 | 250 |
| 2024 | +0.0614 | [0.006,0.096] | +0.0794 | 252 |
| 2025 | +0.0200 | [-0.028,0.059] | +0.0188 | 250 |
| 2026 | +0.0421 | [-0.012,0.109] | +0.0348 | 197 |

**ens_def** — roll-12m median rankIC -0.0131, roll-IC-IR -0.93, frac-12m>0 0.32, trend/yr +0.00039, first-half -0.0167 vs second-half -0.0017, years IC>0: 4

| year | rankIC | boot95% CI | pearson IC | n |
|---|---|---|---|---|
| 2017 | -0.0171 | [-0.079,0.051] | -0.0238 | 251 |
| 2018 | +0.0193 | [-0.039,0.095] | +0.0079 | 251 |
| 2019 | -0.0195 | [-0.123,0.087] | -0.0176 | 252 |
| 2020 | -0.0524 | [-0.194,0.067] | -0.0577 | 253 |
| 2021 | -0.0136 | [-0.116,0.083] | -0.0110 | 252 |
| 2022 | +0.0525 | [-0.027,0.169] | +0.0553 | 251 |
| 2023 | -0.0403 | [-0.166,0.075] | -0.0456 | 250 |
| 2024 | +0.0161 | [-0.041,0.056] | +0.0049 | 252 |
| 2025 | -0.0478 | [-0.171,0.064] | -0.0628 | 250 |
| 2026 | +0.0111 | [-0.067,0.082] | -0.0161 | 197 |


## 3. Overfitting — selection across the trials

- OOF winner's own OOF IR: +1.135; fresh-2026
  xs-IR: +0.023 (OOF xs-IR was
  +0.988).
- DSR on the 2026 daily series: 0.585
  (SR* +0.556, n_trials
  183).
- CSCV on the 22 phase-7 OOF candidates (12870 splits,
  S=16):
  - total-return metric: PBO 0.251, mean OOS relative rank
    0.344, frac OOS-top-quartile
    0.544;
  - excess-vs-SPY metric: PBO 0.001, mean OOS relative rank
    0.056, frac OOS-top-quartile
    0.968.

PBO = fraction of combinatorial splits in which the in-sample-best
candidate lands in the bottom half out-of-sample — the probability the
selection itself is overfit.


## 4. Data checks

| check | result |
|---|---|
| momentum look-ahead audit (all bars after 2021-06-08 NaN'd, signal recomputed) | identical = True (max |Δ| 0.00e+00) |
| vq look-ahead audit (fundamentals truncated to filed<2026-01-02, rebuild) | identical pre-cut = True (max |Δ| 0.00e+00); dup (dt,inst) rows 0 |
| survivorship | 751 instrument rows, 248 dead names kept in universe, 495 still trading at dump end 2026-11-13 |
| vq staleness | median last-filing age 63d, p90 944d, max 5986d; 91 symbols >120d stale; live snapshot age now 11d |
| corporate actions | 31 one-day >50% moves in dump; 0 delist-liquidations in the 2026 book |
| live vs sim vq | snapshot asof 2026-11-06; corr(live, PIT) 0.767 / rank-corr 0.779 over 702 names |

live vq = monthly snapshot cross-section; sim vq = daily PIT filed<D series — same math, different freshness. Live bars = Alpaca IEX vs sim SIP — same vendor, different feed; fills: live ~10:00 ET market order vs sim next-open (documented approximation).

**The vq audit caught a real bug.** The first run of this check failed:
truncating filings at `filed<2026-01-02` changed pre-cut scores by up to
~1.0 rank — `fundamentals._latest_point/_ttm` deduplicated each period
end across the WHOLE fact frame before the `filed<D` filter, so an
amendment filed later overwrote the visible value AND made the period
invisible until the amendment's date (7,361 versioned cells affected;
56/60 sampled symbols' book values differed). Fixed to PIT per-end
versioning (`filed<D` selects the version, then the latest end) — the
audit above is the post-fix rerun. The phase-7 recorded numbers used
the pre-fix vq; every phase-8 number in this doc uses the fixed one.


## 5. Small-account effects ($12.70 book)

- Sleeve budget caps the book at 6 names
  (floor($6.35 / $1)) — phase 6 already showed the
  edge wants ~30.
- 2026 book: base 151 trades, final equity
  $14.10; without the $1 floor
  183 trades (floor-suppressed:
  32), final
  $14.18 → $1-floor drag
  $+0.08; without the 2% band
  151 trades, final
  $14.10 → band effect
  $+0.00. The band is inert at this book size —
  a $1-minimum trade is already $1/$12.70 ≈ 7.9% of equity, past the
  2% band, so the floor always binds first.
- Breadth tracking error (same signal, cost-free, $1000 book so the $1
  floor cannot bind): 6-name sleeve AR +0.145 vs
  30-name AR +0.176; annualised tracking error
  6-vs-30 0.095.

## 6. Ranked root causes (from the numbers above)

Ranking is computed from the measured numbers: item 1 is the isolated
effect of the vq fix (recorded-vs-rerun delta on the same feed), items
2-3 share the 2026 sleeve-vs-equal-weight-universe return
decomposition (alloc + select = the gap exactly), items 4-7 are
structural causes measured on different axes (probabilities, USD
drag, IC trend) ordered by the strength of their evidence.


| # | root cause | key evidence |
|---|---|---|
| 1 | **vq point-in-time data bug** — later-filed amendments overwrote/hid the visible fundamental versions | recorded xs-IR -1.187 -> +0.023 after the fix (Δ +1.211); 7,361 versioned cells; every phase-7 number used the buggy signal |
| 2 | Sector concentration — the sleeve's over/under-weights vs an equal-weight book drove the loss | allocation -0.0412 ann vs EW universe — 98% of the alloc+select gap |
| 3 | Stock selection inside sectors — picks underperformed their own sector peers | selection -0.0009 ann vs EW universe — 2% of the gap |
| 4 | Selection overfitting — the OOF winner was one of 22 tried; in-sample winners decay OOS | PBO 0.25 total-ret / 0.00 xs; OOF xs-IR +0.988 -> 2026 +0.023; DSR 0.585 |
| 5 | Weak/unstable signal level — no decay measured, but the edge is thin — the mom leg's trailing rankIC | first-half +0.0018 vs second-half +0.0359, trend +0.00216/yr; latest year +0.0275 |
| 6 | Micro-account drag — $1 floor + 2% band on a $13 book | floor +0.08 USD, band +0.00 USD on $12.70; breadth TE 0.095 |
| 7 | Data caveats (vq staleness, snapshot-vs-PIT parity, survivorship) | see section 4 — 0.779 live-vs-PIT rank corr; 91 names >120d stale |
