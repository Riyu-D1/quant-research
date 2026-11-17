# Phase-8 audit — is the +27.8%/yr real?

Run: `python scripts/phase8_audit.py` → `results/phase8_audit.json`
(~9 min, all local on the committed honest feed `data/qlib/us_alpaca`).

Audited object: `B50_momvq` — 50% SPY core + 50% six-name equal-weight
momentum+value/quality sleeve, daily micro-book sim, 2017–2026
(2471 days), as recorded in `results/phase8_us_walkforward.json`.

## Verdict: partly real

The equity path is honest — it reproduces **bit-for-bit** and the
look-ahead/PIT machinery is clean. But the headline oversells what the
evidence supports:

- ~half the return is SPY beta by construction (50% core).
- The reported DSR=0.998 is computed on **total returns**, which tests
  "did a half-SPY book beat zero" — nearly tautological. On **excess
  returns vs SPY** the same test gives **0.36**: the sleeve's edge is
  not statistically distinguishable from selection noise at 392 trials.
- The edge is concentrated: APP+CVNA+SMCI = ~62% of 2024 book gain.
- It is fragile to sleeve breadth: `sleeve_n=8` collapses AR to +7.7%,
  *below* SPY (+14.9%) — and n=6 was chosen by the $1-account rule,
  never validated for breadth.
- No independent corroboration: the only second feed (community dump)
  disagrees sign-for-sign, attributable to data quality (below).

`configs/live_signal.yaml` demoted to `validated: false`. The paper
cron stays on — the paper book is itself the cheapest corroborating
out-of-sample evidence available.

## Item 1 — return arithmetic

All ten yearly ARs reproduce **exactly** (identical float64 bits).
Decomposition shows no double counting: book = sleeve_contrib +
core_contrib + cost drag; cash earns 0 (`cash_yield_in_code: false`);
there is no dividend leg to double-count — dividends live inside the
adjusted price series once. `unpriced_hold_days: 0`, 2053 trades, all
"rebalance".

One real defect, immaterial but worth fixing in the engine: the sim
can run marginally negative cash (min **−$1.23**, 78 days, max
executed weight-sum **1.072**). Buys are sized on pre-trade equity
targets; when a sold name freed less than the new pick's target
notional (sell at open×(1−cost) vs buy target on equity), cash drifts
a few cents negative. Alpaca would size-down/reject — so live can't
replicate those days exactly. ≈0.01% of book equity; does not explain
the return.

| year | B50 AR | SPY AR | xs | sleeve-only AR |
|------|--------|--------|-----|----------------|
| 2017 | +38.6% | +21.8% | +16.8 | +42.6% |
| 2018 |  +3.4% |  −5.0% |  +8.4 |  +7.6% |
| 2019 | +29.3% | +31.1% |  −1.8 | +20.7% |
| 2020 | +23.4% | +18.4% |  +5.0 | +30.5% |
| 2021 | +32.4% | +28.6% |  +3.8 | +26.6% |
| 2022 | −12.4% | −18.2% |  +5.8 |  +5.3% |
| 2023 | +46.9% | +26.4% | +20.5 | +49.8% |
| 2024 | +96.1% | +24.9% | +71.2 | +103.0% |
| 2025 | +34.0% | +17.9% | +16.1 | +35.9% |
| 2026 | +11.4% | +13.3% |  −1.9 |  +7.1% |

(Annualized-convention columns; xs is the arithmetic gap vs SPY.
`match_frac` 8/10, worst xs year 2026 −1.9pt — matches the gate.)

## Item 2 — look-ahead

20 seeded rebalance dates dumped pick-by-pick: momentum numerator/
denominator dates, every vq fact with its `filed` date, membership
windows, fill opens. **0 violations** — all inputs precede fills
(signal day t, next-open fill). Amendment probe: 30 versioned facts,
0 failures — the `fundamentals.py` fix (group by period end → sort by
`filed` → latest filing strictly before the signal date) is
structurally correct. The original whole-frame dedup bug it replaced
silently hid earlier visible versions (7361 cells, 56/60 sampled
symbols — see `PHASE8_DIAGNOSIS.md`).

## Item 3 — survivorship

248 dead names in the feed, 247 with real price histories; worst
tails preserved (FRC −97.5% over its final 60d, ARNC −93.9%, SIVB
−60.4% in a day, UTX −83.7%, TGNA −81.1%). **0 delist liquidations in
this sim** — B50 never held a name through its final day (membership
windows bound the tradable span; the sleeve churned out first). Dead
names are present and tradable while listed — the feed is honest —
but this particular path never stress-tested the delist-exit leg.

## Item 4 — why the live sleeve didn't change

Live sleeve stayed `APA FTI VLO MPC OGN LYB` because all six still
rank ≥0.90 on the **fixed** PIT vq — the buggy snapshot wasn't
inverting the ranking, just distorting it (old snapshot vs fixed PIT
spearman **0.764**). Rebuilt with fixed code at the same feed day the
correlation only rises to **0.793** — most of the gap was structural
(the snapshot ranks all-ever symbols incl. dead names on stale
fundamentals and last historical closes, the PIT panel ranks the
day's live universe), not the bug. Net live impact of the vq fix on
today's picks: small — the same six names stay top. Two genuine
live-vs-sim gaps surfaced:

- **Fixed here:** the snapshot was built 2026-11-06 with pre-fix code
  — rebuilt with the fix (asof 2026-11-13). Separately, the
  `refresh_snapshot` CI job pushed to `phase4/adaptive` — post-merge
  the bot checks out `master`; job retargeted in this PR.
- **Flagged:** the live universe contains demoted names (FTI left the
  index 2021-02, OGN 2023-10) with live Alpaca bars — the sim's
  membership-windowed dump can't score them there, so live picks can
  express exposures the validated sim never covered. Policy call, not
  a code bug (the all-listed universe is deliberate —
  `live_holdout.py` documents it); owner may want to restrict
  `live_universe.txt` to current members.
- Cosmetic: the snapshot ranks dead names on stale fundamentals (KSU,
  SCG, AIV all top-10). Harmless — the bot requires fresh bars — but
  it means raw snapshot top-6 ≠ candidate top-6.

## Item 5 — DSR

Stored recomputes exactly: DSR=0.998184 (SR_ann 1.2503, SR* 0.3043,
T=2471, skew −0.285, kurt 10.34, n_trials 392). But on excess-vs-SPY
returns: **DSR=0.3635** (SR_ann 1.050, SR* 1.162, skew +0.020, kurt
7.508). The stored number is internally consistent and externally
meaningless — a 50% SPY book inherits SPY's IR 0.87 for free, so
"beat the max-SR hurdle" is trivially cleared. The honest test says
the alpha is unproven, not that it's disproven.

## Item 6 — robustness (all walk-forward, SPY ref AR +14.9% IR 0.866)

| case | AR | IR | MDD | note |
|------|-----|------|------|------|
| base | +27.8% | 1.250 | −31.6% | reproduced |
| costs ×2 | +26.0% | 1.184 | −31.7% | robust |
| costs ×3 | +24.4% | 1.122 | −31.7% | robust |
| next-close fills | +28.4% | 1.269 | −31.9% | robust (slightly better) |
| sleeve_n=5 | +34.2% | 1.354 | −32.9% | stronger, more concentrated |
| **sleeve_n=8** | **+7.7%** | 0.896 | −17.3% | **below SPY** |

2024 attribution ($12.70-start book): APP +$3.00, CVNA +$2.83, SMCI
+$1.63 → top-3 = ~62% of the year's ~$12.1 gain; ERIE/GM/VLO/MPC/ABNB
all lost money. The 2024 monster year is real on this data but is
three names deep — not broad evidence of stable alpha.

## Item 7 — the community dump disagreement (wf IR +0.40)

Community B50 2012–2020: AR +8.1%, IR +0.40; overlap years show
opposite-sign excess vs the honest feed (2017 −8.8 vs +14.1pt, 2020
−24.1 vs +4.9pt). The disagreement is data, not fragility:

- Community prices are price-return only — its SPY made +70.1%
  2017–2020 vs +79.7% honest (~2.2%/yr missing dividends, more for
  the high-yield names vq favors).
- 248 one-day moves >40% in 2017–2020 (vs 37 honest) — corporate-
  action adjustment holes corrupt exactly the volatile names momentum
  picks.
- Coverage truncates at 2020-11-10; no membership windows (745 inst,
  1.01 windows/name) → a different, effectively static universe →
  different picks (sample date: 2/6 names shared).

So the community run neither confirms nor refutes the honest result —
but it removes the only second-source corroboration.

## Bugs found / fixed in this PR

| # | issue | status |
|---|-------|--------|
| 1 | `refresh_snapshot` job committed to `phase4/adaptive`; post-merge the bot reads `master` — refreshes would never reach the bot | **fixed** (job now targets master) |
| 2 | committed `vq_snapshot.parquet` was built 11-06 with the pre-fix vq code (0.764 vs fixed PIT; 0.793 after rebuild — gap is mostly peer-set) | **fixed** (rebuilt, asof 11-13) |
| 3 | `validated: true` claimed more than the evidence supports | **fixed** (false; re-run re-pins — see config comment) |
| 4 | sim allows negative cash / weight-sum ≤1.072 (−$1.23 max, 78 days) — live can't replicate | documented, immaterial; engine guard is owner's call |
| 5 | live universe admits demoted names the sim can't score (FTI/OGN) | flagged — policy decision on `live_universe.txt` |
| 6 | snapshot top ranks polluted by dead names (KSU/SCG/AIV) | cosmetic — fresh-bar gate filters them before picks |

## Bottom line

`+27.8%/yr` is **partly real**: the arithmetic and PIT plumbing are
honest and reproduce exactly; every gate leg cleared legitimately; the
result survives 3× costs and next-close fills. What is NOT supported
is the claim of a validated edge: the honest excess-DSR is 0.36, the
alpha is concentrated (one year, three names), breadth-fragile (n=8 <
SPY), and uncorroborated. `validated` set to **false**; the paper book
keeps running as the cheapest out-of-sample test available.
