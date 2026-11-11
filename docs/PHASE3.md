# Phase 3 — top-trader disclosure features for US stocks

Point-in-time features from public disclosures of "top traders": corporate
insiders (SEC Form 3/4/5), large fund managers (SEC 13F-HR), and members of
the US House of Representatives (Clerk periodic transaction reports). The
question: do these add signal over the Phase-2 price stack, which had no US
alpha (holdout IR −0.73, IC ≈ 0)?

## Honesty contract

- Feature at trading day **D** uses only filings whose information date is a
  trading day **strictly before D**. A filing on D is visible from the next
  trading day. Enforced by `_window_*`/`searchsorted` in
  `src/quant_research/events.py` and audited by
  `tests/test_phase3.py::test_no_lookahead_recompute`, which recomputes
  sampled real panel rows from the raw event log restricted to
  `filing_date < D`.
- Insider events key on **EDGAR filing date** (acceptance), never the
  transaction date. `insider_features` drops rows whose filing date fails to
  parse.
- 13F rows key on **filing date**, not report (quarter-end) date — filings
  lag the quarter by up to 45 days.
- Congressional rows key on the House **filing date** recorded in the Clerk
  index, not the transaction date.
- OOF folds are the same expanding windows as Phase 2 (2021–2024). The
  2025-01+ holdout is scored exactly once per candidate and appended to
  `results/holdout_log.jsonl` with the git SHA.

## Sources (all free)

| Signal | Source | Fetch | Known at |
|---|---|---|---|
| Insider txns | SEC Form 3/4/5 quarterly bulk datasets (`sec.gov/dera/data/form-345`) | `phase3_fetch.py insiders` | `SUBMISSION.FILING_DATE` |
| 13F holdings | EDGAR submissions API + `infotable.xml` per filing | `phase3_fetch.py f13f` | accession filing date |
| Shares out. | EDGAR companyfacts / submissions | `phase3_fetch.py shares` | filing date |
| Congress | House Clerk `disclosures-clerk.house.gov` FD index + PTR PDFs | `phase3_fetch.py house` | index filing date |

- **Senate**: `efdsearch.senate.gov` returned edge-level HTTP 403 (Akamai)
  from every tested environment — treated as unworkable. House PTRs only;
  no free Senate mirror with full 2021–2024 history was found acceptable
  (Capitol Stock Watch mirror covers only a recent ~6.9k-row window, not
  enough history for the OOF period). Documented limitation: congress
  features under-count Senate activity.
- **House PTR coverage**: ~50% of PTRs are scanned paper filings whose PDFs
  carry no text layer; `parse_ptr_text` returns [] and the doc is logged in
  `manifest.house_failed`. Coverage bias toward e-filers — documented, not
  corrected.
- **Symbol mapping**: `cik_map.parquet` maps universe symbols → issuer CIK
  via `company_tickers.json` (listed) plus Form 345 `ISSUERTRADINGSYMBOL`
  lookup inside each delisted symbol's PIT membership window (delisted).
  5 delisted tickers unresolved (CBS, FRC, SBNY, VIAB, VIAC) — they get
  zero event features.
- 13F issuer names → symbols via normalized-name matching + a small
  predeclared class-letter preference table (`_CLASS_PREF` in events.py).
  Unresolvable issuers contribute no features.

## Predeclared 13F fund list (fixed before any return inspection)

Chosen a priori as widely-followed institutional managers, spanning styles
(value, quant, activist, macro). Written into `scripts/phase3_fetch.py`
`FUNDS` before the first evaluation run:

Berkshire Hathaway, Renaissance Technologies, Citadel Advisors, Bridgewater
Associates, Two Sigma Investments, D.E. Shaw & Co, Millennium Management,
Point72 Asset Management, Baupost Group, Pershing Square Capital,
Tiger Global Management, Coatue Management, Viking Global Investors,
Lone Pine Capital, Soros Fund Management, Third Point,
Elliott Investment Management, AQR Capital Management,
ARK Investment Management, Appaloosa Management.

CIK corrections made during ingestion (before any feature evaluation):
Two Sigma 1475791→1179392, Coatue 1426753→1135730,
Viking 1103803→1103804, ARK 1579982→1697748,
Appaloosa 1006438→1656456 (legacy entity, 13F filings ended 2016).

## Features (windows are calendar days; `fd < D`)

Insider (`ins_*`): open-market buy/sell $ (90d), net $ (90d), distinct
buyers/sellers (90d), cluster = distinct P buyers (30d), role-weighted net
(CEO/CFO/chair 2.0, officer 1.5, director 1.0, 10% owner 0.75, other 0.5),
net $ / market cap (latest-filed shares × close), trading days since last
P filing. 10b5-1-flagged **sales** excluded (`AFF10B5ONE` + footnote);
flagged buys retained.

13F (`f13f_*`): # tracked funds holding, change vs 91d, Σ portfolio weight,
change vs 91d, # newly-appearing holders (91d). Latest-known per fund by
filing date; amendments supersede per (fund, report_date).

Congress (`cong_*`): # buys / sells (45d), signed $ at amount-range
midpoint (45d).

Rule baselines (`rule_*`, used as trade signals not model features):
`rule_cluster_hold20/60` (cluster ≥2 insiders held N trading days),
`rule_ins_buy_hold60`, `rule_cong_buy_hold60`.

## Candidate sets

- `events` — event features only → `lgb_ev`
- `alpha158_events` — Alpha158 left-joined with events → `lgb_ev158`,
  `xgb_ev158`
- `rule:*` — rule baselines via `--rules`

OOF strategies: `top5` (≈5 names — approximates the £10/5-name Alpaca
account) and `topk` (50/5, Phase-2 parity). Costs: commission-free + 10bps
per side spread/slippage (`open_cost`/`close_cost` 0.001).

## Paper-trading gate

Paper trading (`scripts/paper_trade.py`, dispatch-only workflow) is wired
ONLY if a candidate (a) beats the OOF bar, (b) has positive holdout return,
(c) holdout DSR > 0.5. Otherwise Phase 3 stops and reports. Credentials are
env-only (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `NOTION_TOKEN`,
`NOTION_DB_ID`) and the Alpaca endpoint is hard-guarded to paper.
