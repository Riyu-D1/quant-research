# Phase-9 literature scan — what has credible out-of-sample evidence?

Scope: ideas that could improve B50_momvq (50% SPY + 6-name 12-1
momentum + PIT value/quality sleeve, daily rebalance, S&P 500 universe)
using **free data** (Alpaca bars + news headlines from 2020-09, SEC
EDGAR) and **free compute** (GitHub Actions CPU, Kaggle GPU, local or
free models only).

Method note: the brief allowed Riyansh's Claude Max via the `claude`
CLI as a research assistant. The CLI is installed but returned
`Invalid API key · Please run /login` on this machine, so it was not
used. To use it, Riyansh needs to run `claude` then `/login` once. This
scan is my own. I checked the LLM-era citations (2023-2026) by web search
(arXiv/SSRN/journal pages). The classic factor citations are standard
references.

Grades. **A** means credible out-of-sample evidence: post-publication or
independent replication, long samples or many markets, and realistic
costs. **B** means mixed: it works in some samples, but implementable
versions are contested. **C** means in-sample only, short-window, or
narrow-universe backtests with no credible OOS evidence yet.

## 1. Factor and risk improvements

| idea | key papers | grade | verdict for this book |
|---|---|---|---|
| Cross-sectional momentum | Jegadeesh & Titman (1993, JF); Asness, Moskowitz & Pedersen (2013, JF) "Value and Momentum Everywhere" | **A** | Survives decades and asset classes out of sample. The sleeve's core leg is sound. |
| Momentum crashes; vol-scaled momentum | Daniel & Moskowitz (2016, JFE); Barroso & Santa-Clara (2015, JFE) "Momentum has its moments" | **A/B** | Scaling momentum by its own trailing vol roughly halves crash risk in long samples. The evidence is strongest for long-short WML. A long-only sleeve has less crash exposure to remove. **Shortlisted** (vol overlay). |
| Volatility-managed portfolios (general) | Moreira & Muir (2017, JF); critique: Cederburg, O'Doherty, Wang & Yan (2020, JFE 138:95-117) | **B** | Across 103 strategies, real-time vol-managed versions do **not** beat the unmanaged ones out of sample. The spanning-regression alphas are not implementable. Expect little or nothing from the overlay beyond the momentum-specific effect. |
| Value, profitability / quality | Fama & French (1993; 2015); Novy-Marx (2013, JFE); Asness, Frazzini & Pedersen (2019, RAS) "Quality minus junk" | **A** (quality) / **B** (value since 2007) | Profitability and quality replicate well. Value had a long post-2007 drawdown. A combined vq leg is defensible. |
| Post-publication decay, replication | McLean & Pontiff (2016, JF): about 26% lower out of sample, about 58% lower post-publication; Hou, Xue & Zhang (2020, RFS): most anomalies fail with microcaps excluded; Jensen, Kelly & Pedersen (2023, JF): most factors replicate under a Bayesian view | **A** (as a warning) | Haircut any backtest by at least 1/3 to 1/2. Large-cap, value-weighted momentum and quality survive best. |
| Factor timing | Asness (2016) "The siren song of factor timing"; Haddad, Kozak & Santosh (2020, RFS) "Factor timing" | **B** | Valuation-spread timing is weak. Haddad et al. find predictability in the dominant PCs, not in single-factor switching. Treat aggressive timing with suspicion. |
| Factor momentum / adaptive factor weights | Gupta & Kelly (2019, JPM) "Factor momentum everywhere"; Ehsani & Linnainmaa (2022, JF) "Factor momentum and the momentum factor" | **A/B** | Factors with positive trailing returns tend to keep earning. This is the most credible basis for recent-IC-weighted factor blends. Gains are modest and need many factors. **Shortlisted** (IC-adaptive weights). |
| Breadth / concentration, 1/N | DeMiguel, Garlappi & Uppal (2009, RFS); Grinold (1989) fundamental law | **A** (as theory) | Estimated optimal weights rarely beat 1/N out of sample. IR grows with √breadth when the signal holds, and 6 names is very concentrated. **Shortlisted** (breadth sweep + inverse-vol weights). |
| Risk parity / inverse-vol | Asness, Frazzini & Pedersen (2012, FAJ) "Leverage aversion and risk parity"; Maillard, Roncalli & Teiletche (2010) | **B** | Strong across asset classes. Inside a single-equity sleeve it mostly tilts toward low-vol names, which is a separate factor bet. Tested as R50_n15_ivol. |
| Trend / regime filter | Faber (2007) 10-month SMA; Moskowitz, Ooi & Pedersen (2012, JFE) TSMOM; Hurst, Ooi & Pedersen (2017) century of trend | **A** (multi-asset) / **B** (single-equity overlay) | Trend works as a diversified multi-asset strategy. As an on/off filter on one equity index it mostly cuts drawdowns at the cost of whipsaw. Phase 7/8 already tested a binary guard (C50: AR 24.7% < B50, failed gate). Phase 9 tests a graded version. |
| Regime switching | Ang & Bekaert (2002, RFS); Hamilton (1989) | **B** | Regimes are easy to identify in hindsight and hard to identify in real time. Gains are small after the detection lag. |

## 2. Modern ML

| idea | key papers | grade | verdict |
|---|---|---|---|
| Trees / NNs on characteristics | Gu, Kelly & Xiu (2020, RFS) "Empirical asset pricing via machine learning" | **A/B** | Robust OOS R² gains in a 30-year US panel. Most of the long-short spread comes from small and illiquid names and from monthly rebalancing of the whole cross-section. |
| ML vs economic restrictions | Avramov, Cheng & Metzker (2023, Mgmt Sci) "Machine learning vs. economic restrictions" | **A** (as a warning) | ML profits shrink sharply when microcaps, distressed and unrated firms are excluded and costs are charged. An S&P 500 universe is exactly that restricted setting, so expect little. |
| China A-shares | Leippold, Wang & Zhou (2022, JFE) | **B** | Gains are concentrated in retail-heavy small caps. They do not transfer to the S&P 500. |
| Purged / embargoed CV, CPCV | López de Prado (2018) *Advances in Financial ML* ch. 7 and 12 | method | Necessary for any learned model with overlapping labels. Used for the GBM hyper-parameter pick. |
| Deflated Sharpe, PBO | Bailey & López de Prado (2014, JPM); Bailey, Borwein, López de Prado & Zhu (2017, J. Comp. Finance) | method | DSR over **all** trials (392 prior + every phase-9 window) is the adoption bar. |
| "Virtue of complexity" | Kelly, Malamud & Zhou (2024, JF) | **B** | Over-parameterised market-timing regressions; contested (e.g., Nagel 2025 critiques). Not relevant to a 6-name selection problem. |
| Time-series foundation models | Chronos (Ansari et al. 2024), TimesFM (Das et al. 2024); evaluations: "Re(Visiting) Time Series Foundation Models in Finance" (arXiv 2511.18578) | **C** | Zero-shot TSFMs generally do poorly on daily returns. Gains over a random walk are small and sparse. Excluded. |
| Our own history | phases 2-7 of this repo: Alpha158 GBDT stacks, deep nets (TRA/ALSTM/HIST) | **C** here | US holdouts failed: EX-022 GBDT stack IR −0.73; phase-7 ML failed on 2026. Any GBM test must beat that prior. |

**Shortlisted:** LightGBM on the *live-computable* factor ranks, refit
monthly with purged CV. The features are a small, economically motivated
set (not Alpha158), and the trees act as a non-linear factor combiner.
Low expectations.

## 3. LLM / NLP sentiment and text

| idea | key papers | grade | verdict |
|---|---|---|---|
| Media sentiment predicts returns | Tetlock (2007, JF); Loughran & McDonald (2011, JF) finance dictionary | **A** | Short-horizon (days) predictability with fast reversal. Strongest in small caps. |
| FinBERT | Araci (2019, arXiv 1908.10063); Huang, Wang & Yang (2023, CAR) "FinBERT: A large language model for extracting information from financial text" | **A/B** | Beats dictionaries at labelling sentiment. Its training data predates 2020, so there is **no look-ahead into a 2021+ test**. Return evidence is short-horizon. |
| GPT-scored headlines | Lopez-Lira & Tang (JFE 2026; arXiv 2304.07619) | **B** | GPT-4 scores predict next-day drift Oct-2021 to May-2024, after the GPT training cutoff. The effect is strongest in small stocks and negative news, and declines as LLM adoption rises. Horizon is 1-2 days. |
| LLM embeddings of news | Chen, Kelly & Xiu (2022-25, SSRN 4416687) "Expected returns and large language models" | **A/B** | 16 markets, after costs. For large stocks the predictability "dissipates quickly", so it needs daily, broad, long-short trading. |
| Look-ahead in pretrained LMs | Glasserman & Lin (2023, JFDS) "Assessing look-ahead bias in stock return predictions generated by GPT sentiment analysis"; Sarkar & Vafa (2024, SSRN 4754678) "Lookahead bias in pretrained language models" | **A** (as a warning) | Backtests of a modern LLM over its own training window are biased. The bias comes both from leaked outcomes and from a "distraction effect" of firm knowledge. Anonymising text helps. **Claude/GPT scoring of 2020-2026 headlines would be contaminated and cannot be backtested honestly.** |
| LLM financial-statement analysis | Kim, Muhn & Nikolaev (2024, arXiv 2407.17866) | **B/C** | GPT-4 on anonymised statements predicts earnings direction about as well as analysts and ML. The trading results are in-sample relative to GPT-4's training data. |
| 10-K text changes | Cohen, Malloy & Nguyen (2020, JF) "Lazy Prices" | **A** | Monthly, free via EDGAR, no LLM needed. The data build is heavy (full 10-K/10-Q text) and outside this phase. Flagged as the best future text idea. |
| Earnings-call transcripts | e.g. Sarkar & Vafa; many SSRN papers | n/a | No free transcript source. Excluded. |

**Shortlisted:** FinBERT (ProsusAI/finbert) on Alpaca headlines as one
more sleeve leg, run on CPU in Actions shards or on Kaggle GPU. The
literature points to short horizons and small caps, which is the wrong
setting for a slow S&P 500 sleeve, so expect little.

## 4. LLM agents and LLM alpha generators

| system | paper | evaluation | grade |
|---|---|---|---|
| FinGPT | Yang, Liu & Wang (2023, arXiv 2306.06031) | NLP benchmarks and framework; no long-horizon trading OOS | C |
| BloombergGPT | Wu et al. (2023, arXiv 2303.17564) | NLP benchmarks only | n/a (not a trading result) |
| TradingAgents | Xiao, Sun, Luo & Wang (2024, arXiv 2412.20138) | about 3 months (Jan–Mar 2024), 3 mega-caps, Sharpe 5.6–8.2, no costs; inside the backbone LLM's training window | **C** |
| FinMem | Yu et al. (2023, arXiv 2311.13743) | single stocks, months-long windows | C |
| FinAgent | Zhang et al. (2024, KDD) | short windows, few assets | C |
| Alpha-GPT / AlphaGen | Wang et al. (2023, arXiv 2308.00016); Yu et al. (2023, KDD) "Generating synergistic formulaic alpha collections via RL" | mostly CSI300 / CSI500, in-sample mining then a short test; LLM priors encode published factors | C |
| **Long-horizon re-test** | Li, Kim, Cucuringu & Ma, "Can LLM-based financial investing strategies outperform the market in long run?" (FINSABER; KDD 2026; arXiv 2505.07078) | about 20 years, 100+ symbols incl. delisted, bias-mitigated | **A (negative)**: reported LLM advantages "deteriorate significantly"; buy-and-hold beats the agents in every bias-mitigated setup; too conservative in bulls, too aggressive in bears |

Verdict: **no LLM agent or LLM factor miner has credible
out-of-sample evidence.** The one careful long-horizon re-evaluation is
negative. They are also impossible to backtest honestly with a
present-day model, because the training data covers the test years.
Excluded from testing, and the rationale is documented here.

## 5. Shortlist (feasible, free, testable honestly)

Ranked by prior likelihood of surviving an honest yearly walk-forward
with costs on this book:

1. **Breadth / concentration** (n = 8, 10, 15, 20; inverse-vol n=15).
   This is a diversification fix, not new alpha. The arithmetic note:
   the audit's "n=8 → +7.7%" ran on the $12.70 micro book, where 8 names
   × $0.79 fall under Alpaca's $1 floor. The paper book is about $100k,
   so breadth has to be re-tested there.
2. **Vol-managed sleeve + graded trend regime** (Barroso & Santa-Clara
   style, freed share → SPY, no leverage). Credible for momentum
   specifically, weak in general (Cederburg et al.).
3. **Online IC-weighted factor blend** over six live-computable legs
   (factor-momentum logic), monthly, prior = B50's 50/50 mom/vq.
4. **LightGBM** on the same factor ranks, monthly expanding refit,
   label horizon purged + embargo, hyper-parameters by purged CV each
   January. Also a rank-ensemble of 3 + 4.
5. **FinBERT headline sentiment** as an extra leg (2021+ only; scored
   on CPU in Actions or `MODELS="phase4_news_score"` on Kaggle).

The end model (AD50) combines 3 + 4 + the vol overlay. A drift monitor
falls back to B50, and then to SPY. See `docs/PHASE9_IMPROVE.md`.

## 6. The three biggest pitfalls (and how phase 9 handles them)

1. **LLM training-data look-ahead.** A model trained on 2020-2025 text
   "knows" how those stories ended (Glasserman & Lin; Sarkar & Vafa).
   *Handled:* only FinBERT, whose training data predates the test years,
   is used for scoring. Claude, GPT and agent pipelines are not
   backtested.
2. **Multiple testing.** 392 trials before this phase, plus every phase-9
   candidate-year. *Handled:* DSR is computed with the full running count
   on total, excess-vs-SPY **and** active-vs-B50 returns. All candidates
   and the adoption rule were committed before the first run.
3. **Small-sample concentration.** One year and three names (2024:
   APP/CVNA/SMCI) carry the headline. *Handled:* the report includes
   per-year beats against B50, top-3-name share of sleeve P&L, and a block
   bootstrap of the active return. The adoption rule also needs ≥60% of
   years to beat B50, not just the total.
