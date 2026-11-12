"""Phase-4 news sentiment scoring — Kaggle GPU entry point.

    .venv/bin/python scripts/phase4_score_news.py                 # full run
    .venv/bin/python scripts/phase4_score_news.py --smoke         # CI plumbing

Flow: download the `phase4-news` workflow artifact (headlines.parquet
produced by the phase4 `fetch` job) via GH_TOKEN -> pip-install
transformers if absent -> ProsusAI/finbert over unique headline+summary
texts -> results/news_sentiment/{scored_articles,daily_sentiment}.parquet.

The notebook's PUSH_RESULTS=1 cell then commits results/ to a dated
results/kaggle-* branch — that's how the scores come back to the repo.

Flags:
  --smoke        CI path: real artifact downloads, deterministic hash
                 scores (no transformers/torch), --limit applies. Proves
                 fetch->filter->dedup->write plumbing on CPU in seconds.
  --stub-score   skip transformers + FinBERT; deterministic hash scores
  --limit N      keep at most N article-symbol rows (post-universe filter)
  --articles P   read headlines parquet from P instead of the artifact

Every failure exits non-zero — `!`-style notebook cells that swallow
exit codes are a footgun; the notebook uses subprocess(check=True).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from quant_research import news as NW

RAW = REPO / "data" / "news" / "raw"
OUT = REPO / "results" / "news_sentiment"
REPO_SLUG = "Riyu-D1/quant-research"


def ensure_transformers() -> None:
    try:
        import transformers
        return
    except ImportError:
        pass
    print("[phase4-score] installing transformers + sentencepiece")
    # uv-created venvs carry no pip — go through uv (keeps the locked torch);
    # on Kaggle there's no uv but pip exists, so fall back to `python -m pip`.
    r = subprocess.run(["uv", "pip", "install", "--python", sys.executable,
                        "transformers>=4.40,<5", "sentencepiece"], check=False)
    if r.returncode != 0:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "transformers>=4.40,<5", "sentencepiece"],
                       check=False)
    try:
        import transformers  # noqa: F401
    except ImportError:
        sys.exit("[phase4-score] transformers still missing after install "
                 "attempts — cannot score")


def load_articles(args, token: str):
    import pandas as pd
    headlines = REPO / "data" / "news" / "headlines.parquet"
    if args.articles:
        headlines = Path(args.articles)
    if not headlines.exists():
        if not token:
            sys.exit("headlines.parquet missing and no GH_TOKEN to fetch "
                     "the phase4-news artifact — run the phase4 `fetch` job "
                     "first")
        # artifact members are zip-root-flat (upload-artifact strips the
        # path to its least common ancestor) — fetch_artifact_file finds
        # the member by basename and places it correctly.
        NW.fetch_artifact_file(REPO_SLUG, "phase4-news", "headlines.parquet",
                               token, headlines)
    if not headlines.exists():
        sys.exit(f"[phase4-score] headlines still missing at {headlines} "
                 "after artifact download")
    articles = pd.read_parquet(headlines)
    if articles.empty:
        sys.exit(f"[phase4-score] {headlines} has zero rows")
    print(f"[phase4-score] {len(articles)} headline-symbol rows "
          f"from {headlines}")

    # Only universe-tagged articles feed the signal downstream (daily_signal
    # joins on store instruments) — restricting the FinBERT pass to them
    # shrinks the unique-text set ~3-5x. Universe = phase-3 cik_map.
    cmap = REPO / "data" / "events" / "cik_map.parquet"
    try:
        if not cmap.exists() and token:
            NW.fetch_artifact_file(REPO_SLUG, "phase3-events",
                                   "cik_map.parquet", token, cmap)
        if cmap.exists():
            universe = {NW.canon(s) for s in
                        pd.read_parquet(cmap)["symbol"]}
            before = len(articles)
            articles = articles[articles["symbol"].isin(universe)]
            print(f"[phase4-score] universe filter: {before} -> "
                  f"{len(articles)} rows ({len(universe)} tickers)")
            if articles.empty:
                sys.exit("[phase4-score] universe filter dropped every row "
                         "— symbol canon mismatch (cik_map vs headlines)?")
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[phase4-score] universe filter skipped: {e}")

    if args.limit:
        articles = articles.head(int(args.limit))
        print(f"[phase4-score] --limit {args.limit}: {len(articles)} rows")
    return articles


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true",
                   help="CI plumbing check: stub scores, no transformers")
    p.add_argument("--stub-score", action="store_true",
                   help="deterministic hash scores instead of FinBERT")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--articles", default=None,
                   help="read headlines parquet from this path")
    p.add_argument("--model", default=NW.FINBERT)
    args = p.parse_args()

    token = os.environ.get("GH_TOKEN", "")
    articles = load_articles(args, token)

    if args.smoke or args.stub_score:
        scored = NW.stub_score(articles, OUT / "scored_articles.parquet")
    else:
        ensure_transformers()
        scored = NW.score_finbert(articles, OUT / "scored_articles.parquet",
                                  model=args.model)
    if scored.empty:
        sys.exit("[phase4-score] scoring produced zero rows")

    daily = NW.daily_table(scored)
    if daily.empty:
        sys.exit("[phase4-score] daily aggregation produced zero rows")
    daily.to_parquet(OUT / "daily_sentiment.parquet", index=False)
    print(f"[phase4-score] daily table {daily.shape} -> "
          f"{OUT/'daily_sentiment.parquet'}")
    (OUT / "README.txt").write_text(
        "FinBERT (ProsusAI/finbert) scores for Alpaca news headlines.\n"
        "daily_sentiment.parquet: (symbol, pub_date, mean_score, n) — keyed "
        "on PUBLISH calendar date; the engine shifts to the next trading "
        "day (strictly-after convention).\n")
    if not (OUT / "daily_sentiment.parquet").exists():
        sys.exit("[phase4-score] daily_sentiment.parquet not written")


if __name__ == "__main__":
    main()
