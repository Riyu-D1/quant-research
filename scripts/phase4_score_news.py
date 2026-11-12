"""Phase-4 news sentiment scoring — Kaggle GPU entry point.

    .venv/bin/python scripts/phase4_score_news.py

Flow: download the `phase4-news` workflow artifact (headlines.parquet
produced by the phase4 `fetch` job) via GH_TOKEN -> pip-install
transformers if absent -> ProsusAI/finbert over unique headline+summary
texts -> results/news_sentiment/{scored_articles,daily_sentiment}.parquet.

The notebook's PUSH_RESULTS=1 cell then commits results/ to a dated
results/kaggle-* branch — that's how the scores come back to the repo.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from quant_research import news as NW

RAW = REPO / "data" / "news" / "raw"
OUT = REPO / "results" / "news_sentiment"


def ensure_transformers() -> None:
    try:
        import transformers  # noqa: F401
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


def main() -> None:
    headlines = REPO / "data" / "news" / "headlines.parquet"
    if not headlines.exists():
        token = os.environ.get("GH_TOKEN", "")
        if not token:
            sys.exit("headlines.parquet missing and no GH_TOKEN to fetch "
                     "the phase4-news artifact — run the phase4 `fetch` job "
                     "first")
        # artifact zips preserve repo-relative paths (data/news/...) —
        # extract at the repo root so headlines.parquet lands correctly
        NW.gh_artifact_download("Riyu-D1/quant-research", "phase4-news",
                                token, REPO)
    import pandas as pd
    articles = pd.read_parquet(headlines)
    print(f"[phase4-score] {len(articles)} headline-symbol rows")

    # Only universe-tagged articles feed the signal downstream (daily_signal
    # joins on store instruments) — restricting the FinBERT pass to them
    # shrinks the unique-text set ~3-5x. Universe = phase-3 cik_map.
    token = os.environ.get("GH_TOKEN", "")
    cmap = REPO / "data" / "events" / "cik_map.parquet"
    try:
        if not cmap.exists() and token:
            NW.gh_artifact_download("Riyu-D1/quant-research", "phase3-events",
                                    token, REPO / "data" / "events",
                                    members=["cik_map.parquet"])
        if cmap.exists():
            universe = {NW.canon(s) for s in
                        pd.read_parquet(cmap)["symbol"]}
            before = len(articles)
            articles = articles[articles["symbol"].isin(universe)]
            print(f"[phase4-score] universe filter: {before} -> "
                  f"{len(articles)} rows ({len(universe)} tickers)")
    except Exception as e:  # noqa: BLE001
        print(f"[phase4-score] universe filter skipped: {e}")
    ensure_transformers()
    scored = NW.score_finbert(articles, OUT / "scored_articles.parquet")
    daily = NW.daily_table(scored)
    daily.to_parquet(OUT / "daily_sentiment.parquet", index=False)
    print(f"[phase4-score] daily table {daily.shape} -> "
          f"{OUT/'daily_sentiment.parquet'}")
    (OUT / "README.txt").write_text(
        "FinBERT (ProsusAI/finbert) scores for Alpaca news headlines.\n"
        "daily_sentiment.parquet: (symbol, pub_date, mean_score, n) — keyed "
        "on PUBLISH calendar date; the engine shifts to the next trading "
        "day (strictly-after convention).\n")


if __name__ == "__main__":
    main()
