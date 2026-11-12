"""Monthly retrain/reweight — the adaptive part, in Actions.

    .venv/bin/python scripts/monthly_reweight.py            # real run
    .venv/bin/python scripts/monthly_reweight.py --dry-run  # prints only

What it does (all on data available strictly up to today):
  1. fetches ~18 months of daily bars for the live universe + SPY
  2. recomputes per-signal trailing rank-IC (mom/rev on bars) and the
     ew_ic_weights adaptive mix; snapshot signals (vq/news) keep equal
     weight — they have no live IC history, and equal weighting is the
     documented fallback in paper_trade.compute_signals
  3. re-checks the deployability gate on trailing data: if the config is
     validated but the trailing-12m book IR goes non-positive, the new
     weights.json marks validated=false so nothing silently rides on a
     stale stamp (the config's flag is NOT auto-edited — a human flips it
     back on after the next holdout)
  4. writes state/live/weights.json + a weights-change Notion entry with
     the diff and the reason

Env: ALPACA paper keys for bars; NOTION_* for the journal entry.
--dry-run needs nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_research import adaptive as AD  # noqa: E402
from quant_research import live as LV  # noqa: E402
from quant_research import notion_journal as NJ  # noqa: E402


def log(msg: str) -> None:
    print(f"[reweight {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO / "configs" /
                                           "live_signal.yaml"))
    p.add_argument("--state-dir", default=str(REPO / "state" / "live"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    wfile = state_dir / "weights.json"
    old = json.loads(wfile.read_text()) if wfile.exists() else {}

    if args.dry_run:
        new = {"mom": 0.62, "vq": 0.38}
        print(json.dumps({"old": old.get("weights", {}), "new": new,
                          "gate": "dry"}, indent=1))
        return

    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    universe = [l.strip() for l in
                (REPO / cfg["universe"]).read_text().splitlines()
                if l.strip() and not l.startswith("#")]
    end = pd.Timestamp.now()
    closes = LV.alpaca_bars(key, secret, universe + ["SPY"],
                            end - pd.Timedelta(days=560), end)
    day = closes.index.max()
    log(f"bars through {day.date()} — {closes.shape}")

    bars_sigs = [s for s in cfg["signals"] if s in ("mom", "rev")]
    snap_sigs = [s for s in cfg["signals"] if s not in ("mom", "rev")]
    days_lb = closes.index[-min(len(closes.index), 252):]

    if cfg.get("weighting") == "ewic" and bars_sigs:
        sig_series = LV.bars_signals(closes, days_lb, bars_sigs)
        labels = LV.labels_from_bars(closes, days_lb)
        ic_hist = AD.ic_history(sig_series, labels)
        hl = cfg["ic_months"] * 21 * cfg.get("halflife_frac", 0.5)
        w = AD.ew_ic_weights(ic_hist, day, cfg["ic_months"], hl, bars_sigs)
        bw = {s: float(w.get(s, 0.0)) for s in bars_sigs}
        # snapshot signals split an equal share of the book
        share = (1 - sum(bw.values())) / max(len(snap_sigs), 1) \
            if snap_sigs else 0.0
        new = {**{s: max(v, 0.0) for s, v in bw.items()},
               **{s: share for s in snap_sigs}}
        tot = sum(new.values()) or 1.0
        new = {k: v / tot for k, v in new.items()}
    else:
        new = {s: 1.0 / len(cfg["signals"]) for s in cfg["signals"]}

    # trailing-12m book check — keeps `validated` honest between holdouts
    comp = LV.bars_signals(closes, days_lb, bars_sigs) if bars_sigs else {}
    trailing_ir = None
    if comp:
        book = sum(comp[s].rank(pct=True, axis=1) * w_
                   for s, w_ in new.items() if s in comp)
        top = book.rank(axis=1, ascending=False) <= cfg["n_positions"]
        ret = closes[book.columns].pct_change().shift(-1)
        daily = ret.where(top).mean(axis=1) * \
            (cfg["regime_cfg"]["actions"][cfg["regime"]])
        spy_r = closes["SPY"].pct_change().shift(-1)
        both = pd.concat([daily, spy_r], axis=1).dropna()
        if len(both) > 60:
            xs = both.iloc[:, 0] - both.iloc[:, 1]
            trailing_ir = float(xs.mean() / (xs.std() or 1) * np.sqrt(252))
    gate = "n/a"
    if cfg.get("validated") and trailing_ir is not None:
        gate = ("hold" if trailing_ir > 0 else
                "FAIL — trailing IR <= 0; weights stamped unvalidated")
    log(f"trailing xs-IR {trailing_ir} | gate {gate}")
    eff_validated = bool(cfg.get("validated")) and gate != (
        "FAIL — trailing IR <= 0; weights stamped unvalidated")

    rec = {"month": day.strftime("%Y-%m"), "weights": new,
           "config": args.config, "trailing_ir": trailing_ir,
           "gate": gate, "validated_effective": eff_validated,
           "day": str(day.date())}
    wfile.write_text(json.dumps(rec, indent=1))
    log(f"weights -> {wfile}: {new}")

    j = NJ.NotionJournal(os.environ.get("NOTION_TOKEN", ""),
                         os.environ.get("NOTION_DB_ID", ""),
                         os.environ.get("NOTION_PARENT_PAGE_ID", ""))
    ctx = {"validated": eff_validated,
           "weights_str": ", ".join(f"{k}:{v:.2f}" for k, v in new.items()),
           "regime": cfg["regime"],
           "summary": f"monthly reweight {day.strftime('%Y-%m')} | "
                      f"trailing xs-IR {trailing_ir} | gate {gate}"}
    if not j.day_logged(pd.Timestamp(f"{day.strftime('%Y-%m')}-01"),
                        "weights-change"):
        j.log_weights_change(f"{day.strftime('%Y-%m')}-01",
                             old.get("weights", {}), new, ctx)
        log("weights-change entry written")


if __name__ == "__main__":
    main()
