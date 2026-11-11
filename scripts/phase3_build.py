"""Phase-3: build the point-in-time events feature panel.

    python scripts/phase3_build.py [--market us] [--instruments sp500]

Reads data/events/{insider,f13f,shares_out,congress}_events.parquet +
cik_map.parquet (from phase3_fetch.py) and the qlib dump's
calendar/instruments/close bins; writes data/events/features_<market>.parquet
indexed (instrument, datetime), feature cols in events.ALL_COLS + LABEL0.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quant_research import events as ev  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--market", default="us")
    p.add_argument("--instruments", default="sp500")
    p.add_argument("--data-dir", type=Path,
                   default=REPO / "data/qlib/us_alpaca")
    p.add_argument("--events-dir", type=Path, default=REPO / "data/events")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out = args.out or args.events_dir / f"features_{args.market}.parquet"
    t0 = time.time()
    panel = ev.build_panel(args.data_dir, args.instruments,
                           args.events_dir, out)
    print(f"panel: {len(panel)} rows, {panel.index.get_level_values(0).nunique()} "
          f"instruments -> {out} ({time.time()-t0:.0f}s)")
    nz = (panel[ev.FEATURE_COLS].abs().sum(axis=1) > 0).mean()
    print(f"rows with any nonzero event feature: {nz:.1%}")


if __name__ == "__main__":
    main()
