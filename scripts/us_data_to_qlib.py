"""Convert a directory of US daily OHLCV CSVs into a qlib provider directory.

Input:  --input-dir containing one CSV per symbol (e.g. stooq `aapl.us.csv`)
        with columns date,open,high,low,close,volume (case-insensitive).
Output: qlib layout —
        calendars/day.txt, instruments/<universe>.txt,
        features/<sym>/{open,high,low,close,volume,factor}.day.bin (float32).

Usage:
    python scripts/us_data_to_qlib.py --input-dir data/us_csv --output-dir data/qlib/us_v2 \
        --universe sp500 --start 2005-01-01
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume"]


def load_csvs(input_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(input_dir.glob("*.csv")):
        sym = f.stem.split(".")[0].lower()
        df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        if "date" not in df or "close" not in df:
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        out[sym] = df.set_index("date").sort_index()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--universe", default="all")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    args = p.parse_args()

    frames = load_csvs(args.input_dir)
    if not frames:
        raise SystemExit("no usable CSVs found")

    cal = sorted(set().union(*[set(df.index) for df in frames.values()]))
    if args.start:
        cal = [d for d in cal if d >= args.start]
    if args.end:
        cal = [d for d in cal if d <= args.end]
    day_idx = {d: i for i, d in enumerate(cal)}

    root = args.output_dir
    (root / "calendars").mkdir(parents=True, exist_ok=True)
    (root / "instruments").mkdir(parents=True, exist_ok=True)
    (root / "features").mkdir(parents=True, exist_ok=True)
    (root / "calendars/day.txt").write_text("\n".join(cal) + "\n")

    inst_rows = []
    for sym, df in frames.items():
        sub = df.reindex(cal)
        sdir = root / "features" / sym
        sdir.mkdir(exist_ok=True)
        for fld in FIELDS:
            vals = sub[fld].astype(np.float32).values if fld in sub else \
                np.full(len(cal), np.nan, dtype=np.float32)
            vals.tofile(sdir / f"{fld}.day.bin")
        # factor: adjust for splits via adj close if present, else 1.0
        if "adj close" in df.columns or "adjclose" in df.columns:
            ac = df.get("adj close", df.get("adjclose"))
            factor = (ac / df["close"]).reindex(cal).ffill().fillna(1.0)
        else:
            factor = pd.Series(1.0, index=cal)
        factor.astype(np.float32).values.tofile(sdir / "factor.day.bin")
        present = df.index[(df.index >= cal[0]) & (df.index <= cal[-1])]
        if len(present):
            inst_rows.append(f"{sym.upper()}\t{present.min()}\t{present.max()}")

    (root / "instruments" / f"{args.universe}.txt").write_text(
        "\n".join(inst_rows) + "\n")
    (root / "instruments" / "all.txt").write_text(
        "\n".join(inst_rows) + "\n")
    print(f"wrote {len(inst_rows)} instruments, {len(cal)} calendar days -> {root}")


if __name__ == "__main__":
    main()
