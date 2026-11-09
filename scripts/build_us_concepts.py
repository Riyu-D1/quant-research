"""Build HIST concept assets for the US (sp500) universe.

HIST (qlib.contrib.model.pytorch_hist) needs two .npy files:

  * stock_index — dict {INSTRUMENT_CODE: row} covering EVERY instrument the
    dataset can emit (any code missing from the map produces NaN row ids and
    the model crashes at tensor-index time).
  * stock2concept — float32 (n_instruments, n_concepts) binary membership
    matrix. Upstream ships a 4635-concept csi300 matrix; we approximate for
    sp500 with GICS sector + sub-industry memberships from
    configs/benchmarks/assets/sp500_sector_map.csv (built by
    `scripts/alpaca_us_data.py universe`), plus an UNMAPPED concept and one
    all-market concept so the model always has >=1 non-trivial relation.

Regenerate the sector map after rebuilding the universe csv:
    cut -d, -f1,4,5 build/universe_sp500.csv | sort -u \
        > configs/benchmarks/assets/sp500_sector_map.csv  # then fix header

Usage:
    python scripts/build_us_concepts.py \
        --instruments data/qlib/us_alpaca/instruments/sp500.txt \
        --out-dir configs/benchmarks/assets
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# sector for well-known delisted names absent from the current-members map
MANUAL_SECTORS = {
    "LEH": "Financials", "BSC": "Financials", "RSH": "Consumer Discretionary",
    "WYE": "Health Care", "MON": "Materials", "TWX": "Communication Services",
    "CA": "Information Technology", "ESRX": "Health Care",
    "LLL": "Industrials", "CELG": "Health Care", "AGN": "Health Care",
    "RTN": "Industrials", "XLNX": "Information Technology",
    "ATVI": "Communication Services", "SGEN": "Health Care",
    "SIVB": "Financials", "FRC": "Financials", "AABA": "Information Technology",
    "HRS": "Information Technology", "DWDP": "Materials", "PCG": "Utilities",
    "NFX": "Energy", "MAT": "Consumer Discretionary", "GT": "Consumer Discretionary",
    "BHF": "Financials", "FLR": "Industrials",
}
UNMAPPED = "UNMAPPED"
MARKET = "MARKET_SP500"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instruments", type=Path, required=True,
                   help="instruments/<universe>.txt of the target dump")
    p.add_argument("--sector-map", type=Path,
                   default=Path("configs/benchmarks/assets/sp500_sector_map.csv"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("configs/benchmarks/assets"))
    p.add_argument("--prefix", default="qlib_sp500")
    args = p.parse_args()

    syms = []
    for line in args.instruments.read_text().splitlines():
        if line.strip():
            s = line.split("\t")[0].strip().upper()
            if s not in syms:
                syms.append(s)
    syms = sorted(syms)

    smap = {}
    if args.sector_map.exists():
        df = pd.read_csv(args.sector_map)
        for c in ("sector", "sub_industry"):
            if c not in df.columns:
                df[c] = ""
        smap = (df.fillna("").set_index("symbol")
                [["sector", "sub_industry"]].to_dict("index"))

    stock_index = {s: i for i, s in enumerate(syms)}
    sectors = sorted({v["sector"] or UNMAPPED for v in smap.values()}
                     | set(MANUAL_SECTORS.values()) | {UNMAPPED})
    subs = sorted({v["sub_industry"] or UNMAPPED for v in smap.values()}
                  | {UNMAPPED})
    concepts = [MARKET] + [f"SEC:{s}" for s in sectors] + \
               [f"SUB:{s}" for s in subs]
    cidx = {c: i for i, c in enumerate(concepts)}

    mat = np.zeros((len(syms), len(concepts)), dtype=np.float32)
    for s, i in stock_index.items():
        info = smap.get(s) or smap.get(s.replace("-", "."), {})
        sec = info.get("sector") or MANUAL_SECTORS.get(s, UNMAPPED)
        sub = info.get("sub_industry") or UNMAPPED
        mat[i, cidx[MARKET]] = 1.0
        mat[i, cidx[f"SEC:{sec}"]] = 1.0
        mat[i, cidx[f"SUB:{sub}"]] = 1.0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = args.out_dir / f"{args.prefix}_stock_index.npy"
    con_path = args.out_dir / f"{args.prefix}_stock2concept.npy"
    np.save(idx_path, stock_index, allow_pickle=True)
    np.save(con_path, mat)
    n_unmapped = int((mat[:, cidx[f"SEC:{UNMAPPED}"]] == 1).sum())
    print(f"[concepts] {len(syms)} instruments, {len(concepts)} concepts "
          f"({len(sectors)} sectors + {len(subs)} sub-industries + market), "
          f"{n_unmapped} unmapped-sector -> {idx_path}, {con_path}")


if __name__ == "__main__":
    main()
