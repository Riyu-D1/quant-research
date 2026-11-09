"""Raw momentum IC probe — reads qlib .bin files directly (no qlib init).

Cross-sectional IC of 5d/20d momentum vs the benchmark label
(close[t] / close[t-1] - 1, i.e. Ref($close,-2)/Ref($close,-1)-1 shifted)
per calendar day. Cheap sanity check on whether a dump carries next-day alpha.

    python scripts/momentum_probe.py --data-dir data/qlib/us_data --instruments sp500.txt
    python scripts/momentum_probe.py --data-dir data/qlib/cn_data --instruments csi300.txt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def mom_ic(root: Path, inst_file: str, start: str, end: str, max_syms: int):
    cal = pd.read_csv(root / "calendars/day.txt", header=None,
                      names=["d"], sep=r"\s+")["d"]
    day_idx = {d: i for i, d in enumerate(cal)}
    test_days = [d for d in cal if start <= d <= end]

    syms = [l.split("\t")[0].lower() for l in
            (root / "instruments" / inst_file).read_text().splitlines() if l.strip()]
    closes = {}
    for s in syms[:max_syms]:
        f = root / "features" / s / "close.day.bin"
        if f.exists():
            a = np.fromfile(f, dtype=np.float32)
            if len(a) >= len(cal):
                closes[s] = a[: len(cal)]

    ics5, ics20 = [], []
    for d in test_days:
        t = day_idx[d]
        m5, m20, lab = [], [], []
        for s, c in closes.items():
            if t + 2 < len(c) and t - 20 >= 0 and min(
                    c[t-1], c[t-2], c[t-5], c[t-20]) > 0:
                m5.append(c[t-5] / c[t-2] - 1)
                m20.append(c[t-20] / c[t-2] - 1)
                lab.append(c[t] / c[t-1] - 1)
        if len(lab) > 20:
            i5 = pd.Series(m5).corr(pd.Series(lab))
            i20 = pd.Series(m20).corr(pd.Series(lab))
            if np.isfinite(i5):
                ics5.append(i5)
                ics20.append(i20)
    return np.mean(ics5), np.mean(ics20), len(closes), len(ics5)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--instruments", default="all.txt")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2019-07-01")
    p.add_argument("--max-syms", type=int, default=200)
    a = p.parse_args()
    i5, i20, n, nd = mom_ic(a.data_dir, a.instruments, a.start, a.end, a.max_syms)
    print(f"{a.data_dir} {a.instruments} [{a.start}..{a.end}] "
          f"mom5 IC {i5:.4f} | mom20 IC {i20:.4f} | syms={n} days={nd}")
