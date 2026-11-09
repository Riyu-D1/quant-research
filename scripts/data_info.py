"""Summarise a qlib data dump: calendar range, instrument universes, feature set."""
import argparse
from pathlib import Path


def summarize(name: str, root: Path) -> None:
    if not root.exists():
        print(f"[{name}] MISSING: {root}")
        return
    cal = (root / "calendars" / "day.txt").read_text().split()
    inst_dir = root / "instruments"
    universes = {}
    for f in sorted(inst_dir.glob("*.txt")):
        n_syms = 0
        dynamic = 0
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            n_syms += 1
            parts = line.split("\t")
            if len(parts) >= 3 and (parts[1] != cal[0] or parts[2] != cal[-1]):
                dynamic += 1
        universes[f.stem] = (n_syms, dynamic)
    n_features = len(list((root / "features").glob("*")))
    print(f"[{name}] {root}")
    print(f"  calendar: {cal[0]} .. {cal[-1]}  ({len(cal)} days)")
    print(f"  instruments dirs/files: {n_features} symbols with feature data")
    for u, (n, dyn) in universes.items():
        print(f"  universe {u}: {n} listings, {dyn} with limited date windows (survivorship-aware)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cn", type=Path, default=None)
    p.add_argument("--us", type=Path, default=None)
    args = p.parse_args()
    for name, root in [("cn", args.cn), ("us", args.us)]:
        if root:
            summarize(name, root)


if __name__ == "__main__":
    main()
