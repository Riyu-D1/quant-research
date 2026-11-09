"""Download a qlib community data dump (wraps qlib.tests.data.GetData)."""
import argparse
from qlib.tests.data import GetData


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-dir", required=True)
    p.add_argument("--region", default="cn", choices=["cn", "us"])
    p.add_argument("--interval", default="1d")
    args = p.parse_args()
    GetData().qlib_data(
        name="qlib_data",
        target_dir=args.target_dir,
        interval=args.interval,
        region=args.region,
        delete_old=True,
    )


if __name__ == "__main__":
    main()
