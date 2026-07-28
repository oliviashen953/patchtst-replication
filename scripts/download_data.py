"""Fetch the ETTh1 benchmark CSV into data/.

ETTh1 is public, from the Informer authors' dataset repository:
    https://github.com/zhouhaoyi/ETDataset

    python scripts/download_data.py
"""

from __future__ import annotations

import argparse
import os
import urllib.request

URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)
EXPECTED_ROWS = 17420
EXPECTED_COLUMNS = 8  # date + 7 channels

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(HERE, "data", "ETTh1.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download if present")
    args = parser.parse_args()

    if os.path.exists(DEST) and not args.force:
        print(f"{DEST} already exists (use --force to re-download)")
        return

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, DEST)

    with open(DEST) as handle:
        header = handle.readline().strip().split(",")
        rows = sum(1 for _ in handle)

    if len(header) != EXPECTED_COLUMNS or rows != EXPECTED_ROWS:
        raise SystemExit(
            f"unexpected file: {rows} rows x {len(header)} cols "
            f"(expected {EXPECTED_ROWS} x {EXPECTED_COLUMNS})"
        )

    print(f"wrote {DEST}")
    print(f"  {rows} rows, channels: {', '.join(header[1:])}")


if __name__ == "__main__":
    main()
