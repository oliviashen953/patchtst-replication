"""Collect Step 9 results into a comparison table against the paper.

    python experiments/collate.py
    python experiments/collate.py --tag base --csv results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
HORIZONS = [96, 192, 336, 720]


def load(tag: str | None) -> list[dict]:
    records = []
    for path in sorted(RESULTS.glob("*.json")):
        record = json.loads(path.read_text())
        if tag is None or record.get("tag") == tag:
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=None, help="only this run tag")
    parser.add_argument("--csv", default=None, help="also write a CSV here")
    args = parser.parse_args()

    records = load(args.tag)
    if not records:
        raise SystemExit(
            f"no results in {RESULTS} -- run experiments/run_etth1.py first"
        )

    by_horizon: dict[int, list[dict]] = {}
    for r in records:
        by_horizon.setdefault(r["pred_len"], []).append(r)

    print()
    print("ETTh1 multivariate forecasting, lookback L=336 (PatchTST/42)")
    print()
    header = (
        f"{'H':>5}  {'ours MSE':>9} {'paper':>7} {'Δ':>8}   "
        f"{'ours MAE':>9} {'paper':>7} {'Δ':>8}   {'ep':>4} {'min':>6}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    d_mse, d_mae = [], []
    for h in HORIZONS:
        runs = by_horizon.get(h)
        if not runs:
            print(f"{h:>5}  {'(not run)':>9}")
            continue
        # average over seeds if there are several
        mse = sum(r["test"]["mse"] for r in runs) / len(runs)
        mae = sum(r["test"]["mae"] for r in runs) / len(runs)
        ref = runs[0]["paper_patchtst_42"]
        epochs = sum(r["best_epoch"] for r in runs) / len(runs)
        minutes = sum(r["wall_seconds"] for r in runs) / len(runs) / 60
        dm, da = mse - ref["mse"], mae - ref["mae"]
        d_mse.append(dm)
        d_mae.append(da)
        flag = "" if abs(dm) <= 0.02 else ("  <-- off" if abs(dm) > 0.05 else "  ~")
        print(
            f"{h:>5}  {mse:>9.4f} {ref['mse']:>7.3f} {dm:>+8.4f}   "
            f"{mae:>9.4f} {ref['mae']:>7.3f} {da:>+8.4f}   "
            f"{epochs:>4.0f} {minutes:>6.1f}{flag}"
        )
        rows.append(
            dict(pred_len=h, mse=mse, mae=mae, paper_mse=ref["mse"],
                 paper_mae=ref["mae"], delta_mse=dm, delta_mae=da,
                 n_seeds=len(runs), best_epoch=epochs, minutes=minutes)
        )

    if d_mse:
        print("-" * len(header))
        print(f"{'mean':>5}  {'':>9} {'':>7} {sum(d_mse)/len(d_mse):>+8.4f}   "
              f"{'':>9} {'':>7} {sum(d_mae)/len(d_mae):>+8.4f}")
        print()
        worst = max(abs(x) for x in d_mse)
        if worst <= 0.02:
            print("  Within 0.02 MSE at every horizon -- a faithful replication.")
        elif worst <= 0.05:
            print("  Within 0.05 MSE -- close. Likely the LR-schedule deviation")
            print("  (we use cosine; the official ETTh1 script uses lradj type3).")
        else:
            print("  Off by more than 0.05 MSE somewhere. Check, in this order:")
            print("    1. model size -- ETTh1 needs d_model=16/n_heads=4/d_ff=128,")
            print("       NOT the paper's 128/16/256 defaults (Appendix A.1.4)")
            print("    2. that you are comparing against PatchTST/42, not /64")
            print("    3. double normalization -- targets are already standardized")
            print("    4. the split: 8209/2785/2785 windows at H=96")

    if args.csv:
        import csv

        path = Path(args.csv)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  wrote {path}")
    print()


if __name__ == "__main__":
    main()
