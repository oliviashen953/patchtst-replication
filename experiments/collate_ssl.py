"""Summarize the Step 12 self-supervised results.

    python experiments/collate_ssl.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results_ssl"
HORIZONS = [96, 192, 336, 720]

# Everything here is single-seed. Below this, call it a tie.
TIE = 0.002

# Table 12, ETTh1 block. The 'sup' column is the paper's L=336 PatchTST/42 row,
# NOT a model matching the self-supervised geometry -- see run_ssl.py.
PAPER = {
    96:  {"finetune": 0.366, "linear_probe": 0.371, "sup": 0.375},
    192: {"finetune": 0.431, "linear_probe": 0.411, "sup": 0.414},
    336: {"finetune": 0.450, "linear_probe": 0.445, "sup": 0.431},
    720: {"finetune": 0.472, "linear_probe": 0.487, "sup": 0.449},
}


def load() -> list[dict]:
    return [json.loads(p.read_text()) for p in RESULTS.glob("*.json")]


def pick(runs, stage, pred_len, metric="mse"):
    for r in runs:
        if r.get("stage") == stage and r.get("pred_len") == pred_len:
            return r["test"][metric]
    return None


def f(v, w=10, nd=4):
    return f"{v:{w}.{nd}f}" if v is not None else f"{'--':>{w}}"


def main() -> None:
    runs = load()
    if not runs:
        raise SystemExit(f"no results in {RESULTS}")

    pre = next((r for r in runs if r.get("stage") == "pretrain"), None)
    if pre:
        print(f"\nPretraining: masked reconstruction MSE "
              f"{pre['first_val_loss']:.4f} -> {pre['best_val_loss']:.4f} "
              f"over {pre['epochs_run']} epochs "
              f"(best at {pre['best_epoch']}), "
              f"{pre['wall_seconds'] / 60:.1f} min")

    print("\nAll single-seed. Differences below "
          f"{TIE} MSE are ties.\n")

    # ------------------------------------------------------ ours vs ours
    print("=" * 78)
    print("Does pretraining help? Same architecture, same geometry, same budget")
    print("=" * 78)
    print(f"{'H':>5}{'scratch':>11}{'lin.probe':>11}{'finetune':>11}"
          f"{'probe-scr':>12}{'ft-scr':>10}   verdict")
    for h in HORIZONS:
        s = pick(runs, "scratch", h)
        p = pick(runs, "linear_probe", h)
        t = pick(runs, "finetune", h)
        dp = p - s if (p is not None and s is not None) else None
        dt = t - s if (t is not None and s is not None) else None
        if dt is None:
            verdict = ""
        elif dt < -TIE:
            verdict = "pretraining helps"
        elif dt > TIE:
            verdict = "pretraining hurts"
        else:
            verdict = "tie"
        print(f"{h:>5}{f(s, 11)}{f(p, 11)}{f(t, 11)}"
              f"{f(dp, 12)}{f(dt, 10)}   {verdict}")

    # ------------------------------------------------------- vs the paper
    print("\n" + "=" * 78)
    print("Versus Table 12 (ETTh1). Paper pretrains 100 epochs at L=512, P=S=12")
    print("=" * 78)
    print(f"{'H':>5}{'arm':>14}{'ours':>10}{'paper':>10}{'delta':>10}")
    for h in HORIZONS:
        for stage in ["linear_probe", "finetune", "scratch"]:
            ours = pick(runs, stage, h)
            ref = PAPER[h]["sup" if stage == "scratch" else stage]
            delta = ours - ref if ours is not None else None
            label = stage if stage != "scratch" else "scratch/sup*"
            print(f"{h:>5}{label:>14}{f(ours, 10)}{ref:>10.3f}{f(delta, 10)}")

    print("\n* the paper's 'Sup.' column for ETTh1 is its L=336 PatchTST/42 row,")
    print("  a different lookback, patch geometry and model size. Our 'scratch'")
    print("  is the like-for-like control; the delta against 'Sup.' is context,")
    print("  not a replication target.")

    # ------------------------------------------------------------- timing
    print("\n" + "=" * 78)
    print(f"{'run':>22}{'epochs':>9}{'best ep':>9}{'minutes':>10}")
    for r in sorted(runs, key=lambda r: r.get("name", "")):
        if r.get("stage") == "pretrain":
            print(f"{r['name']:>22}{r['epochs_run']:>9}{r['best_epoch']:>9}"
                  f"{r['wall_seconds'] / 60:>10.1f}")
        else:
            epochs = sum(p["epochs_run"] for p in r["phases"])
            best = r["phases"][-1]["best_epoch"]
            print(f"{r['name']:>22}{epochs:>9}{best:>9}"
                  f"{r['wall_seconds'] / 60:>10.1f}")
    print()


if __name__ == "__main__":
    main()
