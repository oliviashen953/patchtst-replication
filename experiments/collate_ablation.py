"""Summarize the Step 10 ablations.

    python experiments/collate_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
HORIZONS = [96, 192, 336, 720]
LOOKBACKS = [96, 192, 336, 512, 720]


def load() -> dict:
    out = {}
    for path in RESULTS.glob("*.json"):
        r = json.loads(path.read_text())
        out[r["tag"]] = r
    return out


def main() -> None:
    runs = load()
    if not runs:
        raise SystemExit(f"no results in {RESULTS}")

    def mse(tag):
        r = runs.get(tag)
        return r["test"]["mse"] if r else None

    def fmt(v):
        return f"{v:.4f}" if v is not None else "   --  "

    print("\n" + "=" * 66)
    print("A. Does channel independence matter?   (ETTh1, L=336)")
    print("=" * 66)
    print(f"{'H':>5} {'independent':>13} {'mixing':>10} {'Δ':>9}  verdict")
    print("-" * 66)
    wins = 0
    n = 0
    for h in HORIZONS:
        a, b = mse("base") if h == 96 else None, mse(f"mix_h{h}")
        a = mse("base") if h == 96 else None
        # base run is stored per-horizon under tag "base"; find it
        base = next((r for r in runs.values()
                     if r["tag"] == "base" and r["pred_len"] == h), None)
        a = base["test"]["mse"] if base else None
        if a is None or b is None:
            print(f"{h:>5} {fmt(a):>13} {fmt(b):>10}")
            continue
        n += 1
        if a < b:
            wins += 1
        verdict = "independence wins" if a < b else "MIXING wins"
        print(f"{h:>5} {fmt(a):>13} {fmt(b):>10} {b - a:>+9.4f}  {verdict}")
    if n:
        print(f"\n  channel independence wins {wins}/{n} horizons")
        if wins == n:
            print("  -> reproduces the paper's ablation: on ETT-style data, where")
            print("     channels are parallel sensors, isolating them helps.")

    print("\n" + "=" * 66)
    print("B. Does RevIN matter?   (ETTh1, L=336)")
    print("=" * 66)
    print(f"{'H':>5} {'with RevIN':>13} {'without':>10} {'Δ':>9}")
    print("-" * 66)
    for h in HORIZONS:
        base = next((r for r in runs.values()
                     if r["tag"] == "base" and r["pred_len"] == h), None)
        a = base["test"]["mse"] if base else None
        b = mse(f"norevin_h{h}")
        if a is None or b is None:
            print(f"{h:>5} {fmt(a):>13} {fmt(b):>10}")
            continue
        print(f"{h:>5} {fmt(a):>13} {fmt(b):>10} {b - a:>+9.4f}")

    print("\n" + "=" * 66)
    print("C. Does a longer lookback help?   (ETTh1, H=96)")
    print("=" * 66)
    print("  The paper's sharpest claim: PatchTST IMPROVES with a longer")
    print("  lookback, where other Transformers degrade. Patching is what")
    print("  makes the long lookback affordable, so this curve is the actual")
    print("  argument for the whole method.")
    print()
    print(f"{'L':>6} {'n_patches':>10} {'test MSE':>10}   trend")
    print("-" * 66)
    prev = None
    values = []
    for L in LOOKBACKS:
        r = runs.get(f"look_L{L}")
        if not r:
            print(f"{L:>6} {'--':>10} {'--':>10}")
            continue
        m = r["test"]["mse"]
        npatch = r.get("n_parameters")
        arrow = "" if prev is None else ("  better" if m < prev else "  worse")
        print(f"{L:>6} {r['model'].get('seq_len', L)//8:>10} {m:>10.4f}{arrow}")
        values.append((L, m))
        prev = m
    if len(values) >= 3:
        best_L, best_m = min(values, key=lambda t: t[1])
        first_m = values[0][1]
        print()
        print(f"  best at L={best_L} (MSE {best_m:.4f}); L={values[0][0]} gives {first_m:.4f}")
        if best_L > values[0][0]:
            print("  -> longer lookback helps. Reproduces the paper's central claim.")
        else:
            print("  -> longer lookback does NOT help here. That contradicts the")
            print("     paper's claim and would need investigating before reporting.")
    print()


if __name__ == "__main__":
    main()
