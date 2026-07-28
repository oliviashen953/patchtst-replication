"""Summarize the Step 10 ablations.

    python experiments/collate_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
HORIZONS = [96, 192, 336, 720]
LOOKBACKS = [96, 192, 336, 512, 720]

# Below this, treat a difference as a tie: everything here is single-seed.
TIE = 0.002


def load() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]


def pick(runs, tag, pred_len=None, seq_len=None):
    """Results are keyed by (tag, pred_len) -- a tag alone is NOT unique,
    since the four baseline horizons all share tag 'base'.

    Two or more matches is an error, not a coin flip. `collate_ssl.py` used to
    return the first match from an unsorted glob, and a 3-epoch smoke run
    carrying the same (stage, pred_len) as a real one got reported as the
    result. Whichever run the filesystem happened to yield first won. Raising
    here means that failure announces itself instead of printing a plausible
    wrong number.
    """
    hits = [
        r for r in runs
        if r["tag"] == tag
        and (pred_len is None or r["pred_len"] == pred_len)
        and (seq_len is None or r["model"].get("seq_len") == seq_len)
    ]
    if len(hits) > 1:
        names = ", ".join(sorted(r["name"] for r in hits))
        raise SystemExit(
            f"ambiguous: {len(hits)} runs match tag={tag!r} pred_len={pred_len} "
            f"seq_len={seq_len} -- {names}. Remove or re-tag the extras."
        )
    return hits[0]["test"]["mse"] if hits else None


def f(v, w=12):
    return f"{v:{w}.4f}" if v is not None else f"{'--':>{w}}"


def main() -> None:
    runs = load()
    if not runs:
        raise SystemExit(f"no results in {RESULTS}")

    print("\nAll single-seed. Differences below "
          f"{TIE} MSE are reported as ties.\n")

    # ---------------------------------------------------------------- A
    print("=" * 70)
    print("A. Does channel independence matter?   (ETTh1, L=336)")
    print("=" * 70)
    print(f"{'H':>5} {'independent':>12} {'mixing':>12} {'Δ':>9}  verdict")
    print("-" * 70)
    ci_wins = mix_wins = ties = 0
    for h in HORIZONS:
        a, b = pick(runs, "base", h), pick(runs, f"mix_h{h}", h)
        if a is None or b is None:
            print(f"{h:>5} {f(a)} {f(b)}")
            continue
        d = b - a
        if abs(d) < TIE:
            verdict, ties = "tie", ties + 1
        elif a < b:
            verdict, ci_wins = "independence", ci_wins + 1
        else:
            verdict, mix_wins = "MIXING", mix_wins + 1
        print(f"{h:>5} {f(a)} {f(b)} {d:>+9.4f}  {verdict}")
    print(f"\n  independence {ci_wins}, mixing {mix_wins}, ties {ties}")
    if mix_wins > ci_wins:
        print("  NOTE: this does NOT reproduce the paper's Table 7, which finds")
        print("  channel independence better. Treat as a discrepancy to explain,")
        print("  not a finding -- it needs seeds and a check that our channel-")
        print("  independent path is not itself limited.")

    # ---------------------------------------------------------------- B
    print("\n" + "=" * 70)
    print("B. Does RevIN matter?   (ETTh1, L=336)")
    print("=" * 70)
    print(f"{'H':>5} {'with RevIN':>12} {'without':>12} {'Δ':>9}  verdict")
    print("-" * 70)
    for h in HORIZONS:
        a, b = pick(runs, "base", h), pick(runs, f"norevin_h{h}", h)
        if a is None or b is None:
            print(f"{h:>5} {f(a)} {f(b)}")
            continue
        d = b - a
        verdict = "tie" if abs(d) < TIE else ("RevIN helps" if d > 0 else "RevIN hurts")
        print(f"{h:>5} {f(a)} {f(b)} {d:>+9.4f}  {verdict}")

    # ---------------------------------------------------------------- C
    print("\n" + "=" * 70)
    print("C. Does a longer lookback help?   (ETTh1, H=96)")
    print("=" * 70)
    print("  The paper's sharpest claim: PatchTST improves with a longer")
    print("  lookback where other Transformers degrade. Patching is what makes")
    print("  the long lookback affordable, so this curve is the actual argument")
    print("  for the method.\n")
    print(f"{'L':>6} {'test MSE':>12}   trend")
    print("-" * 70)
    curve = []
    prev = None
    for L in LOOKBACKS:
        m = pick(runs, f"look_L{L}", 96, seq_len=L)
        if m is None:
            print(f"{L:>6} {f(m)}")
            continue
        if prev is None:
            trend = ""
        elif m < prev - TIE:
            trend = "  better"
        elif m > prev + TIE:
            trend = "  WORSE"
        else:
            trend = "  flat"
        print(f"{L:>6} {f(m)}{trend}")
        curve.append((L, m))
        prev = m

    if len(curve) >= 3:
        best_L, best_m = min(curve, key=lambda t: t[1])
        first_L, first_m = curve[0]
        last_L, last_m = curve[-1]
        print()
        print(f"  shortest L={first_L}: {first_m:.4f}")
        print(f"  best     L={best_L}: {best_m:.4f}")
        print(f"  longest  L={last_L}: {last_m:.4f}")
        print()
        improved = first_m - best_m > TIE
        degrades = last_m - best_m > TIE
        if improved and not degrades:
            print("  -> monotone improvement with lookback. Reproduces the claim.")
        elif improved and degrades:
            print(f"  -> PARTIAL. Improves out to L={best_L}, then degrades.")
            print("     The paper reports monotone gains, so this only supports")
            print("     the claim over the shorter range. Plausible causes: the")
            print("     reduced ETTh1 model (d_model=16) may lack the capacity to")
            print("     use a very long history, and longer L costs training")
            print("     windows. Needs seeds before treating as a real difference.")
        else:
            print("  -> longer lookback does NOT help. Contradicts the paper;")
            print("     investigate before reporting.")
    print()


if __name__ == "__main__":
    main()
