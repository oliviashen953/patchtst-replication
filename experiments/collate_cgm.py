"""Summarize the Step 11 CGM 2x2.

    python experiments/collate_cgm.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results_cgm"


def main() -> None:
    runs = {}
    for path in RESULTS.glob("*.json"):
        r = json.loads(path.read_text())
        runs[r["tag"]] = r
    if not runs:
        raise SystemExit(f"no results in {RESULTS} -- run scripts/cgm_slurm.sh")

    def get(tag, key="cgm_rmse_mgdl"):
        r = runs.get(tag)
        return r["test"][key] if r else None

    def fmt(v, w=9):
        return f"{v:{w}.3f}" if v is not None else f"{'--':>{w}}"

    print("\nCGM forecasting, 4 h lookback -> 1 h ahead, synthetic data")
    print("RMSE on the CGM channel, mg/dL (lower is better)\n")
    print(f"{'':>22} {'channel-indep':>14} {'mixing':>10} {'Δ':>9}")
    print("-" * 60)

    rows = []
    for label, ci_tag, mix_tag in (
        ("drivers informative", "ci_drivers", "mix_drivers"),
        ("drivers zeroed (ctrl)", "ci_control", "mix_control"),
    ):
        a, b = get(ci_tag), get(mix_tag)
        d = (b - a) if (a is not None and b is not None) else None
        print(f"{label:>22} {fmt(a,14)} {fmt(b,10)} "
              f"{(f'{d:+9.3f}' if d is not None else f'{chr(45)*2:>9}')}")
        rows.append((label, a, b, d))

    print()
    real, ctrl = rows[0][3], rows[1][3]
    if real is None or ctrl is None:
        print("  (incomplete -- need all four runs)")
        return

    print("Reading it:")
    if real < -0.05:
        print(f"  * With informative meal/bolus, MIXING wins by {-real:.3f} mg/dL.")
        if ctrl > -0.05:
            print(f"  * With the drivers zeroed, that advantage disappears "
                  f"(Δ = {ctrl:+.3f}).")
            print()
            print("  => The gain comes from the CAUSAL INFORMATION in meal/bolus,")
            print("     not from mixing simply being a larger model. Channel")
            print("     independence -- which wins on ETTh1 -- costs accuracy here,")
            print("     because it structurally forbids the model from seeing the")
            print("     inputs that drive the target.")
        else:
            print(f"  * But mixing ALSO wins with the drivers zeroed "
                  f"(Δ = {ctrl:+.3f}), so at least part of the gain is capacity,")
            print("    not causal information. Cannot attribute it to meal/bolus.")
    else:
        print(f"  * Mixing does NOT beat channel independence here "
              f"(Δ = {real:+.3f}).")
        print("    The prediction fails: even with genuinely causal drivers,")
        print("    isolating channels was not harmful. Worth investigating before")
        print("    drawing any conclusion about real CGM.")

    print()
    print("Caveats: synthetic data, single seed, one architecture size.")
    print("Nothing here transfers to real CGM without real data.")
    print()


if __name__ == "__main__":
    main()
