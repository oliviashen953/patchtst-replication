"""Step 13 — the job list for the ETTh1 sweeps, one line per run.

    python scripts/sweep_jobs.py            # print every job, numbered
    python scripts/sweep_jobs.py --count    # just how many there are
    python scripts/sweep_jobs.py --args 7   # the CLI arguments for job 7
    python scripts/sweep_jobs.py --group table7

Keeping the list in Python rather than in the SLURM script means the array
index maps to a job deterministically, the same list can be printed for a
human, and adding a group does not touch the submission script.

Groups, and what each one is for
--------------------------------
table7   Ablation of patching and channel-independence (paper Table 7 / 10).
         Step 10 covered (a) P+CI and (c) P only; these are the two missing
         no-patching cells. The paper marks some cells '-' for running out of
         memory on a 48GB A40 even at batch size 1, so an OOM here is a result,
         not a failure.
seeds    Paper Table 14: the same configuration at five seeds. This is the
         experiment that answers our standing "single seed" caveat.
table9   Paper Table 9: varying look-back window, all four horizons. Step 10C
         swept L over {96,192,336,512,720} at T=96 only -- which is not the
         paper's grid. Theirs is L in {24,48,96,192,336,720} (512 is the
         PatchTST/64 lookback, not a Table 9 column), and it covers every
         horizon, so this group runs the grid they actually used.
fig4     Paper Figure 4: MSE against patch length, L=336, T=96.
fig5     Paper Figure 5: MSE against (n_layers, d_model), T=96.
patch64  PatchTST/64 -- L=512, the paper's headline variant, against our /42.
arch     The four upstream defaults we do not match, one at a time plus all
         together, with a control that isolates the change of encoder
         implementation from the change of settings.
"""

from __future__ import annotations

import argparse

HORIZONS = [96, 192, 336, 720]
SEEDS = [2021, 2022, 2023, 2024, 2025]

# (n_layers, d_model) exactly as Figure 5 labels them 1..6.
FIG5_GRID = [(3, 128), (3, 256), (4, 128), (4, 256), (5, 128), (5, 256)]
FIG4_PATCH_LENS = [2, 4, 8, 12, 16, 24, 32, 40]

# Table 9 / Figure 2's x axis. Note 24 and 48 are shorter than one patch stride
# pair at P=16,S=8 -- L=24 gives just 3 patches -- which is the point: it is the
# left end of the curve where the model is starved of history.
TABLE9_LOOKBACKS = [24, 48, 96, 192, 336, 720]

# Each entry is one upstream default we currently deviate from. `arch_ctrl` is
# the control: the same settings as our baseline, run through the new encoder
# implementation, so a difference in any other row cannot be blamed on the
# rewrite.
ARCH_VARIANTS = [
    ("arch_ctrl", ["--encoder-impl", "tst"]),
    ("arch_bn", ["--encoder-impl", "tst", "--norm", "batch"]),
    ("arch_postnorm", ["--encoder-impl", "tst", "--post-norm"]),
    ("arch_resattn", ["--encoder-impl", "tst", "--res-attention"]),
    ("arch_attndrop0", ["--encoder-impl", "tst", "--attn-dropout", "0"]),
    ("arch_noaffine", ["--no-revin-affine"]),
    ("arch_upstream", ["--encoder-impl", "tst", "--norm", "batch", "--post-norm",
                       "--res-attention", "--attn-dropout", "0",
                       "--no-revin-affine"]),
]


def jobs() -> list[tuple[str, str, list[str]]]:
    """(group, tag, args) for every run, in a fixed order."""
    out: list[tuple[str, str, list[str]]] = []

    # -- Table 7: the two no-patching cells --------------------------------
    for h in HORIZONS:
        out.append(("table7", f"nopatch_ci_h{h}",
                    ["--pred-len", str(h), "--no-patching"]))
    for h in HORIZONS:
        out.append(("table7", f"nopatch_mix_h{h}",
                    ["--pred-len", str(h), "--no-patching", "--channel-mixing"]))

    # -- Table 14: seed variance -------------------------------------------
    for seed in SEEDS:
        for h in HORIZONS:
            out.append(("seeds", f"seed{seed}",
                        ["--pred-len", str(h), "--seed", str(seed)]))

    # -- Table 9: varying look-back window, every horizon -------------------
    for lookback in TABLE9_LOOKBACKS:
        for h in HORIZONS:
            out.append(("table9", f"look9_L{lookback}",
                        ["--pred-len", str(h), "--seq-len", str(lookback)]))

    # -- Figure 4: varying patch length ------------------------------------
    for p in FIG4_PATCH_LENS:
        out.append(("fig4", f"patchlen_P{p}",
                    ["--pred-len", "96", "--patch-len", str(p)]))

    # -- Figure 5: varying model size --------------------------------------
    for layers, d_model in FIG5_GRID:
        out.append(("fig5", f"size_L{layers}_D{d_model}",
                    ["--pred-len", "96", "--n-layers", str(layers),
                     "--d-model", str(d_model), "--n-heads", "16"]))

    # -- PatchTST/64 --------------------------------------------------------
    for h in HORIZONS:
        out.append(("patch64", f"patchtst64_h{h}",
                    ["--pred-len", str(h), "--seq-len", "512"]))

    # -- Architecture fidelity ---------------------------------------------
    for tag, extra in ARCH_VARIANTS:
        for h in HORIZONS:
            out.append(("arch", f"{tag}_h{h}",
                        ["--pred-len", str(h)] + extra))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--args", type=int, metavar="INDEX",
                        help="print the CLI arguments for one job")
    parser.add_argument("--group", help="restrict the listing to one group")
    opts = parser.parse_args()

    all_jobs = jobs()

    if opts.args is not None:
        if not 0 <= opts.args < len(all_jobs):
            raise SystemExit(f"index {opts.args} out of range 0..{len(all_jobs) - 1}")
        _, tag, args = all_jobs[opts.args]
        # --probe-test logs per-epoch TEST mse for inspection only; selection
        # still consults validation alone, and the probe loader carries its own
        # RNG so the run stays bit-identical. It is what makes the selection
        # diagnostic available for every one of these runs.
        print(" ".join(args + ["--tag", tag, "--probe-test"]))
        return

    if opts.count:
        print(len(all_jobs))
        return

    for i, (group, tag, args) in enumerate(all_jobs):
        if opts.group and group != opts.group:
            continue
        print(f"{i:3d}  {group:8s}  {tag:22s}  {' '.join(args)}")


if __name__ == "__main__":
    main()
