"""Summarize the Step 13 ETTh1 sweeps.

    python experiments/collate_sweep.py               # every group
    python experiments/collate_sweep.py --group arch

One section per paper target. Anything missing prints as `--` rather than
being silently skipped, and a run that died out of memory prints as `OOM`,
because for the no-patching cells that is the same thing the paper reports.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
HORIZONS = [96, 192, 336, 720]
SEEDS = [2021, 2022, 2023, 2024, 2025]

# Table 3, read off the paper. PatchTST/42 is our comparison (L=336, 42
# patches); /64 is the L=512 variant that the patch64 group targets.
PAPER_42 = {96: (0.375, 0.399), 192: (0.414, 0.421),
            336: (0.431, 0.436), 720: (0.449, 0.466)}
PAPER_64 = {96: (0.370, 0.400), 192: (0.413, 0.429),
            336: (0.422, 0.440), 720: (0.447, 0.468)}

# Table 10 (the full version of Table 7), ETTh1 block: MSE per horizon for
# each of the four ablation cases.
#
# Read this row before reading ours. On ETTh1 the paper's own CI-only column
# BEATS or ties its full model at every horizon, and its text concedes the
# point -- patching plus channel-independence wins "especially on larger
# datasets (Weather, Traffic, and Electricity) where the models are less
# susceptible to overfitting". ETTh1 is not one of those.
PAPER_TABLE10 = {
    96:  {"P+CI": 0.375, "CI": 0.365, "P": 0.416, "neither": 0.455},
    192: {"P+CI": 0.414, "CI": 0.403, "P": 0.459, "neither": 0.503},
    336: {"P+CI": 0.431, "CI": 0.430, "P": 0.484, "neither": 0.514},
    720: {"P+CI": 0.449, "CI": 0.449, "P": 0.500, "neither": 0.531},
}

# Table 9, ETTh1 block: MSE by look-back window, per horizon.
#
# Note where the minimum falls. At three of four horizons the paper's own
# ETTh1 numbers get WORSE past L=336 -- the monotone improvement it describes
# is a statement about Figure 2's three large datasets, not about this one.
PAPER_TABLE9 = {
    96:  {24: 0.464, 48: 0.410, 96: 0.393, 192: 0.382, 336: 0.375, 720: 0.376},
    192: {24: 0.521, 48: 0.469, 96: 0.445, 192: 0.428, 336: 0.414, 720: 0.413},
    336: {24: 0.570, 48: 0.516, 96: 0.484, 192: 0.451, 336: 0.431, 720: 0.445},
    720: {24: 0.575, 48: 0.509, 96: 0.480, 192: 0.452, 336: 0.449, 720: 0.458},
}
TABLE9_LOOKBACKS = [24, 48, 96, 192, 336, 720]

FIG5_GRID = [(3, 128), (3, 256), (4, 128), (4, 256), (5, 128), (5, 256)]
FIG4_PATCH_LENS = [2, 4, 8, 12, 16, 24, 32, 40]
ARCH_TAGS = ["arch_ctrl", "arch_bn", "arch_postnorm", "arch_resattn",
             "arch_attndrop0", "arch_noaffine", "arch_upstream"]
ARCH_WHAT = {
    "arch_ctrl": "control: same settings, new encoder impl",
    "arch_bn": "BatchNorm instead of LayerNorm",
    "arch_postnorm": "post-norm instead of pre-norm",
    "arch_resattn": "residual attention (RealFormer)",
    "arch_attndrop0": "attention dropout 0, not 0.3",
    "arch_noaffine": "RevIN without affine parameters",
    "arch_upstream": "all five upstream defaults at once",
}


def load() -> dict[str, dict]:
    """{name: record} for every ETTh1 result on disk."""
    out = {}
    for p in sorted(RESULTS.glob("etth1_*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            print(f"  ! {p.name} is not valid JSON, skipping")
            continue
        out[r["name"]] = r
    return out


def get(runs, tag, pred_len, seed=2021):
    return runs.get(f"etth1_{tag}_h{pred_len}_s{seed}")


def oom_tags() -> set[str]:
    """Tags whose SLURM task died with CUDA OOM.

    The paper marks such cells '-' in Table 7 ("runs out of GPU memory (NVIDIA
    A40 48GB) even with batch size 1"), so this is a finding to report, not an
    error to hide. Detected by scanning the sweep logs.
    """
    found = set()
    logs = Path(__file__).resolve().parent.parent / "logs"
    for err in logs.glob("sweep_*.err"):
        text = err.read_text(errors="ignore")
        if "CUDA out of memory" not in text and "OutOfMemoryError" not in text:
            continue
        out = err.with_suffix(".out")
        if not out.exists():
            continue
        for line in out.read_text(errors="ignore").splitlines():
            if line.startswith("args:") and "--tag" in line:
                parts = line.split()
                found.add(parts[parts.index("--tag") + 1])
    return found


def cell(runs, oom, tag, h, metric="mse", seed=2021):
    if tag in oom:
        return f"{'OOM':>9}"
    r = get(runs, tag, h, seed)
    return f"{r['test'][metric]:9.4f}" if r else f"{'--':>9}"


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def do_table7(runs, oom):
    section("Table 7 / 10 — patching (P) and channel-independence (CI), test MSE")
    print("Step 10 supplied the two patched cells; the no-patching cells are new.")
    print(f"\n{'H':>5}{'P+CI (a)':>11}{'CI only (b)':>13}{'P only (c)':>12}"
          f"{'neither (d)':>13}   best")
    for h in HORIZONS:
        cells = {
            "P+CI": ("base", None),
            "CI": (f"nopatch_ci_h{h}", None),
            "P": (f"mix_h{h}", None),
            "neither": (f"nopatch_mix_h{h}", None),
        }
        vals = {}
        for k, (tag, _) in cells.items():
            r = None if tag in oom else get(runs, tag, h)
            vals[k] = r["test"]["mse"] if r else None
        best = min((v, k) for k, v in vals.items() if v is not None)[1] \
            if any(v is not None for v in vals.values()) else "--"
        row = "".join(
            f"{'OOM':>11}" if cells[k][0] in oom else
            (f"{vals[k]:11.4f}" if vals[k] is not None else f"{'--':>11}")
            for k in ["P+CI", "CI", "P", "neither"])
        print(f"{h:>5}{row}   {best}")
        paper_row = "".join(f"{PAPER_TABLE10[h][k]:11.3f}"
                            for k in ["P+CI", "CI", "P", "neither"])
        paper_best = min((v, k) for k, v in PAPER_TABLE10[h].items())[1]
        print(f"{'':>5}{paper_row}   {paper_best}   <- paper, Table 10")
    print("\n  The paper's own ETTh1 CI-only column beats or ties its full model at")
    print("  every horizon. Its claim that patching+CI wins is made for the larger")
    print("  datasets; on ETTh1 its own numbers do not show it. '-'/OOM cells are")
    print("  reported by the paper too, for the same reason.")


def do_seeds(runs, oom):
    section("Table 14 — seed variance, test MSE")
    print(f"{'H':>5}" + "".join(f"{s:>10}" for s in SEEDS)
          + f"{'mean':>10}{'std':>9}{'paper/42':>10}{'mean-paper':>12}")
    for h in HORIZONS:
        vals = []
        cells = []
        for s in SEEDS:
            r = get(runs, f"seed{s}", h, seed=s)
            if r:
                vals.append(r["test"]["mse"])
                cells.append(f"{r['test']['mse']:10.4f}")
            else:
                cells.append(f"{'--':>10}")
        if vals:
            m = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            tail = f"{m:10.4f}{sd:9.4f}{PAPER_42[h][0]:10.3f}{m - PAPER_42[h][0]:+12.4f}"
        else:
            tail = f"{'--':>10}{'--':>9}{PAPER_42[h][0]:10.3f}{'--':>12}"
        print(f"{h:>5}" + "".join(cells) + tail)
    print("\n  This is the experiment that retires the repo's 'single seed' caveat.")
    print("  A gap smaller than the std across seeds is not a gap.")


def do_table9(runs, oom):
    section("Table 9 — varying look-back window L, test MSE (ours / paper)")
    print("Step 10C used L in {96,192,336,512,720} at T=96 only. That is not the")
    print("paper's grid: 512 is the PatchTST/64 lookback, and 24 and 48 are the")
    print("left end of Figure 2's x axis, where the model is starved of history.")
    print(f"\n{'T':>5}{'':>3}" + "".join(f"{'L=' + str(L):>16}" for L in TABLE9_LOOKBACKS))
    for h in HORIZONS:
        ours, paper = [], []
        for L in TABLE9_LOOKBACKS:
            r = get(runs, f"look9_L{L}", h)
            ours.append(f"{r['test']['mse']:16.4f}" if r else f"{'--':>16}")
            paper.append(f"{PAPER_TABLE9[h][L]:16.3f}")
        print(f"{h:>5}{'ours':>3}" + "".join(ours))
        print(f"{'':>5}{'pap':>3}" + "".join(paper))
        best_ours = min(
            ((get(runs, f"look9_L{L}", h) or {}).get("test", {}).get("mse", 9e9), L)
            for L in TABLE9_LOOKBACKS)
        best_paper = min((PAPER_TABLE9[h][L], L) for L in TABLE9_LOOKBACKS)
        ours_best = f"L={best_ours[1]}" if best_ours[0] < 9e9 else "--"
        print(f"{'':>8}  minimum at: ours {ours_best},  paper L={best_paper[1]}")
    print("\n  The paper's ETTh1 minimum is at L=336 for three of four horizons,")
    print("  and its L=720 column is WORSE than its L=336 column at 96, 336 and")
    print("  720. Its 'performance improves with a longer look-back' claim is")
    print("  made about Figure 2's three large datasets, not about ETTh1.")


def do_fig4(runs, oom):
    section("Figure 4 — varying patch length P, L=336, T=96")
    print(f"{'P':>5}{'stride':>8}{'patches':>9}{'test MSE':>11}")
    for p in FIG4_PATCH_LENS:
        r = get(runs, f"patchlen_P{p}", 96)
        if r:
            m = r["model"]
            # The JSON stores the geometry, not the token count; recompute it
            # the same way patchtst.patching.num_patches does.
            n = (m["seq_len"] - m["patch_len"]) // m["stride"] + 1 + 1
            print(f"{p:>5}{m['stride']:>8}{n:>9}{r['test']['mse']:11.4f}")
        else:
            print(f"{p:>5}{'--':>8}{'--':>9}{'--':>11}")
    print("\n  Stride is P/2 throughout: the paper does not state it for this")
    print("  figure, so the overlap ratio of its P=16,S=8 setting is held fixed.")


def do_fig5(runs, oom):
    section("Figure 5 — varying model size, T=96")
    print(f"{'#':>3}{'n_layers':>10}{'d_model':>9}{'params':>12}{'test MSE':>11}")
    for i, (layers, d_model) in enumerate(FIG5_GRID, start=1):
        r = get(runs, f"size_L{layers}_D{d_model}", 96)
        if r:
            print(f"{i:>3}{layers:>10}{d_model:>9}{r['n_parameters']:>12,}"
                  f"{r['test']['mse']:11.4f}")
        else:
            print(f"{i:>3}{layers:>10}{d_model:>9}{'--':>12}{'--':>11}")
    print("\n  d_ff = 2*d_model and n_heads = 16, the paper's defaults. Note these")
    print("  are far larger than the reduced ETTh1 model (d_model=16) that")
    print("  Appendix A.1.4 prescribes, so higher MSE here is the expected result.")


def do_patch64(runs, oom):
    section("PatchTST/64 (L=512) against our /42 (L=336) and the paper")
    print(f"{'H':>5}{'ours/64':>10}{'paper/64':>10}{'delta':>9}"
          f"{'ours/42':>10}{'paper/42':>10}{'delta':>9}")
    for h in HORIZONS:
        r64 = get(runs, f"patchtst64_h{h}", h)
        r42 = get(runs, "base", h)
        a = f"{r64['test']['mse']:10.4f}" if r64 else f"{'--':>10}"
        da = f"{r64['test']['mse'] - PAPER_64[h][0]:+9.4f}" if r64 else f"{'--':>9}"
        b = f"{r42['test']['mse']:10.4f}" if r42 else f"{'--':>10}"
        db = f"{r42['test']['mse'] - PAPER_42[h][0]:+9.4f}" if r42 else f"{'--':>9}"
        print(f"{h:>5}{a}{PAPER_64[h][0]:10.3f}{da}{b}{PAPER_42[h][0]:10.3f}{db}")
    print("\n  The paper's /64 beats its /42 at every horizon. Step 10C found our")
    print("  lookback gains flattening past L=336, so this is a direct test.")


def do_arch(runs, oom):
    section("Architecture fidelity — upstream defaults we do not match, test MSE")
    for tag in ARCH_TAGS:
        print(f"  {tag:16s} {ARCH_WHAT[tag]}")
    print(f"\n{'H':>5}{'base':>10}" + "".join(f"{t.replace('arch_', ''):>11}"
                                              for t in ARCH_TAGS) + f"{'paper':>9}")
    for h in HORIZONS:
        base = get(runs, "base", h)
        row = f"{base['test']['mse']:10.4f}" if base else f"{'--':>10}"
        for tag in ARCH_TAGS:
            r = get(runs, f"{tag}_h{h}", h)
            row += f"{r['test']['mse']:11.4f}" if r else f"{'--':>11}"
        print(f"{h:>5}{row}{PAPER_42[h][0]:9.3f}")

    print(f"\n{'H':>5}{'variant':>16}{'best epoch':>12}{'test@oracle':>13}"
          f"{'selection cost':>16}")
    print("  Does any variant move the epoch validation keeps? That is the")
    print("  mechanism the H=720 gap was traced to.")
    for h in HORIZONS:
        base = get(runs, "base", h)
        if base:
            cost = ((base.get("probe_at_val_best") or 0)
                    - (base.get("probe_best_mse") or 0))
            print(f"{h:>5}{'base':>16}{base['best_epoch']:>12}"
                  f"{base.get('probe_best_mse') or float('nan'):13.4f}{cost:16.4f}")
        for tag in ARCH_TAGS:
            r = get(runs, f"{tag}_h{h}", h)
            if not r:
                continue
            cost = (r.get("probe_at_val_best") or 0) - (r.get("probe_best_mse") or 0)
            print(f"{'':>5}{tag.replace('arch_', ''):>16}{r['best_epoch']:>12}"
                  f"{r.get('probe_best_mse') or float('nan'):13.4f}{cost:16.4f}")


GROUPS = {"table7": do_table7, "seeds": do_seeds, "table9": do_table9,
          "fig4": do_fig4,
          "fig5": do_fig5, "patch64": do_patch64, "arch": do_arch}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(GROUPS))
    opts = parser.parse_args()

    runs = load()
    if not runs:
        raise SystemExit(f"no results in {RESULTS}")
    oom = oom_tags()
    if oom:
        print(f"\nOOM (reported as such, not dropped): {sorted(oom)}")

    for name in ([opts.group] if opts.group else list(GROUPS)):
        GROUPS[name](runs, oom)
    print()


if __name__ == "__main__":
    main()
