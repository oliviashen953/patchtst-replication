"""Figures — regenerate every plot in figures/ from the committed result JSONs.

    python experiments/figures.py            # all figures
    python experiments/figures.py --only f2  # just one

Style follows the paper, not my taste
-------------------------------------
The official repo contains exactly one plotting function -- `visual()` in
`PatchTST_supervised/utils/tools.py` -- and it is six lines of matplotlib
defaults: plot the ground truth, plot the prediction, legend, save. `f7` below
reproduces it exactly, including the upstream convention of concatenating the
lookback window onto the front of both series and plotting the last channel.

The metric figures follow Figure 2 of the paper (`pic/varying_L.png` upstream):
matplotlib's default colour cycle, a full box frame, dotted vertical gridlines
only, one marker shape per series, plain titles, and a single framed legend
below the panels. No annotations, no callouts, no value labels -- the numbers
live in the tables in RESULTS.md, and the figures show the shape.

Per-series marker shapes are not decoration: they carry the series identity
redundantly with colour, which is what makes the default cycle safe to reuse.

Provenance
----------
Nothing here trains anything. Every number is read back out of
experiments/results*/ , so a figure cannot disagree with RESULTS.md unless the
JSONs themselves changed. The one exception is `f7`, which loads saved
checkpoints and runs inference on CPU -- and checkpoints are gitignored, so f7
needs the runs present locally while f1-f6 work from a fresh clone.

Anything labelled `oracle` selects its epoch on the TEST split. Those series are
diagnostics and are never reportable results; they exist to separate "the model
cannot do better" from "validation picked the wrong epoch".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
RESULTS_CGM = ROOT / "experiments" / "results_cgm"
RESULTS_SSL = ROOT / "experiments" / "results_ssl"
FIGURES = ROOT / "figures"

HORIZONS = [96, 192, 336, 720]

# PatchTST/42 on ETTh1, Table 3. Our L=336/P=16/S=8 gives 42 patches, so /42 is
# the right column -- /64 (L=512) is a stronger model and comparing to it would
# flatter or damn us for the wrong reason.
PAPER_42 = {96: (0.375, 0.399), 192: (0.414, 0.421),
            336: (0.431, 0.436), 720: (0.449, 0.466)}

# Table 9, ETTh1: the paper's own look-back sweep. Its curve turns up past
# L=336 at three of four horizons, exactly as ours does -- the monotone claim
# in the text is about Figure 2's three large datasets, not this one.
PAPER_TABLE9 = {
    96:  {24: 0.464, 48: 0.410, 96: 0.393, 192: 0.382, 336: 0.375, 720: 0.376},
    192: {24: 0.521, 48: 0.469, 96: 0.445, 192: 0.428, 336: 0.414, 720: 0.413},
    336: {24: 0.570, 48: 0.516, 96: 0.484, 192: 0.451, 336: 0.431, 720: 0.445},
    720: {24: 0.575, 48: 0.509, 96: 0.480, 192: 0.452, 336: 0.449, 720: 0.458},
}

# Paper Figure 2 conventions.
MARKERS = ["o", "x", "^", "s", "+", "D"]

plt.rcParams.update({
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.linestyle": ":",
    "grid.color": "0.75",
    "grid.linewidth": 0.9,
    "axes.axisbelow": True,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# --------------------------------------------------------------------- loading

def load(directory: Path, pattern: str) -> list[dict]:
    """Every real run matching `pattern`. Smoke tests are excluded by name.

    That exclusion is the Step 12 bug: smoke_ft.json carries stage='finetune'
    and pred_len=96 exactly like the real run, so anything keyed on
    (stage, pred_len) picked up whichever the glob yielded first.
    """
    return [json.loads(p.read_text()) for p in sorted(directory.glob(pattern))
            if not p.name.startswith("smoke_")]


def etth1(tag: str) -> dict[int, dict]:
    """{pred_len: record} for one Step 9/10 tag, keyed by horizon.

    Step 10 submitted its ablations one horizon per task, so their tags carry
    the horizon too ('mix_h720'). Match on the family, then key on the horizon
    the record itself reports rather than on the string.
    """
    runs = load(RESULTS, "etth1_*.json")
    out = {r["pred_len"]: r for r in runs
           if r.get("tag") == tag or r.get("tag", "").startswith(tag + "_h")}
    if not out:
        raise SystemExit(f"no runs tagged {tag!r} in {RESULTS}")
    missing = [h for h in HORIZONS if h not in out]
    if missing:
        raise SystemExit(f"tag {tag!r} is missing horizons {missing}")
    return out


# ----------------------------------------------------------------- plot helpers

def series_plot(ax, x, series, *, xlabel, ylabel, title):
    """One panel of paper-Figure-2-style lines over a categorical x axis."""
    pos = np.arange(len(x))
    for i, (label, values) in enumerate(series):
        ax.plot(pos, values, marker=MARKERS[i % len(MARKERS)], label=label)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(x=0.08)


def legend_below(fig, ax, ncol=None):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center",
               ncol=ncol or len(labels), frameon=True, fancybox=True)


def save(fig, name: str, dpi: int) -> Path:
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


# ------------------------------------------------------- f1: the headline table

def f1_replication(dpi: int):
    """Step 9: ours vs PatchTST/42, MSE and MAE, all four horizons."""
    base = etth1("base")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, metric, idx in [(axes[0], "mse", 0), (axes[1], "mae", 1)]:
        series_plot(
            ax, HORIZONS,
            [("ours", [base[h]["test"][metric] for h in HORIZONS]),
             ("PatchTST/42 (paper)", [PAPER_42[h][idx] for h in HORIZONS])],
            xlabel="T", ylabel=metric.upper(), title=f"ETTh1 {metric.upper()}, L=336")
    legend_below(fig, axes[0])
    return save(fig, "f1_replication", dpi)


# ---------------------------------------------- f2: the curves that explain it

def f2_curves(dpi: int):
    """Per-epoch train / validation / test for each horizon, with both the
    validation-selected and the test-oracle epoch marked."""
    base = etth1("base")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), constrained_layout=True)

    for ax, h in zip(axes.ravel(), HORIZONS):
        r = base[h]
        hist = r["history"]
        ep = np.arange(len(hist["val_mse"]))

        ax.plot(ep, hist["train_loss"], label="train loss")
        ax.plot(ep, hist["val_mse"], label="validation MSE")
        if hist["probe_mse"]:
            ax.plot(ep, hist["probe_mse"], label="test MSE (probe)")
        ax.axvline(r["best_epoch"], color="0.4", linestyle="--", linewidth=1.2,
                   label="epoch kept (validation)")
        if hist["probe_mse"]:
            ax.axvline(r["probe_best_epoch"], color="0.4", linestyle=":",
                       linewidth=1.2, label="epoch a test oracle would keep")

        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE")
        ax.set_title(f"ETTh1 T={h}")
        ax.grid(axis="x", visible=False)   # epoch axis is continuous, not ticks
    legend_below(fig, axes[0, 0], ncol=3)
    return save(fig, "f2_training_curves", dpi)


# ------------------------------------------------------- f3: the selection cost

def f3_selection(dpi: int):
    """What validation-selection costs at each horizon, against the paper."""
    base = etth1("base")
    if not base[720]["history"]["probe_mse"]:
        raise SystemExit("no probe data -- rerun with PROBE=1 sbatch scripts/train_slurm.sh")

    val_sel = [base[h]["test"]["mse"] for h in HORIZONS]
    oracle = [base[h]["probe_best_mse"] for h in HORIZONS]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    series_plot(
        axes[0], HORIZONS,
        [("ours, validation-selected", val_sel),
         ("ours, test-oracle epoch", oracle),
         ("PatchTST/42 (paper)", [PAPER_42[h][0] for h in HORIZONS])],
        xlabel="T", ylabel="MSE", title="ETTh1 MSE by selection rule")
    series_plot(
        axes[1], HORIZONS,
        [("MSE lost to selection", [v - o for v, o in zip(val_sel, oracle)])],
        xlabel="T", ylabel="MSE", title="Cost of selecting on validation")
    axes[1].legend()
    legend_below(fig, axes[0])
    return save(fig, "f3_selection_cost", dpi)


# ------------------------------------------------------------- f4: ablations

def f4_ablations(dpi: int):
    """Step 10: channel independence, RevIN, and the lookback sweep."""
    base, mix, norevin = etth1("base"), etth1("mix"), etth1("norevin")
    # Key on (T, L), NOT on L alone. The lookback runs are tagged look_L<L> at
    # BOTH horizons, so keying on seq_len silently collapses look_L96_h96 and
    # look_L96_h720 onto one entry and keeps whichever the glob yielded last.
    # Same failure that put a smoke run in the Step 12 table; see collate.py.
    # Step 13 re-ran this on the paper's own grid (look9_*). Prefer those; the
    # Step 10C runs (look_*) used L=512, which is the PatchTST/64 lookback and
    # not a Table 9 column.
    look = {(r["pred_len"], r["model"]["seq_len"]): r
            for r in load(RESULTS, "etth1_look9_*.json")}

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), constrained_layout=True)
    series_plot(
        axes[0], HORIZONS,
        [("channel-independent", [base[h]["test"]["mse"] for h in HORIZONS]),
         ("channel-mixing", [mix[h]["test"]["mse"] for h in HORIZONS])],
        xlabel="T", ylabel="MSE", title="ETTh1, channel independence")
    axes[0].legend()
    series_plot(
        axes[1], HORIZONS,
        [("with RevIN", [base[h]["test"]["mse"] for h in HORIZONS]),
         ("without RevIN", [norevin[h]["test"]["mse"] for h in HORIZONS])],
        xlabel="T", ylabel="MSE", title="ETTh1, RevIN")
    axes[1].legend()
    # The paper's own Figure 2 axes: MSE against L, one panel per horizon.
    # Figure 2's caption fixes them -- "L = 24, 48, 96, 192, 336, 720 ... T = 96,
    # 720" -- so both panels are drawn even though only T=96 was swept at first.
    # L=512 is ours, not the paper's; it is the PatchTST/64 geometry.
    for ax, T in ((axes[2], 96), (axes[3], 720)):
        ls = sorted(L for (t, L) in look if t == T)
        if not ls:
            ax.set_visible(False)
            continue
        series_plot(
            ax, ls,
            [("PatchTST (ours)", [look[(T, L)]["test"]["mse"] for L in ls]),
             ("PatchTST/42 (paper, Table 9)", [PAPER_TABLE9[T][L] for L in ls])],
            xlabel="L", ylabel="MSE", title=f"ETTh1 T={T}, varying look-back")
        ax.legend()
    return save(fig, "f4_ablations", dpi)


# ------------------------------------------------------------------ f5: CGM

def f5_cgm(dpi: int):
    """Step 11: the 2x2, in mg/dL, with the capacity control beside it."""
    runs = {r["tag"]: r for r in load(RESULTS_CGM, "*.json")}
    need = ["ci_drivers", "mix_drivers", "ci_control", "mix_control"]
    if any(t not in runs for t in need):
        raise SystemExit(f"missing CGM runs: {[t for t in need if t not in runs]}")

    def rmse(tag):
        return runs[tag]["test"]["cgm_rmse_mgdl"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)

    ax = axes[0]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, [rmse("ci_drivers"), rmse("ci_control")], w,
           label="channel-independent")
    ax.bar(x + w / 2, [rmse("mix_drivers"), rmse("mix_control")], w,
           label="channel-mixing")
    ax.set_xticks(x)
    ax.set_xticklabels(["drivers informative", "drivers zeroed"])
    ax.set_ylabel("CGM RMSE (mg/dL)")
    ax.set_title("Synthetic CGM, 4 h to 1 h ahead")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    params = [runs["ci_drivers"]["n_parameters"], runs["mix_drivers"]["n_parameters"]]
    ax.bar([0, 1], params, 0.5, color=["C0", "C1"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["channel-indep", "channel-mixing"])
    ax.set_ylabel("parameters")
    ax.set_title("Capacity of the two architectures")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.grid(axis="x", visible=False)

    legend_below(fig, axes[0])
    return save(fig, "f5_cgm", dpi)


# ------------------------------------------------------------------ f6: SSL

def f6_ssl(dpi: int):
    """Step 12: the verdict under validation selection, and under a fixed rule."""
    runs = load(RESULTS_SSL, "*.json")
    arms = ["scratch", "linear_probe", "finetune"]
    labels = {"scratch": "scratch", "linear_probe": "linear probe",
              "finetune": "fine-tune"}

    def val_sel(stage, h):
        return next((r["test"]["mse"] for r in runs
                     if r.get("stage") == stage and r.get("pred_len") == h), None)

    def oracle(stage, h):
        return next((r["phases"][-1].get("probe_best_mse") for r in runs
                     if r.get("stage") == stage and r.get("pred_len") == h), None)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True,
                             sharey=True)
    for ax, getter, title in [
        (axes[0], val_sel, "Selected on validation"),
        (axes[1], oracle, "Selected on test (oracle)"),
    ]:
        series_plot(ax, HORIZONS,
                    [(labels[a], [getter(a, h) for h in HORIZONS]) for a in arms],
                    xlabel="T", ylabel="MSE", title=title)
    legend_below(fig, axes[0])
    return save(fig, "f6_ssl", dpi)


# -------------------------------------------------------- f7: actual forecasts

def official_visual(ax, true, preds=None):
    """The upstream plot, unchanged.

    `PatchTST_supervised/utils/tools.py` in yuqinie98/PatchTST:

        def visual(true, preds=None, name='./pic/test.pdf'):
            plt.figure()
            plt.plot(true, label='GroundTruth', linewidth=2)
            if preds is not None:
                plt.plot(preds, label='Prediction', linewidth=2)
            plt.legend()
            plt.savefig(name, bbox_inches='tight')

    Same two series, same labels, same linewidth, same default colours -- only
    bound to an axis so several windows can share one figure.
    """
    ax.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        ax.plot(preds, label="Prediction", linewidth=2)


def f7_forecasts(dpi: int, device: str = "cpu"):
    """Qualitative check, drawn the way the official repo draws it.

    Upstream calls visual() on every 20th test batch with the lookback window
    concatenated onto the front of both series and the last channel selected:

        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

    That convention is why the two lines overlap on the left of every panel:
    the first `seq_len` points are the same input in both.
    """
    sys.path.insert(0, str(ROOT))
    import torch
    from patchtst.data import make_datasets
    from patchtst.model import PatchTST

    base = etth1("base")
    shown = [96, 720]
    n_cols = 3
    fig, axes = plt.subplots(len(shown), n_cols, figsize=(13, 6.4),
                             constrained_layout=True)

    for row, h in enumerate(shown):
        rec = base[h]
        ckpt_path = ROOT / rec["train"]["checkpoint"]
        if not ckpt_path.exists():
            raise SystemExit(f"{ckpt_path} missing -- rerun scripts/train_slurm.sh")

        kwargs = dict(rec["model"])
        model = PatchTST(n_channels=7, pred_len=h, **kwargs)
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["state_dict"])
        model.to(device).eval()

        _, _, test_set = make_datasets(str(ROOT / "data" / "ETTh1.csv"),
                                       seq_len=kwargs["seq_len"], pred_len=h)
        # Evenly spaced windows, so these are not hand-picked best cases.
        idxs = np.linspace(0, len(test_set) - 1, n_cols + 2)[1:-1].astype(int)

        for col, i in enumerate(idxs):
            ax = axes[row, col]
            x, y = test_set[int(i)]
            with torch.no_grad():
                pred = model(x.unsqueeze(0).to(device))[0].cpu().numpy()

            # Upstream's channel -1 and upstream's concatenation.
            gt = np.concatenate((x.numpy()[:, -1], y.numpy()[:, -1]), axis=0)
            pd = np.concatenate((x.numpy()[:, -1], pred[:, -1]), axis=0)
            official_visual(ax, gt, pd)
            ax.set_title(f"ETTh1 T={h}, test window {i}")
            ax.grid(axis="x", visible=False)

    legend_below(fig, axes[0, 0])
    return save(fig, "f7_forecasts", dpi)


def f8_architecture(dpi: int):
    """Step 13: the encoder norm closes the H=720 gap, and why."""
    base = etth1("base")
    variants = [("control", "arch_ctrl"), ("BatchNorm", "arch_bn"),
                ("post-norm", "arch_postnorm"), ("res-attn", "arch_resattn"),
                ("attn-drop 0", "arch_attndrop0"), ("all five", "arch_upstream")]

    def mse(tag, h):
        r = load(RESULTS, f"etth1_{tag}_h{h}_h{h}_s2021.json")
        return r[0]["test"]["mse"] if r else float("nan")

    def rec(tag, h):
        r = load(RESULTS, f"etth1_{tag}_h{h}_h{h}_s2021.json")
        return r[0] if r else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)

    series_plot(
        axes[0], HORIZONS,
        [("ours, as published in Step 9", [base[h]["test"]["mse"] for h in HORIZONS])]
        + [(label, [mse(tag, h) for h in HORIZONS]) for label, tag in variants]
        + [("PatchTST/42 (paper)", [PAPER_42[h][0] for h in HORIZONS])],
        xlabel="T", ylabel="MSE", title="ETTh1 MSE by encoder variant")
    axes[0].legend(fontsize=7.5, ncol=2)

    # The mechanism: with BatchNorm the epoch validation keeps is the epoch
    # test would have kept, so the selection error disappears.
    ax = axes[1]
    labels, costs = ["Step 9\nbase"], [
        (base[720].get("probe_at_val_best") or 0) - (base[720].get("probe_best_mse") or 0)]
    for label, tag in variants:
        r = rec(tag, 720)
        if r:
            labels.append(label.replace(" ", "\n"))
            costs.append((r.get("probe_at_val_best") or 0) - (r.get("probe_best_mse") or 0))
    ax.bar(np.arange(len(costs)), costs)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MSE lost to checkpoint selection")
    ax.set_title("T=720: selection error, by variant")
    ax.grid(axis="x", visible=False)
    return save(fig, "f8_architecture", dpi)


FIGS = {
    "f1": f1_replication,
    "f8": f8_architecture,
    "f2": f2_curves,
    "f3": f3_selection,
    "f4": f4_ablations,
    "f5": f5_cgm,
    "f6": f6_ssl,
    "f7": f7_forecasts,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(FIGS),
                        help="figure keys to build (default: all)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    for k in args.only or sorted(FIGS):
        print(f"[{k}] {FIGS[k].__doc__.splitlines()[0]}")
        FIGS[k](args.dpi)


if __name__ == "__main__":
    main()
