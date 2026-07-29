"""Step 9 — Reproduce the paper's ETTh1 benchmark row.

    python experiments/run_etth1.py --pred-len 96 --device cuda

Writes one JSON per run to experiments/results/, which experiments/collate.py
then turns into a comparison table.

Which configuration this reproduces
-----------------------------------
Our lookback is L=336 with P=16, S=8, giving 42 patches -- so the right
comparison is **PatchTST/42**, not PatchTST/64 (which uses L=512). Comparing
against the wrong column is an easy way to conclude your implementation is
broken when it is not.

The ETTh1 hyperparameters are NOT the paper's defaults. Appendix A.1.4 says
that for very small datasets (ILI, ETTh1, ETTh2) a reduced model is used to
mitigate overfitting: n_heads=4, d_model=16, d_ff=128, versus the defaults of
16/128/256. ETTh1 has only 8209 training windows, so the default-size model
overfits badly. This is the single most likely reason a from-scratch attempt
misses the paper's number.

Known deviations from the official setup
----------------------------------------
Recorded honestly rather than hidden, because they may cost a little accuracy:

  1. LR schedule. The official ETTh1 script uses `lradj type3` (constant for a
     few epochs, then decayed by a fixed factor). We use cosine annealing.
  2. Dropout. The paper text states 0.2 everywhere, but the released ETTh1
     script uses 0.3 (and fc_dropout 0.3). We follow the script, since that is
     what produced the published numbers.
  3. Early stopping. The official default patience is 100 with 100 epochs, so
     it effectively never triggers -- they train the full 100 epochs and keep
     the best-validation checkpoint. We do the same by default.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from patchtst.data import make_datasets
from patchtst.model import PatchTST
from patchtst.train import TrainConfig, fit, test

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "ETTh1.csv"
RESULTS = Path(__file__).resolve().parent / "results"

# PatchTST/42 on ETTh1, from Table 3 of the paper. Reference targets only.
PAPER_PATCHTST_42 = {
    96: {"mse": 0.375, "mae": 0.399},
    192: {"mse": 0.414, "mae": 0.421},
    336: {"mse": 0.431, "mae": 0.436},
    720: {"mse": 0.449, "mae": 0.466},
}

# The reduced ETTh1 model from Appendix A.1.4 -- NOT the paper defaults.
ETTH1_MODEL = dict(
    seq_len=336,
    patch_len=16,
    stride=8,
    d_model=16,
    n_heads=4,
    n_layers=3,
    d_ff=128,
    dropout=0.3,
    head_dropout=0.0,
    individual=False,
    revin=True,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-len", type=int, required=True,
                        choices=[96, 192, 336, 720])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-3,
                        help="upstream exp_main.py uses plain Adam, i.e. 0.0")
    parser.add_argument("--lradj", default="cosine", choices=["cosine", "type3"],
                        help="type3 matches the official ETTh1 script")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tag", default="base", help="label for this run")
    # Ablation switches, used by Step 10.
    parser.add_argument("--no-revin", action="store_true",
                        help="disable reversible instance normalization")
    parser.add_argument("--channel-mixing", action="store_true",
                        help="fold channels into the SEQUENCE, so attention "
                             "crosses channels (the thing PatchTST avoids)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="override the lookback window L (default 336)")
    # Step 13 sweeps. Every one of these defaults to the Step 9 configuration,
    # so omitting them reproduces the baseline exactly.
    parser.add_argument("--patch-len", type=int, default=None,
                        help="P. Figure 4 sweeps this over [2..40]")
    parser.add_argument("--stride", type=int, default=None,
                        help="S. Defaults to P/2, preserving the 50%% overlap "
                             "of the paper's P=16,S=8")
    parser.add_argument("--no-patching", action="store_true",
                        help="Table 7 cases (b) and (d): one token per timestep "
                             "(P=S=1, no end padding), i.e. the original TST")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None,
                        help="defaults to 2*d_model when --d-model is given")
    # Architecture fidelity: each of these is an upstream default we do not
    # currently match. See RESULTS.md, "Known deviations".
    parser.add_argument("--encoder-impl", default="torch", choices=["torch", "tst"],
                        help="'tst' is the upstream-shaped layer, needed for any "
                             "of the three flags below")
    parser.add_argument("--norm", default="layer", choices=["layer", "batch"],
                        help="upstream default is BatchNorm")
    parser.add_argument("--post-norm", action="store_true",
                        help="norm after the residual add, as upstream does "
                             "(pre_norm=False)")
    parser.add_argument("--res-attention", action="store_true",
                        help="RealFormer residual attention, on by default upstream")
    parser.add_argument("--attn-dropout", type=float, default=None,
                        help="upstream keeps this at 0 while dropout=0.3 elsewhere")
    parser.add_argument("--no-revin-affine", action="store_true",
                        help="upstream's --affine defaults to 0")
    parser.add_argument("--probe-test", action="store_true",
                        help="record per-epoch TEST mse for inspection. Never "
                             "used for selection -- the best checkpoint is still "
                             "chosen on validation. Answers whether the "
                             "val-optimal epoch is the test-optimal one.")
    args = parser.parse_args()

    if not CSV.exists():
        raise SystemExit(f"{CSV} missing -- run: python scripts/download_data.py")

    torch.manual_seed(args.seed)
    model_kwargs = dict(ETTH1_MODEL)
    if args.no_revin:
        model_kwargs["revin"] = False
    if args.channel_mixing:
        model_kwargs["channel_mixing"] = True
    if args.seq_len is not None:
        model_kwargs["seq_len"] = args.seq_len

    if args.no_patching:
        # One token per timestep. pad_end would otherwise buy a 337th token
        # that is a duplicate of the last one, which is not what "no patching"
        # means -- the original TST has exactly L tokens.
        model_kwargs.update(patch_len=1, stride=1, pad_end=False)
    else:
        if args.patch_len is not None:
            model_kwargs["patch_len"] = args.patch_len
            # The paper does not state the stride for Figure 4. Holding the
            # overlap ratio fixed at the P=16,S=8 setting is the assumption
            # that keeps the sweep about patch length and nothing else.
            model_kwargs["stride"] = max(1, args.patch_len // 2)
        if args.stride is not None:
            model_kwargs["stride"] = args.stride

    if args.d_model is not None:
        model_kwargs["d_model"] = args.d_model
        model_kwargs["d_ff"] = args.d_ff if args.d_ff is not None else 2 * args.d_model
    if args.d_ff is not None:
        model_kwargs["d_ff"] = args.d_ff
    if args.n_heads is not None:
        model_kwargs["n_heads"] = args.n_heads
    if args.n_layers is not None:
        model_kwargs["n_layers"] = args.n_layers

    model_kwargs["encoder_impl"] = args.encoder_impl
    model_kwargs["norm"] = args.norm
    if args.post_norm:
        model_kwargs["pre_norm"] = False
    if args.res_attention:
        model_kwargs["res_attention"] = True
    if args.attn_dropout is not None:
        model_kwargs["attn_dropout"] = args.attn_dropout
    if args.no_revin_affine:
        model_kwargs["revin_affine"] = False

    train_set, val_set, test_set = make_datasets(
        str(CSV), seq_len=model_kwargs["seq_len"], pred_len=args.pred_len
    )

    model = PatchTST(n_channels=7, pred_len=args.pred_len, **model_kwargs)

    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"etth1_{args.tag}_h{args.pred_len}_s{args.seed}"
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        lradj=args.lradj,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        checkpoint=str(RESULTS / f"{name}.pt"),
    )

    print(f"[{name}] {model.n_parameters():,} params, "
          f"{len(train_set)}/{len(val_set)}/{len(test_set)} windows, "
          f"device={args.device}", flush=True)

    started = time.time()
    history = fit(model, train_set, val_set, config,
                  probe_set=test_set if args.probe_test else None)
    metrics = test(model, test_set, config)
    elapsed = time.time() - started

    probe_best_epoch = (
        min(range(len(history.probe_mse)), key=history.probe_mse.__getitem__)
        if history.probe_mse else None
    )

    reference = PAPER_PATCHTST_42[args.pred_len]
    record = {
        "name": name,
        "tag": args.tag,
        "pred_len": args.pred_len,
        "seed": args.seed,
        "test": metrics,
        "paper_patchtst_42": reference,
        "delta_mse": metrics["mse"] - reference["mse"],
        "delta_mae": metrics["mae"] - reference["mae"],
        "best_epoch": history.best_epoch,
        "best_val_mse": history.best_val_mse,
        "epochs_run": len(history.val_mse),
        "stopped_early": history.stopped_early,
        "wall_seconds": elapsed,
        "n_parameters": model.n_parameters(),
        "model": model_kwargs,
        # Repo-relative: an absolute path here would bake the machine's home
        # directory into a committed result file.
        "train": {**asdict(config),
                  "checkpoint": str(Path(config.checkpoint).relative_to(ROOT))},
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "device_name": (
            torch.cuda.get_device_name(0) if args.device.startswith("cuda") else platform.processor()
        ),
        "history": {"val_mse": history.val_mse, "val_mae": history.val_mae,
                    "train_loss": history.train_loss,
                    # Empty unless --probe-test. Diagnostic only.
                    "probe_mse": history.probe_mse},
        # What a TEST-selected oracle would have kept. NOT a reportable result --
        # it selects on test. Its job is to separate "the model cannot do better"
        # from "validation picked the wrong epoch".
        "probe_best_epoch": probe_best_epoch,
        "probe_best_mse": min(history.probe_mse) if history.probe_mse else None,
        "probe_at_val_best": (
            history.probe_mse[history.best_epoch]
            if history.probe_mse and 0 <= history.best_epoch < len(history.probe_mse)
            else None),
    }

    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(record, indent=2))

    print(f"[{name}] test MSE {metrics['mse']:.4f} (paper {reference['mse']:.3f}, "
          f"{record['delta_mse']:+.4f})  |  MAE {metrics['mae']:.4f} "
          f"(paper {reference['mae']:.3f}, {record['delta_mae']:+.4f})", flush=True)
    print(f"[{name}] best epoch {history.best_epoch}/{len(history.val_mse)}, "
          f"{elapsed/60:.1f} min -> {out}", flush=True)

    if probe_best_epoch is not None:
        cost = record["probe_at_val_best"] - record["probe_best_mse"]
        print(f"[{name}] selection: val picked epoch {history.best_epoch} "
              f"(test there {record['probe_at_val_best']:.4f}); a test-oracle "
              f"would have picked epoch {probe_best_epoch} "
              f"(test {record['probe_best_mse']:.4f}), i.e. {cost:+.4f} left on "
              f"the table by selecting on val", flush=True)


if __name__ == "__main__":
    main()
