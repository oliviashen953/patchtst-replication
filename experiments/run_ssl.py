"""Step 12 — Self-supervised pretraining and fine-tuning on ETTh1.

    python experiments/run_ssl.py --stage pretrain --device cuda
    python experiments/run_ssl.py --stage linear_probe --pred-len 96 --device cuda
    python experiments/run_ssl.py --stage finetune     --pred-len 96 --device cuda
    python experiments/run_ssl.py --stage scratch      --pred-len 96 --device cuda

Four arms, one shared pretrained checkpoint:

  pretrain      masked reconstruction, 100 epochs, no labels
  linear_probe  freeze the backbone, train the forecasting head for 20 epochs
  finetune      10 epochs of linear probing, then 20 epochs end-to-end
  scratch       the SAME architecture trained from random init -- the control

The protocol is Section 5.3 of the paper: "we first apply self-supervised
pre-training ... for 100 epochs ... With (a), we only train the model head for
20 epochs while freezing the rest of the network; With (b), we apply linear
probing for 10 epochs to update the model head and then end-to-end fine-tuning
the entire network for 20 epochs."

Why `scratch` exists
--------------------
Table 12 of the paper compares its self-supervised ETTh1 numbers against a
"Sup." column of 0.375 / 0.414 / 0.431 / 0.449 -- which is exactly the
PatchTST/42 row of Table 3, i.e. the L=336 supervised model with overlapping
P=16, S=8 patches and the reduced ETTh1 architecture. The self-supervised runs
use L=512 with non-overlapping P=S=12 and the full-size model. So the paper's
own supervised/self-supervised comparison on ETTh1 varies several things at
once. `scratch` is the like-for-like control: identical architecture, identical
geometry, identical schedule, random init instead of pretrained.

Configuration
-------------
From the official `patchtst_pretrain.py` defaults, which are what the
representation-learning section ran: L=512, patch 12, stride 12, d_model 128,
n_heads 16, n_layers 3, d_ff 512, dropout 0.2, head_dropout 0.2, batch 64,
lr 1e-4, mask ratio 0.4, RevIN without affine parameters. Note these are NOT
the reduced ETTh1 settings Step 9 used -- the self-supervised codebase has no
per-dataset override, so this arm runs a ~1.1M-parameter model on a dataset
with 8k training windows.
"""

from __future__ import annotations

import argparse
import json
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
from patchtst.pretrain import (
    MaskedPatchTST,
    PretrainConfig,
    freeze_backbone,
    pretrain_fit,
    transfer_backbone,
)
from patchtst.train import TrainConfig, fit, test

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "ETTh1.csv"
RESULTS = Path(__file__).resolve().parent / "results_ssl"

# The representation-learning geometry and model, from the official
# patchtst_pretrain.py defaults. Shared by all four arms.
SSL_MODEL = dict(
    seq_len=512,
    patch_len=12,
    stride=12,          # == patch_len: non-overlapping, as Section 4.2 requires
    d_model=128,
    n_heads=16,
    n_layers=3,
    d_ff=512,
    dropout=0.2,
    revin=True,
    revin_affine=False,
)

# Table 12, ETTh1 block. Reference targets only.
PAPER_TABLE12_ETTH1 = {
    96:  {"finetune": {"mse": 0.366, "mae": 0.397},
          "linear_probe": {"mse": 0.371, "mae": 0.400},
          "sup": {"mse": 0.375, "mae": 0.399}},
    192: {"finetune": {"mse": 0.431, "mae": 0.443},
          "linear_probe": {"mse": 0.411, "mae": 0.428},
          "sup": {"mse": 0.414, "mae": 0.421}},
    336: {"finetune": {"mse": 0.450, "mae": 0.456},
          "linear_probe": {"mse": 0.445, "mae": 0.446},
          "sup": {"mse": 0.431, "mae": 0.436}},
    720: {"finetune": {"mse": 0.472, "mae": 0.484},
          "linear_probe": {"mse": 0.487, "mae": 0.478},
          "sup": {"mse": 0.449, "mae": 0.466}},
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def build_forecaster(pred_len: int, head_dropout: float) -> PatchTST:
    return PatchTST(
        n_channels=7,
        pred_len=pred_len,
        head_dropout=head_dropout,
        individual=False,
        pad_end=False,      # non-overlapping patches, no synthetic padded patch
        align="end",        # keep the most recent timesteps
        **SSL_MODEL,
    )


def run_pretrain(args) -> None:
    train_set, val_set, _ = make_datasets(
        str(CSV), seq_len=SSL_MODEL["seq_len"], pred_len=args.pred_len
    )
    model = MaskedPatchTST(
        n_channels=7, mask_ratio=args.mask_ratio,
        head_dropout=args.head_dropout, **SSL_MODEL
    )
    config = PretrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mask_ratio=args.mask_ratio,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        checkpoint=str(RESULTS / f"{args.name}.pt"),
    )

    print(f"[{args.name}] {model.n_parameters():,} params, "
          f"{model.n_patches} patches, {len(train_set)} train windows, "
          f"masking {args.mask_ratio:.0%}", flush=True)

    started = time.time()
    history = pretrain_fit(model, train_set, val_set, config)
    elapsed = time.time() - started

    record = {
        "name": args.name,
        "stage": "pretrain",
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "first_val_loss": history.val_loss[0] if history.val_loss else None,
        "epochs_run": len(history.val_loss),
        "wall_seconds": elapsed,
        "n_parameters": model.n_parameters(),
        "model": SSL_MODEL,
        "train": asdict(config) | {
            "checkpoint": str(Path(config.checkpoint).relative_to(ROOT))
        },
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "history": {"train_loss": history.train_loss, "val_loss": history.val_loss},
    }
    (RESULTS / f"{args.name}.json").write_text(json.dumps(record, indent=2))

    print(f"[{args.name}] masked recon MSE {history.val_loss[0]:.4f} -> "
          f"{history.best_val_loss:.4f} (best epoch {history.best_epoch}), "
          f"{elapsed / 60:.1f} min", flush=True)


def run_downstream(args) -> None:
    """linear_probe | finetune | scratch — all end in test() on the same split."""
    train_set, val_set, test_set = make_datasets(
        str(CSV), seq_len=SSL_MODEL["seq_len"], pred_len=args.pred_len
    )
    model = build_forecaster(args.pred_len, args.head_dropout)

    transferred = 0
    if args.stage != "scratch":
        ckpt = RESULTS / f"{args.pretrained}.pt"
        if not ckpt.exists():
            raise SystemExit(
                f"{ckpt} missing -- run --stage pretrain first"
            )
        transferred = transfer_backbone(str(ckpt), model)
        print(f"[{args.name}] transferred {transferred} backbone tensors "
              f"from {ckpt.name}", flush=True)

    def make_config(epochs: int, tag: str) -> TrainConfig:
        return TrainConfig(
            epochs=epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=epochs,        # never stop early; keep the best checkpoint
            lradj="cosine",
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            checkpoint=str(RESULTS / f"{args.name}_{tag}.pt"),
        )

    started = time.time()
    phases = []

    if args.stage == "linear_probe":
        # (a) train the head only, 20 epochs.
        freeze_backbone(model, True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{args.name}] probing {trainable:,}/{model.n_parameters():,} "
              f"params", flush=True)
        history = fit(model, train_set, val_set, make_config(args.probe_epochs, "probe"))
        phases.append(("probe", history))

    elif args.stage == "finetune":
        # (b) 10 epochs of linear probing FIRST, then 20 end-to-end. The paper
        # cites Kumar et al. (2022): probing before fine-tuning beats fine-
        # tuning directly, because a randomly initialized head sends large
        # early gradients through the pretrained features and distorts them.
        freeze_backbone(model, True)
        phases.append(("probe", fit(model, train_set, val_set,
                                    make_config(args.warmup_epochs, "warmup"))))
        freeze_backbone(model, False)
        phases.append(("finetune", fit(model, train_set, val_set,
                                       make_config(args.finetune_epochs, "ft"))))

    else:  # scratch
        history = fit(model, train_set, val_set,
                      make_config(args.scratch_epochs, "scratch"))
        phases.append(("scratch", history))

    final_config = make_config(1, "eval")
    metrics = test(model, test_set, final_config)
    elapsed = time.time() - started

    reference = PAPER_TABLE12_ETTH1[args.pred_len]
    key = {"linear_probe": "linear_probe", "finetune": "finetune",
           "scratch": "sup"}[args.stage]
    paper = reference[key]

    record = {
        "name": args.name,
        "stage": args.stage,
        "pred_len": args.pred_len,
        "seed": args.seed,
        "test": metrics,
        "paper_table12": paper,
        # The 'sup' reference is the paper's L=336 PatchTST/42 row, so treat the
        # scratch delta as indicative only -- see this file's docstring.
        "delta_mse": metrics["mse"] - paper["mse"],
        "delta_mae": metrics["mae"] - paper["mae"],
        "transferred_tensors": transferred,
        "phases": [
            {"name": name,
             "epochs_run": len(h.val_mse),
             "best_epoch": h.best_epoch,
             "best_val_mse": h.best_val_mse,
             "val_mse": h.val_mse}
            for name, h in phases
        ],
        "wall_seconds": elapsed,
        "n_parameters": model.n_parameters(),
        "model": SSL_MODEL,
        "train": asdict(final_config) | {"checkpoint": None},
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "device_name": (
            torch.cuda.get_device_name(0)
            if args.device.startswith("cuda") else platform.processor()
        ),
    }
    (RESULTS / f"{args.name}.json").write_text(json.dumps(record, indent=2))

    print(f"[{args.name}] test MSE {metrics['mse']:.4f} (paper {paper['mse']:.3f}, "
          f"{record['delta_mse']:+.4f})  |  MAE {metrics['mae']:.4f} "
          f"(paper {paper['mae']:.3f}, {record['delta_mae']:+.4f})  "
          f"{elapsed / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["pretrain", "linear_probe", "finetune", "scratch"])
    parser.add_argument("--pred-len", type=int, default=96,
                        choices=[96, 192, 336, 720],
                        help="for --stage pretrain this only sets the window "
                             "count; no horizon is ever predicted")
    parser.add_argument("--epochs", type=int, default=100,
                        help="pretraining epochs (paper: 100)")
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--scratch-epochs", type=int, default=30,
                        help="10 + 20, matching the fine-tuning arm's budget")
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="the self-supervised code uses plain Adam")
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pretrained", default="pretrain_etth1_mask40",
                        help="name of the pretrained checkpoint to transfer")
    parser.add_argument("--tag", default=None, help="override the output name")
    args = parser.parse_args()

    if not CSV.exists():
        raise SystemExit(f"{CSV} missing -- run: python scripts/download_data.py")

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.tag:
        args.name = args.tag
    elif args.stage == "pretrain":
        args.name = args.pretrained
    else:
        args.name = f"{args.stage}_h{args.pred_len}_s{args.seed}"

    torch.manual_seed(args.seed)

    if args.stage == "pretrain":
        run_pretrain(args)
    else:
        run_downstream(args)


if __name__ == "__main__":
    main()
