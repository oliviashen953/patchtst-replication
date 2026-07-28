"""Step 11 — Does channel independence still make sense on CGM?

    python experiments/run_cgm.py --channel-mixing --tag mix_drivers

The question
------------
ETTh1's channels are parallel sensors; none causes another, and channel
independence wins. CGM's `meal` and `bolus` channels are *causal drivers* of
`cgm`. Channel independence structurally forbids the model from looking at them
when forecasting glucose.

So this is a 2x2:

                     drivers informative     drivers zeroed (control)
    channel-indep         (a)                        (c)
    channel-mixing        (b)                        (d)

Prediction: (b) beats (a), because mixing can exploit meal/bolus. And (d)
should NOT beat (c) -- with the causal link removed there is nothing to
exploit, so any advantage in (b) that also shows up in (d) is an artifact of
capacity rather than of causal information.

The control matters. Channel mixing adds parameters (a longer position table),
so it could win for the boring reason of being a bigger model. Running the same
comparison with the drivers zeroed separates the two explanations.

Data is seeded and synthetic -- see patchtst/cgm.py. No patient data is in this
repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from patchtst.cgm import FEATURE_COLS, make_cgm_datasets
from patchtst.model import PatchTST
from patchtst.train import TrainConfig, fit

RESULTS = Path(__file__).resolve().parent / "results_cgm"


@torch.no_grad()
def cgm_metrics(model, dataset, mean, std, device, batch_size=128) -> dict:
    """MSE on the standardized scale, plus RMSE/MAE in mg/dL on the CGM channel."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sse = count = 0.0
    sse_mgdl = sae_mgdl = n_mgdl = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        sse += ((pred - y) ** 2).sum().item()
        count += y.numel()
        # channel 0 is cgm; undo the standardization to get mg/dL
        p = pred[:, :, 0].cpu().numpy() * std[0] + mean[0]
        t = y[:, :, 0].cpu().numpy() * std[0] + mean[0]
        sse_mgdl += float(((p - t) ** 2).sum())
        sae_mgdl += float(np.abs(p - t).sum())
        n_mgdl += p.size
    return {
        "mse": sse / max(1.0, count),
        "cgm_rmse_mgdl": float(np.sqrt(sse_mgdl / max(1.0, n_mgdl))),
        "cgm_mae_mgdl": sae_mgdl / max(1.0, n_mgdl),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-mixing", action="store_true")
    parser.add_argument("--zero-drivers", action="store_true",
                        help="control: remove the causal meal/bolus effect")
    parser.add_argument("--seq-len", type=int, default=48, help="4 h at 5-min sampling")
    parser.add_argument("--pred-len", type=int, default=12, help="1 h ahead")
    parser.add_argument("--n-days", type=int, default=120)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="cgm")
    args = parser.parse_args()

    effect = 0.0 if args.zero_drivers else 1.0
    torch.manual_seed(args.seed)

    train_set, val_set, test_set = make_cgm_datasets(
        n_days=args.n_days, seed=args.data_seed,
        seq_len=args.seq_len, pred_len=args.pred_len,
        meal_effect=effect, bolus_effect=effect,
    )

    model = PatchTST(
        n_channels=len(FEATURE_COLS),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        patch_len=8, stride=4,          # 40-min patches, 20-min stride
        d_model=32, n_heads=4, n_layers=3, d_ff=128,
        dropout=0.2, revin=True,
        channel_mixing=args.channel_mixing,
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"cgm_{args.tag}_s{args.seed}"
    config = TrainConfig(
        epochs=args.epochs, batch_size=128, learning_rate=1e-4,
        patience=15, seed=args.seed, device=args.device, num_workers=2,
        checkpoint=str(RESULTS / f"{name}.pt"),
    )

    print(f"[{name}] mixing={args.channel_mixing} drivers={'off' if args.zero_drivers else 'on'} "
          f"{model.n_parameters():,} params, {len(train_set)}/{len(val_set)}/{len(test_set)} windows",
          flush=True)

    started = time.time()
    history = fit(model, train_set, val_set, config)
    metrics = cgm_metrics(
        model, test_set, train_set.mean, train_set.std, torch.device(args.device)
    )
    elapsed = time.time() - started

    record = {
        "name": name, "tag": args.tag,
        "channel_mixing": args.channel_mixing,
        "drivers": "off" if args.zero_drivers else "on",
        "test": metrics,
        "best_epoch": history.best_epoch,
        "best_val_mse": history.best_val_mse,
        "epochs_run": len(history.val_mse),
        "n_parameters": model.n_parameters(),
        "seq_len": args.seq_len, "pred_len": args.pred_len,
        "seed": args.seed, "data_seed": args.data_seed,
        "wall_seconds": elapsed,
        "train": asdict(config),
    }
    (RESULTS / f"{name}.json").write_text(json.dumps(record, indent=2))

    print(f"[{name}] test MSE {metrics['mse']:.4f}  |  CGM RMSE "
          f"{metrics['cgm_rmse_mgdl']:.2f} mg/dL  |  MAE "
          f"{metrics['cgm_mae_mgdl']:.2f} mg/dL  |  {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
