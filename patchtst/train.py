"""Step 8 — Training loop.

Paper anchor: Section 5.2 and Appendix A.2 (experimental settings).

Nothing conceptually new here -- this is standard supervised training. But the
protocol details are what decide whether your Step 9 numbers are comparable to
the paper's, so they are worth getting exactly right:

  * **Loss is MSE** on the standardized scale, computed over the whole horizon
    at once. Not per-step, not weighted by horizon position.
  * **Adam**, with a cosine-annealed learning rate.
  * **Early stopping on VALIDATION loss**, never on test. The test split is
    touched exactly once, at the end, with the best-validation checkpoint
    reloaded. Peeking at test to pick a stopping point is the most common way
    replication numbers end up quietly optimistic.
  * **Report both MSE and MAE.** The benchmark tables list both, and they can
    disagree about which model wins.

A note on the scale of the reported numbers
-------------------------------------------
The MSE/MAE in the paper's tables are on the STANDARDIZED scale -- the data as
the scaler left it, not raw mg/dL-style units. That is why an ETTh1 MSE around
0.37 is a sensible target rather than something in the hundreds. Since our
model denormalizes its forecast (Step 7), and the targets from Step 1 are
already standardized, both sides are in the same units and the comparison is
apples to apples. Do not add another normalization here.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    grad_clip_norm: float = 1.0
    patience: int = 10
    min_delta: float = 0.0
    min_lr: float = 1e-5
    lradj: str = "cosine"   # "cosine" | "type3"
    seed: int = 42
    device: str = "cpu"
    num_workers: int = 0
    checkpoint: str | None = None


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_mse: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)
    # Per-epoch TEST error, recorded only when fit() is given a probe_set.
    # Diagnostic output -- nothing in this file ever reads it back.
    probe_mse: list[float] = field(default_factory=list)
    probe_mae: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_mse: float = float("inf")
    stopped_early: bool = False


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = 1.0,
) -> float:
    """One pass over the training data. Returns the mean batch loss."""
    model.train()
    total, n_batches = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Step 8 exercise — one optimizer step.
        #
        #   optimizer.zero_grad()
        #   pred = model(x)
        #   loss = nn.functional.mse_loss(pred, y)
        #   loss.backward()
        #   if grad_clip_norm:  torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        #   optimizer.step()
        #
        # Then accumulate: total += loss.item(); n_batches += 1
        # zero_grad() FIRST. PyTorch accumulates gradients by default, so
        # forgetting it silently sums every batch's gradients together -- your
        # loss will look erratic and you will blame the learning rate.
        #
        # Answer:
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total += loss.item()
        n_batches += 1

    return total / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Return {'mse': .., 'mae': ..} over the whole loader.

    Note the @torch.no_grad() decorator on this function: without it, every
    validation pass would build a graph it never uses, wasting memory and time.
    """
    model.eval()
    sse, sae, count = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Step 8 exercise — accumulate squared and absolute error.
        #
        # Answer:
        pred = model(x)
        sse += ((pred - y) ** 2).sum().item()
        sae += (pred - y).abs().sum().item()
        count += y.numel()

    return {"mse": sse / max(1, count), "mae": sae / max(1, count)}


def fit(
    model: nn.Module,
    train_set,
    val_set,
    config: TrainConfig,
    probe_set=None,
) -> History:
    """Train with cosine LR decay and early stopping on validation MSE.

    `probe_set` is a diagnostic hook, and it is the one place in this repo where
    the test split is touched during training. Pass it and every epoch's test
    MSE/MAE is recorded into `history.probe_mse` / `.probe_mae` -- but the
    selection rule below reads `metrics["mse"]` from the VALIDATION loader only,
    and never the probe. The point is to be able to ask, after the fact, whether
    the val-optimal epoch and the test-optimal epoch are the same epoch. If they
    are far apart, the reported number is limited by checkpoint *selection*
    rather than by training.

    Look but do not touch: the moment a probe value influences what you keep,
    the test split stops being a test split.
    """
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = model.to(device)

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )
    # The probe must not perturb training, and `shuffle=False` is NOT enough to
    # guarantee that: DataLoader draws one number from the *global* RNG every
    # time an iterator is created, to seed its workers -- shuffling or not. So
    # probing each epoch would shift the train loader's shuffle order from the
    # next epoch onward, and the instrumented run would stop being comparable to
    # the uninstrumented one. Giving the probe its own generator keeps the draw
    # off the global stream. (Measured: without this, val MSE diverges from the
    # unprobed run by ~1e-5 by epoch 2 -- small, but it is a different run.)
    probe_loader = (
        DataLoader(probe_set, batch_size=config.batch_size, shuffle=False,
                   num_workers=config.num_workers,
                   generator=torch.Generator().manual_seed(config.seed))
        if probe_set is not None else None
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if config.lradj == "type3":
        # The official ETTh1 setup: hold the initial rate for 3 epochs, then
        # decay by 0.9 every epoch. That reaches ~0.17x by epoch 20 and ~0.007x
        # by epoch 50, which acts as strong implicit regularization -- the model
        # largely stops moving before it can overfit. Cosine, by contrast, is
        # nearly flat early (~0.9x at epoch 20).
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0 if epoch < 3 else 0.9 ** (epoch - 3),
        )
    elif config.lradj == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.min_lr
        )
    else:
        raise ValueError(f"unknown lradj {config.lradj!r}; use 'cosine' or 'type3'")

    history = History()
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        started = time.time()
        loss = train_epoch(
            model, train_loader, optimizer, device, config.grad_clip_norm
        )
        metrics = evaluate(model, val_loader, device)
        scheduler.step()

        history.train_loss.append(loss)
        history.val_mse.append(metrics["mse"])
        history.val_mae.append(metrics["mae"])
        if probe_loader is not None:
            probe = evaluate(model, probe_loader, device)
            history.probe_mse.append(probe["mse"])
            history.probe_mae.append(probe["mae"])
        history.lr.append(optimizer.param_groups[0]["lr"])
        history.epoch_seconds.append(time.time() - started)

        # Step 8 exercise — track the best model and decide whether to stop.
        #
        # Answer:
        improved = metrics["mse"] < history.best_val_mse - config.min_delta
        if improved:
            history.best_val_mse = metrics["mse"]
            history.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                history.stopped_early = True
                break

        # .clone() matters: state_dict() returns references to live tensors, so
        # without it "best_state" would keep changing as training continues and
        # you would restore the LAST weights, not the best ones.

    # Restore the best weights so the caller evaluates the right model.
    if best_state is not None:
        model.load_state_dict(best_state)

    if config.checkpoint:
        path = Path(config.checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "history": asdict(history),
            },
            path,
        )

    return history


def test(model: nn.Module, test_set, config: TrainConfig) -> dict[str, float]:
    """Evaluate on the test split. Call this ONCE, after fit() has finished."""
    device = torch.device(config.device)
    loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    return evaluate(model.to(device), loader, device)
