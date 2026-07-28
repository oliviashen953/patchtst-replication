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
    seed: int = 42
    device: str = "cpu"
    num_workers: int = 0
    checkpoint: str | None = None


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_mse: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)
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

        # TODO(you): one optimizer step.
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

        # TODO(you): accumulate squared and absolute error.
        #
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
) -> History:
    """Train with cosine LR decay and early stopping on validation MSE."""
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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.min_lr
    )

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
        history.lr.append(optimizer.param_groups[0]["lr"])
        history.epoch_seconds.append(time.time() - started)

        # TODO(you): track the best model and decide whether to stop.
        #
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
