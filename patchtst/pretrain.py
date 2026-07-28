"""Step 12 — Masked self-supervised pretraining.

Paper anchor: Section 4.2 ("Representation Learning"), Figure 1(c), and the
experimental protocol at the top of Section 5.3.

The pretext task
----------------
Take the lookback window, cut it into patches, throw 40% of them away (set to
zero), and train the model to put them back. No labels, no horizon, no forecast
-- just reconstruction. Then keep the encoder, bolt the Step 6 forecasting head
back on, and fine-tune.

Three design choices carry the whole section, and each is a decision the paper
argues for explicitly.

**1. Mask patches, not timesteps.** Zerveas et al. (2021) mask individual time
steps. The paper's objection (Section 4.2, first paragraph) is that this is too
easy: a single missing step is recoverable by interpolating its two neighbours,
so the model can score well without understanding anything. Masking a whole
patch removes a contiguous block, and interpolation across it is genuinely hard.

**2. Non-overlapping patches.** The supervised model uses P=16, S=8, so
consecutive patches share half their timesteps. Under masking that is fatal: the
neighbours of a masked patch literally contain half of its values, and the model
can copy them. Hence the paper's "we divide each input sequence into regular
non-overlapping patches ... for convenience to ensure observed patches do not
contain information of the masked ones". Representation-learning runs therefore
use L=512, P=12, S=12 -> 42 patches. check_step12 turns this into a measurable
claim rather than a stylistic note.

**3. A D x P linear head.** The prediction head is removed and replaced by one
`Linear(d_model, patch_len)` applied to every patch token. The paper spends a
paragraph (Section 4.2) on why this matters: a per-timestep representation would
need a `(L*D) x (M*T)` output matrix, which is enormous and overfits when
downstream data is scarce. Per-patch reconstruction keeps the head at `D*P`.

What is masked, and where
-------------------------
Masking happens *after* RevIN and *before* the encoder, so both the input and
the reconstruction target live in normalized units. RevIN's denormalize is never
called during pretraining -- there is nothing to push back into real units,
because the target is the normalized patch itself. The official code says this
in its own way: `RevInCB(dls.vars, denorm=False)`.

Each (sample, channel) pair gets its OWN mask. Channel independence again: the
encoder sees B*C univariate patch sequences, and there is no reason for channel
3's masked positions to line up with channel 0's.

The loss is computed on masked patches ONLY. Averaging over all patches instead
would let the model coast on the 60% it can see -- copy the input, score well,
learn nothing.

Deviations from the official self-supervised code, recorded honestly
--------------------------------------------------------------------
  1. It picks the learning rate with an LR-finder sweep and trains with
     `fit_one_cycle`. We use the fixed lr=1e-4 of its own `--lr` default with
     the cosine schedule the rest of this repo uses, so pretraining and the
     supervised control differ in as few ways as possible.
  2. Its encoder normalizes with BatchNorm; ours is a vanilla PyTorch encoder
     with LayerNorm. That is inherited from Step 4, not new here.
  3. Validation masking is seeded (see `masked_reconstruction_loss` callers).
     Upstream draws a fresh random mask every validation pass, which makes the
     validation curve noisy for a quantity we use to select a checkpoint. Same
     mask ratio, same distribution -- only the seed is pinned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from .backbone import ChannelIndependentBackbone


def random_patch_mask(
    patches: torch.Tensor,
    mask_ratio: float = 0.4,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out a random subset of patches.

    Args:
        patches: [B, C, N, P], normalized.
        mask_ratio: fraction of patches to remove. The paper uses 0.4.
        generator: optional RNG, for a reproducible validation mask.

    Returns:
        masked: [B, C, N, P], a copy with the removed patches set to zero.
        mask:   [B, C, N] bool, True where a patch was removed.

    The count is EXACT, not a per-patch coin flip: `int(N * (1 - ratio))`
    patches are kept for every (sample, channel), matching the official
    `random_masking`. A Bernoulli mask would give the same ratio on average but
    a variable one per sample, and with N=42 that variation is not small.
    """
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")
    if patches.ndim != 4:
        raise ValueError(f"expected [B, C, N, P], got {tuple(patches.shape)}")

    batch, channels, n_patches, _ = patches.shape
    n_keep = int(n_patches * (1.0 - mask_ratio))

    # Rank the patches of each (sample, channel) by a random key and keep the
    # lowest `n_keep`. Sorting noise is how the official code does it, and it is
    # what makes the kept count exact.
    noise = torch.rand(
        batch, channels, n_patches, device=patches.device, generator=generator
    )
    rank = noise.argsort(dim=-1).argsort(dim=-1)     # rank of each position
    mask = rank >= n_keep                           # True = removed

    masked = patches * (~mask).unsqueeze(-1)        # broadcast over patch_len
    return masked, mask


def masked_reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE over the MASKED patches only.

    recon, target: [B, C, N, P].  mask: [B, C, N] bool, True where masked.

    Mean over patch_len first, then a mask-weighted mean over patches -- so
    every masked patch counts once regardless of P, exactly as in the official
    `PatchMaskCB._loss`.

    Including the visible patches here is the classic way to get a pretraining
    run that looks healthy and teaches nothing: the model drives the visible 60%
    to zero error by copying its input, the total loss falls, and the masked
    part never improves.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    per_patch = ((recon - target) ** 2).mean(dim=-1)     # [B, C, N]
    denom = mask.sum()
    if denom == 0:
        raise ValueError("no patches were masked; loss is undefined")
    return (per_patch * mask).sum() / denom


class PretrainHead(nn.Module):
    """Map every patch token back to its raw values: [.., N, D] -> [.., N, P].

    The entire self-supervised head. `nn.Linear` acts on the last axis, so one
    call reconstructs every patch of every channel at once -- and, like the
    Step 6 forecasting head, the weights are shared across channels.
    """

    def __init__(self, *, d_model: int, patch_len: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.d_model, self.patch_len)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.shape[-1] != self.d_model:
            raise ValueError(
                f"expected last axis {self.d_model}, got {tuple(encoded.shape)}"
            )
        return self.linear(self.dropout(encoded))


class MaskedPatchTST(nn.Module):
    """PatchTST with the forecasting head swapped for a reconstruction head.

    forward(x) -> (recon, target, mask), all in normalized units:
        recon  [B, C, N, P]   what the model rebuilt
        target [B, C, N, P]   the patches it should have rebuilt
        mask   [B, C, N]      True where a patch was removed

    The backbone is the SAME class the forecaster uses, with the same submodule
    names, which is the only reason `transfer_backbone` can be a plain
    state-dict copy later. Defaults here are the self-supervised ones from the
    official `patchtst_pretrain.py`: L=512, P=S=12, d_model=128, d_ff=512,
    mask ratio 0.4, RevIN without affine parameters.
    """

    def __init__(
        self,
        *,
        n_channels: int,
        seq_len: int = 512,
        patch_len: int = 12,
        stride: int = 12,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.2,
        head_dropout: float = 0.2,
        mask_ratio: float = 0.4,
        revin: bool = True,
        revin_affine: bool = False,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.mask_ratio = float(mask_ratio)

        # pad_end=False: a replicated padding patch would be a partly synthetic
        # reconstruction target. align="end": drop the oldest timesteps, not the
        # newest. Both match the official self-supervised code.
        self.backbone = ChannelIndependentBackbone(
            n_channels=n_channels,
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            revin=revin,
            revin_affine=revin_affine,
            pad_end=False,
            align="end",
        )
        self.n_patches = self.backbone.n_patches

        self.head = PretrainHead(
            d_model=d_model, patch_len=patch_len, dropout=head_dropout
        )

    def forward(
        self, x: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [B, seq_len, C] -> (recon, target, mask)."""
        captured: dict[str, torch.Tensor] = {}

        def mask_patches(patches: torch.Tensor) -> torch.Tensor:
            # `patches` arrives normalized. Detach the target: it is data, not
            # something to backpropagate into. With revin_affine=False there are
            # no parameters upstream of it anyway, but leaving the graph
            # attached would let an affine RevIN cheat by shrinking BOTH the
            # prediction and the target toward zero.
            captured["target"] = patches.detach()
            masked, mask = random_patch_mask(patches, self.mask_ratio, generator)
            captured["mask"] = mask
            return masked

        encoded = self.backbone(x, patch_transform=mask_patches)  # [B, C, N, D]
        recon = self.head(encoded)                                # [B, C, N, P]
        return recon, captured["target"], captured["mask"]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def transfer_backbone(
    pretrained: dict | str,
    model: nn.Module,
    strict: bool = True,
) -> int:
    """Copy pretrained BACKBONE weights into a forecasting model, in place.

    `pretrained` is either a state dict or a path to a checkpoint saved by
    `pretrain_fit`. Returns the number of tensors copied.

    The head is deliberately excluded -- `head.linear` reconstructs a patch of
    length P, `head.head` forecasts H steps ahead, and they are different
    layers with different shapes. Everything under `backbone.` is shared, which
    is exactly what "the learned representation" means here.

    `strict=True` raises if nothing matched. That check earns its keep: rename
    a submodule and a silent no-op transfer leaves you fine-tuning a randomly
    initialized model while reporting it as pretrained.
    """
    if isinstance(pretrained, (str, bytes)) or hasattr(pretrained, "__fspath__"):
        blob = torch.load(pretrained, map_location="cpu", weights_only=False)
        pretrained = blob.get("state_dict", blob)

    target = model.state_dict()
    copied, skipped = 0, []
    for name, tensor in pretrained.items():
        if not name.startswith("backbone."):
            continue
        if name not in target:
            skipped.append(name)
            continue
        if target[name].shape != tensor.shape:
            skipped.append(f"{name} {tuple(tensor.shape)}->{tuple(target[name].shape)}")
            continue
        target[name].copy_(tensor)
        copied += 1

    if strict and copied == 0:
        raise RuntimeError(
            "transfer_backbone copied nothing -- the checkpoint and the model "
            "do not share backbone parameter names"
        )
    if skipped:
        print(f"[transfer] skipped {len(skipped)}: {skipped[:4]}", flush=True)
    return copied


def freeze_backbone(model: nn.Module, frozen: bool = True) -> int:
    """Freeze (or unfreeze) everything except the head. Returns params frozen.

    This is linear probing: the representation is held fixed and only the read-
    out is trained, which measures how much of the forecast was already present
    in the pretrained features.

    Freezing sets `requires_grad=False`, which stops gradients but NOT dropout
    or (if you ever add it) BatchNorm's running statistics. Our encoder has
    dropout, so `model.eval()` semantics still matter -- the frozen backbone is
    not fully deterministic during probing. That is upstream's behaviour too.
    """
    n = 0
    for name, param in model.named_parameters():
        if name.startswith("backbone."):
            param.requires_grad = not frozen
            n += param.numel()
    return n


@dataclass
class PretrainConfig:
    epochs: int = 100          # the paper pretrains for 100 epochs
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    mask_ratio: float = 0.4
    min_lr: float = 1e-5
    seed: int = 2021
    val_seed: int = 12345      # pins the validation mask, see module docstring
    device: str = "cpu"
    num_workers: int = 0
    checkpoint: str | None = None


@dataclass
class PretrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")


@torch.no_grad()
def evaluate_reconstruction(
    model: MaskedPatchTST,
    loader: DataLoader,
    device: torch.device,
    seed: int,
) -> float:
    """Mean masked-reconstruction MSE, with a fixed mask for comparability."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total, n_batches = 0.0, 0
    for x, _ in loader:                      # the horizon target is unused here
        x = x.to(device)
        recon, target, mask = model(x, generator=generator)
        total += masked_reconstruction_loss(recon, target, mask).item()
        n_batches += 1
    return total / max(1, n_batches)


def pretrain_fit(
    model: MaskedPatchTST,
    train_set,
    val_set,
    config: PretrainConfig,
) -> PretrainHistory:
    """Masked-reconstruction pretraining. Keeps the best-validation weights.

    The datasets are the ordinary Step 1 forecasting datasets; the `y` half of
    each item is simply ignored. That is what "no labels" looks like in
    practice, and it is what the official code does too -- same dataloader,
    different callback.
    """
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = model.to(device)
    model.mask_ratio = float(config.mask_ratio)

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.min_lr
    )

    history = PretrainHistory()
    best_state = None

    for epoch in range(config.epochs):
        started = time.time()
        model.train()
        total, n_batches = 0.0, 0

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, target, mask = model(x)
            loss = masked_reconstruction_loss(recon, target, mask)
            loss.backward()
            if config.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip_norm
                )
            optimizer.step()
            total += loss.item()
            n_batches += 1

        train_loss = total / max(1, n_batches)
        val_loss = evaluate_reconstruction(
            model, val_loader, device, config.val_seed
        )
        scheduler.step()

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.epoch_seconds.append(time.time() - started)

        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    if config.checkpoint:
        from dataclasses import asdict
        from pathlib import Path

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
