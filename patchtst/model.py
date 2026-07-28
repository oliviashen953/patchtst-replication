"""Step 7 — Assemble PatchTST.

Paper anchor: Figure 1 (the whole architecture).

Everything already exists. This file only wires it together:

    x  [B, L, C]
      -> RevIN.normalize          (inside the backbone)
      -> Patchify                 [B, C, N, P] ### !!!! patch first, then fold C into the batch
      -> fold C into the batch    [B*C, N, P]  ### !!!! then do the channel independence here!!
      -> PatchEncoder             [B*C, N, d_model]
      -> unfold C back out        [B, C, N, d_model]
      -> FlattenHead              [B, pred_len, C]
      -> RevIN.denormalize        <- THIS FILE'S ONE REAL SUBTLETY
    y_hat  [B, pred_len, C]

The denormalize step
--------------------
The backbone normalized the input, so the head's output is in *normalized*
units. The forecast has to be pushed back into real units before it can be
compared with the targets.

That is why the backbone holds the RevIN instance rather than this file
creating its own: denormalize() reuses the mean and standard deviation that
normalize() stored on that same object. A second RevIN would have no statistics
and would raise.

Note that RevIN's statistics are [B, 1, C] and the forecast is [B, pred_len, C],
so they broadcast even though pred_len != seq_len. That is the property Step 2
was careful to preserve, and it pays off here.

Forget the denormalize and nothing crashes -- your model simply predicts
standardized values while the loss compares them against real ones, so training
"works" and the MSE is nonsense. check_step07 tests for exactly this.
"""

from __future__ import annotations

import torch
from torch import nn

from .backbone import ChannelIndependentBackbone
from .head import FlattenHead


class PatchTST(nn.Module):
    """PatchTST: patch-based, channel-independent long-term forecaster.

    Args:
        n_channels: C, number of series.
        seq_len: L, lookback length.
        pred_len: H, forecast horizon.
        patch_len: P.
        stride: S.
        d_model, n_heads, n_layers, d_ff, dropout: encoder settings.
        head_dropout: dropout inside the prediction head.
        individual: one prediction head per channel instead of a shared one.
        revin: use reversible instance normalization.
    """

    def __init__(
        self,
        *,
        n_channels: int,
        seq_len: int = 336,
        pred_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        head_dropout: float = 0.0,
        individual: bool = False,
        revin: bool = True,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)

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
        )
        self.n_patches = self.backbone.n_patches

        self.head = FlattenHead(
            n_channels=n_channels,
            n_patches=self.n_patches,
            d_model=d_model,
            pred_len=pred_len,
            dropout=head_dropout,
            individual=individual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, n_channels] -> y_hat: [B, pred_len, n_channels]."""
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.n_channels):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.n_channels}], "
                f"got {tuple(x.shape)}"
            )

        encoded = self.backbone(x)              # [B, C, n_patches, d_model]
        forecast = self.head(encoded)           # [B, pred_len, C]

        # Push the forecast back into real units. Use the BACKBONE's RevIN
        # object, not a new one -- it is the only thing holding the mean/std
        # that normalize() just stored. Its statistics are [B, 1, C], so they
        # broadcast over pred_len even though pred_len != seq_len.
        if self.backbone.revin is not None:
            forecast = self.backbone.revin.denormalize(forecast)

        return forecast

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: eval mode, no gradients."""
        was_training = self.training
        self.eval()
        try:
            return self(x)
        finally:
            self.train(was_training)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
