"""Step 5 — Channel independence.

Paper anchor: Section 4.2 ("Channel-independence").

The claim
---------
Older multivariate Transformers use *channel mixing*: at each timestep they take
all `C` variables and project them jointly into one token. Every token is a blend
of temperature, humidity, pressure, ... PatchTST does the opposite. Each channel
is forecast **separately**, through **one shared backbone**. No attention ever
crosses channels.

This is counterintuitive -- surely a multivariate model should exploit
cross-variable structure? -- and the paper's Table 7 ablation says otherwise.
Two reasons it wins in practice:

1. **Less overfitting.** Channel mixing has to learn a joint embedding whose
   parameter count grows with `C`. On Traffic (862 channels!) that is a large,
   mostly spurious, thing to fit.
2. **Shared statistics.** Because the backbone is shared, every channel's data
   trains the *same* weights. You effectively get `C` times more training
   sequences for one set of parameters.

The implementation
------------------
Here is the part worth appreciating: channel independence is a *reshape*.

    [B, C, N, P]  --fold C into the batch-->  [B*C, N, P]
                  --shared encoder-->         [B*C, N, d_model]
                  --unfold-->                 [B, C, N, d_model]

The encoder is handed `B*C` independent patch sequences and has no way to know
which came from which channel. Structural, not a penalty term, not a mask -- the
information simply is not present. One line buys the whole property.

A caveat the paper is careful about, and you should be too: channel
independence assumes the channels are parallel series. If some channel
*causally drives* another (an insulin dose driving glucose, say), forbidding
cross-channel attention is a real modelling restriction, not a free win. That
question is exactly what Step 11 revisits on CGM data.
"""

from __future__ import annotations

import torch
from torch import nn

from .encoder import PatchEncoder
from .patching import Patchify, num_patches
from .revin import RevIN


class ChannelIndependentBackbone(nn.Module):
    """RevIN -> patch -> per-channel encoding -> [B, C, N, d_model].

    Everything from raw input up to (but not including) the prediction head.

    Args:
        n_channels: C, number of series.
        seq_len: L, lookback length.
        patch_len: P.
        stride: S.
        d_model, n_heads, n_layers, d_ff, dropout: encoder settings.
        revin: apply reversible instance normalization on the way in.

    side note:
    RevIN is a normalization technique that is reversible,
    meaning that we can undo the normalization by applying the same normalization again.
    this is useful because it allows us to undo the normalization when we need to,
    for example when we need to compute the loss or the metrics.
    it is also useful because it allows us to use the same normalization for the training and the testing.
    """

    def __init__(
        self,
        *,
        n_channels: int,
        seq_len: int = 336,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        revin: bool = True,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.use_revin = bool(revin)

        self.n_patches = num_patches(seq_len, patch_len, stride, pad_end=True)

        self.revin = RevIN(self.n_channels) if self.use_revin else None
        self.patchify = Patchify(patch_len=patch_len, stride=stride, pad_end=True)
        self.encoder = PatchEncoder(
            patch_len=patch_len,
            n_patches=self.n_patches,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, n_channels] -> [B, n_channels, n_patches, d_model]."""
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.n_channels):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.n_channels}], "
                f"got {tuple(x.shape)}"
            )
        batch = x.shape[0]

        if self.revin is not None:
            x = self.revin.normalize(x)

        patches = self.patchify(x)          # [B, C, N, P]

        # THIS IS THE LINE THAT CREATES CHANNEL INDEPENDENCE. After it the
        # encoder sees B*C separate univariate patch sequences and has no way
        # to tell which channel any of them came from -- structural, not a mask
        # or a penalty term.
        #
        # .contiguous() is required: `patches` is still the strided view that
        # Step 3's unfold returned, and reshape needs a real memory layout.
        # Without it: "view size is not compatible with input tensor's size and
        # stride".
        flat = patches.contiguous().reshape(
            batch * self.n_channels, self.n_patches, -1
        )
        encoded = self.encoder(flat)        # [B*C, N, d_model]

        # Unfold the channel axis back out. The argument ORDER matters and the
        # bug is silent: we folded as (batch, channels), so we must unfold in
        # that same order. Swap them and the tensor still has the right shape --
        # only the channels are scrambled, and nothing raises.
        return encoded.reshape(
            batch, self.n_channels, self.n_patches, self.d_model
        )
