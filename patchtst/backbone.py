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

        # THIS IS THE LINE THAT CREATES CHANNEL INDEPENDENCE.
        # the line is:
        # flat = patches.contiguous().reshape(
        #     batch * self.n_channels, self.n_patches, -1
        # )
        # after it the
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

# ● That's the whole answer:

#   ┌───────────────┬─────────────────────────────────────────┐
#   │ Perturb one…  │            Others affected?             │
#   ├───────────────┼─────────────────────────────────────────┤
#   │ batch element │ No — bit-identical. Zero communication. │
#   ├───────────────┼─────────────────────────────────────────┤
#   │ seq position  │ Yes, all of them. Attention mixes them. │
#   └───────────────┴─────────────────────────────────────────┘

#   The batch axis is the one axis with no interaction. The seq axis is where attention lives.

#   So the fold is a choice of which axis to hide channels in

#   You have [B, C, N, P] and must collapse it to 3-D for the encoder. Three options:

#   # ① fold C into BATCH        → attention runs over N patches, per channel
#   patches.reshape(B*C, N, P)          # CHANNEL INDEPENDENT  ← PatchTST

#   # ② fold C into SEQ          → channel 0's patch can attend to channel 3's patch
#   patches.reshape(B, C*N, P)          # channel mixing

#   # ③ fold C into FEATURE      → every token is a blend of all channels
#   patches.permute(...).reshape(B, N, C*P)   # channel mixing

#   Same tensor, same numbers, three different models. Option ② and ③ are roughly what the older Transformers do.

#   Channel independence isn't enforced by a mask or a loss penalty — it's enforced by placement. Once a channel sits in the batch axis, there
#   is no code path by which its values can reach another channel. The attention matrix is [N, N], computed separately for each of the B*C
#   rows. Channel information isn't forbidden from crossing; it's simply not present to cross.

#   That's why it's one line, and why your check_step05 assertion — perturb channel 0, all 6 others bit-identical — comes out exact rather
#   than approximate. It's not "the model learned to mostly ignore other channels." It structurally cannot see them.

#   The flip side, again worth remembering

#   This is also precisely why applying it to CGM is questionable. If meal sits in the batch axis alongside cgm, there is no mechanism by
#   which a meal can inform the glucose forecast — not a weak one, none. On ETTh1 that's an acceptable trade. On CGM you'd be structurally
#   deleting your best predictor.

#   Changing that would mean picking option ② for a subset of channels — which is a real research question, and one you're now equipped to
#   actually implement.


"""
● TL;DR
  
  Channels = variables (columns). ETTh1 has 7 sensors measured on the same hourly clock. Time goes down, channels go across.

  The reshape doesn't change any numbers. [B, C, N, P] → [B*C, N, P] just re-labels: instead of addressing a sequence by (window, channel),
  you address it by row. Same 6 sequences, nothing blended.

  Why that gives channel independence: attention runs over the seq axis, never across the batch axis. Putting channels in the batch axis
  means there's no path for one channel to reach another. Not "the model learns to ignore them" — the information isn't there.

  Same tensor, different axis, different model:

  patches.reshape(B*C, N, P)     # channels in BATCH  → independent  ← PatchTST
  patches.reshape(B, C*N, P)     # channels in SEQ    → mixing

  Matters for your CGM work: channel independence would structurally prevent meal/bolus from informing the cgm forecast. Fine on ETTh1
  (parallel sensors), questionable when channels are causal drivers. That's Step 11.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  What actually changed: only the brackets

  BEFORE  [B=2, C=2, N=2, P=2]           AFTER  [B*C=4, N=2, P=2]

  window 0 ┌ channel 0: [[0,0],[1,1]]     row 0: [[0,0],[1,1]]
           └ channel 1: [[10,10],[11,11]] row 1: [[10,10],[11,11]]
  window 1 ┌ channel 0: [[100,...],...]   row 2: [[100,...],...]
           └ channel 1: [[110,...],...]   row 3: [[110,...],...]

  Before: 2 groups of 2. After: 4 in a flat list. Memory is untouched — same 16 numbers, same order. One layer of nesting was removed.

  Why removing that nesting is the whole idea


"""