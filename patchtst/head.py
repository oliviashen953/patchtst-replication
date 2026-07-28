"""Step 6 — The prediction head.

Paper anchor: Section 4.1 (final paragraph -- "flatten with linear head").

After Step 5 every channel has been encoded into `n_patches` tokens of width
`d_model`. To turn that into a forecast, PatchTST does the simplest possible
thing:

    [B, C, N, d_model]  --flatten last two axes-->  [B, C, N*d_model]
                        --Linear(N*d_model -> H)->  [B, C, H]
                        --transpose-->              [B, H, C]

That is the entire head. Flatten and one linear layer.

Two things are worth noticing about that.

**It is direct multi-step.** All `H` horizon steps come out of one forward pass.
There is no decoder, no autoregression, no teacher forcing, and therefore no
error accumulation over the horizon. This is why Step 1 needed no `label_len`.

**It is deliberately boring.** After an elaborate story about patching and
channel independence, the read-out is a single `nn.Linear`. That is a claim in
itself: if the representation is right, the head does not have to be clever.
The paper is implicitly arguing that everything interesting already happened
upstream.

Individual vs shared heads
--------------------------
`individual=False` (the default) gives every channel the SAME head weights --
consistent with channel independence, where one shared backbone serves all
series. `individual=True` gives each channel its own head, multiplying head
parameters by `C`. The paper exposes both; shared is the default and is what you
want on wide datasets like Traffic (862 channels).
"""

from __future__ import annotations

import torch
from torch import nn


class FlattenHead(nn.Module):
    """Flatten the patch/feature axes and project to the forecast horizon.

    Input : [B, C, n_patches, d_model]
    Output: [B, pred_len, C]      <- note the transpose back to time-major

    Args:
        n_channels: C.
        n_patches: N.
        d_model: encoder width.
        pred_len: H, forecast horizon.
        dropout: applied after flattening.
        individual: one head per channel instead of a shared head.
    """

    def __init__(
        self,
        *,
        n_channels: int,
        n_patches: int,
        d_model: int,
        pred_len: int,
        dropout: float = 0.0,
        individual: bool = False,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_patches = int(n_patches)
        self.d_model = int(d_model)
        self.pred_len = int(pred_len)
        self.individual = bool(individual)
        if min(self.n_channels, self.n_patches, self.d_model, self.pred_len) < 1:
            raise ValueError("all dimensions must be positive")

        self.flatten = nn.Flatten(start_dim=-2)   # [.., N, d_model] -> [.., N*d_model]
        self.dropout = nn.Dropout(dropout)

        in_features = self.n_patches * self.d_model

        if self.individual:
            # One head per channel. Head parameters scale with C.
            self.heads = nn.ModuleList(
                nn.Linear(in_features, self.pred_len) for _ in range(self.n_channels)
            )
        else:
            # One head for EVERY channel -- that is what "shared" means, and it
            # is what keeps the parameter count independent of C.
            self.head = nn.Linear(in_features, self.pred_len)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        expected = (self.n_channels, self.n_patches, self.d_model)
        if encoded.ndim != 4 or encoded.shape[1:] != expected:
            raise ValueError(
                f"expected [batch, {self.n_channels}, {self.n_patches}, "
                f"{self.d_model}], got {tuple(encoded.shape)}"
            )

        flat = self.dropout(self.flatten(encoded))   # [B, C, N*d_model]

        if self.individual:
            # Apply each channel's own head to its own slice, then restack.
            per_channel = [
                self.heads[c](flat[:, c]) for c in range(self.n_channels)
            ]
            out = torch.stack(per_channel, dim=1)    # [B, C, pred_len]
        else:
            # nn.Linear maps the LAST axis, and flat is [B, C, N*d_model], so a
            # single call covers every channel at once -- no loop needed. This
            # one line IS "shared across channels".
            out = self.head(flat)                    # [B, C, pred_len]

        # Return time-major. `out` is [B, C, pred_len] but the rest of the world
        # -- including the Step 1 targets -- expects [B, pred_len, C].
        #
        # This is a nasty bug class whenever C == pred_len: the shape check
        # still passes and only the numbers are wrong. ETTh1 spares us by
        # accident (C=7, pred_len=96), so check_step06 tests it deliberately
        # rather than trusting the shape.
        return out.transpose(1, 2)
